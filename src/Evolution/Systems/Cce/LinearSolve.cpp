// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/LinearSolve.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/StaticCache.hpp"
#include "Utilities/VectorAlgebra.hpp"

namespace Cce {

// This builds up the spectral representation of the matrix associated with the
// linear operator (1 - y) d_y f + 2 f.
// Broadly, this is accomplished by manipulating both the right and left hand
// sides of the Legendre identity:
// (2n + 1) P_n = d_x (P_{n + 1} - P_{n - 1})
// Writing this as operations on the modal coefficients,
//  sum_n (A * m)_n P_n = sum_n (B * m)_n d_x P_n
// So, in particular, we can take advantage of this to obtain a formula for the
// coefficients of the integral given coefficients m_1 of the input:
//   sum_n (m_1)_n P_n = sum_n (B * A^{-1} * m_1)_n d_x P_n,
// Therefore the matrix we wish to act with is B * A^{-1}.
// In this function we calculate the matrix M such that
//   sum_n (m_2)_n P_n = sum_n (M * m_2) ((1 - x) d_x P_n  + 2 * P_n)
// Using
//   sum_n  m_n  (1 + x) P_n  = sum_n (C * m)_n P_n,
// so
//  sum_n (C * A * m)_n P_n = sum_n (B * m)_n ((1 - x) d_x P_n)
// Adding sum_n (2* B * m_n P_n) to both sides,
//   sum_n ((C * A + 2 B ) * m)_n P_n
//        = sum_n (B * m)_n ((1 - x) d_x P_n + 2 * P_n)
// Therefore, the matrix M = B * (C * A + 2 B)^-1 .
static Matrix q_integration_matrix(size_t number_of_points) noexcept {
  Matrix inverse_one_minus_y = Matrix(number_of_points, number_of_points, 0.0);
  for (size_t i = 1; i < number_of_points - 1; ++i) {
    double n = i;
    inverse_one_minus_y(i, i - 1) = -n / (2.0 * n - 1.0);
    inverse_one_minus_y(i, i) = 1.0;
    inverse_one_minus_y(i, i + 1) = -(n + 1.0) / (2.0 * n + 3.0);
  }
  inverse_one_minus_y(0, 0) = 1.0;
  inverse_one_minus_y(0, 1) = -1.0 / 3.0;
  inverse_one_minus_y(number_of_points - 1, number_of_points - 2) =
      -(number_of_points - 1.0) / (2.0 * (number_of_points - 1.0) - 1.0);
  inverse_one_minus_y(number_of_points - 1, number_of_points - 1) = 1.0;

  Matrix indef_int(number_of_points, number_of_points, 0.0);
  for (size_t i = 1; i < number_of_points - 1; ++i) {
    indef_int(i, i - 1) = 1.0;
    indef_int(i, i + 1) = -1.0;
  }
  indef_int(0, 1) = -1.0;
  indef_int(number_of_points - 1, number_of_points - 2) = 1.0;

  Matrix dy_identity_lhs(number_of_points, number_of_points, 0.0);
  for (size_t i = 0; i < number_of_points - 1; ++i) {
    dy_identity_lhs(i, i) = 2.0 * i + 1.0;
  }

  Matrix lhs_mat = inverse_one_minus_y * dy_identity_lhs;

  for (size_t i = 1; i < number_of_points - 1; ++i) {
    lhs_mat(i, i - 1) += 2.0;
    lhs_mat(i, i + 1) += -2.0;
  }
  lhs_mat(0, 1) += -2.0;
  lhs_mat(number_of_points - 1, number_of_points - 2) += 2.0;

  return Spectral::modal_to_nodal_matrix<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
             number_of_points) *
         indef_int * inv(lhs_mat) *
         Spectral::nodal_to_modal_matrix<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
             number_of_points);
}

const Matrix& precomputed_cce_q_integrator(
    size_t number_of_radial_grid_points) noexcept {
  const auto lazy_matrix_cache = make_static_cache<CacheRange<
      1, Spectral::maximum_number_of_points<Spectral::Basis::Legendre> + 1>>(
      [](const size_t generator_number_of_radial_points) noexcept {
        return q_integration_matrix(generator_number_of_radial_points);
      });
  return lazy_matrix_cache(number_of_radial_grid_points);
}

void radial_integrate_pole(
    const gsl::not_null<ComplexDataVector*> integral_result,
    const ComplexDataVector& pole_of_integrand,
    const ComplexDataVector& regular_integrand,
    const ComplexDataVector& boundary, const ComplexDataVector& one_minus_y,
    const size_t l_max) noexcept {
  size_t number_of_radial_grid_points =
      pole_of_integrand.size() /
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  ComplexDataVector integrand =
      pole_of_integrand + one_minus_y * regular_integrand;

  apply_matrices(integral_result,
                 std::array<Matrix, 3>{{Matrix{}, Matrix{},
                                        precomputed_cce_q_integrator(
                                            number_of_radial_grid_points)}},
                 integrand,
                 Spectral::Swsh::swsh_volume_mesh_for_radial_operations(
                     l_max, number_of_radial_grid_points)
                     .extents());

  // apply boundary condition
  const ComplexDataVector boundary_correction =
      .25 * (boundary -
             ComplexDataVector{
                 integral_result->data(),
                 Spectral::Swsh::number_of_swsh_collocation_points(l_max)});
  const ComplexDataVector one_minus_y_squared = pow<2>(
      1.0 -
      std::complex<double>(1.0, 0.0) *
          Spectral::collocation_points<Spectral::Basis::Legendre,
                                       Spectral::Quadrature::GaussLobatto>(
              number_of_radial_grid_points));
  *integral_result += outer_product(boundary_correction, one_minus_y_squared);
}

namespace detail {
void transpose_to_reals_then_imags_radial_stripes(
    const gsl::not_null<DataVector*> result, const ComplexDataVector input,
    const size_t number_of_radial_points,
    const size_t number_of_angular_points) noexcept {
  for (size_t i = 0; i < input.size() * 2; ++i) {
    (*result)[i] = ((i / number_of_radial_points) % 2) == 0
                       ? real(input[number_of_angular_points *
                                        (i % number_of_radial_points) +
                                    i / (2 * number_of_radial_points)])
                       : imag(input[number_of_angular_points *
                                        (i % number_of_radial_points) +
                                    i / (2 * number_of_radial_points)]);
  }
}
}  // namespace detail

}  // namespace Cce
