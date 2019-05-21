// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

// matrix routines prepare the matrix of the desired size, with the appropriate
// boundary conditions, and apply it to the computed right-hand sides. It's
// probably best to just write two routines - one which works for most
// equations, one for the H equation.

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/LinearOperators/IndefiniteIntegral.hpp"
#include "NumericalAlgorithms/LinearOperators/Transpose.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Utilities/Blas.hpp"

extern "C" {
extern void dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);
}

namespace Cce {

/*!
 * \brief Provides access to a lazily cached integration matrix for the \f$Q\f$
 * and \f$W\f$ equations in CCE hypersurface evaluation.
 *
 * \details The provided matrix acts on the integrand collocation points and
 * solves the equation,
 * \f[
 * (1 - y) \partial_y f + 2 f = g,
 * \f]
 * for \f$f\f$ given integrand \f$g\f$.
 */
const Matrix& precomputed_cce_q_integrator(
    size_t number_of_radial_grid_points) noexcept;

/*!
 * \brief A utility function for evaluating the \f$Q\f$ and \f$W\f$ hypersurface
 * integrals during CCE evolution.
 *
 * \details Computes and returns by `not_null` pointer the solution to the
 * equation
 *
 * \f[
 * (1 - y) \partial_y f + 2 f = A + (1 - y) B,
 * \f]
 *
 * where \f$A\f$ is provided as `pole_of_integrand` and \f$B\f$ is provided as
 * `regular_integrand`. The value `one_minus_y` is required for determining the
 * integrand and `l_max` is required to determine the shape of the spin-weighted
 * spherical harmonic mesh.
 */
void radial_integrate_pole(
    const gsl::not_null<ComplexDataVector*> integral_result,
    const ComplexDataVector& pole_of_integrand,
    const ComplexDataVector& regular_integrand,
    const ComplexDataVector& boundary, const ComplexDataVector& one_minus_y,
    const size_t l_max) noexcept;

namespace detail {
// needed because the standard transpose utility cannot create an arbitrary
// ordering of blocks of data. This returns by pointer the configuration useful
// for the linear solve step for H integration
void transpose_to_reals_then_imags_radial_stripes(
    const gsl::not_null<DataVector*> result, const ComplexDataVector input,
    const size_t number_of_radial_points,
    const size_t number_of_angular_points) noexcept;
}  // namespace detail

// @{
/*!
 * \brief Computational structs for evaluating the hypersurface integrals during
 * CCE evolution. These are compatible with use in `db::mutate_apply`.
 *
 * \details
 * The integral evaluated and the corresponding inputs required depend on the
 * CCE quantity being computed. In any of these, the only mutated tag is `Tag`,
 * where the result of the integration is placed. The supported `Tag`s act in
 * the following ways:
 * - If the `Tag` is `Tags::BondiBeta` or `Tags::BondiU`, the integral to be
 * evaluated is simply \f[ \partial_y f = A, \f] where \f$A\f$ is retrieved with
 * `Tags::Integrand<Tag>`.
 * - If the `Tag` is `Tags::BondiQ` or `Tags::BondiW`, the integral to be
 * evaluated is \f[ (1 - y) \partial_y f + 2 f = A + (1 - y) B, \f] where
 * \f$A\f$ is retrieved with `Tags::PoleOfIntegrand<Tag>` and \f$B\f$ is
 * retrieved with `Tags::RegularIntegrand<Tag>`.
 * - If `Tag` is `Tags::BondiH`, the integral to be evaluated is:
 * \f[
 * (1 - y) \partial_y f + L f + L^\prime \bar{f} = A + (1 - y) B,
 * \f]
 * where \f$A\f$ is retrieved with `Tags::PoleOfIntegrand<Tag>`, \f$B\f$ is
 * retrieved with `Tags::RegularIntegrand<Tag>`, \f$L\f$ is retrieved with
 * `Tags::LinearFactor<Tag>`, and \f$L^\prime\f$ is retrieved with
 * `Tags::LinearFactorForConjugate<Tag>`. The presence of \f$L\f$ and
 * \f$L^\prime\f$ ensure that the only current method we have for evaluating the
 * \f$H\f$ hypersurface equation is a direct linear solve, rather than the
 * spectral matrix multiplications which are available for the other integrals.
 *
 * In each case, the boundary value at the world tube for the integration is
 * retrieved from `Tags::BoundaryValue<Tag>`.
 *
 * The additional template parameter `BoundaryPrefix` is to be set to the prefix
 * tag which represents the boundary data you intend to use. This allows the
 * freedom to switch between regularity-preserving boundary data and standard
 * CCE boundary data.
 */
template <template <typename> class BoundaryPrefix, typename Tag>
struct RadialIntegrateBondi {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tag>>;
  using integrand_tags = tmpl::list<Tags::Integrand<Tag>>;

  using return_tags = tmpl::list<Tag>;
  using argument_tags = tmpl::append<integrand_tags, boundary_tags,
                                     tmpl::list<Spectral::Swsh::Tags::LMax>>;
  static void apply(
      const gsl::not_null<Scalar<
          SpinWeighted<ComplexDataVector, db::item_type<Tag>::type::spin>>*>
          integral_result,
      const Scalar<SpinWeighted<ComplexDataVector,
                                db::item_type<Tag>::type::spin>>& integrand,
      const Scalar<SpinWeighted<ComplexDataVector,
                                db::item_type<Tag>::type::spin>>& boundary,
      const size_t l_max) noexcept {
    size_t number_of_radial_grid_points =
        get(integrand).size() /
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    indefinite_integral(make_not_null(&get(*integral_result).data()),
                        get(integrand).data(),
                        Spectral::Swsh::swsh_volume_mesh_for_radial_operations(
                            l_max, number_of_radial_grid_points),
                        2);
    // add in the boundary data to each angular slice
    for (size_t i = 0; i < number_of_radial_grid_points; i++) {
      ComplexDataVector angular_view{
          get(*integral_result).data().data() +
              Spectral::Swsh::number_of_swsh_collocation_points(l_max) * i,
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
      angular_view += get(boundary).data();
    }
  }
};

template <>
struct RadialIntegrateBondi<Tags::BondiQ> {
  using boundary_tags = tmpl::list<Tags::BoundaryValue<Tags::BondiQ>>;
  using integrand_tags = tmpl::list<Tags::PoleOfIntegrand<Tags::BondiQ>,
                                    Tags::RegularIntegrand<Tags::BondiQ>>;

template <template <typename> class BoundaryPrefix>
struct RadialIntegrateBondi<BoundaryPrefix, Tags::BondiQ> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::BondiQ>>;
  using integrand_tags = tmpl::list<Tags::PoleOfIntegrand<Tags::BondiQ>,
                                    Tags::RegularIntegrand<Tags::BondiQ>>;
  using integration_independent_tags = tmpl::list<Tags::OneMinusY>;

  using return_tags = tmpl::list<Tags::BondiQ>;
  using argument_tags =
      tmpl::append<integrand_tags, boundary_tags, integration_independent_tags,
                   tmpl::list<Spectral::Swsh::Tags::LMax>>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          integral_result,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& pole_of_integrand,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& regular_integrand,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& boundary,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const size_t l_max) noexcept {
    radial_integrate_pole(make_not_null(&get(*integral_result).data()),
                          get(pole_of_integrand).data(),
                          get(regular_integrand).data(), get(boundary).data(),
                          get(one_minus_y).data(), l_max);
  }
};

template <template <typename> class BoundaryPrefix>
struct RadialIntegrateBondi<BoundaryPrefix, Tags::BondiW> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::BondiW>>;
  using integrand_tags = tmpl::list<Tags::PoleOfIntegrand<Tags::BondiW>,
                                    Tags::RegularIntegrand<Tags::BondiW>>;
  using integration_independent_tags = tmpl::list<Tags::OneMinusY>;

  using return_tags = tmpl::list<Tags::BondiW>;
  using argument_tags =
      tmpl::append<integrand_tags, boundary_tags, integration_independent_tags,
                   tmpl::list<Spectral::Swsh::Tags::LMax>>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          integral_result,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& pole_of_integrand,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& regular_integrand,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const size_t l_max) noexcept {
    radial_integrate_pole(make_not_null(&get(*integral_result).data()),
                          get(pole_of_integrand).data(),
                          get(regular_integrand).data(), get(boundary).data(),
                          get(one_minus_y).data(), l_max);
  }
};

template <template <typename> class BoundaryPrefix>
struct RadialIntegrateBondi<BoundaryPrefix, Tags::BondiH> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::BondiH>>;
  using integrand_tags =
      tmpl::list<Tags::PoleOfIntegrand<Tags::BondiH>,
                 Tags::RegularIntegrand<Tags::BondiH>,
                 Tags::LinearFactor<Tags::BondiH>,
                 Tags::LinearFactorForConjugate<Tags::BondiH>>;
  using integration_independent_tags = tmpl::list<Tags::OneMinusY>;

  using return_tags = tmpl::list<Tags::BondiH>;
  using argument_tags =
      tmpl::append<integrand_tags, boundary_tags, integration_independent_tags,
                   tmpl::list<Spectral::Swsh::Tags::LMax>>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          integral_result,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& pole_of_integrand,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& regular_integrand,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& linear_factor,
      const Scalar<SpinWeighted<ComplexDataVector, 4>>&
          linear_factor_of_conjugate,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const size_t l_max) noexcept {
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points =
        get(pole_of_integrand).size() / number_of_angular_points;

    Matrix operator_matrix(2 * number_of_radial_points,
                           2 * number_of_radial_points);

    ComplexDataVector integrand =
        get(pole_of_integrand).data() +
        get(one_minus_y).data() * get(regular_integrand).data();

    DataVector transpose_buffer{2 * get(pole_of_integrand).size()};
    DataVector linear_solve_buffer{2 * get(pole_of_integrand).size()};

    // transpose such that each radial slice is
    // (real radial slice 00) (imag radial slice 00) (real radial slice 01) ...
    detail::transpose_to_reals_then_imags_radial_stripes(
        make_not_null(&linear_solve_buffer), integrand, number_of_radial_points,
        number_of_angular_points);

    raw_transpose(make_not_null(transpose_buffer.data()),
                  linear_solve_buffer.data(), number_of_radial_points,
                  2 * number_of_angular_points);

    auto mesh = Mesh<1>{{{number_of_radial_points}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
    const auto& derivative_matrix =
        Spectral::differentiation_matrix(mesh.slice_through(0));
    for (size_t offset = 0; offset < number_of_angular_points; ++offset) {
      // on repeated evaluations, the matrix gets permuted by the dgesv routine.
      // We'll ignore its pivots and just overwrite the whole thing on each
      // pass. There are probably optimizations that can be made which make use
      // of the pivots.
      for (size_t chunk = 0; chunk < 2; ++chunk) {
        for (size_t i = 0; i < number_of_radial_points; ++i) {
          for (size_t j = 0; j < number_of_radial_points; ++j) {
            operator_matrix(i + chunk * number_of_radial_points,
                            j + chunk * number_of_radial_points) =
                derivative_matrix(i, j) *
                real(get(one_minus_y).data()[i * number_of_angular_points]);
          }
        }
      }
      for (size_t i = 0; i < number_of_radial_points; ++i) {
        for (size_t j = 0; j < number_of_radial_points; ++j) {
          operator_matrix(i + number_of_radial_points, j) = 0.0;
          operator_matrix(i, j + number_of_radial_points) = 0.0;
        }
      }

      for (size_t i = 0; i < number_of_radial_points; ++i) {
        // upper left
        operator_matrix(i, i) += real(
            get(linear_factor).data()[offset + i * number_of_angular_points] +
            get(linear_factor_of_conjugate)
                .data()[offset + i * number_of_angular_points]);
        operator_matrix(0, i) = 0.0;
        // upper right
        operator_matrix(i, number_of_radial_points + i) -= imag(
            get(linear_factor).data()[offset + i * number_of_angular_points] -
            get(linear_factor_of_conjugate)
                .data()[offset + i * number_of_angular_points]);
        operator_matrix(0, number_of_radial_points + i) = 0.0;
        // lower left
        operator_matrix(number_of_radial_points + i, i) += imag(
            get(linear_factor).data()[offset + i * number_of_angular_points] +
            get(linear_factor_of_conjugate)
                .data()[offset + i * number_of_angular_points]);
        operator_matrix(number_of_radial_points, i) = 0.0;
        // lower right
        operator_matrix(number_of_radial_points + i,
                        number_of_radial_points + i) +=
            real(get(linear_factor)
                     .data()[offset + i * number_of_angular_points] -
                 get(linear_factor_of_conjugate)
                     .data()[offset + i * number_of_angular_points]);
        operator_matrix(number_of_radial_points, number_of_radial_points + i) =
            0.0;
      }
      operator_matrix(0, 0) = 1.0;
      operator_matrix(number_of_radial_points, number_of_radial_points) = 1.0;
      // put the data currently in integrand into a real DataVector of twice the
      // length
      linear_solve_buffer[offset * 2 * number_of_radial_points] =
          real(get(boundary).data()[offset]);
      linear_solve_buffer[(offset * 2 + 1) * number_of_radial_points] =
          imag(get(boundary).data()[offset]);
      std::vector<int> ipiv(2 * number_of_radial_points);
      int twice_size = 2 * number_of_radial_points;
      int info = 0;
      int one = 1;

      dgesv_(&twice_size, &one, operator_matrix.data(), &twice_size,
             ipiv.data(),
             linear_solve_buffer.data() + offset * 2 * number_of_radial_points,
             &twice_size, &info);
    }
    raw_transpose(make_not_null(reinterpret_cast<double*>(
                      get(*integral_result).data().data())),
                  linear_solve_buffer.data(), number_of_radial_points,
                  2 * number_of_angular_points);
  }
};
// @}
}  // namespace Cce
