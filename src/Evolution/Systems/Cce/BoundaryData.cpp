// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "ApparentHorizons/SpherepackIterator.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "NumericalAlgorithms/LinearOperators/Transpose.hpp"

namespace Cce {

void trigonometric_functions_on_swsh_collocation(
    const gsl::not_null<Scalar<DataVector>*> cos_phi,
    const gsl::not_null<Scalar<DataVector>*> cos_theta,
    const gsl::not_null<Scalar<DataVector>*> sin_phi,
    const gsl::not_null<Scalar<DataVector>*> sin_theta,
    const size_t l_max) noexcept {
  check_and_resize(make_not_null(&get(*cos_phi)),
                   Spectral::Swsh::number_of_swsh_collocation_points(l_max));
  check_and_resize(make_not_null(&get(*cos_theta)),
                   Spectral::Swsh::number_of_swsh_collocation_points(l_max));
  check_and_resize(make_not_null(&get(*sin_phi)),
                   Spectral::Swsh::number_of_swsh_collocation_points(l_max));
  check_and_resize(make_not_null(&get(*sin_theta)),
                   Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  const auto& collocation = Spectral::Swsh::precomputed_collocation<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    get(*sin_theta)[collocation_point.offset] = sin(collocation_point.theta);
    get(*cos_theta)[collocation_point.offset] = cos(collocation_point.theta);
    get(*sin_phi)[collocation_point.offset] = sin(collocation_point.phi);
    get(*cos_phi)[collocation_point.offset] = cos(collocation_point.phi);
  }
}

void cartesian_to_angular_coordinates_and_derivatives(
    const gsl::not_null<tnsr::I<DataVector, 3>*> cartesian_coords,
    const gsl::not_null<tnsr::iJ<DataVector, 3>*> cartesian_to_angular_jacobian,
    const gsl::not_null<tnsr::iJ<DataVector, 3>*>
        inverse_cartesian_to_angular_jacobian,
    const Scalar<DataVector>& cos_phi, const Scalar<DataVector>& cos_theta,
    const Scalar<DataVector>& sin_phi, const Scalar<DataVector>& sin_theta,
    const double extraction_radius) noexcept {
  check_and_resize(cartesian_coords, get(cos_phi).size());
  check_and_resize(cartesian_to_angular_jacobian, get(cos_phi).size());
  check_and_resize(inverse_cartesian_to_angular_jacobian, get(cos_phi).size());

  // note: factor of r scaled out
  get<0>(*cartesian_coords) = get(sin_theta) * get(cos_phi);
  get<1>(*cartesian_coords) = get(sin_theta) * get(sin_phi);
  get<2>(*cartesian_coords) = get(cos_theta);

  // dx/dr   dy/dr  dz/dr
  get<0, 0>(*cartesian_to_angular_jacobian) = get(sin_theta) * get(cos_phi);
  get<0, 1>(*cartesian_to_angular_jacobian) = get(sin_theta) * get(sin_phi);
  get<0, 2>(*cartesian_to_angular_jacobian) = get(cos_theta);
  // dx/dtheta   dy/dtheta  dz/dtheta
  get<1, 0>(*cartesian_to_angular_jacobian) =
      extraction_radius * get(cos_theta) * get(cos_phi);
  get<1, 1>(*cartesian_to_angular_jacobian) =
      extraction_radius * get(cos_theta) * get(sin_phi);
  get<1, 2>(*cartesian_to_angular_jacobian) =
      -extraction_radius * get(sin_theta);
  // (1/sin(theta)) { dx/dphi,   dy/dphi,  dz/dphi }
  get<2, 0>(*cartesian_to_angular_jacobian) = -extraction_radius * get(sin_phi);
  get<2, 1>(*cartesian_to_angular_jacobian) = extraction_radius * get(cos_phi);
  get<2, 2>(*cartesian_to_angular_jacobian) = 0.0;

  // dr/dx   dtheta/dx   dphi/dx * sin(theta)
  get<0, 0>(*inverse_cartesian_to_angular_jacobian) =
      get(cos_phi) * get(sin_theta);
  get<0, 1>(*inverse_cartesian_to_angular_jacobian) =
      get(cos_phi) * get(cos_theta) / extraction_radius;
  get<0, 2>(*inverse_cartesian_to_angular_jacobian) =
      -get(sin_phi) / (extraction_radius);
  // dr/dy   dtheta/dy   dphi/dy * sin(theta)
  get<1, 0>(*inverse_cartesian_to_angular_jacobian) =
      get(sin_phi) * get(sin_theta);
  get<1, 1>(*inverse_cartesian_to_angular_jacobian) =
      get(cos_theta) * get(sin_phi) / extraction_radius;
  get<1, 2>(*inverse_cartesian_to_angular_jacobian) =
      get(cos_phi) / (extraction_radius);
  // dr/dz   dtheta/dz   dphi/dz * sin(theta)
  get<2, 0>(*inverse_cartesian_to_angular_jacobian) = get(cos_theta);
  get<2, 1>(*inverse_cartesian_to_angular_jacobian) =
      -get(sin_theta) / extraction_radius;
  get<2, 2>(*inverse_cartesian_to_angular_jacobian) = 0.0;
}

void cartesian_spatial_metric_and_derivatives(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> cartesian_spatial_metric,
    const gsl::not_null<tnsr::II<DataVector, 3>*> inverse_spatial_metric,
    const gsl::not_null<tnsr::ijj<DataVector, 3>*> d_cartesian_spatial_metric,
    const gsl::not_null<tnsr::ii<DataVector, 3>*> dt_cartesian_spatial_metric,
    const tnsr::ii<DataVector, 3>& spatial_metric_coefficients,
    const tnsr::ii<DataVector, 3>& dr_spatial_metric_coefficients,
    const tnsr::ii<DataVector, 3>& dt_spatial_metric_coefficients,
    const tnsr::iJ<DataVector, 3>& inverse_cartesian_to_angular_jacobian,
    const tnsr::I<DataVector, 3>& cartesian_coords,
    const YlmSpherepack& spherical_harmonics, const bool radial_renormalize,
    const size_t l_max) noexcept {
  size_t size = get<0, 0>(inverse_cartesian_to_angular_jacobian).size();
  check_and_resize(cartesian_spatial_metric, size);
  check_and_resize(d_cartesian_spatial_metric, size);
  check_and_resize(dt_cartesian_spatial_metric, size);

  // It is assumed at this point that the coefficients are provided in the ylm
  // spherepack format, as prepared by the utility in `ReadBoundaryDataH5.hpp`.

  // Allocations
  tnsr::ijj<DataVector, 3> angular_d_cartesian_spatial_metric_gauss_legendre{
      spherical_harmonics.theta_points().size() *
      spherical_harmonics.phi_points().size()};
  tnsr::ijj<DataVector, 3> angular_d_cartesian_spatial_metric{size};
  tnsr::ii<DataVector, 3> dr_cartesian_spatial_metric{size};
  DataVector transpose_buffer{size};
  std::vector<double> interp_result(size);

  const auto& collocation = Spectral::Swsh::precomputed_collocation<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);

  std::vector<std::array<double, 2>> target_points =
      std::vector<std::array<double, 2>>(
          Spectral::Swsh::number_of_swsh_collocation_points(l_max));
  for (const auto& point : collocation) {
    target_points[point.offset] = make_array(point.theta, point.phi);
  }
  auto interpolation_info =
      spherical_harmonics.set_up_interpolation_info(target_points);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      // SpEC worldtube data is stored from Spherepack, so it is most consistent
      // to use Spherepack (rather than libsharp) to extract that data.
      spherical_harmonics.interpolate_from_coefs(
          make_not_null(&interp_result), spatial_metric_coefficients.get(i, j),
          interpolation_info);
      for (size_t v = 0; v < interp_result.size(); ++v) {
        cartesian_spatial_metric->get(i, j)[v] = interp_result[v];
      }

      spherical_harmonics.interpolate_from_coefs(
          make_not_null(&interp_result),
          dt_spatial_metric_coefficients.get(i, j), interpolation_info);
      std::copy(interp_result.begin(), interp_result.end(),
                dt_cartesian_spatial_metric->get(i, j).begin());

      spherical_harmonics.interpolate_from_coefs(
          make_not_null(&interp_result),
          dr_spatial_metric_coefficients.get(i, j), interpolation_info);
      std::copy(interp_result.begin(), interp_result.end(),
                dr_cartesian_spatial_metric.get(i, j).begin());

      ComplexDataVector eth_of_component =
          Spectral::Swsh::swsh_derivative<0, Spectral::Swsh::Tags::Eth>(
              std::complex<double>(1.0, 0.0) *
                  cartesian_spatial_metric->get(i, j),
              l_max);
      angular_d_cartesian_spatial_metric.get(1, i, j) = -real(eth_of_component);
      angular_d_cartesian_spatial_metric.get(2, i, j) = -imag(eth_of_component);
    }
  }

  *inverse_spatial_metric =
      determinant_and_inverse(*cartesian_spatial_metric).second;

  // Some SpEC worldtube data has an incorrectly normalized radial derivative,
  // which must be fixed
  DataVector correction_factor{size, 0.0};
  if (radial_renormalize) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        correction_factor += inverse_spatial_metric->get(i, j) *
                             cartesian_coords.get(i) * cartesian_coords.get(j);
      }
    }
    correction_factor = sqrt(correction_factor);
  } else {
    correction_factor = 1.0;
  }

  // convert derivatives to cartesian form
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        d_cartesian_spatial_metric->get(k, i, j) =
            correction_factor *
            inverse_cartesian_to_angular_jacobian.get(k, 0) *
            dr_cartesian_spatial_metric.get(i, j);
        for (size_t A = 0; A < 2; ++A) {
          d_cartesian_spatial_metric->get(k, i, j) +=
              inverse_cartesian_to_angular_jacobian.get(k, A + 1) *
              angular_d_cartesian_spatial_metric.get(A + 1, i, j);
        }
      }
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < i; ++j) {
      cartesian_spatial_metric->get(i, j) = cartesian_spatial_metric->get(j, i);
      dt_cartesian_spatial_metric->get(i, j) =
          dt_cartesian_spatial_metric->get(j, i);
      for (size_t k = 0; k < 3; ++k) {
        d_cartesian_spatial_metric->get(k, i, j) =
            d_cartesian_spatial_metric->get(k, j, i);
      }
    }
  }
}

void cartesian_shift_and_derivatives(
    gsl::not_null<tnsr::I<DataVector, 3>*> cartesian_shift,
    gsl::not_null<tnsr::iJ<DataVector, 3>*> d_cartesian_shift,
    gsl::not_null<tnsr::I<DataVector, 3>*> dt_cartesian_shift,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const tnsr::I<DataVector, 3>& shift_coefficients,
    const tnsr::I<DataVector, 3>& dr_shift_coefficients,
    const tnsr::I<DataVector, 3>& dt_shift_coefficients,
    const tnsr::iJ<DataVector, 3>& inverse_cartesian_to_angular_jacobian,
    const tnsr::I<DataVector, 3>& cartesian_coords,
    const YlmSpherepack& spherical_harmonics, const bool radial_renormalize,
    const size_t l_max) noexcept {
  size_t size = get<0, 0>(inverse_cartesian_to_angular_jacobian).size();
  check_and_resize(cartesian_shift, size);
  check_and_resize(d_cartesian_shift, size);
  check_and_resize(dt_cartesian_shift, size);

  // Allocations
  tnsr::iJ<DataVector, 3> angular_d_cartesian_shift_gauss_legendre{
      spherical_harmonics.theta_points().size() *
      spherical_harmonics.phi_points().size()};
  tnsr::iJ<DataVector, 3> angular_d_cartesian_shift{size};
  tnsr::I<DataVector, 3> dr_cartesian_shift{size};

  DataVector transpose_buffer{size};

  std::vector<double> interp_result(size);

  const auto& collocation = Spectral::Swsh::precomputed_collocation<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);

  std::vector<std::array<double, 2>> target_points =
      std::vector<std::array<double, 2>>(
          Spectral::Swsh::number_of_swsh_collocation_points(l_max));
  for (const auto& point : collocation) {
    target_points[point.offset] = make_array(point.theta, point.phi);
  }
  auto interpolation_info =
      spherical_harmonics.set_up_interpolation_info(target_points);

  for (size_t i = 0; i < 3; ++i) {
    spherical_harmonics.interpolate_from_coefs(make_not_null(&interp_result),
                                               shift_coefficients.get(i),
                                               interpolation_info);
    std::copy(interp_result.begin(), interp_result.end(),
              cartesian_shift->get(i).begin());

    spherical_harmonics.interpolate_from_coefs(make_not_null(&interp_result),
                                               dr_shift_coefficients.get(i),
                                               interpolation_info);
    std::copy(interp_result.begin(), interp_result.end(),
              dr_cartesian_shift.get(i).begin());

    spherical_harmonics.interpolate_from_coefs(make_not_null(&interp_result),
                                               dt_shift_coefficients.get(i),
                                               interpolation_info);
    std::copy(interp_result.begin(), interp_result.end(),
              dt_cartesian_shift->get(i).begin());

    ComplexDataVector eth_of_component =
        Spectral::Swsh::swsh_derivative<0, Spectral::Swsh::Tags::Eth>(
            std::complex<double>(1.0, 0.0) * cartesian_shift->get(i), l_max);
    angular_d_cartesian_shift.get(1, i) = -real(eth_of_component);
    angular_d_cartesian_shift.get(2, i) = -imag(eth_of_component);
  }

  // Apply possible correction factor to improperly normalized SpEC worldtube
  // data
  DataVector correction_factor{size, 0.0};
  if (radial_renormalize) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        correction_factor += inverse_spatial_metric.get(i, j) *
                             cartesian_coords.get(i) * cartesian_coords.get(j);
      }
    }
    correction_factor = sqrt(correction_factor);
  } else {
    correction_factor = 1.0;
  }

  // convert derivatives to cartesian form
  for (size_t i = 0; i < 3; ++i) {
    for (size_t k = 0; k < 3; ++k) {
      d_cartesian_shift->get(k, i) =
          correction_factor * inverse_cartesian_to_angular_jacobian.get(k, 0) *
          dr_cartesian_shift.get(i);
      for (size_t A = 0; A < 2; ++A) {
        d_cartesian_shift->get(k, i) +=
            inverse_cartesian_to_angular_jacobian.get(k, A + 1) *
            angular_d_cartesian_shift.get(A + 1, i);
      }
    }
  }
}

void cartesian_lapse_and_derivatives(
    const gsl::not_null<Scalar<DataVector>*> cartesian_lapse,
    const gsl::not_null<tnsr::i<DataVector, 3>*> d_cartesian_lapse,
    const gsl::not_null<Scalar<DataVector>*> dt_cartesian_lapse,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const Scalar<DataVector>& lapse_coefficients,
    const Scalar<DataVector>& dr_lapse_coefficients,
    const Scalar<DataVector>& dt_lapse_coefficients,
    const tnsr::iJ<DataVector, 3>& inverse_cartesian_to_angular_jacobian,
    const tnsr::I<DataVector, 3>& cartesian_coords,
    const YlmSpherepack& spherical_harmonics, const bool radial_renormalize,
    const size_t l_max) noexcept {
  size_t size = get<0, 0>(inverse_cartesian_to_angular_jacobian).size();
  check_and_resize(cartesian_lapse, size);
  check_and_resize(d_cartesian_lapse, size);
  check_and_resize(dt_cartesian_lapse, size);

  // Allocations
  tnsr::i<DataVector, 3> angular_d_cartesian_lapse_gauss_legendre{
      spherical_harmonics.theta_points().size() *
      spherical_harmonics.phi_points().size()};
  tnsr::i<DataVector, 3> angular_d_cartesian_lapse{size};
  Scalar<DataVector> dr_cartesian_lapse{size};

  DataVector transpose_buffer{size};

  std::vector<double> interp_result(size);

  const auto& collocation = Spectral::Swsh::precomputed_collocation<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);

  std::vector<std::array<double, 2>> target_points =
      std::vector<std::array<double, 2>>(
          Spectral::Swsh::number_of_swsh_collocation_points(l_max));
  for (const auto& point : collocation) {
    target_points[point.offset] = make_array(point.theta, point.phi);
  }
  auto interpolation_info =
      spherical_harmonics.set_up_interpolation_info(target_points);

  spherical_harmonics.interpolate_from_coefs(make_not_null(&interp_result),
                                             get(lapse_coefficients),
                                             interpolation_info);
  std::copy(interp_result.begin(), interp_result.end(),
            get(*cartesian_lapse).begin());

  spherical_harmonics.interpolate_from_coefs(make_not_null(&interp_result),
                                             get(dr_lapse_coefficients),
                                             interpolation_info);
  std::copy(interp_result.begin(), interp_result.end(),
            get(dr_cartesian_lapse).begin());

  spherical_harmonics.interpolate_from_coefs(make_not_null(&interp_result),
                                             get(dt_lapse_coefficients),
                                             interpolation_info);
  std::copy(interp_result.begin(), interp_result.end(),
            get(*dt_cartesian_lapse).begin());

  ComplexDataVector eth_of_component =
      Spectral::Swsh::swsh_derivative<0, Spectral::Swsh::Tags::Eth>(
          std::complex<double>(1.0, 0.0) * get(*cartesian_lapse), l_max);
  angular_d_cartesian_lapse.get(1) = -real(eth_of_component);
  angular_d_cartesian_lapse.get(2) = -imag(eth_of_component);

  // Apply possible correction factor to improperly normalized SpEC worldtube
  // data
  DataVector correction_factor{size, 0.0};
  if (radial_renormalize) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        correction_factor += inverse_spatial_metric.get(i, j) *
                             cartesian_coords.get(i) * cartesian_coords.get(j);
      }
    }
    correction_factor = sqrt(correction_factor);
  } else {
    correction_factor = 1.0;
  }

  // convert derivatives to cartesian form
  for (size_t k = 0; k < 3; ++k) {
    d_cartesian_lapse->get(k) =
        correction_factor * inverse_cartesian_to_angular_jacobian.get(k, 0) *
        get(dr_cartesian_lapse);
    for (size_t A = 0; A < 2; ++A) {
      d_cartesian_lapse->get(k) +=
          inverse_cartesian_to_angular_jacobian.get(k, A + 1) *
          angular_d_cartesian_lapse.get(A + 1);
    }
  }
}

void generalized_harmonic_quantities(
    const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi,
    const gsl::not_null<tnsr::aa<DataVector, 3>*> dt_psi,
    const gsl::not_null<tnsr::aa<DataVector, 3>*> psi,
    const gsl::not_null<tnsr::II<DataVector, 3>*> inverse_spatial_metric,
    const tnsr::ii<DataVector, 3>& cartesian_spatial_metric,
    const tnsr::ijj<DataVector, 3>& d_cartesian_spatial_metric,
    const tnsr::ii<DataVector, 3>& dt_cartesian_spatial_metric,
    const tnsr::I<DataVector, 3>& cartesian_shift,
    const tnsr::iJ<DataVector, 3>& d_cartesian_shift,
    const tnsr::I<DataVector, 3>& dt_cartesian_shift,
    const Scalar<DataVector>& cartesian_lapse,
    const tnsr::i<DataVector, 3>& d_cartesian_lapse,
    const Scalar<DataVector>& dt_cartesian_lapse) noexcept {
  check_and_resize(dt_psi, get(dt_cartesian_lapse).size());

  GeneralizedHarmonic::phi(
      phi, cartesian_lapse, d_cartesian_lapse, cartesian_shift,
      d_cartesian_shift, cartesian_spatial_metric, d_cartesian_spatial_metric);

  get<0, 0>(*dt_psi) = -2.0 * get(cartesian_lapse) * get(dt_cartesian_lapse);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get<0, 0>(*dt_psi) +=
          dt_cartesian_spatial_metric.get(i, j) * cartesian_shift.get(i) *
              cartesian_shift.get(j) +
          2.0 * cartesian_spatial_metric.get(i, j) * cartesian_shift.get(i) *
              dt_cartesian_shift.get(j);
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    dt_psi->get(0, i + 1) = 0.0;
    for (size_t j = 0; j < 3; ++j) {
      dt_psi->get(0, i + 1) +=
          dt_cartesian_spatial_metric.get(i, j) * cartesian_shift.get(j) +
          cartesian_spatial_metric.get(i, j) * dt_cartesian_shift.get(j);
    }
    dt_psi->get(i + 1, 0) = dt_psi->get(0, i + 1);
  }

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      dt_psi->get(i + 1, j + 1) = dt_cartesian_spatial_metric.get(i, j);
    }
  }

  gr::spacetime_metric(psi, cartesian_lapse, cartesian_shift,
                       cartesian_spatial_metric);

  *inverse_spatial_metric =
      determinant_and_inverse(cartesian_spatial_metric).second;
}

void null_metric_and_derivative(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> du_null_metric,
    const gsl::not_null<tnsr::aa<DataVector, 3>*> null_metric,
    const tnsr::iJ<DataVector, 3>& cartesian_to_angular_jacobian,
    const tnsr::aa<DataVector, 3>& dt_psi,
    const tnsr::aa<DataVector, 3>& psi) noexcept {
  if (UNLIKELY((get<0, 0>(*null_metric).size() != get<0, 0>(psi).size()))) {
    *null_metric = tnsr::aa<DataVector, 3>{get<0, 0>(psi).size()};
  }
  if (UNLIKELY((get<0, 0>(*du_null_metric).size() != get<0, 0>(psi).size()))) {
    *du_null_metric = tnsr::aa<DataVector, 3>{get<0, 0>(psi).size()};
  }

  get<0, 0>(*null_metric) = get<0, 0>(psi);
  get<0, 0>(*du_null_metric) = get<0, 0>(dt_psi);

  get<0, 1>(*null_metric) = -1.0;
  get<0, 1>(*du_null_metric) = 0.0;

  for (size_t i = 0; i < 3; ++i) {
    null_metric->get(1, i + 1) = 0.0;
    du_null_metric->get(1, i + 1) = 0.0;
  }

  for (size_t A = 0; A < 2; ++A) {
    null_metric->get(0, A + 2) =
        cartesian_to_angular_jacobian.get(A + 1, 0) * psi.get(0, 1);
    du_null_metric->get(0, A + 2) =
        cartesian_to_angular_jacobian.get(A + 1, 0) * dt_psi.get(0, 1);
    for (size_t i = 1; i < 3; ++i) {
      null_metric->get(0, A + 2) +=
          cartesian_to_angular_jacobian.get(A + 1, i) * psi.get(0, i + 1);
      du_null_metric->get(0, A + 2) +=
          cartesian_to_angular_jacobian.get(A + 1, i) * dt_psi.get(0, i + 1);
    }
  }

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = A; B < 2; ++B) {
      null_metric->get(A + 2, B + 2) =
          cartesian_to_angular_jacobian.get(A + 1, 0) *
          cartesian_to_angular_jacobian.get(B + 1, 0) * psi.get(1, 1);
      du_null_metric->get(A + 2, B + 2) =
          cartesian_to_angular_jacobian.get(A + 1, 0) *
          cartesian_to_angular_jacobian.get(B + 1, 0) * dt_psi.get(1, 1);

      for (size_t i = 1; i < 3; ++i) {
        null_metric->get(A + 2, B + 2) +=
            cartesian_to_angular_jacobian.get(A + 1, i) *
            cartesian_to_angular_jacobian.get(B + 1, i) * psi.get(i + 1, i + 1);
        du_null_metric->get(A + 2, B + 2) +=
            cartesian_to_angular_jacobian.get(A + 1, i) *
            cartesian_to_angular_jacobian.get(B + 1, i) *
            dt_psi.get(i + 1, i + 1);
      }

      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i + 1; j < 3; ++j) {
          // note: this can be simplified from symmetry, but then Pypp doesn't
          // work
          null_metric->get(A + 2, B + 2) +=
              (cartesian_to_angular_jacobian.get(A + 1, i) *
                   cartesian_to_angular_jacobian.get(B + 1, j) +
               cartesian_to_angular_jacobian.get(A + 1, j) *
                   cartesian_to_angular_jacobian.get(B + 1, i)) *
              psi.get(i + 1, j + 1);
          du_null_metric->get(A + 2, B + 2) +=
              (cartesian_to_angular_jacobian.get(A + 1, i) *
                   cartesian_to_angular_jacobian.get(B + 1, j) +
               cartesian_to_angular_jacobian.get(A + 1, j) *
                   cartesian_to_angular_jacobian.get(B + 1, i)) *
              dt_psi.get(i + 1, j + 1);
        }
      }
    }
  }

  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b < a; ++b) {
      null_metric->get(a, b) = null_metric->get(b, a);
      du_null_metric->get(a, b) = du_null_metric->get(b, a);
    }
  }
}

void inverse_null_metric(
    const gsl::not_null<tnsr::AA<DataVector, 3>*> inverse_null_metric,
    const tnsr::aa<DataVector, 3>& null_metric) noexcept {
  if (UNLIKELY((get<0, 0>(*inverse_null_metric).size() !=
                get<0, 0>(null_metric).size()))) {
    *inverse_null_metric =
        tnsr::AA<DataVector, 3>{get<0, 0>(null_metric).size()};
  }
  // The angular part is invertible on its own, so we just do that manually
  // Allocation

  *inverse_null_metric = determinant_and_inverse(null_metric).second;
  // auto angular_determinant = DataVector{get<0, 0>(null_metric).size()};
  // angular_determinant = get<2, 2>(null_metric) * get<3, 3>(null_metric) -
  // get<2, 3>(null_metric) * get<3, 2>(null_metric);

  // get<2, 2>(*inverse_null_metric) =
  // get<3, 3>(null_metric) / angular_determinant;
  // get<3, 3>(*inverse_null_metric) =
  // get<2, 2>(null_metric) / angular_determinant;
  // get<2, 3>(*inverse_null_metric) =
  // -get<2, 3>(null_metric) / angular_determinant;

  // get<0, 1>(*inverse_null_metric) = -1.0;

  // get<0, 0>(*inverse_null_metric) = 0.0;
  // get<0, 2>(*inverse_null_metric) = 0.0;
  // get<0, 3>(*inverse_null_metric) = 0.0;

  // get<1, 2>(*inverse_null_metric) =
  // get<2, 2>(*inverse_null_metric) * get<2, 0>(null_metric) +
  // get<2, 3>(*inverse_null_metric) * get<3, 0>(null_metric);
  // get<1, 3>(*inverse_null_metric) =
  // get<2, 3>(*inverse_null_metric) * get<2, 0>(null_metric) +
  // get<3, 3>(*inverse_null_metric) * get<3, 0>(null_metric);

  // get<1, 1>(*inverse_null_metric) =
  // -get<0, 0>(null_metric) +
  // get<1, 2>(*inverse_null_metric) * get<2, 0>(null_metric) +
  // get<1, 3>(*inverse_null_metric) * get<3, 0>(null_metric);
  // for (size_t a = 0; a < 4; ++a) {
  // for (size_t b = 0; b < a; ++b) {
  // inverse_null_metric->get(a, b) = inverse_null_metric->get(b, a);
  // }
  // }
}

void worldtube_normal_and_derivatives(
    const gsl::not_null<tnsr::iJ<DataVector, 3>*> angular_d_worldtube_normal,
    const gsl::not_null<tnsr::I<DataVector, 3>*> worldtube_normal,
    const gsl::not_null<tnsr::I<DataVector, 3>*> dt_worldtube_normal,
    const tnsr::iJ<DataVector, 3>& cartesian_to_angular_jacobian,
    const Scalar<DataVector>& cos_phi, const Scalar<DataVector>& cos_theta,
    const tnsr::iaa<DataVector, 3>& phi, const tnsr::aa<DataVector, 3>& psi,
    const tnsr::aa<DataVector, 3>& dt_psi, const Scalar<DataVector>& sin_phi,
    const Scalar<DataVector>& sin_theta,
    const tnsr::II<DataVector, 3> inverse_spatial_metric) noexcept {
  size_t size = get<0, 0>(psi).size();
  if (UNLIKELY((get<0, 0>(*angular_d_worldtube_normal).size() != size))) {
    *angular_d_worldtube_normal = tnsr::iJ<DataVector, 3>{get(cos_phi).size()};
  }
  if (UNLIKELY(get<0>(*worldtube_normal).size() != size)) {
    *worldtube_normal = tnsr::I<DataVector, 3>{get(cos_phi).size()};
  }
  if (UNLIKELY(get<0>(*dt_worldtube_normal).size() != size)) {
    *dt_worldtube_normal = tnsr::I<DataVector, 3>{get(cos_phi).size()};
  }

  // Allocation
  tnsr::i<DataVector, 3> sigma{size};
  get<0>(sigma) = get(cos_phi) * square(get(sin_theta));
  get<1>(sigma) = get(sin_phi) * square(get(sin_theta));
  get<2>(sigma) = get(sin_theta) * get(cos_theta);

  // Allocation
  DataVector norm_of_sigma{size};
  norm_of_sigma = square(get<0>(sigma)) * get<0, 0>(inverse_spatial_metric);
  for (size_t i = 1; i < 3; ++i) {
    norm_of_sigma += square(sigma.get(i)) * inverse_spatial_metric.get(i, i);
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i + 1; j < 3; ++j) {
      norm_of_sigma +=
          2.0 * sigma.get(i) * sigma.get(j) * inverse_spatial_metric.get(i, j);
    }
  }
  norm_of_sigma = sqrt(norm_of_sigma);

  get<0>(sigma) /= norm_of_sigma;
  get<1>(sigma) /= norm_of_sigma;
  get<2>(sigma) /= norm_of_sigma;

  for (size_t i = 0; i < 3; ++i) {
    worldtube_normal->get(i) = inverse_spatial_metric.get(i, 0) * sigma.get(0);
    for (size_t j = 1; j < 3; ++j) {
      worldtube_normal->get(i) +=
          inverse_spatial_metric.get(i, j) * sigma.get(j);
    }
  }

  // Allocation
  // tnsr::iJ<DataVector, 3> angular_d_sigma{get<0, 0>(psi).size()};
  // get<0, 0>(angular_d_sigma) = 0.0;
  // get<0, 1>(angular_d_sigma) = 0.0;
  // get<0, 2>(angular_d_sigma) = 0.0;

  // get<1, 0>(angular_d_sigma) =
  //     2.0 * get(cos_phi) * get(cos_theta) * get(sin_theta) / norm_of_sigma;
  // get<1, 1>(angular_d_sigma) =
  //     2.0 * get(sin_phi) * get(cos_theta) * get(sin_theta) / norm_of_sigma;
  // get<1, 2>(angular_d_sigma) =
  //     (square(get(cos_theta)) - square(get(sin_theta))) / norm_of_sigma;

  // // note : scaled as 1/sin(theta)
  // get<2, 0>(angular_d_sigma) = -get(sin_phi) * get(sin_theta) /
  // norm_of_sigma; get<2, 1>(angular_d_sigma) = get(cos_phi) * get(sin_theta) /
  // norm_of_sigma; get<2, 2>(angular_d_sigma) = 0.0;

  // for (size_t i = 0; i < 3; ++i) {
  //   angular_d_worldtube_normal->get(0, i) = 0.0;
  // }

  // for (size_t A = 0; A < 2; ++A) {
  //   for (size_t i = 0; i < 3; ++i) {
  //     // first term
  //     angular_d_worldtube_normal->get(A + 1, i) =
  //         angular_d_sigma.get(A + 1, 0) *
  //         (inverse_spatial_metric.get(i, 0) -
  //          worldtube_normal->get(i) * worldtube_normal->get(0));
  //     for (size_t n = 1; n < 3; ++n) {
  //       angular_d_worldtube_normal->get(A + 1, i) +=
  //           angular_d_sigma.get(A + 1, n) *
  //           (inverse_spatial_metric.get(i, n) -
  //            worldtube_normal->get(i) * worldtube_normal->get(n));
  //     }
  //     // second term
  //     // TODO: consider optimizations from e.g. symmetries and the
  //     for (size_t j = 0; j < 3; ++j) {
  //       for (size_t m = 0; m < 3; ++m) {
  //         for (size_t n = 0; n < 3; ++n) {
  //           angular_d_worldtube_normal->get(A + 1, i) +=
  //               cartesian_to_angular_jacobian.get(A + 1, j) *
  //               phi.get(j, m + 1, n + 1) * worldtube_normal->get(m) *
  //               (0.5 * worldtube_normal->get(i) * worldtube_normal->get(n) -
  //                inverse_spatial_metric.get(i, n));
  //         }
  //       }
  //     }
  //   }
  // }

  for (size_t i = 0; i < 3; ++i) {
    // TODO: optimize
    dt_worldtube_normal->get(i) = 0.0;
    for (size_t m = 0; m < 3; ++m) {
      for (size_t n = 0; n < 3; ++n) {
        dt_worldtube_normal->get(i) +=
            (0.5 * worldtube_normal->get(i) * worldtube_normal->get(m) -
             inverse_spatial_metric.get(i, m)) *
            worldtube_normal->get(n) * dt_psi.get(m + 1, n + 1);
      }
    }
  }
}

void null_vector_l_and_derivatives(
    const gsl::not_null<tnsr::iA<DataVector, 3>*> angular_d_null_l,
    const gsl::not_null<tnsr::A<DataVector, 3>*> du_null_l,
    const gsl::not_null<tnsr::A<DataVector, 3>*> null_l,
    const tnsr::iJ<DataVector, 3>& angular_d_worldtube_normal,
    const tnsr::I<DataVector, 3>& dt_worldtube_normal,
    const tnsr::iJ<DataVector, 3>& cartesian_to_angular_jacobian,
    const tnsr::i<DataVector, 3>& d_lapse, const tnsr::iaa<DataVector, 3>& phi,
    const tnsr::iJ<DataVector, 3>& d_shift, const Scalar<DataVector>& dt_lapse,
    const tnsr::aa<DataVector, 3>& dt_psi,
    const tnsr::I<DataVector, 3>& dt_shift, const Scalar<DataVector>& lapse,
    const tnsr::aa<DataVector, 3>& psi, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& worldtube_normal) noexcept {
  if (UNLIKELY((get<0, 0>(*angular_d_null_l).size() != get(lapse).size()))) {
    *angular_d_null_l = tnsr::iA<DataVector, 3>{get(lapse).size()};
  }
  if (UNLIKELY((get<0>(*du_null_l).size() != get(lapse).size()))) {
    *du_null_l = tnsr::A<DataVector, 3>{get(lapse).size()};
  }
  if (UNLIKELY((get<0>(*null_l).size() != get(lapse).size()))) {
    *null_l = tnsr::A<DataVector, 3>{get(lapse).size()};
  }

  // Allocation
  DataVector denominator = get(lapse);
  for (size_t i = 0; i < 3; ++i) {
    // off-diagonal
    for (size_t j = i + 1; j < 3; ++j) {
      // TODO: use symmetry, currently breaks the pypp test
      denominator -=
          psi.get(i + 1, j + 1) * (shift.get(i) * worldtube_normal.get(j) +
                                   shift.get(j) * worldtube_normal.get(i));
    }
    // diagonal
    denominator -=
        psi.get(i + 1, i + 1) * shift.get(i) * worldtube_normal.get(i);
  }
  get<0>(*null_l) = 1.0 / (denominator * get(lapse));
  for (size_t i = 0; i < 3; ++i) {
    null_l->get(i + 1) =
        (worldtube_normal.get(i) - shift.get(i) / get(lapse)) / denominator;
  }

  // TODO REMOVE
  // // Allocation
  // TODO: remove
  // tnsr::i<DataVector, 2> angular_deriv_of_denominator{get(lapse).size()};

  // // angular directions denoted with capitals
  // for (size_t A = 0; A < 2; ++A) {
  //   angular_deriv_of_denominator.get(A) =
  //       -cartesian_to_angular_jacobian.get(A + 1, 0) * get<0>(d_lapse);
  //   for (size_t i = 1; i < 3; ++i) {
  //     angular_deriv_of_denominator.get(A) -=
  //         cartesian_to_angular_jacobian.get(A + 1, i) * d_lapse.get(i);
  //   }
  //   // vary i < j (off-diagonal symmetric part), independent of k
  //   for (size_t i = 0; i < 3; ++i) {
  //     for (size_t j = i + 1; j < 3; ++j) {
  //       // symmetry
  //       angular_deriv_of_denominator.get(A) +=
  //           (angular_d_worldtube_normal.get(A + 1, j) * shift.get(i) +
  //            angular_d_worldtube_normal.get(A + 1, i) * shift.get(j)) *
  //           psi.get(i + 1, j + 1);
  //     }
  //     angular_deriv_of_denominator.get(A) +=
  //         angular_d_worldtube_normal.get(A + 1, i) * psi.get(i + 1, i + 1) *
  //         shift.get(i);
  //   }
  //   // remaining k-dependent parts
  //   for (size_t k = 0; k < 3; ++k) {
  //     for (size_t i = 0; i < 3; ++i) {
  //       for (size_t j = i + 1; j < 3; ++j) {
  //         // symmetry
  //         angular_deriv_of_denominator.get(A) +=
  //             cartesian_to_angular_jacobian.get(A + 1, k) *
  //             (worldtube_normal.get(j) * shift.get(i) +
  //              worldtube_normal.get(i) * shift.get(j)) *
  //             phi.get(k, i + 1, j + 1);
  //         angular_deriv_of_denominator.get(A) +=
  //             cartesian_to_angular_jacobian.get(A + 1, k) *
  //             (worldtube_normal.get(j) * d_shift.get(k, i) +
  //              worldtube_normal.get(i) * d_shift.get(k, j)) *
  //             psi.get(i + 1, j + 1);
  //       }
  //       // diagonal k-dependent part
  //       angular_deriv_of_denominator.get(A) +=
  //           worldtube_normal.get(i) *
  //           cartesian_to_angular_jacobian.get(A + 1, k) *
  //           (shift.get(i) * phi.get(k, i + 1, i + 1) +
  //            d_shift.get(k, i) * psi.get(i + 1, i + 1));
  //     }
  //   }
  //   angular_deriv_of_denominator.get(A) /= square(denominator);
  // }

  // // note: angular_deriv_of_denominator contains the sign change from
  // // differentiating the inverse

  // // the time component of the angular_d_null_l
  // for (size_t A = 0; A < 2; ++A) {
  //   angular_d_null_l->get(A + 1, 0) = angular_deriv_of_denominator.get(A);
  //   for (size_t i = 0; i < 3; ++i) {
  //     angular_d_null_l->get(A + 1, 0) -=
  //         d_lapse.get(i) * cartesian_to_angular_jacobian.get(A + 1, i) /
  //         (get(lapse) * denominator);
  //   }
  //   angular_d_null_l->get(A + 1, 0) /= get(lapse);
  // }

  // // spatial components (m) of the angular_d_null_l
  // for (size_t A = 0; A < 2; ++A) {
  //   for (size_t m = 0; m < 3; ++m) {
  //     angular_d_null_l->get(A + 1, m + 1) =
  //         angular_deriv_of_denominator.get(A) *
  //             (worldtube_normal.get(m) - shift.get(m) / get(lapse)) +
  //         angular_d_worldtube_normal.get(A + 1, m) / denominator;
  //   }
  // }

  // tnsr::iA<DataVector, 3> angular_d_spacetime_normal{get(lapse).size(), 0.0};
  // for (size_t A = 0; A < 2; ++A) {
  //   for (size_t i = 0; i < 3; ++i) {
  //     angular_d_spacetime_normal.get(A + 1, 0) +=
  //         cartesian_to_angular_jacobian.get(A + 1, i) *
  //         (-d_lapse.get(i) / square(get(lapse)));
  //     for (size_t j = 0; j < 3; ++j) {
  //       angular_d_spacetime_normal.get(A + 1, i + 1) +=
  //           cartesian_to_angular_jacobian.get(A + 1, j) *
  //           (-d_shift.get(j, i) / get(lapse) +
  //            shift.get(i) * d_lapse.get(j) / square(get(lapse)));
  //     }
  //   }
  // }

  // for (size_t A = 0; A < 2; ++A) {
  //   for (size_t m = 0; m < 3; ++m) {
  //     angular_d_null_l->get(A + 1, m + 1) +=
  //         angular_d_spacetime_normal.get(A + 1, m + 1) / denominator;
  //   }
  // }
  // for (size_t a = 0; a < 4; ++a) {
  //   angular_d_null_l->get(0, a) = 0.0;
  // }

  // Allocation
  DataVector du_denominator{get(lapse).size()};

  du_denominator = -get(dt_lapse);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i + 1; j < 3; ++j) {
      // symmetry
      du_denominator += (dt_shift.get(i) * worldtube_normal.get(j) +
                         dt_shift.get(j) * worldtube_normal.get(i)) *
                            psi.get(i + 1, j + 1) +
                        (shift.get(i) * worldtube_normal.get(j) +
                         shift.get(j) * worldtube_normal.get(i)) *
                            dt_psi.get(i + 1, j + 1) +
                        (shift.get(i) * dt_worldtube_normal.get(j) +
                         shift.get(j) * dt_worldtube_normal.get(i)) *
                            psi.get(i + 1, j + 1);
    }
    // diagonal
    du_denominator +=
        dt_shift.get(i) * psi.get(i + 1, i + 1) * worldtube_normal.get(i) +
        shift.get(i) * dt_psi.get(i + 1, i + 1) * worldtube_normal.get(i) +
        shift.get(i) * psi.get(i + 1, i + 1) * dt_worldtube_normal.get(i);
  }
  du_denominator /= square(denominator);

  get<0>(*du_null_l) =
      (du_denominator - get(dt_lapse) / (get(lapse) * denominator)) /
      get(lapse);
  for (size_t i = 0; i < 3; ++i) {
    du_null_l->get(i + 1) =
        (dt_worldtube_normal.get(i) - dt_shift.get(i) / get(lapse)) /
        denominator;
    du_null_l->get(i + 1) +=
        shift.get(i) * get(dt_lapse) / (square(get(lapse)) * denominator);
    du_null_l->get(i + 1) +=
        (-shift.get(i) / get(lapse) + worldtube_normal.get(i)) * du_denominator;
  }
}

void dlambda_null_metric_and_inverse(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> dlambda_null_metric,
    const gsl::not_null<tnsr::AA<DataVector, 3>*> dlambda_inverse_null_metric,
    const tnsr::iA<DataVector, 3> angular_d_null_l,
    const tnsr::iJ<DataVector, 3>& cartesian_to_angular_jacobian,
    const tnsr::iaa<DataVector, 3>& phi, const tnsr::aa<DataVector, 3>& dt_psi,
    const tnsr::A<DataVector, 3>& du_null_l,
    const tnsr::AA<DataVector, 3>& inverse_null_metric,
    const tnsr::A<DataVector, 3>& null_l,
    const tnsr::aa<DataVector, 3>& psi) noexcept {
  // first, the (down-index) null metric
  if (UNLIKELY(
          (get<0, 0>(*dlambda_null_metric).size() != get<0, 0>(psi).size()))) {
    *dlambda_null_metric = tnsr::aa<DataVector, 3>(get<0, 0>(psi).size());
  }
  if (UNLIKELY((get<0, 0>(*dlambda_inverse_null_metric).size() !=
                get<0, 0>(psi).size()))) {
    *dlambda_inverse_null_metric =
        tnsr::AA<DataVector, 3>(get<0, 0>(psi).size());
  }
  get<0, 0>(*dlambda_null_metric) = get<0>(null_l) * get<0, 0>(dt_psi) +
                                    2.0 * get<0>(du_null_l) * get<0, 0>(psi);
  for (size_t i = 0; i < 3; ++i) {
    get<0, 0>(*dlambda_null_metric) +=
        null_l.get(i + 1) * phi.get(i, 0, 0) +
        2.0 * du_null_l.get(i + 1) * psi.get(i + 1, 0);
  }
  // A0 component
  for (size_t A = 0; A < 2; ++A) {
    dlambda_null_metric->get(0, A + 2) =
        cartesian_to_angular_jacobian.get(A + 1, 0) *
            (get<0>(du_null_l) * get<1, 0>(psi) +
             get<0>(null_l) * get<1, 0>(dt_psi)) +
        angular_d_null_l.get(A + 1, 1) * get<1, 0>(psi) +
        angular_d_null_l.get(A + 1, 0) * get<0, 0>(psi);
    for (size_t k = 1; k < 3; ++k) {
      dlambda_null_metric->get(0, A + 2) +=
          cartesian_to_angular_jacobian.get(A + 1, k) *
              (get<0>(du_null_l) * psi.get(k + 1, 0) +
               get<0>(null_l) * dt_psi.get(k + 1, 0)) +
          angular_d_null_l.get(A + 1, k + 1) * psi.get(k + 1, 0);
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t k = 0; k < 3; ++k) {
        dlambda_null_metric->get(0, A + 2) +=
            cartesian_to_angular_jacobian.get(A + 1, k) *
            (du_null_l.get(i + 1) * psi.get(k + 1, i + 1) +
             null_l.get(i + 1) * phi.get(i, k + 1, 0));
      }
    }
    dlambda_null_metric->get(A + 2, 0) = dlambda_null_metric->get(0, A + 2);
  }
  // zero the null directions
  get<0, 1>(*dlambda_null_metric) = 0.0;
  for (size_t a = 1; a < 4; ++a) {
    dlambda_null_metric->get(1, a) = 0.0;
  }

  /// REWRITE

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      dlambda_null_metric->get(A + 2, B + 2) = 0.0;
    }
  }

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = A; B < 2; ++B) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          dlambda_null_metric->get(A + 2, B + 2) +=
              null_l.get(0) * cartesian_to_angular_jacobian.get(A + 1, i) *
              cartesian_to_angular_jacobian.get(B + 1, j) *
              dt_psi.get(i + 1, j + 1);
        }
      }
    }
  }

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = A; B < 2; ++B) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          for (size_t k = 0; k < 3; ++k) {
            dlambda_null_metric->get(A + 2, B + 2) +=
                null_l.get(k + 1) *
                cartesian_to_angular_jacobian.get(A + 1, i) *
                cartesian_to_angular_jacobian.get(B + 1, j) *
                phi.get(k, i + 1, j + 1);
          }
        }
      }
    }
  }

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = A; B < 2; ++B) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t a = 0; a < 4; ++a) {
          dlambda_null_metric->get(A + 2, B + 2) +=
              (angular_d_null_l.get(A + 1, a) *
                   cartesian_to_angular_jacobian.get(B + 1, i) +
               angular_d_null_l.get(B + 1, a) *
                   cartesian_to_angular_jacobian.get(A + 1, i)) *
              psi.get(a, i + 1);
        }
      }
    }
  }
  ///

  // for (size_t A = 0; A < 2; ++A) {
  //   for (size_t B = A; B < 2; ++B) {
  //     dlambda_null_metric->get(A + 2, B + 2) =
  //         get<0, 1>(psi) * (angular_d_null_l.get(A + 1, 0) *
  //                               cartesian_to_angular_jacobian.get(B + 1, 0) +
  //                           angular_d_null_l.get(B + 1, 0) *
  //                               cartesian_to_angular_jacobian.get(A + 1, 0));
  //     for (size_t l = 1; l < 3; ++l) {
  //       dlambda_null_metric->get(A + 2, B + 2) +=
  //           psi.get(0, l + 1) *
  //           (angular_d_null_l.get(A + 1, 0) *
  //                cartesian_to_angular_jacobian.get(B + 1, l) +
  //            angular_d_null_l.get(B + 1, 0) *
  //                cartesian_to_angular_jacobian.get(A + 1, l));
  //     }
  //     for (size_t l = 0; l < 3; ++l) {
  //       for (size_t k = 0; k < 3; ++k) {
  //         dlambda_null_metric->get(A + 2, B + 2) +=
  //             psi.get(k + 1, l + 1) *
  //                 (angular_d_null_l.get(A + 1, k + 1) *
  //                      cartesian_to_angular_jacobian.get(B + 1, l) +
  //                  angular_d_null_l.get(B + 1, k + 1) *
  //                      cartesian_to_angular_jacobian.get(A + 1, l)) +
  //             get<0>(null_l) * cartesian_to_angular_jacobian.get(A + 1, l) *
  //                 cartesian_to_angular_jacobian.get(B + 1, k) *
  //                 dt_psi.get(l + 1, k + 1);

  //         for (size_t i = 0; i < 3; ++i) {
  //           dlambda_null_metric->get(A + 2, B + 2) +=
  //               null_l.get(i + 1) *
  //               cartesian_to_angular_jacobian.get(A + 1, l) *
  //               cartesian_to_angular_jacobian.get(B + 1, k) *
  //               phi.get(i, l + 1, k + 1);
  //         }
  //       }
  //     }
  //   }
  // }
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b < a; ++b) {
      dlambda_null_metric->get(a, b) = dlambda_null_metric->get(b, a);
    }
  }

  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b < 4; ++b) {
      dlambda_inverse_null_metric->get(a, b) = 0.0;
    }
  }
  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = A; B < 2; ++B) {
      for (size_t C = 0; C < 2; ++C) {
        for (size_t D = 0; D < 2; ++D) {
          dlambda_inverse_null_metric->get(A + 2, B + 2) +=
              -inverse_null_metric.get(A + 2, C + 2) *
              inverse_null_metric.get(B + 2, D + 2) *
              dlambda_null_metric->get(C + 2, D + 2);
        }
      }
    }
  }

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      dlambda_inverse_null_metric->get(1, A + 2) +=
          inverse_null_metric.get(A + 2, B + 2) *
          dlambda_null_metric->get(0, B + 2);
      for (size_t C = 0; C < 2; ++C) {
        dlambda_inverse_null_metric->get(1, A + 2) +=
            -inverse_null_metric.get(A + 2, B + 2) *
            inverse_null_metric.get(1, C + 2) *
            dlambda_null_metric->get(C + 2, B + 2);
      }
    }
  }

  dlambda_inverse_null_metric->get(1, 1) += -dlambda_null_metric->get(0, 0);

  for (size_t A = 0; A < 2; ++A) {
    dlambda_inverse_null_metric->get(1, 1) +=
        2.0 * inverse_null_metric.get(1, A + 2) *
        dlambda_null_metric->get(0, A + 2);
    for (size_t B = 0; B < 2; ++B) {
      dlambda_inverse_null_metric->get(1, 1) +=
          -inverse_null_metric.get(1, A + 2) *
          inverse_null_metric.get(1, B + 2) *
          dlambda_null_metric->get(A + 2, B + 2);
    }
  }

  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b < a; ++b) {
      dlambda_inverse_null_metric->get(a, b) =
          dlambda_inverse_null_metric->get(b, a);
    }
  }
}

void bondi_r(const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> r,
             const tnsr::aa<DataVector, 3>& null_metric) noexcept {
  if (UNLIKELY((get(*r).data().size() != get<0, 0>(null_metric).size()))) {
    get(*r).data() = ComplexDataVector{get<0, 0>(null_metric).size()};
  }
  // the inclusion of the std::complex<double> informs the Blaze expression
  // templates to turn the result into a ComplexDataVector
  get(*r).data() = std::complex<double>(1.0, 0) *
                   pow(get<2, 2>(null_metric) * get<3, 3>(null_metric) -
                           square(get<2, 3>(null_metric)),
                       0.25);
}

void d_bondi_r(const gsl::not_null<tnsr::a<DataVector, 3>*> d_bondi_r,
               const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
               const tnsr::aa<DataVector, 3>& dlambda_null_metric,
               const tnsr::aa<DataVector, 3>& du_null_metric,
               const tnsr::AA<DataVector, 3>& inverse_null_metric, size_t l_max,
               const YlmSpherepack spherical_harmonic) noexcept {
  if (UNLIKELY(
          (get<0>(*d_r).size() != get<0, 0>(inverse_null_metric).size()))) {
    *d_r = tnsr::a<DataVector, 3>(get<0, 0>(inverse_null_metric).size());
  }
  const auto& r = get<Tags::R>(*variables_with_r);
  // compute the time derivative part
  get<0>(*d_r) =
      0.25 * real(get(r).data()) *
      (get<2, 2>(inverse_null_metric) * get<2, 2>(du_null_metric) +
       2.0 * get<2, 3>(inverse_null_metric) * get<2, 3>(du_null_metric) +
       get<3, 3>(inverse_null_metric) * get<3, 3>(du_null_metric));
  // compute the lambda derivative part
  get<1>(*d_r) =
      0.25 * real(get(r).data()) *
      (get<2, 2>(inverse_null_metric) * get<2, 2>(dlambda_null_metric) +
       2.0 * get<2, 3>(inverse_null_metric) * get<2, 3>(dlambda_null_metric) +
       get<3, 3>(inverse_null_metric) * get<3, 3>(dlambda_null_metric));

  ComplexDataVector eth_of_r =
      Spectral::Swsh::swsh_derivative<0, Spectral::Swsh::Tags::Eth>(
          get(r).data(), l_max);
  d_r->get(2) = -real(eth_of_r);
  d_r->get(3) = -imag(eth_of_r);
}


void dyads(
    const gsl::not_null<tnsr::i<ComplexDataVector, 2>*> down_dyad,
    const gsl::not_null<tnsr::i<ComplexDataVector, 2>*> up_dyad) noexcept {
  // implicit factors of sin_theta omitted
  get<0>(*down_dyad) = -1.0;
  get<1>(*down_dyad) = std::complex<double>(0.0, -1.0);
  get<0>(*up_dyad) = -1.0;
  get<1>(*up_dyad) = std::complex<double>(0.0, -1.0);
}

void beta_initial_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
    const tnsr::a<DataVector, 3>& d_r) noexcept {
  if (UNLIKELY(get(*beta).data().size() != get<0>(d_r).size())) {
    get(*beta).data() = ComplexDataVector{get<1>(d_r).size()};
  }
  get(*beta).data() = std::complex<double>(-0.5, 0.0) * log(get<1>(d_r));
}

void bondi_u_initial_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
    const tnsr::i<ComplexDataVector, 2>& down_dyad,
    const tnsr::a<DataVector, 3>& d_r,
    const tnsr::AA<DataVector, 3>& inverse_null_metric) noexcept {
  if (UNLIKELY(get(*u).data().size() != get<0>(d_r).size())) {
    get(*u).data() = ComplexDataVector{get<0>(d_r).size()};
  }

  get(*u).data() = -get<0>(down_dyad) * get<1, 2>(inverse_null_metric) -
                   get<1>(down_dyad) * get<1, 3>(inverse_null_metric);

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      get(*u).data() -= d_r.get(2 + A) * down_dyad.get(B) *
                        inverse_null_metric.get(A + 2, B + 2) / get<1>(d_r);
    }
  }
}

void bondi_w_initial_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> w,
    const tnsr::a<DataVector, 3>& d_r,
    const tnsr::AA<DataVector, 3>& inverse_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r) noexcept {
  if (UNLIKELY(get(*w).data().size() != get(r).data().size())) {
    get(*w).data() = ComplexDataVector{get(r).data().size()};
  }
  get(*w).data() =
      std::complex<double>(1.0, 0.0) *
      (-1.0 + get<1>(d_r) * get<1, 1>(inverse_null_metric) - 2.0 * get<0>(d_r));

  for (size_t A = 0; A < 2; ++A) {
    get(*w).data() += 2.0 * d_r.get(A + 2) * inverse_null_metric.get(1, A + 2);
  }

  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      get(*w).data() += d_r.get(A + 2) * d_r.get(B + 2) *
                        inverse_null_metric.get(A + 2, B + 2) / get<1>(d_r);
    }
  }
  get(*w).data() /= std::complex<double>(1.0, 0.0) * get(r).data();
}

void bondi_j_initial_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const tnsr::aa<DataVector, 3>& null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>> r,
    const tnsr::i<ComplexDataVector, 2>& up_dyad) noexcept {
  if (UNLIKELY(get(*j).data().size() != get(r).data().size())) {
    get(*j).data() = ComplexDataVector{get(r).data().size()};
  }
  get(*j).data() =
      0.5 *
      (square(get<0>(up_dyad)) * get<2, 2>(null_metric) +
       2.0 * get<0>(up_dyad) * get<1>(up_dyad) * get<2, 3>(null_metric) +
       square(get<1>(up_dyad)) * get<3, 3>(null_metric)) /
      square(get(r).data());
}

void dr_bondi_j(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dr_j,
    const tnsr::aa<DataVector, 3>& dlambda_null_metric,
    const tnsr::a<DataVector, 3>& d_r,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
    const tnsr::i<ComplexDataVector, 2>& up_dyad) noexcept {
  if (UNLIKELY(get(*dr_j).data().size() != get(r).data().size())) {
    get(*dr_j).data() = ComplexDataVector{get(r).data().size()};
  }
  get(*dr_j) = -2.0 * get(j) / get(r);
  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      get(*dr_j).data() += 0.5 * up_dyad.get(A) * up_dyad.get(B) *
                           dlambda_null_metric.get(A + 2, B + 2) /
                           (square(get(r).data()) * get<1>(d_r));
    }
  }
}

void d2lambda_bondi_r(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> d2lambda_r,
    const tnsr::a<DataVector, 3>& d_r,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r) noexcept {
  if (UNLIKELY(get(*d2lambda_r).data().size() != get(j).data().size())) {
    get(*d2lambda_r).data() = ComplexDataVector{get(j).data().size()};
  }
  get(*d2lambda_r) =
      -0.25 * get(r) *
      (get(dr_j) * conj(get(dr_j)) -
       0.25 * square(conj(get(j)) * get(dr_j) + get(j) * conj(get(dr_j))) /
           (1.0 + get(j) * conj(get(j))));
  get(*d2lambda_r).data() *= square(get<1>(d_r));
}

void bondi_q_initial_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> q,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& d2lambda_r,
    const tnsr::AA<DataVector, 3>& dlambda_inverse_null_metric,
    const tnsr::a<DataVector, 3>& d_r,
    const tnsr::i<ComplexDataVector, 2> down_dyad,
    const tnsr::i<DataVector, 2> angular_d_dlambda_r,
    const tnsr::AA<DataVector, 3>& inverse_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& u) noexcept {
  if (UNLIKELY(get(*q).data().size() != get(j).data().size())) {
    get(*q).data() = ComplexDataVector{get(j).data().size()};
  }
  // Allocation
  Scalar<SpinWeighted<ComplexDataVector, 1>> dlambda_u;

  get(dlambda_u).data() = ComplexDataVector{get(j).data().size()};

  get(dlambda_u).data() = -get(u).data() * get(d2lambda_r).data() / get<1>(d_r);

  for (size_t A = 0; A < 2; ++A) {
    get(dlambda_u) -= (dlambda_inverse_null_metric.get(1, A + 2) +
                       get(d2lambda_r).data() *
                           inverse_null_metric.get(1, A + 2) / get<1>(d_r)) *
                      down_dyad.get(A);
    for (size_t B = 0; B < 2; ++B) {
      get(dlambda_u) -=
          (d_r.get(B + 2) * dlambda_inverse_null_metric.get(A + 2, B + 2) /
           get<1>(d_r)) *
          down_dyad.get(A);
      get(dlambda_u) -= angular_d_dlambda_r.get(B) *
                        inverse_null_metric.get(A + 2, B + 2) *
                        down_dyad.get(A) / get<1>(d_r);
    }
  }

  get(*q).data() =
      square(get(r).data()) *
      (get(j).data() * conj(get(dlambda_u).data()) +
       sqrt(1.0 + get(j).data() * conj(get(j).data())) * get(dlambda_u).data());

  ComplexDataVector dlambda_u_check{get(*q).data().size(), 0.0};
  dlambda_u_check =
      square(get<1>(d_r)) / square(get(r).data()) *
      (sqrt(1.0 + get(j).data() * conj(get(j).data())) * get(*q).data() -
       get(j).data() * conj(get(*q).data()));
}

void bondi_h_initial_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> h,
    const tnsr::a<DataVector, 3>& d_r,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
    const tnsr::aa<DataVector, 3>& du_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
    const tnsr::i<ComplexDataVector, 2>& up_dyad) noexcept {
  if (UNLIKELY(get(*h).data().size() != get(j).data().size())) {
    get(*h).data() = ComplexDataVector{get(j).data().size()};
  }
  // TEST FOR SPEC H
  // get(*h).data() = 0.0;
  get(*h).data() = -2.0 * get<0>(d_r) / get(r).data() * get(j).data();
  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      get(*h).data() += (0.5 / square(get(r).data())) * up_dyad.get(A) *
                        up_dyad.get(B) * du_null_metric.get(A + 2, B + 2);
    }
  }
}
}  // namespace Cce
