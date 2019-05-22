// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/SerialEvolveHelpers.hpp"

namespace Cce{

ComplexDataVector interpolate_to_bondi_r(ComplexDataVector to_interpolate,
                                         ComplexDataVector boundary_r,
                                         double target_r,
                                         size_t l_max) noexcept {
  ComplexDataVector values_on_surface{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  size_t number_of_radial_points =
      to_interpolate.size() /
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  ComplexDataVector transpose_buffer{to_interpolate.size()};
  transpose(make_not_null(&transpose_buffer), to_interpolate,
            Spectral::Swsh::number_of_swsh_collocation_points(l_max),
            number_of_radial_points);
  DataVector radial_collocation_points =
      Spectral::collocation_points<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto>(
          number_of_radial_points);
  for (size_t i = 0;
       i < Spectral::Swsh::number_of_swsh_collocation_points(l_max); ++i) {
    ComplexDataVector radial_view{
        transpose_buffer.data() + i * number_of_radial_points,
        number_of_radial_points};
    DataVector real_part = real(radial_view);
    DataVector imag_part = imag(radial_view);
    boost::math::barycentric_rational<double> real_interpolant(
        radial_collocation_points.data(), real_part.data(),
        number_of_radial_points, number_of_radial_points - 1);
    boost::math::barycentric_rational<double> imag_interpolant(
        radial_collocation_points.data(), imag_part.data(),
        number_of_radial_points, number_of_radial_points - 1);

    values_on_surface[i] = std::complex<double>(
        real_interpolant(1.0 - 2.0 * real(boundary_r[i]) / target_r),
        imag_interpolant(1.0 - 2.0 * real(boundary_r[i]) / target_r));
  }
  return values_on_surface;
}
}  // namespace Cce
