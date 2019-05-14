// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/Evolution/Systems/Cce/CceComputationTestHelpers.hpp"

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {
namespace TestHelpers {

// explicit power method avoids some troublesome Blaze behavior with powers of
// complex
ComplexDataVector power(const ComplexDataVector& val,
                        const size_t exponent) noexcept {
  ComplexDataVector pow{val.size(), 1.0};
  for (size_t i = 0; i < exponent; ++i) {
    pow *= val;
  }
  return pow;
}

void generate_volume_data_from_separated_values(
    const gsl::not_null<ComplexDataVector*> volume_data,
    const gsl::not_null<ComplexDataVector*> one_divided_by_r,
    const ComplexDataVector& angular_collocation,
    const ComplexModalVector& radial_coefficients, const size_t l_max,
    const size_t number_of_radial_grid_points) noexcept {
  for (size_t i = 0; i < number_of_radial_grid_points; ++i) {
    ComplexDataVector volume_angular_view{
        volume_data->data() +
            i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    ComplexDataVector one_divided_by_r_angular_view{
        one_divided_by_r->data() +
            i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    volume_angular_view = angular_collocation * radial_coefficients[0];
    for (size_t radial_power = 1; radial_power < radial_coefficients.size();
         ++radial_power) {
      volume_angular_view += angular_collocation *
                             radial_coefficients[radial_power] *
                             power(one_divided_by_r_angular_view, radial_power);
    }
  }
}
}  // namespace TestHelpers
}  // namespace Cce
