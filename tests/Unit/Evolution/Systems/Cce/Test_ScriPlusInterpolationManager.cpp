// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/VectorAlgebra.hpp"
#include "Evolution/Systems/Cce/ScriPlusInterpolationManager.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {
namespace {

template <typename VectorType>
void test_interpolate_quadratic() {
  MAKE_GENERATOR(generator);
  UniformCustomDistribution<double> value_dist{-5.0, 5.0};
  const double linear_coefficient = value_dist(generator);
  const double quadratic_coefficient = value_dist(generator);
  const size_t vector_size = 5;
  const size_t data_points = 40;

  const VectorType random_vector = make_with_random_values<VectorType>(
      make_not_null(&generator), make_not_null(&value_dist), vector_size);

  UniformCustomDistribution<double> time_dist{-0.5, 0.5};

  ScriPlusInterpolationManager<FlexibleBarycentricInterpolator, VectorType>
      interpolation_manager{5, vector_size};

  // Construct data at a peculiar time set
  //  f(u_bondi) = f_0 *(1.0 + a * u_bondi + b * u_bondi^2);
  //  where u_bondi is recorded with a small random deviation away from the
  //  actual time
  //
  for(size_t i = 0; i < data_points; ++i) {
    const DataVector time_vector = make_with_random_values<DataVector>(
        make_not_null(&generator), make_not_null(&time_dist), vector_size);
    interpolation_manager.insert_data(
        time_vector + static_cast<double>(i),
        random_vector *
            (1.0 + linear_coefficient * (static_cast<double>(i) + time_vector) +
             quadratic_coefficient *
                 square(static_cast<double>(i) + time_vector)));
    if(i > 7 and i < data_points - 8) {
      interpolation_manager.insert_target_time(static_cast<double>(i) +
                                               time_dist(generator));
    }
    while(interpolation_manager.first_time_is_ready_to_interpolate()) {
      auto interpolation_result =
          interpolation_manager.interpolate_and_pop_first_time();
      CHECK_ITERABLE_APPROX(
          interpolation_result.first,
          random_vector *
              (1.0 + linear_coefficient * interpolation_result.second +
               quadratic_coefficient * square(interpolation_result.second)));
    }
    // the culling of the data means that the interpolation manager should never
    // have too many more points than it needs to get a good interpolation.
    CHECK(interpolation_manager.data_sizes() < 12);
  }
  while (interpolation_manager.first_time_is_ready_to_interpolate()) {
    auto interpolation_result =
        interpolation_manager.interpolate_and_pop_first_time();
    CHECK_ITERABLE_APPROX(
        interpolation_result.first,
        random_vector *
            (1.0 + linear_coefficient * interpolation_result.second +
             quadratic_coefficient * square(interpolation_result.second)));
  }
  CHECK(interpolation_manager.target_times_size() == 0);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.ScriPlusInterpolationManager",
                  "[Unit][Evolution]") {
  test_interpolate_quadratic<DataVector>();
  test_interpolate_quadratic<ComplexDataVector>();
}
}  // namespace
}  // namespace Cce
