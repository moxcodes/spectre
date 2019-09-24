// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace Cce {

void pypp_test_worltube_computation_steps() noexcept {
  pypp::SetupLocalPythonEnvironment local_python_env{"Evolution/Systems/Cce/"};

  pypp::check_with_random_values<1>(
      &calculate_cartesian_to_angular_coordinates_and_derivatives,
      "BoundaryData",
      {"compute_cartesian_to_angular_coordinates",
       "compute_cartesian_to_angular_jacobian",
       "compute_cartesian_to_angular_inverse_jacobian"},
      {{{0.01, 10.0}}}, DataVector{1});

  pypp::check_with_random_values<1>(
      &calculate_null_metric_and_derivative, "BoundaryData",
      {"calculate_du_null_metric", "calculate_null_metric"}, {{{0.01, 10.0}}},
      DataVector{1});

  pypp::check_with_random_values<1>(
      &calculate_inverse_null_metric, "BoundaryData",
      {"calculate_inverse_null_metric"}, {{{0.01, 10.0}}}, DataVector{1});

  pypp::check_with_random_values<1>(
      &calculate_worldtube_normal_and_derivatives, "BoundaryData",
      {"calculate_angular_d_worldtube_normal", "calculate_worldtube_normal"},
      {{{0.01, 10.0}}}, DataVector{1});

  pypp::check_with_random_values<1>(
      &calculate_null_vector_l_and_derivatives, "BoundaryData",
      {"calculate_angular_d_null_vector_l", "calculate_du_null_vector_l",
       "calculate_null_vector_l"},
      {{{0.01, 10.0}}}, DataVector{1});

  pypp::check_with_random_values<1>(&calculate_dlambda_null_metric_and_inverse,
                                    "BoundaryData",
                                    {"calculate_dlambda_null_metric",
                                     "calculate_inverse_dlambda_null_metric"},
                                    {{{0.01, 10.0}}}, DataVector{1});

}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.BoundaryData",
                  "[Unit][Evolution]") {
  // TODO test the spherical harmonic transform steps  at the start and any
  // functions not testable with pypp

  pypp_test_worldtube_computation_steps();
}

}  // namespace Cce