// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace InitializeJ {

InitializeJCoordinatesForVolumeValue::InitializeJCoordinatesForVolumeValue(
    const double angular_coordinate_tolerance,
    const size_t max_iterations) noexcept
    : angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations} {}

std::unique_ptr<InitializeJ> InitializeJCoordinatesForVolumeValue::get_clone()
    const noexcept {
  return std::make_unique<InitializeJCoordinatesForVolumeValue>(
      angular_coordinate_tolerance_, max_iterations_);
}

void InitializeJCoordinatesForVolumeValue::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& /*boundary_dr_j*/,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, const size_t l_max,
    const size_t number_of_radial_points) const noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const SpinWeighted<ComplexDataVector, 2> j_boundary_view;
  make_const_view(make_not_null(&j_boundary_view), get(*j), 0,
                  number_of_angular_points);

  adjust_angular_coordinates_for_j(j, cartesian_cauchy_coordinates,
                                   angular_cauchy_coordinates, get(boundary_j),
                                   l_max, angular_coordinate_tolerance_,
                                   max_iterations_, false, j_boundary_view);
}

void InitializeJCoordinatesForVolumeValue::pup(PUP::er& p) noexcept {
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
}

/// \cond
PUP::able::PUP_ID InitializeJCoordinatesForVolumeValue::my_PUP_ID = 0;
/// \endcond
}  // namespace InitializeJ
}  // namespace Cce
