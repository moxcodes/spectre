// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce {
namespace InitializeJ {
// Note: not actually options-creatable, but does satisfy the interface
// requirements
struct InitializeJCoordinatesForVolumeValue : InitializeJ {
  WRAPPED_PUPable_decl_template(
      InitializeJCoordinatesForVolumeValue);  // NOLINT
  explicit InitializeJCoordinatesForVolumeValue(
      CkMigrateMessage* /*unused*/) noexcept {}

  InitializeJCoordinatesForVolumeValue(double angular_coordinate_tolerance,
                                       size_t max_iterations) noexcept;

  std::unique_ptr<InitializeJ> get_clone() const noexcept override;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max,
      size_t number_of_radial_points) const noexcept override;

  void pup(PUP::er& /*p*/) noexcept override;

 private:
  double angular_coordinate_tolerance_ = 1.0e-10;
  size_t max_iterations_ = 300;
};
}  // namespace InitializeJ
}  // namespace Cce
