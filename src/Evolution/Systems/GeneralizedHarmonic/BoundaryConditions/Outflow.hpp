// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace GeneralizedHarmonic::BoundaryConditions {
template <size_t Dim>
class Outflow final : public BoundaryCondition<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Outflow boundary condition that only verifies the characteristic speeds "
      "are all directed out of the domain."};

  Outflow() = default;
  Outflow(Outflow&&) noexcept = default;
  Outflow& operator=(Outflow&&) noexcept = default;
  Outflow(const Outflow&) = default;
  Outflow& operator=(const Outflow&) = default;
  ~Outflow() override = default;

  explicit Outflow(CkMigrateMessage* msg) noexcept;

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Outflow);

  auto get_clone() const noexcept -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Outflow;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags = tmpl::list<
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>;
  using dg_gridless_tags = tmpl::list<>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;

  static std::optional<std::string> dg_outflow(
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
      /*outward_directed_normal_vector*/,

      const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift) noexcept {
    const auto char_speeds = characteristic_speeds(
        gamma_1, lapse, shift, outward_directed_normal_covector);
    Scalar<DataVector> normal_dot_mesh_velocity;
    if (face_mesh_velocity.has_value()) {
      normal_dot_mesh_velocity = dot_product(outward_directed_normal_covector,
                                             face_mesh_velocity.value());
    }
    double min_speed = std::numeric_limits<double>::signaling_NaN();
    for (size_t i = 0; i < char_speeds.size(); ++i) {
      if (face_mesh_velocity.has_value()) {
            min_speed = min(char_speeds[i] - get(normal_dot_mesh_velocity));
      } else {
        min_speed = min(char_speeds[i]);
      }
      if (min_speed < 0.0) {
        return {MakeString{}
                << "Outflow boundary condition violated with speed index " << i
                << " ingoing: " << min_speed << "\n speed: " << char_speeds[i]
                << "\nn_i: " << outward_directed_normal_covector
                << "\n"
                   "See GeneralizedHarmonic::characteristic_speeds for the "
                   "index ordering of characteristic speeds\n"};
      }
    }
    return std::nullopt;
  }
};
}  // namespace GeneralizedHarmonic::BoundaryConditions
