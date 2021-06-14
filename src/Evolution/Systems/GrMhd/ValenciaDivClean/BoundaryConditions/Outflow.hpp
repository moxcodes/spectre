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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
/// A `BoundaryCondition` that only verifies that all characteristic speeds are
/// directed out of the domain; no boundary data is altered by this boundary
/// condition.
class Outflow final : public BoundaryCondition {
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
  using dg_interior_temporary_tags =
      tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                 gr::Tags::Lapse<DataVector>>;
  using dg_gridless_tags = tmpl::list<>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;

  static std::optional<std::string> dg_outflow(
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
      /*outward_directed_normal_vector*/,

      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const Scalar<DataVector>& lapse) noexcept {
    double min_speed = std::numeric_limits<double>::signaling_NaN();
    Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>> buffer{
        get(lapse).size()};
    auto& normal_dot_shift = get<::Tags::TempScalar<0>>(buffer);
    dot_product(make_not_null(&normal_dot_shift),
                outward_directed_normal_covector, shift);
    auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<1>>(buffer);
    if (face_mesh_velocity.has_value()) {
      dot_product(make_not_null(&normal_dot_mesh_velocity),
                  outward_directed_normal_covector, face_mesh_velocity.value());
      get(normal_dot_shift) += get(normal_dot_mesh_velocity);
    }
    // The characteristic speeds are bounded by \pm \alpha - \beta_n, and
    // saturate that bound, so there is no need to check the hydro-dependent
    // characteristic speeds.
    min_speed = std::min(min(-get(lapse) - get(normal_dot_shift)),
                         min(get(lapse) - get(normal_dot_shift)));
    if (min_speed < 0.0) {
      return {MakeString{} << "Outflow boundary condition violated. Speed: "
                           << min_speed << "\nn_i: "
                           << outward_directed_normal_covector << "\n"};
    }
    return std::nullopt;
  }
};
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
