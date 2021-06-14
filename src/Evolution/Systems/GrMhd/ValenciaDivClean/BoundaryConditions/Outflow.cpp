// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Outflow.hpp"

#include <memory>

namespace grmhd::ValenciaDivClean::BoundaryConditions {
Outflow::Outflow(CkMigrateMessage* const msg) noexcept
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Outflow::get_clone() const noexcept {
  return std::make_unique<Outflow>(*this);
}

void Outflow::pup(PUP::er& p) { BoundaryCondition::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID Outflow::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
