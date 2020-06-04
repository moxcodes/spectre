// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/Domain.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Domain/Creators/DomainCreator.hpp"

namespace Lb{
namespace Distribution {

// TODO move to cpp
template <size_t Dim>
struct RoundRobin {
  RoundRobin(const Domain<Dim>& /*domain*/,
             const std::vector<std::array<size_t, Dim>>&
             /*initial_refinement_levels*/,
             const size_t number_of_procs) noexcept
      : number_of_procs_{number_of_procs} {}

  template <typename Metavariables>
  int which_proc(const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                 const ElementId<Dim>& /*element_id*/) noexcept {
    return counter_++ % static_cast<int>(number_of_procs_);
  }
  size_t number_of_procs_;
  int counter_ = 0;
};
}  // namespace Distribution
}  // namespace Lb
