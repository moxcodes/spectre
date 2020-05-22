// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "ErrorHandling/Error.hpp"
#include "Time/Tags.hpp"


namespace Cce {
namespace InterfaceManagers{

enum class InterpolationStrategy { EveryStep, EverySubstep };

template <typename DbTagList>
bool should_interpolate_for_strategy(
    const db::DataBox<DbTagList>& box,
    const InterpolationStrategy strategy) noexcept {
  if(strategy == InterpolationStrategy::EverySubstep) {
    return true;
  }
  if(strategy == InterpolationStrategy::EveryStep) {
    return db::get<::Tags::TimeStepper<>>(box).can_change_step_size(
        db::get<::Tags::TimeStepId>(box),
        db::get<::Tags::HistoryEvolvedVariables<>>(box));
  }
  ERROR("Interpolation strategy not recognized");
}
}  // namespace InterfaceManagers
}  // namespace Cce
