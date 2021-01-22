// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Actions/SelfStartActions.hpp"
#include "Time/TimeStepId.hpp"

namespace SelfStart {
bool in_self_start(const TimeStepId& time_step_id) noexcept {
  return time_step_id.slab_number() < 0;
}
}
