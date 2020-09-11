// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"

#include "Parallel/Printf.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Terminates if the current `::Tags::TimeStepId` has time value later or
 * equal to `Tags::EndTime`.
 *
 * \details Uses:
 * - DataBox:
 *   - `Cce::Tags::EndTime`
 *   - `Tags::TimeStepId`
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: nothing
 *
 */
struct ExitIfEndTimeReached {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf("CCE time: %f\n",
                     db::get<::Tags::TimeStepId>(box).substep_time().value());
    return std::tuple<db::DataBox<DbTags>&&, bool>(
        std::move(box),
        db::get<::Tags::TimeStepId>(box).substep_time().value() >=
            db::get<Tags::EndTime>(box));
  }
};

}  // namespace Actions
}  // namespace Cce
