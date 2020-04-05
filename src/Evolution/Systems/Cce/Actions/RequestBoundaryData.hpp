// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/Actions/BoundaryComputeAndSendToEvolution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"

#include "Parallel/Printf.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Requests boundary data be sent from `WorldtubeBoundaryComponent` to
 * `EvolutionComponent` (template parameters).
 *
 * \details Calls the simple action
 * `Cce::Actions::BoundaryComputeAndSendToEvolution<WorldtubeBoundaryComponent,
 * EvolutionComponent>` on the `WorldtubeBoundaryComponent`, which performs
 * boundary computations then sends data to the `EvolutionComponent` via
 * `Cce::Actions::ReceiveWorldtubeData`. Requests the current
 * `Tags::TimeStepId`. For the majority of these requests, it's better to issue
 * them as early as possible to maximize the degree of parallelism for the
 * system, so most calls should use `Cce::Actions::RequestNextBoundaryData`,
 * because it can be called the substep prior to when the data will actually be
 * used.
 *
 * Uses:
 *  - DataBox:
 *    - `Tags::TimeStepId`
 *
 * \ref DataBoxGroup changes
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: nothing
 */
template <typename WorldtubeBoundaryComponent, typename EvolutionComponent>
struct RequestBoundaryData {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    Parallel::simple_action<Actions::BoundaryComputeAndSendToEvolution<
        WorldtubeBoundaryComponent, EvolutionComponent>>(
        Parallel::get_parallel_component<WorldtubeBoundaryComponent>(cache),
        db::get<::Tags::TimeStepId>(box));
    return std::forward_as_tuple(std::move(box));
  }
};

/*!
 * \ingroup ActionsGroup
 * \brief Requests boundary data be sent from `WorldtubeBoundaryComponent` to
 * `EvolutionComponent`.
 *
 * \details Calls the simple action
 * `Cce::Actions::BoundaryComputeAndSendToEvolution<WorldtubeBoundaryComponent,
 * EvolutionComponent>` on the `WorldtubeBoundaryComponent`, which performs
 * boundary computations then sends data to the `EvolutionComponent` via
 * `Cce::Actions::ReceiveWorldtubeData`. Requests the
 * `Tags::Next<Tags::TimeStepId>` (for the next timestep).
 *
 * Uses:
 *  - DataBox:
 *    - `Tags::Next<Tags::TimeStepId>`
 *
 * \ref DataBoxGroup changes
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: nothing
 */
template <typename WorldtubeBoundaryComponent, typename EvolutionComponent>
struct RequestNextBoundaryData;

template <typename WorldtubeBoundaryComponent,
          template <typename...> class EvolutionTemplate, typename RunStage,
          typename Metavariables>
struct RequestNextBoundaryData<
    WorldtubeBoundaryComponent,
    EvolutionTemplate<RunStage, WorldtubeBoundaryComponent, Metavariables>> {

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // only request the data if the next step is not after the end time.
    const bool end_time_reached =
        (db::get<::Tags::TimeStepId>(box).step_time().value() +
         db::get<::Tags::TimeStep>(box).value()) >=
            db::get<Tags::EndTimeFromFile<RunStage>>(box) and
        db::get<::Tags::TimeStepId>(box).is_at_slab_boundary();
    if (end_time_reached) {
      Parallel::printf("End time reached. not requesting the next time\n");
    }
    if (not end_time_reached) {
      Parallel::simple_action<Actions::BoundaryComputeAndSendToEvolution<
          WorldtubeBoundaryComponent,
          EvolutionTemplate<RunStage, WorldtubeBoundaryComponent,
                            Metavariables>>>(
          Parallel::get_parallel_component<WorldtubeBoundaryComponent>(cache),
          db::get<::Tags::Next<::Tags::TimeStepId>>(box));
    }
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Cce
