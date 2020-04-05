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

namespace Cce {
/// \cond
template <typename RunStage, class Metavariables>
struct H5WorldtubeBoundary;
template <typename RunStage, typename BoundaryComponent, class Metavariables>
struct CharacteristicEvolution;
/// \endcond
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
template <typename RunStage>
struct ExitIfEndTimeReached;

template <>
struct ExitIfEndTimeReached<MainRun> {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf("time: %f\n",
                     db::get<::Tags::TimeStepId>(box).substep_time().value());

    return std::tuple<db::DataBox<DbTags>&&, bool>(
        std::move(box),
        db::get<::Tags::TimeStepId>(box).substep_time().value() >=
            db::get<Tags::EndTimeFromFile<MainRun>>(box));
  }
};

template <>
struct ExitIfEndTimeReached<InitializationRun> {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf("pre-run time: %f\n",
                     db::get<::Tags::TimeStepId>(box).substep_time().value());

    const bool end_time_reached =
        (db::get<::Tags::TimeStepId>(box).step_time().value() +
         db::get<::Tags::TimeStep>(box).value()) >=
            db::get<Tags::EndTimeFromFile<InitializationRun>>(box) and
        db::get<::Tags::TimeStepId>(box).is_at_slab_boundary();
    // If the initialization run is complete, send the J hypersurface data to
    // the initialization routine.
    if (end_time_reached) {
      Parallel::printf(
          "Sending data to main evolution component for initialization\n");
      Parallel::receive_data<Cce::ReceiveTags::JHypersurfaceData>(
          Parallel::get_parallel_component<CharacteristicEvolution<
              MainRun, H5WorldtubeBoundary<MainRun, Metavariables>,
              Metavariables>>(cache),
          0_st,
          tuples::TaggedTuple<Tags::BondiJ, Tags::LMax<InitializationRun>,
                              Tags::NumberOfRadialPoints<InitializationRun>>(
              db::get<Tags::BondiJ>(box),
              db::get<Spectral::Swsh::Tags::LMaxBase>(box),
              db::get<Spectral::Swsh::Tags::NumberOfRadialPointsBase>(box)),
          true);
    }

    return std::tuple<db::DataBox<DbTags>&&, bool>(
        std::move(box), end_time_reached);
  }
};

}  // namespace Actions
}  // namespace Cce
