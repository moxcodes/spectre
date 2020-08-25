// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Parallel::Actions {
template <typename Metavariables>
struct ManagePhaseControl {
  using const_global_cache_tags =
      tmpl::list<Tags::AlgorithmControlTriggers<Metavariables>>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& algorithm_control_triggers =
        Parallel::get<Tags::AlgorithmControlTriggers<Metavariables>>(cache)
            .triggers;
    AlgorithmControl<Metavariables> algorithm_control{};
    if (tuples::get<Tags::PauseTrigger<Metavariables>>(
            algorithm_control_triggers)
            ->is_triggered(box)) {
      algorithm_control.execution_flag = AlgorithmExecution::Pause;
    }

    if (tuples::get<Tags::GlobalSyncTrigger<Metavariables>>(
            algorithm_control_triggers)
            ->is_triggered(box)) {
      algorithm_control.execution_flag = AlgorithmExecution::SleepForSyncPhases;
    }

    if (tuples::get<Tags::HaltTrigger<Metavariables>>(
            algorithm_control_triggers)
            ->is_triggered(box)) {
      algorithm_control.execution_flag = AlgorithmExecution::Halt;
    }

    tmpl::for_each<typename Metavariables::global_sync_phases>(
        [&algorithm_control, &algorithm_control_triggers,
         &box](auto tag_v) noexcept {
          using phase_constant = typename decltype(tag_v)::type;
          if (get<Tags::PhaseTrigger<Metavariables, phase_constant>>(
                  algorithm_control_triggers)
                  ->is_triggered(box)) {
            if(not static_cast<bool>(algorithm_control.global_sync_phases)) {
              algorithm_control.global_sync_phases =
                  std::unordered_set<typename Metavariables::Phase>{};
            }
            (*algorithm_control.global_sync_phases)
                .insert(phase_constant::value);
          }
        });

    return std::make_tuple(std::move(box), std::move(algorithm_control));
  }
};
}  // namespace Parallel::Actions
