// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Parallel::Actions {
template <typename Metavariables>
struct CheckForGlobalSync {
  using const_global_cache_tags = tmpl::flatten<tmpl::list<
      Tags::GlobalSyncTrigger<typename Metavariables::global_sync_triggers>,
      Tags::TerminationTrigger<typename Metavariables::global_sync_triggers>,
      tmpl::transform<
          typename Metavariables::global_sync_phases,
          tmpl::bind<
              Tags::PhaseTrigger, tmpl::pin<Metavariables>, tmpl::_1,
              tmpl::pin<typename Metavariables::global_sync_triggers>>>>>;
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    bool should_stop = db::get<Tags::GlobalSyncTrigger<
        typename Metavariables::global_sync_triggers>>(box)
                           .is_triggered(box);
    bool should_terminate = db::get<Tags::TerminationTrigger<
        typename Metavariables::global_sync_triggers>>(box)
                                .is_triggered(box);
    std::vector<typename Metavariables::Phase> requested_phases{};
    tmpl::for_each<typename Metavariables::global_sync_phases>(
        [&requested_phases, &box](auto tag_v) noexcept {
          using phase_constant = typename decltype(tag_v)::type;
          if (db::get<Tags::PhaseTrigger<
                  Metavariables, phase_constant,
                  typename Metavariables::global_sync_triggers>>(box)
                  .is_triggered(box)) {
            requested_phases.push_back(phase_constant::value);
          }
        });
    return std::make_tuple(
        std::move(box), should_stop or should_terminate,
        tmpl::index_of<ActionList, CheckForGlobalSync>::value + 1,
        should_terminate ? std::vector<typename Metavariables::Phase>{}
                         : requested_phases,
        should_stop and not should_terminate, should_terminate);
  }
};

}  // namespace Parallel::Actions
