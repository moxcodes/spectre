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
  using const_global_cache_tags = tmpl::list<
      Tags::GlobalSyncTrigger<typename Metavariables::global_sync_triggers>>;
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
    if (should_stop) {
      Parallel::get_parallel_component<ParallelComponent>(cache)
          .stop_for_sync_phases();
    }
    return std::make_tuple(std::move(box), should_stop);
  }
};
}  // namespace Parallel::Actions
