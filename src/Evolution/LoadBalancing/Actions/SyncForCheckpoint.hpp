// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <unordered_map>
#include <vector>

#include "AlgorithmArray.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Evolution/LoadBalancing/Actions/EmulateLoad.hpp"
#include "Evolution/LoadBalancing/Actions/InitializeGraphDumpLabel.hpp"
#include "Evolution/LoadBalancing/Actions/InitializeLoadBalancingTestArray.hpp"
#include "Evolution/LoadBalancing/Actions/LoadBalancingTestCommunication.hpp"
#include "Evolution/LoadBalancing/Actions/StepManagement.hpp"
#include "Evolution/LoadBalancing/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Tags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "Utilities/Gsl.hpp"

namespace Lb {
namespace Actions {


template <typename Metavariables>
struct SyncForCheckpoint {
  using const_global_cache_tags =
      tmpl::list<Tags::CheckpointTrigger<typename Metavariables::triggers>>;

  template <typename DataBox, typename... InboxTags, typename ActionList,
            typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementId<Metavariables::volume_dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if (db::get<Tags::CheckpointTrigger<typename Metavariables::triggers>>(
            box).is_triggered(box)) {
      db::mutate<Parallel::Tags::IsSyncingForCheckpoint>(
          make_not_null(&box),
          [](const gsl::not_null<bool*> is_syncing) noexcept {
            *is_syncing = true;
          });
      cache.get_main_proxy().sync_run_checkpoint();
    }
    return std::forward_as_tuple(std::move(box));
  }
};

struct RestartFromCheckpointWhenReady {
  using inbox_tags =
      tmpl::list<Parallel::ReceiveTags::IsRestartingFromCheckpointSync>;

  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(DataBox& box, tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementId<Metavariables::volume_dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    tuples::get<Parallel::ReceiveTags::IsRestartingFromCheckpointSync>(inboxes)
        .erase(0_st);
    db::mutate<Parallel::Tags::IsSyncingForCheckpoint>(
        make_not_null(&box),
        [](const gsl::not_null<bool*> is_syncing) noexcept {
          *is_syncing = false;
        });
    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    if (db::get<Parallel::Tags::IsSyncingForCheckpoint>(box) and
        tuples::get<Parallel::ReceiveTags::IsRestartingFromCheckpointSync>(
            inboxes)
                .count(0_st) != 1) {
      return false;
    }
    return true;
  }
};

template <typename Metavariables>
using CheckpointSyncAndRestart = tmpl::list<SyncForCheckpoint<Metavariables>,
                                            RestartFromCheckpointWhenReady>;

}  // namespace Actions
}  // namespace Lb
