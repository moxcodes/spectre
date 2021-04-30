// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Evolution/LoadBalancing/Actions/EmulateLoad.hpp"
#include "Evolution/LoadBalancing/Actions/InitializeGraphDumpLabel.hpp"
#include "Evolution/LoadBalancing/Actions/InitializeLoadBalancingTestArray.hpp"
#include "Evolution/LoadBalancing/Actions/LoadBalancingTestCommunication.hpp"
#include "Evolution/LoadBalancing/Actions/StepManagement.hpp"
#include "Evolution/LoadBalancing/Tags.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "Utilities/Gsl.hpp"

namespace Lb {
namespace detail {
CREATE_HAS_STATIC_MEMBER_VARIABLE(use_z_order_distribution)
CREATE_HAS_STATIC_MEMBER_VARIABLE_V(use_z_order_distribution)
}  // namespace detail


template <typename Metavariables>
struct LoadBalancingTestArray {
  static constexpr size_t volume_dim = Metavariables::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  using array_index = ElementId<volume_dim>;
  using metavariables = Metavariables;
  static constexpr bool use_at_sync = Metavariables::use_at_sync;

  using initialization_action_list =
      tmpl::list<Actions::InitializeLoadBalancingTestArray<volume_dim>,
                 Actions::InitializeGraphDumpLabel<Metavariables>,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>;
  using evolution_action_list =
      tmpl::list<Actions::ExitIfComplete,
                 Actions::SendDataToNeighbors<volume_dim>,
                 Actions::ReceiveDataFromNeighbors<volume_dim>,
                 Actions::EmulateLoad<>, Actions::IncrementTime,
                 PhaseControl::Actions::ExecutePhaseChange<
                     typename Metavariables::phase_changes,
                     typename Metavariables::triggers>>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        initialization_action_list>,
                 Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Evolve,
                                        evolution_action_list>>;

  using array_allocation_tags =
      tmpl::list<domain::Tags::InitialRefinementLevels<volume_dim>>;

  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>,
      array_allocation_tags>;

  using const_global_cache_tags =
      tmpl::push_back<Parallel::get_const_global_cache_tags_from_actions<
                          phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>&
          global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<LoadBalancingTestArray<Metavariables>>(
        local_cache)
        .start_phase(next_phase);
  }

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
          initialization_items) noexcept;
};

struct SetMigratable {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) noexcept {
    auto& lb_element_array =
        Parallel::get_parallel_component<LoadBalancingTestArray<Metavariables>>(
            cache);
    lb_element_array(array_index).ckLocal()->setMigratable(true);
  }
};

template <class Metavariables>
void LoadBalancingTestArray<Metavariables>::allocate_array(
    Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
    const tuples::tagged_tuple_from_typelist<initialization_tags>&
        initialization_items) noexcept {
  auto& local_cache = *(global_cache.ckLocalBranch());
  auto& lb_element_array =
      Parallel::get_parallel_component<LoadBalancingTestArray>(local_cache);
  const auto& domain =
      Parallel::get<domain::Tags::Domain<volume_dim>>(local_cache);
  const auto& initial_refinement_levels =
      get<domain::Tags::InitialRefinementLevels<volume_dim>>(
          initialization_items);
  bool use_z_order_distribution = true;
  if constexpr (detail::has_use_z_order_distribution_v<Metavariables>) {
    use_z_order_distribution = Metavariables::use_z_order_distribution;
  }
  int which_proc = 0;
  const domain::BlockZCurveProcDistribution<volume_dim> element_distribution{
      static_cast<size_t>(sys::number_of_procs()), initial_refinement_levels};
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs = initial_refinement_levels[block.id()];
    const std::vector<ElementId<volume_dim>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);
    if (use_z_order_distribution) {
      for (const auto& element_id : element_ids) {
        const size_t target_proc = element_distribution.get_proc_for_element(
            block.id(), ElementId<volume_dim>(element_id));
        lb_element_array(ElementId<volume_dim>(element_id))
            .insert(global_cache, initialization_items, target_proc);
      }
    } else {
      const int number_of_procs = sys::number_of_procs();
      for (size_t i = 0; i < element_ids.size(); ++i) {
        lb_element_array(ElementId<volume_dim>(element_ids[i]))
            .insert(global_cache, initialization_items, which_proc);
        which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
      }
    }
  }
  lb_element_array.doneInserting();
}
}  // namespace Lb
