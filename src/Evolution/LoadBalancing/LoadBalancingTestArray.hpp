// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <unordered_map>
#include <vector>

#include "AlgorithmArray.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Evolution/LoadBalancing/Actions/EmulateLoad.hpp"
#include "Evolution/LoadBalancing/Actions/InitializeLoadBalancingTestArray.hpp"
#include "Evolution/LoadBalancing/Actions/StepManagement.hpp"
#include "Evolution/LoadBalancing/Actions/LoadBalancingTestCommunication.hpp"
#include "Evolution/LoadBalancing/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "Utilities/Gsl.hpp"

namespace Lb {

template <typename Metavariables>
struct LoadBalancingTestArray {
  static constexpr size_t volume_dim = Metavariables::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  using array_index =  ElementId<volume_dim>;
  using metavariables = Metavariables;

  using initialization_action_list =
      tmpl::list<Actions::InitializeLoadBalancingTestArray<volume_dim>,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>;
  using evolution_action_list =
      tmpl::list<Actions::ExitIfComplete,
                 Actions::SendDataToNeighbors<volume_dim>,
                 Actions::ReceiveDataFromNeighbors<volume_dim>,
                 Actions::EmulateLoad<>,
                 Actions::IncrementTime>;

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

  using const_global_cache_tag_list =
      Parallel::get_const_global_cache_tags_from_actions<
          phase_dependent_action_list>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>&
      global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<LoadBalancingTestArray<Metavariables>>(
        local_cache)
        .start_phase(next_phase);
  }

  static void allocate_array(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
          initialization_items) noexcept;
};

template <class Metavariables>
void LoadBalancingTestArray<Metavariables>::allocate_array(
    Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
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
  typename Metavariables::distribution_strategy distribution{
      domain, initial_refinement_levels,
      static_cast<size_t>(Parallel::number_of_procs())};
  for (const auto& block : domain.blocks()) {
    const std::vector<ElementId<volume_dim>> element_ids =
        initial_element_ids(block.id(), initial_refinement_levels[block.id()]);
    for (size_t i = 0; i < element_ids.size(); ++i) {
      lb_element_array(ElementId<volume_dim>{element_ids[i]})
          .insert(global_cache, initialization_items,
                  distribution.which_proc(
                      local_cache, ElementId<volume_dim>{element_ids[i]}));
    }
  }
}
}  // namespace Lb
