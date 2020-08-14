// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/OptionTags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"


namespace DgElementArray_detail {

template <typename Metavariables, typename Component,
          typename check = std::void_t<>>
struct has_registration_list_impl : std::false_type {};

template <typename Metavariables, typename Component>
struct has_registration_list_impl<
    Metavariables, Component,
    std::void_t<
        typename Metavariables::template registration_list<Component>::type>>
    : std::true_type {};

template <typename Metavariables, typename Component>
constexpr bool has_registration_list =
    has_registration_list_impl<Metavariables, Component>::value;

}  // namespace DgElementArray_detail

/*!
 * \brief The parallel component responsible for managing the DG elements that
 * compose the computational domain
 *
 * This parallel component will perform the actions specified by the
 * `PhaseDepActionList`.
 */
template <class Metavariables, class PhaseDepActionList>
struct DgElementArray {
  static constexpr size_t volume_dim = Metavariables::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using phase_dependent_action_list = PhaseDepActionList;
  using array_index = ElementId<volume_dim>;

  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<volume_dim>>;

  using array_allocation_tags =
      tmpl::list<domain::Tags::InitialRefinementLevels<volume_dim>>;

  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>,
      array_allocation_tags>;

  template <typename DbTagList, typename ArrayIndex>
  static void pup(PUP::er& p, db::DataBox<DbTagList>& box,
                  typename Metavariables::Phase /*phase*/,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& local_array_index) noexcept {
    // The deregistration and registration below does not actually insert
    // anything into the PUP::er stream, so nothing is done on a sizing pup.
    if constexpr (DgElementArray_detail::has_registration_list<
                      Metavariables, DgElementArray>) {
      using registration_list =
          typename Metavariables::template registration_list<
              DgElementArray>::type;
      if (p.isPacking()) {
        tmpl::for_each<registration_list>(
            [&box, &cache, &local_array_index](auto registration_v) noexcept {
              using registration = typename decltype(registration_v)::type;
              registration::template perform_deregistration<DgElementArray>(
                  box, cache, local_array_index);
            });
      }
      if (p.isUnpacking()) {
        tmpl::for_each<registration_list>(
            [&box, &cache, &local_array_index](auto registration_v) noexcept {
              using registration = typename decltype(registration_v)::type;
              registration::template perform_registration<DgElementArray>(
                  box, cache, local_array_index);
            });
      }
    }
  }

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
          initialization_items) noexcept;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<DgElementArray>(local_cache)
        .start_phase(next_phase);
  }
};

template <class Metavariables, class PhaseDepActionList>
void DgElementArray<Metavariables, PhaseDepActionList>::allocate_array(
    Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
    const tuples::tagged_tuple_from_typelist<initialization_tags>&
        initialization_items) noexcept {
  auto& local_cache = *(global_cache.ckLocalBranch());
  auto& dg_element_array =
      Parallel::get_parallel_component<DgElementArray>(local_cache);
  const auto& domain =
      Parallel::get<domain::Tags::Domain<volume_dim>>(local_cache);
  const auto& initial_refinement_levels =
      get<domain::Tags::InitialRefinementLevels<volume_dim>>(
          initialization_items);
  int which_proc = 0;
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs = initial_refinement_levels[block.id()];
    const std::vector<ElementId<volume_dim>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);
    const int number_of_procs = sys::number_of_procs();
    for (size_t i = 0; i < element_ids.size(); ++i) {
      dg_element_array(ElementId<volume_dim>(element_ids[i]))
          .insert(global_cache, initialization_items, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
  }
  dg_element_array.doneInserting();
}
