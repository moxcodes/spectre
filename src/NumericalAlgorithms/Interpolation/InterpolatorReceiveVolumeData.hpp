// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <sstream>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/AddTemporalIdsToInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Adds volume data from an `Element`.
///
/// Attempts to interpolate if it already has received target points from
/// any InterpolationTargets.
///
/// Uses:
/// - DataBox:
///   - `Tags::NumberOfElements`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::VolumeVarsInfo<Metavariables>`
///   - `Tags::InterpolatedVarsHolders<Metavariables>`
struct InterpolatorReceiveVolumeData {
  template <
      typename ParallelComponent, typename DbTags, typename Metavariables,
      typename ArrayIndex, size_t VolumeDim,
      Requires<tmpl::list_contains_v<DbTags, Tags::NumberOfElements>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const typename Metavariables::temporal_id::type& temporal_id,
      const ElementId<VolumeDim>& element_id, const ::Mesh<VolumeDim>& mesh,
      Variables<typename Metavariables::interpolator_source_vars>&&
          vars) noexcept {
    bool add_temporal_ids_to_targets = false;
    db::mutate<Tags::VolumeVarsInfo<Metavariables>>(
        make_not_null(&box),
        [
          &add_temporal_ids_to_targets, &temporal_id, &element_id, &mesh, &vars
        ](const gsl::not_null<
            db::item_type<Tags::VolumeVarsInfo<Metavariables>>*>
              container) noexcept {
          if (container->find(temporal_id) == container->end()) {
            add_temporal_ids_to_targets = true;
            container->emplace(
                temporal_id,
                std::unordered_map<
                    ElementId<VolumeDim>,
                    typename Tags::VolumeVarsInfo<Metavariables>::Info>{});
          }
          container->at(temporal_id)
              .emplace(std::make_pair(
                  element_id,
                  typename Tags::VolumeVarsInfo<Metavariables>::Info{
                      mesh, std::move(vars)}));
        });

    // Try to interpolate data for all InterpolationTargets.
    tmpl::for_each<typename Metavariables::interpolation_target_tags>([
      &add_temporal_ids_to_targets, &box, &cache, &temporal_id
    ](auto tag_v) noexcept {
      using tag = typename decltype(tag_v)::type;

      // The first time (and only the first time) that this interpolator
      // is called at this temporal_id, tell all the
      // InterpolationTargets that they will interpolate at this
      // temporal_id.
      if (add_temporal_ids_to_targets) {
        auto& target = Parallel::get_parallel_component<
            InterpolationTarget<Metavariables, tag>>(cache);
        Parallel::simple_action<AddTemporalIdsToInterpolationTarget<tag>>(
            target, std::vector<typename Metavariables::temporal_id::type>{
                        temporal_id});
      }

      try_to_interpolate<tag>(make_not_null(&box), make_not_null(&cache),
                              temporal_id);
    });

    // The following is output for debugging.
    size_t volume_data_allocated = 0;
    const auto& volume_info = db::get<Tags::VolumeVarsInfo<Metavariables>>(box);
    for(const auto& time_id: volume_info) {
      for(const auto& elem_id: time_id.second) {
        volume_data_allocated += elem_id.second.vars.size();
      }
    }
    size_t interp_data_allocated = 0;
    const auto& holders =
        db::get<Tags::InterpolatedVarsHolders<Metavariables>>(box);
    tmpl::for_each<typename Metavariables::interpolation_target_tags>([&](
        auto tag) noexcept {
      using Tag = typename decltype(tag)::type;
      const auto& holder = get<Vars::HolderTag<Tag, Metavariables>>(holders);
      for (const auto& info : holder.infos) {
        for (const auto& var: info.second.vars) {
          interp_data_allocated += var.size();
        }
      }
    });
    std::ostringstream oss;
    oss << temporal_id;
    Parallel::printf(
        "Proc %zu node %zu: End of InterpolatorReceiveVolumeData "
        "time %s: : volume holds %zu doubles, interp holds %zu doubles\n",
        Parallel::my_proc(), Parallel::my_node(),
        oss.str(),
        volume_data_allocated, interp_data_allocated);
  }
};

}  // namespace Actions
}  // namespace intrp
