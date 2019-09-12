// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/Systems/Cce/InitializeCce.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ScriPlusInterpolationManager.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Parallel/Info.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Time/Tags.hpp"

namespace Cce {

namespace Tags {

template <typename ToInterpolate, typename ObservationTag>
struct InterpolationManager : db::SimpleTag {
  using type = ScriPlusInterpolationManager<ToInterpolate, ObservationTag>;
  static std::string name() noexcept { return "InterpolationManager"; }
};
}  // namespace Tags

namespace Actions {

struct InitializeCharacteristicScri {
  using const_global_cache_tags = tmpl::list<Tags::ScriInterpolationPoints>;

  template <typename Metavariables>
  using interpolation_managers = tmpl::flatten<tmpl::list<
      tmpl::transform<typename Metavariables::scri_values_to_observe,
                      tmpl::bind<Tags::InterpolationManager, ComplexDataVector,
                                 tmpl::_1>>>>;

  template <typename DbTags, typename... InterpolationManagerTags>
  static auto create_interpolation_managers_box(
      db::DataBox<DbTags>& box, const size_t l_max,
      const size_t number_of_scri_interpolation_points,
      tmpl::list<InterpolationManagerTags...>
      /*meta*/) noexcept {
    return Initialization::merge_into_databox<
        InitializeCharacteristicScri,
        db::AddSimpleTags<InterpolationManagerTags...>, db::AddComputeTags<>>(
        std::move(box),
        db::item_type<InterpolationManagerTags>{
            number_of_scri_interpolation_points,
            Spectral::Swsh::number_of_swsh_collocation_points(l_max),
            FlexibleBarycentricInterpolator{}}...);
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<not tmpl::list_contains_v<
                DbTags, tmpl::front<interpolation_managers<Metavariables>>>> =
                nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const size_t l_max = Parallel::get<Tags::LMax>(cache);
    const size_t number_of_scri_interpolation_points =
        Parallel::get<Tags::ScriInterpolationPoints>(cache);
    auto initialize_box = create_interpolation_managers_box(
        box, l_max, number_of_scri_interpolation_points,
        interpolation_managers<Metavariables>{});
    return std::make_tuple(std::move(initialize_box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                DbTags, tmpl::front<interpolation_managers<Metavariables>>>> =
                nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Cce
