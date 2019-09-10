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
  using type = ScriPlusInterpolationManager<ToInterpolate>;
  static std::string name() noexcept { return "InterpolationManager"; }
};
}  // namespace Tags

namespace Actions {

struct InitializeCharacteristicScri {
  using const_global_cache_tags = tmpl::list<Tags::ScriInterpolationPoints>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const size_t l_max = Parallel::get<Tags::LMax>(cache);
    const size_t number_of_scri_interpolation_points =
        Parallel::get<Tags::ScriInterpolationPoints>(cache);
    auto initialize_box =
        db::create<db::AddSimpleTags<Tags::InterpolationManager<
                       ComplexDataVector, Tags::News>>,
                   db::AddComputeTags<>>(
            ScriPlusInterpolationManager<ComplexDataVector>{
                number_of_scri_interpolation_points,
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
                FlexibleBarycentricInterpolator{}});
    return std::make_tuple(std::move(initialize_box));
  }
};
}  // namespace Actions
}  // namespace Cce
