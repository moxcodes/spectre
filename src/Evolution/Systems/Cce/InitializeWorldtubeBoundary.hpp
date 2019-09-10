// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/CharacteristicExtractor.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"

namespace Cce {

namespace Tags {

struct H5WorldtubeBoundaryDataManager : db::SimpleTag {
  using type = CceH5BoundaryDataManager;
  static std::string name() noexcept {
    return "H5WorldtubeBoundaryDataManager";
  }
};
}  // namespace Tags

namespace InitializationTags {
struct H5WorldtubeBoundaryDataManager : db::SimpleTag {
  using type = CceH5BoundaryDataManager;
  using option_tags =
      tmpl::list<OptionTags::LMax, OptionTags::BoundaryDataFilename,
                 OptionTags::H5LookaheadPoints, OptionTags::H5Interpolator>;

  static CceH5BoundaryDataManager create_from_options(
      const double l_max, const std::string filename,
      const size_t number_of_lookahead_points,
      const std::unique_ptr<Interpolator>& interpolator) noexcept {
    return CceH5BoundaryDataManager(filename, l_max, number_of_lookahead_points,
                                    *interpolator);
  }
  static std::string name() noexcept {
    return "H5WorldtubeBoundaryDataManager";
  }
};

}  // namespace InitializationTags

struct InitializeH5WorldtubeBoundary {
  using initialization_tags =
      tmpl::list<InitializationTags::H5WorldtubeBoundaryDataManager>;

  using const_global_cache_tags = tmpl::list<Tags::LMax>;

  template <class Metavariables>
  using h5_boundary_manager_simple_tags = db::AddSimpleTags<
      ::Tags::Variables<
          typename Metavariables::cce_boundary_communication_tags>,
      Tags::H5WorldtubeBoundaryDataManager>;

  template <class Metavariables>
  using return_tag_list =
      tmpl::append<h5_boundary_manager_simple_tags<Metavariables>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                DbTags, InitializationTags::H5WorldtubeBoundaryDataManager>> =
                nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& l_max = Parallel::get<Tags::LMax>(cache);
    Variables<typename Metavariables::cce_boundary_communication_tags>
        boundary_variables{
            Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    CceH5BoundaryDataManager boundary_data_manager;
    db::mutate<InitializationTags::H5WorldtubeBoundaryDataManager>(
        make_not_null(&box),
        [&boundary_data_manager](const gsl::not_null<CceH5BoundaryDataManager*>
                                     initialization_data_manager) {
          boundary_data_manager = std::move(*initialization_data_manager);
        });
    auto initial_box = Initialization::merge_into_databox<
        InitializeH5WorldtubeBoundary,
        h5_boundary_manager_simple_tags<Metavariables>, db::AddComputeTags<>,
        Initialization::MergePolicy::Overwrite>(
        std::move(box), std::move(boundary_variables),
        std::move(boundary_data_manager));

    return std::make_tuple(std::move(initial_box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<not tmpl::list_contains_v<
                DbTags, InitializationTags::H5WorldtubeBoundaryDataManager>> =
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
}  // namespace Cce
