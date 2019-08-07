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

template <typename Interpolator>
struct H5WorldtubeBoundaryDataManager : db::SimpleTag{
  using type = CceH5BoundaryDataManager<Interpolator>;
  static std::string name() noexcept {
    return "H5WorldtubeBoundaryDataManager(" +
           pretty_type::get_name<Interpolator>() + ")";
  }
};
}  // namespace Tags

struct InitializeH5WorldtubeBoundary {
  // TODO hard-coded options
  static std::string input_filename() noexcept { return "CceR0100"; }
  using interpolator = CubicInterpolator;
  static const size_t number_of_lookahead_points = 200;
  // \TODO

  template <class Metavariables>
  using h5_boundary_manager_simple_tags = db::AddSimpleTags<
      Tags::H5WorldtubeBoundaryDataManager<interpolator>,
      ::Tags::Variables<
          typename Metavariables::cce_boundary_communication_tags>>;

  template <class Metavariables>
  using return_tag_list =
      tmpl::append<h5_boundary_manager_simple_tags<Metavariables>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& l_max = Parallel::get<OptionTags::LMax>(cache);

    CceH5BoundaryDataManager<interpolator> h5_boundary_data_manager{
        input_filename() + ".h5", l_max, number_of_lookahead_points};

    Variables<typename Metavariables::cce_boundary_communication_tags>
        boundary_variables{
            Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    auto initial_box =
        db::create<h5_boundary_manager_simple_tags<Metavariables>,
                   db::AddComputeTags<>>(std::move(h5_boundary_data_manager),
                                         std::move(boundary_variables));
    return std::make_tuple(std::move(initial_box));
  }
};
}  // namespace Cce
