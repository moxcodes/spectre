// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/LoadBalancing/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TaggedTuple.hpp"


namespace Lb {
namespace Actions {

struct InitializeGraphDumpLabel {
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            typename ArrayIndex>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
#ifdef SPECTRE_CHARM_PROJECTIONS
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeGraphDumpLabel, db::AddSimpleTags<Tags::GraphDumpLabel>,
            db::AddComputeTags<>, Initialization::MergePolicy::Overwrite>(
            std::move(box), 0_st));
#endif
#ifndef SPECTRE_CHARM_PROJECTIONS
    return std::make_tuple(std::move(box));
#endif
  }
};

}  // namespace Actions
}  // namespace Lb
