// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/MakeString.hpp"

namespace gh {
namespace Actions {

struct CheckpointGh {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf(MakeString{}
                     << "In checkpoint action; array index : " << array_index
                     << " at time "
                     << db::get<Tags::TimeStepId>(box).substep_time().value()
                     << "\n");
    return std::make_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace gh
