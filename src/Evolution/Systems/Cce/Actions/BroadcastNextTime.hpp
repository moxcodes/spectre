// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"

#include "Parallel/Printf.hpp"

namespace Cce {
namespace Actions {


template <typename Tag>
struct BroadcastCceTime {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    tmpl::for_each<
        typename Metavariables::components_to_receive_cce_next_time_steps>(
        [&cache, &box](auto component_v) noexcept {
          using component = typename decltype(component_v)::type;
          auto& receiver_proxy =
              Parallel::get_parallel_component<component>(cache);
          // note that this will need to be revised if there is at any point
          // more than one CCE evolution component in the same executable
          Parallel::printf("sending time broadcast : %f\n",
                           db::get<Tag>(box).substep_time().value());
          Parallel::receive_data<intrp::ReceiveTags::NextTime<
              typename Metavariables::CceWorldtubeTarget>>(
              receiver_proxy, db::get<::Tags::TimeStepId>(box),
              db::get<Tag>(box), true);
        });
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Cce
