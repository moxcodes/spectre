// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/Boundary.hpp"
#include "Parallel/Info.hpp"
#include "Utilities/Rational.hpp"

namespace Cce {

namespace Tags {
// Receive tags
struct FinalTimeReceiveTag {
  using temporal_id = int;
  using type = std::unordered_map<temporal_id, double>;
};
}  // namespace Tags

struct InitializeCharacteristic {
  /* options are undergoing changes... we're just going to hard-code the options
   * for now until it becomes obvious how to deal with the options in a real way
   */
  // TODO
  const double start_time = 1000.0;
  const double end_time_input = -1.0;
  const double target_step_size = 1.0;

  const std::string input_file = "CceR0100.h5";
  // \TODO
  using evolution_tags = db::AddSimpleTags<Tags::TimeId>;
  using return_tag_list = tmpl::append<evolution_tags>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // Evolution quantity initialization
    double end_time = end_time_input;
    if(end_time == -1.0) {
      // the temporal_id here isn't really used, there's only one double being
      // communicated during the initialization.
      end_time = tuples::get<Tags::FinalTimeReceiveTag>(inboxes)[0];
    }
    // currently hard-coded to fixed step size
    const Slab single_step_slab{start_time, start_time + target_step_size};
    // In order to finish this we need to figure out how advancing slabs works
    const Time initial_time = single_step_slab.start();
    const TimeDelta fixed_time_step =
        TimeDelta{single_step_slab, Rational{1, 1}};

    auto new_box =
        db::create_from<db::RemoveTags<>, evolution_tags, db::AddComputeTags<>>(

        );

    //
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    // if input has asked the simulation to go to the end, we need to work out
    // where the end is in the input data.
    if(end_time == -1.0) {
      auto& time_obtained == tuples::get<Tags::FinalTimeReceiveTag>(inboxes);
      return time_obtained.size() == 1;
    }
    return true;
  }
};
}  // namespace Cce
