// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Parallel/Info.hpp"
#include "Time/Tags.hpp"
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
  /* Options are being updated. The correct way to store options will be to put
   * the option tag list in the initialization action (here), and they'll be
   * automatically extracted and parsed from any action placed in the special
   * "Initialization" phase*/
  // TODO
  static constexpr double start_time = 1000.0;
  static constexpr double end_time_input = -1.0;
  static constexpr double target_step_size = 1.0;

  const std::string input_file = "CceR0100.h5";
  // \TODO

  template <typename Metavariables>
  struct EvolutionTags {
    using coordinate_variables_tag =
        typename Metavariables::evolved_coordinates_variables_tag;
    using dt_coordinate_variables_tag =
        db::add_tag_prefix<::Tags::dt, coordinate_variables_tag>;
    using evolution_simple_tags = db::AddSimpleTags<
        ::Tags::TimeId, ::Tags::Next<::Tags::TimeId>, ::Tags::TimeStep,
        ::Tags::HistoryEvolvedVariables<coordinate_variables_tag,
                                        dt_coordinate_variables_tag>,
        ::Tags::HistoryEvolvedTensor<
            typename Metavariables::evolved_swsh_tag,
            typename Metavariables::evolved_swsh_dt_tag>>;
    using evolution_compute_tags = db::AddComputeTags<::Tags::Time>;

    template <typename TagList>
    static auto initialize(
        db::DataBox<TagList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const double initial_time_value, const double /*end_time_value*/,
        const double step_size) noexcept {
      // currently hard-coded to fixed step size
      const Slab single_step_slab{initial_time_value,
                                  initial_time_value + step_size};
      const Time initial_time = single_step_slab.start();
      const TimeDelta fixed_time_step =
          TimeDelta{single_step_slab, Rational{1, 1}};
      const TimeId initial_time_id{true, 0, initial_time};

      const auto& time_stepper = Parallel::get<OptionTags::TimeStepper>(cache);
      const TimeId second_time_id =
          time_stepper.next_time_id(initial_time_id, fixed_time_step);

      typename db::item_type<::Tags::HistoryEvolvedVariables<
          coordinate_variables_tag, dt_coordinate_variables_tag>>
          coordinate_history;

      typename db::item_type<::Tags::HistoryEvolvedTensor<
          typename Metavariables::evolved_swsh_tag,
          typename Metavariables::evolved_swsh_dt_tag>>
          swsh_history;

      return db::create_from<db::RemoveTags<>, evolution_simple_tags,
                             evolution_compute_tags>(
          std::move(box), initial_time_id, second_time_id,
          std::move(coordinate_history), std::move(swsh_history));
    }
  };

  template <class Metavariables>
  using return_tag_list = tmpl::append<
      typename EvolutionTags<Metavariables>::evolution_simple_tags,
      typename EvolutionTags<Metavariables>::evolution_compute_tags>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // Evolution quantity initialization
    double end_time = end_time_input;
    if (end_time == -1.0) {
      // the temporal_id here isn't really used, there's only one double being
      // communicated during the initialization.
      end_time = tuples::get<Tags::FinalTimeReceiveTag>(inboxes)[0];
      // TODO this should clear out the time obtained as necessary
    }

    auto evolution_box = EvolutionTags<Metavariables>::initialize(
        std::move(box), cache, start_time, end_time, target_step_size);
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
    if (end_time_input == -1.0) {
      auto& time_obtained = tuples::get<Tags::FinalTimeReceiveTag>(inboxes);
      return time_obtained.size() == 1;
    }
    return true;
  }
};
}  // namespace Cce
