// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace PhaseControl {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Check if any triggers are activated, and perform phase changes as
 * needed.
 *
 * This action iterates over the `Tags::PhaseChangeAndTriggers`, sending
 * reduction data for the phase decision for each triggered `PhaseChange`, then
 * halt algorithm execution so that the `Main` chare can make a phase decision
 * if any were triggered.
 *
 * Uses:
 * - GlobalCache: `Tags::PhaseChangeAndTriggers`
 * - DataBox: As specified by the `PhaseChange` option-created objects.
 *   - `PhaseChange` objects are permitted to perform mutations on the
 *     \ref DataBoxGroup "DataBox" to store persistent state information.
 */
template <typename PhaseChangeRegistrars, typename TriggerRegistrars>
struct ExecutePhaseChange {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&, Parallel::AlgorithmExecution> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*component*/) noexcept {
    const auto& phase_control_and_triggers = Parallel::get<
        Tags::PhaseChangeAndTriggers<PhaseChangeRegistrars, TriggerRegistrars>>(
        cache);
    bool should_halt = false;
    for (const auto& [trigger, phase_controls] : phase_control_and_triggers) {
      if (trigger->is_triggered(box)) {
        for (const auto& phase_control : phase_controls) {
          phase_control->template contribute_phase_data<ParallelComponent>(
              make_not_null(&box), cache, array_index);
        }
        should_halt = true;
      }
    }
    // if we halt, we need to make sure that the Main chare knows that it is
    // because we are requesting phase change arbitration, regardless of what
    // data was actually sent to make that decision.
    if (should_halt) {
      if constexpr (std::is_same_v<typename ParallelComponent::chare_type,
                    Parallel::Algorithms::Array>) {
        Parallel::contribute_to_phase_change_reduction<ParallelComponent>(
            tuples::TaggedTuple<TagsAndCombines::UsePhaseChangeArbitration>{
                true},
            cache, array_index);
      } else {
        Parallel::contribute_to_phase_change_reduction<ParallelComponent>(
            tuples::TaggedTuple<TagsAndCombines::UsePhaseChangeArbitration>{
                true},
            cache);
      }
    }
    return {std::move(box), should_halt
                                ? Parallel::AlgorithmExecution::Halt
                                : Parallel::AlgorithmExecution::Continue};
  }
};
}  // namespace Actions

/*!
 * \brief Use the runtime data aggregated in `phase_change_decision_data` to
 * decide which phase to execute next.
 *
 * \details This function will iterate through each of the option-created pairs
 * of `PhaseChange`s, and obtain from each a
 * `std::optional<std::pair<Metavariables::Phase,
 * PhaseControl::ArbitrationStrategy>`. Any `std::nullopt` is skipped. If all
 * `PhaseControl`s provide `std::nullopt`, the phase will either keep its
 * current value (if the halt was caused by one of the trigers associated with
 * an  option-created `PhaseChange`), or this function will return a
 * `std::nullopt` as well (otherwise), indicating that the phase should proceed
 * according to other information, such as global ordering.
 *
 * In the case of a `PhaseControl::ArbitrationStrategy::RunPhaseImmediately`,
 * the first such return value is immediately run, and no further `PhaseChange`s
 * are queried for their input.

 */
template <typename PhaseChangeRegistrars, typename TriggerRegistrars,
          typename... DecisionTags, typename Metavariables>
typename std::optional<typename Metavariables::Phase> arbitrate_phase_control(
    const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
        phase_change_decision_data,
    typename Metavariables::Phase current_phase,
    const Parallel::GlobalCache<Metavariables>& cache) noexcept {
  const auto& phase_control_and_triggers = Parallel::get<
      Tags::PhaseChangeAndTriggers<PhaseChangeRegistrars, TriggerRegistrars>>(
      cache);
  bool phase_chosen = false;
  for (const auto& [trigger, phase_controls] : phase_control_and_triggers) {
    // avoid unused variable warning
    (void)trigger;
    for (const auto& phase_control : phase_controls) {
      const auto phase_result = phase_control->arbitrate_phase_control(
          phase_change_decision_data, current_phase, cache);
      if (phase_result.has_value()) {
        if (phase_result.value().second ==
            ArbitrationStrategy::RunPhaseImmediately) {
          tuples::get<TagsAndCombines::UsePhaseChangeArbitration>(
              *phase_change_decision_data) = false;
          return phase_result.value().first;
        }
        current_phase = phase_result.value().first;
        phase_chosen = true;
      }
    }
  }
  if (tuples::get<TagsAndCombines::UsePhaseChangeArbitration>(
          *phase_change_decision_data) == false and
      not phase_chosen) {
    return std::nullopt;
  }
  // if no phase control object suggests a specific phase, return to execution
  // in the current phase.
  tuples::get<TagsAndCombines::UsePhaseChangeArbitration>(
      *phase_change_decision_data) = false;
  return current_phase;
}

/*!
 * \brief Initialize the Main chare's `phase_change_decision_data` for the
 * option-selected `PhaseChange`s.
 *
 * \details This struct provides a convenient method of specifying the
 * initialization of the `phase_change_decision_data`. To instruct the Main
 * chare to use this initialization routine, define the type alias:
 * ```
 * using initialize_phase_change_decision_data =
 *   PhaseControl::initialize_phase_change_decision_data<
 *       phase_change_registrars, trigger_registrars>;
 * ```
 */
template <typename PhaseChangeRegistrars, typename TriggerRegistrars>
struct initialize_phase_change_decision_data {
  template <typename... DecisionTags, typename Metavariables>
  static void apply(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data,
      const Parallel::GlobalCache<Metavariables>& cache) noexcept {
    tuples::get<TagsAndCombines::UsePhaseChangeArbitration>(
        *phase_change_decision_data) = false;
    const auto& phase_control_and_triggers = Parallel::get<
        Tags::PhaseChangeAndTriggers<PhaseChangeRegistrars, TriggerRegistrars>>(
        cache);
    for (const auto& [trigger, phase_controls] : phase_control_and_triggers) {
      for (const auto& phase_control : phase_controls) {
        phase_control->initialize_phase_change_decision_data(
            phase_change_decision_data);
      }
    }
  }
};
}  // namespace PhaseControl
