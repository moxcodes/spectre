// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep  // for Tags::Next
#include "Parallel/GlobalCache.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class TimeDelta;
class TimeStepId;
namespace Tags {
template <typename Tag>
struct Next;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

/// Adjust the step size for local time stepping
///
/// \note this is a free function version of `Actions::ChangeStepSize`.
/// This free function alternative permits the inclusion of the time step
/// procedure in the middle of another action. When used as a free function, the
/// calling action is responsible for specifying the const global cache tags
/// needed by this function (`Tags::StepChoosers<StepChooserRegistrars>`,
/// `Tags::StepController`).
template <typename StepChooserRegistrars, typename DbTags,
          typename Metavariables>
bool change_step_size(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const Parallel::GlobalCache<Metavariables>& cache) noexcept {
  using step_choosers_tag = Tags::StepChoosers<StepChooserRegistrars>;

  const LtsTimeStepper& time_stepper = db::get<Tags::TimeStepper<>>(*box);
  const auto& step_choosers = db::get<step_choosers_tag>(*box);
  const auto& step_controller = db::get<Tags::StepController>(*box);

  const auto& time_id = db::get<Tags::TimeStepId>(*box);
  const auto& history = db::get<Tags::HistoryEvolvedVariables<>>(*box);

  if (not time_stepper.can_change_step_size(time_id, history)) {
    return true;
  }

  const auto& current_step = db::get<Tags::TimeStep>(*box);

  const double last_step_size = history.size() > 0
                                    ? db::get<Tags::TimeStep>(*box).value()
                                    : std::numeric_limits<double>::infinity();

  // The step choosers return the magnitude of the desired step, so
  // we always want the minimum requirement, but we have to negate
  // the final answer if time is running backwards.
  double desired_step = std::numeric_limits<double>::infinity();
  bool step_accepted = true;
  for (const auto& step_chooser : step_choosers) {
    const auto [step_choice, step_choice_accepted] =
        step_chooser->desired_step(box, last_step_size, cache);
    desired_step = std::min(desired_step, step_choice);
    step_accepted = step_accepted and step_choice_accepted;
  }
  if (not current_step.is_positive()) {
    desired_step = -desired_step;
  }

  const auto new_step = step_controller.choose_step(
      db::get<Tags::Next<Tags::TimeStepId>>(*box).step_time(), desired_step);
  if (new_step != current_step) {
    db::mutate<Tags::Next<Tags::TimeStep>>(
        box, [&new_step](const gsl::not_null<TimeDelta*> next_step) noexcept {
          *next_step = new_step;
        });
  }
  // if step accepted, just proceed. Otherwise, change Time::Next and jump
  // back to the first instance of `UpdateU`.
  if (step_accepted) {
    if constexpr (Metavariables::debug_volume_step_observation) {
      db::mutate<Tags::VolumeStep>(
          box,
          [](const gsl::not_null<Scalar<DataVector>*> volume_time_step,
             const TimeDelta& time_step) noexcept {
            get(*volume_time_step) = time_step.value();
          },
          db::get<Tags::TimeStep>(*box));
    }
    return true;
  } else {
    db::mutate<Tags::Next<Tags::TimeStepId>, Tags::TimeStep>(
        box, [&time_stepper, &time_id, &new_step](
                 const gsl::not_null<TimeStepId*> next_time_id,
                 const gsl::not_null<TimeDelta*> time_step) noexcept {
          *next_time_id = time_stepper.next_time_id(
              time_id, new_step.with_slab(time_id.step_time().slab()));
          *time_step = new_step.with_slab(time_id.step_time().slab());
        });
    return false;
  }
}

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Adjust the step size for local time stepping
///
/// Uses:
/// - GlobalCache:
///   - Tags::StepChoosers<StepChooserRegistrars>
///   - Tags::StepController
/// - DataBox:
///   - Tags::HistoryEvolvedVariables
///   - Tags::TimeStep
///   - Tags::TimeStepId
///   - Tags::TimeStepper<>
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: Tags::Next<Tags::TimeStepId>, Tags::TimeStep
template <typename StepChooserRegistrars>
struct ChangeStepSize {
  using step_choosers_tag = Tags::StepChoosers<StepChooserRegistrars>;
  using const_global_cache_tags =
      tmpl::list<step_choosers_tag, Tags::StepController>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&, bool, size_t> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    static_assert(Metavariables::local_time_stepping,
                  "ChangeStepSize can only be used with local time-stepping.");
    const bool step_successful =
        change_step_size<StepChooserRegistrars>(make_not_null(&box), cache);
    if (step_successful) {
      return {std::move(box), false,
              tmpl::index_of<ActionList, ChangeStepSize>::value};
    } else {
      if constexpr (std::is_same_v<
                        tmpl::index_if<
                            ActionList,
                            tt::is_a_lambda<Actions::UpdateU, tmpl::_1>>,
                        tmpl::no_such_type_>) {
        ERROR(
            "Step not successful, and there is no UpdateU action to return "
            "to.");
      } else {
        return {
            std::move(box), false,
            tmpl::index_if<ActionList,
                           tt::is_a_lambda<Actions::UpdateU, tmpl::_1>>::value};
      }
    }
  }
};
}  // namespace Actions
