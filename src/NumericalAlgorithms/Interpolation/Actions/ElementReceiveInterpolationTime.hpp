// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/optional.hpp>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"


namespace intrp {
namespace Actions {

template <typename InterpolationTargetTag>
struct ElementReceiveInterpolationTime {
  using inbox_tags =
      tmpl::list<intrp::ReceiveTags::NextTime<InterpolationTargetTag>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static constexpr size_t dim = Metavariables::volume_dim;
    // Get element logical coordinates.
    const auto& block_logical_coords =
        get<Vars::PointInfoTag<InterpolationTargetTag, dim>>(
            db::get<Tags::InterpPointInfo<Metavariables>>(box));
    const std::vector<ElementId<dim>> element_ids{{array_index}};
    const auto element_coord_holders =
        element_logical_coordinates(element_ids, block_logical_coords);
    Parallel::printf("checking at time %f, array index %d\n",
                     db::get<::Tags::TimeStepId>(box).substep_time().value(),
                     array_index);

    // There is exactly one element_id in the list of element_ids.
    if (element_coord_holders.count(element_ids[0]) == 0) {
      // There are no points in this element, so we don't need
      // to shuffle around the time steps, just empty inbox
      tuples::get<ReceiveTags::NextTime<InterpolationTargetTag>>(inboxes)
          .clear();
      db::mutate<Tags::NextInterpolationTimeInFuture<InterpolationTargetTag>>(
          make_not_null(&box),
          [](const gsl::not_null<bool*>
                 next_interpolation_time_in_future) noexcept {
            *next_interpolation_time_in_future = true;
          });
      return std::forward_as_tuple(std::move(box));
    }

    if (not static_cast<bool>(
            db::get<Tags::NextInterpolationTimeStepId<InterpolationTargetTag>>(
                box))) {
      // the inbox for this is a std::map, so the first element is the earliest
      db::mutate<Tags::NextInterpolationTimeStepId<InterpolationTargetTag>>(
          make_not_null(&box),
          [&inboxes](const gsl::not_null<boost::optional<TimeStepId>*>
                         next_cce_time) noexcept {
            *next_cce_time = boost::optional<TimeStepId>(
                (*tuples::get<
                      intrp::ReceiveTags::NextTime<InterpolationTargetTag>>(
                      inboxes)
                      .begin())
                    .second);
            Parallel::printf("received and loaded new time : %f\n",
                             (**next_cce_time).substep_time().value());
            tuples::get<intrp::ReceiveTags::NextTime<InterpolationTargetTag>>(
                inboxes)
                .erase(
                    tuples::get<
                        intrp::ReceiveTags::NextTime<InterpolationTargetTag>>(
                        inboxes)
                        .begin());
          });
    }
    db::mutate<Tags::NextInterpolationTimeInFuture<InterpolationTargetTag>>(
        make_not_null(&box),
        [](const gsl::not_null<bool*> next_interpolation_time_in_future,
           const boost::optional<TimeStepId>& next_interpolation_time_step,
           const TimeStepId& this_time_step, const TimeStepId& next_time_step,
           const TimeStepper& timestepper) noexcept {
          *next_interpolation_time_in_future =
              (*next_interpolation_time_step).substep_time().value() >
                  next_time_step.substep_time().value() or
              this_time_step.substep() + 1 != timestepper.number_of_substeps();
          if (this_time_step.substep() + 1 ==
                  timestepper.number_of_substeps() and
              (*next_interpolation_time_step).substep_time().value() <
                  this_time_step.step_time().value()) {
            Parallel::printf(
                "next interpolation time: %f, this time step: %f\n",
                (*next_interpolation_time_step).substep_time().value(),
                this_time_step.step_time().value());
            ERROR(
                "The DG evolution has advanced beyond the next requested "
                "interpolation time. This indicates a flaw in the logic of "
                "`ElementReceiveInterpolationTime` or the action list control "
                "flow. Interpolation time: ");
          }
        },
        db::get<Tags::NextInterpolationTimeStepId<InterpolationTargetTag>>(box),
        db::get<::Tags::TimeStepId>(box),
        db::get<::Tags::Next<::Tags::TimeStepId>>(box),
        db::get<::Tags::TimeStepper<TimeStepper>>(box));
    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index) noexcept {
    static constexpr size_t dim = Metavariables::volume_dim;
    // Get element logical coordinates.
    const auto& block_logical_coords =
        get<Vars::PointInfoTag<InterpolationTargetTag, dim>>(
            db::get<Tags::InterpPointInfo<Metavariables>>(box));
    const std::vector<ElementId<dim>> element_ids{{array_index}};
    const auto element_coord_holders =
        element_logical_coordinates(element_ids, block_logical_coords);

    // There is exactly one element_id in the list of element_ids.
    if (element_coord_holders.count(element_ids[0]) == 0) {
      // There are no points in this element, so we don't need
      // to wait for the CCE time
      return true;
    }
    return tuples::get<ReceiveTags::NextTime<InterpolationTargetTag>>(inboxes)
                   .size() > 0 or
           static_cast<bool>(
               db::get<
                   Tags::NextInterpolationTimeStepId<InterpolationTargetTag>>(
                   box));
  }
};

}  // namespace Actions
}  // namespace intrp
