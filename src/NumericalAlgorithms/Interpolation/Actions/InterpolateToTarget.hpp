// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/InterpolationTargetVarsFromElement.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Interpolation/PointInfoTag.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Interpolates and sends points to an InterpolationTarget.
///
/// This is invoked on DgElementArray.
///
/// Uses:
/// - DataBox:
///   - `intrp::Tags::InterpPointInfo<Metavariables>`
///   - `Tags::Mesh<Metavariables::volume_dim>`
///   - Variables tagged by
///     InterpolationTargetTag::vars_to_interpolate_to_target
///   - `::Tags::TimeStepId`
///   - `::Tags::HistoryEvolvedVariables<>`
/// - ConstGlobalCache
///   - `::Tags::TimeStepper<>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: nothing
template <typename InterpolationTargetTag>
struct InterpolateToTarget {
  template <typename DbTags, typename Metavariables, typename... InboxTags,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Exit early if we are on a substep and we don't want to interpolate
    // substeps
    if (not InterpolationTargetTag::should_interpolate(box)) {
      return std::forward_as_tuple(std::move(box));
    }
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
      // to do anything.
      return std::forward_as_tuple(std::move(box));
    }

    // There are points in this element, so interpolate to them and
    // send the interpolated data to the target.  This is done
    // in several steps:
    const auto& element_coord_holder = element_coord_holders.at(element_ids[0]);

    using compute_items_in_new_box = tmpl::list_difference<
        typename InterpolationTargetTag::compute_items_on_source, DbTags>;

    // 1. Create a new DataBox that contains
    // InterpolationTargetTag::compute_items_on_source
    auto new_box =
        db::create_from<db::RemoveTags<>, db::AddSimpleTags<>,
                        db::AddComputeTags<compute_items_in_new_box>>(
            std::move(box));

    // 2. Set up local variables to hold vars_to_interpolate and fill it
    const auto& mesh = db::get<domain::Tags::Mesh<dim>>(new_box);
    Variables<typename InterpolationTargetTag::vars_to_interpolate_to_target>
        local_vars(mesh.number_of_grid_points());

    tmpl::for_each<
        typename InterpolationTargetTag::vars_to_interpolate_to_target>(
        [&new_box, &local_vars ](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          get<tag>(local_vars) = db::get<tag>(new_box);
        });

    // 3. Set up interpolator
    intrp::Irregular<dim> interpolator(
        mesh, element_coord_holder.element_logical_coords);

    // 4. Interpolate and send interpolated data to target
    auto& receiver_proxy = Parallel::get_parallel_component<
        InterpolationTarget<Metavariables, InterpolationTargetTag>>(cache);
    Parallel::simple_action<
        Actions::InterpolationTargetVarsFromElement<InterpolationTargetTag>>(
        receiver_proxy,
        std::vector<Variables<
            typename InterpolationTargetTag::vars_to_interpolate_to_target>>(
            {interpolator.interpolate(local_vars)}),
        std::vector<std::vector<size_t>>({element_coord_holder.offsets}),
        db::get<typename Metavariables::temporal_id>(new_box));

    // 5. Put back original DataBox.
    box = db::create_from<db::RemoveTags<compute_items_in_new_box>,
                          db::AddSimpleTags<>>(std::move(new_box));

    return std::forward_as_tuple(std::move(box));
  }
};

/// \ingroup ActionsGroup
/// \brief Interpolates and sends points to an InterpolationTarget.
///
/// This is invoked on DgElementArray.
///
/// Uses:
/// - DataBox:
///   - `intrp::Tags::InterpPointInfo<Metavariables>`
///   - `Tags::Mesh<Metavariables::volume_dim>`
///   - Variables tagged by
///     InterpolationTargetTag::vars_to_interpolate_to_target
///   - `::Tags::TimeStepId`
///   - `::Tags::HistoryEvolvedVariables<>`
/// - ConstGlobalCache
///   - `::Tags::TimeStepper<>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: nothing
template <typename InterpolationTargetTag>
struct InterpolateDenseOutputToTarget {
  template <typename DbTags, typename Metavariables, typename... InboxTags,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
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

    // There is exactly one element_id in the list of element_ids.
    if (element_coord_holders.count(element_ids[0]) == 0 or
        get<Tags::NextInterpolationTimeInFuture<InterpolationTargetTag>>(box)) {
      // There are no points in this element, so we don't need
      // to do anything.
      return std::forward_as_tuple(std::move(box));
    }

    const TimeStepId next_interpolation_target_time =
        *db::get<Tags::NextInterpolationTimeStepId<InterpolationTargetTag>>(
            box);

    // first, we need to dense output the evolved variables
    auto evolution_variables =
        db::get<typename Metavariables::system::variables_tag>(box);
    db::get<::Tags::TimeStepper<TimeStepper>>(box).dense_update_u(
        make_not_null(&evolution_variables),
        db::get<::Tags::HistoryEvolvedVariables<>>(box),
        next_interpolation_target_time.substep_time().value());

    // There are points in this element, so interpolate to them and
    // send the interpolated data to the target.  This is done
    // in several steps:
    const auto& element_coord_holder = element_coord_holders.at(element_ids[0]);


    // Set up local variables to hold vars_to_interpolate and fill it
    const auto& mesh = db::get<domain::Tags::Mesh<dim>>(box);
    Variables<typename InterpolationTargetTag::vars_to_interpolate_to_target>
        local_vars(mesh.number_of_grid_points());

    tmpl::for_each<
        typename InterpolationTargetTag::vars_to_interpolate_to_target>(
        [&evolution_variables, &local_vars](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          get<tag>(local_vars) = get<tag>(evolution_variables);
        });

    // Set up interpolator
    intrp::Irregular<dim> interpolator(
        mesh, element_coord_holder.element_logical_coords);

    // Interpolate and send interpolated data to target
    auto& receiver_proxy = Parallel::get_parallel_component<
        InterpolationTarget<Metavariables, InterpolationTargetTag>>(cache);
    Parallel::simple_action<
        Actions::InterpolationTargetVarsFromElement<InterpolationTargetTag>>(
        receiver_proxy,
        std::vector<Variables<
            typename InterpolationTargetTag::vars_to_interpolate_to_target>>(
            {interpolator.interpolate(local_vars)}),
        std::vector<std::vector<size_t>>({element_coord_holder.offsets}),
        next_interpolation_target_time);

    Parallel::printf(
        "Sending data at time : %f, array index: %d\n",
        (*db::get<Tags::NextInterpolationTimeStepId<InterpolationTargetTag>>(
             box))
            .substep_time()
            .value(),
        array_index);

    db::mutate<Tags::NextInterpolationTimeInFuture<InterpolationTargetTag>,
               Tags::NextInterpolationTimeStepId<InterpolationTargetTag>>(
        make_not_null(&box),
        [](const gsl::not_null<bool*> next_interpolation_time_in_future,
           const gsl::not_null<boost::optional<TimeStepId>*>
               next_interpolation_time) noexcept {
          *next_interpolation_time = boost::none;
          *next_interpolation_time_in_future = true;
        });

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace intrp
