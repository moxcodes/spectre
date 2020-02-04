// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Time/Tags.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionGroup
 * \brief Places the data from the current hypersurface necessary to compute
 * `Tag` in the `ScriPlusInterpolationManager` associated with the `Tag`.
 *
 * \details Adds both the appropriate scri+ value(s) and a number of target
 * inertial times to interpolate of quantity equal to the
 * `InitializationTags::ScriOutputDensity` determined from options, equally
 * spaced between the current time and the next time in the algorithm.
 *
 * Uses:
 * - `::Tags::TimeStepId`
 * - `::Tags::Next<::Tags::TimeStepId>`
 * - `Cce::InitializationTags::ScriOutputDensity`
 * - if `Tag` is `::Tags::Multiplies<Lhs, Rhs>`:
 *   - `Lhs` and `Rhs`
 * - if `Tag` has `Cce::Tags::Du<Argument>`:
 *   - `Argument`
 * - otherwise uses `Tag`
 *
 * \ref DataBoxGroup changes:
 * - Modifies:
 *   - `Tags::InterpolationManager<ComplexDataVector, Tag>`
 * - Adds: nothing
 * - Removes: nothing
 */
template <typename Tag>
struct EnqueueInterpolationScriData {
  using const_global_cache_tags =
      tmpl::list<InitializationTags::ScriOutputDensity>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if (db::get<::Tags::TimeStepId>(box).substep() == 0) {
      db::mutate<Tags::InterpolationManager<ComplexDataVector, Tag>>(
          make_not_null(&box),
          [](const gsl::not_null<
                 ScriPlusInterpolationManager<ComplexDataVector, Tag>*>
                 interpolation_manager,
             const Scalar<DataVector>& inertial_retarded_time,
             const db::item_type<Tag>& scri_data, const TimeStepId& time_id,
             const TimeStepId& next_time_id, const size_t density) noexcept {
            interpolation_manager->insert_data(get(inertial_retarded_time),
                                               get(scri_data).data());
            const double this_step_time = time_id.substep_time().value();
            const double next_step_time = next_time_id.substep_time().value();
            for (size_t i = 0; i < density; ++i) {
              interpolation_manager->insert_target_time(
                  this_step_time + (next_step_time - this_step_time) *
                                       static_cast<double>(i) /
                                       static_cast<double>(density));
            }
          },
          db::get<Tags::InertialRetardedTime>(box), db::get<Tag>(box),
          db::get<::Tags::TimeStepId>(box),
          db::get<::Tags::Next<::Tags::TimeStepId>>(box),
          db::get<InitializationTags::ScriOutputDensity>(box));
    }
    return std::forward_as_tuple(std::move(box));
  }
};
/// \cond
template <typename Tag>
struct EnqueueInterpolationScriData<Tags::Du<Tag>> {
  using const_global_cache_tags =
      tmpl::list<InitializationTags::ScriOutputDensity>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if (db::get<::Tags::TimeStepId>(box).substep() == 0) {
      db::mutate<Tags::InterpolationManager<ComplexDataVector, Tags::Du<Tag>>>(
          make_not_null(&box),
          [](const gsl::not_null<ScriPlusInterpolationManager<ComplexDataVector,
                                                              Tags::Du<Tag>>*>
                 interpolation_manager,
             const Scalar<DataVector>& inertial_retarded_time,
             const db::item_type<Tag>& scri_data, const TimeStepId& time_id,
             const TimeStepId& next_time_id, const size_t density) {
            interpolation_manager->insert_data(get(inertial_retarded_time),
                                               get(scri_data).data());
            const double this_step_time = time_id.substep_time().value();
            const double next_step_time = next_time_id.substep_time().value();
            for (size_t i = 0; i < density; ++i) {
              interpolation_manager->insert_target_time(
                  this_step_time + (next_step_time - this_step_time) *
                                       static_cast<double>(i) /
                                       static_cast<double>(density));
            }
          },
          db::get<Tags::InertialRetardedTime>(box), db::get<Tag>(box),
          db::get<::Tags::TimeStepId>(box),
          db::get<::Tags::Next<::Tags::TimeStepId>>(box),
          db::get<InitializationTags::ScriOutputDensity>(box));
    }
    return std::forward_as_tuple(std::move(box));
  }
};

template <typename LhsTag, typename RhsTag>
struct EnqueueInterpolationScriData<::Tags::Multiplies<LhsTag, RhsTag>> {
  using const_global_cache_tags =
      tmpl::list<InitializationTags::ScriOutputDensity>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if (db::get<::Tags::TimeStepId>(box).substep() == 0) {
      db::mutate<Tags::InterpolationManager<
          ComplexDataVector, ::Tags::Multiplies<LhsTag, RhsTag>>>(
          make_not_null(&box),
          [](const gsl::not_null<ScriPlusInterpolationManager<
                 ComplexDataVector, ::Tags::Multiplies<LhsTag, RhsTag>>*>
                 interpolation_manager,
             const Scalar<DataVector>& inertial_retarded_time,
             const db::item_type<LhsTag>& lhs_data,
             const db::item_type<RhsTag>& rhs_data, const TimeStepId& time_id,
             const TimeStepId& next_time_id, const size_t density) noexcept {
            interpolation_manager->insert_data(get(inertial_retarded_time),
                                               get(lhs_data).data(),
                                               get(rhs_data).data());
            const double this_step_time = time_id.substep_time().value();
            const double next_step_time = next_time_id.substep_time().value();
            for (size_t i = 0; i < density; ++i) {
              interpolation_manager->insert_target_time(
                  this_step_time + (next_step_time - this_step_time) *
                                       static_cast<double>(i) /
                                       static_cast<double>(density));
            }
          },
          db::get<Tags::InertialRetardedTime>(box), db::get<LhsTag>(box),
          db::get<RhsTag>(box), db::get<::Tags::TimeStepId>(box),
          db::get<::Tags::Next<::Tags::TimeStepId>>(box),
          db::get<InitializationTags::ScriOutputDensity>(box));
    }
    return std::forward_as_tuple(std::move(box));
  }
};

template <typename LhsTagArgument, typename RhsTag>
struct EnqueueInterpolationScriData<
    ::Tags::Multiplies<Tags::Du<LhsTagArgument>, RhsTag>> {
  using const_global_cache_tags =
      tmpl::list<InitializationTags::ScriOutputDensity>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if (db::get<::Tags::TimeStepId>(box).substep() == 0) {
      db::mutate<Tags::InterpolationManager<
          ComplexDataVector,
          ::Tags::Multiplies<Tags::Du<LhsTagArgument>, RhsTag>>>(
          make_not_null(&box),
          [](const gsl::not_null<ScriPlusInterpolationManager<
                 ComplexDataVector,
                 ::Tags::Multiplies<Tags::Du<LhsTagArgument>, RhsTag>>*>
                 interpolation_manager,
             const Scalar<DataVector>& inertial_retarded_time,
             const db::item_type<LhsTagArgument>& lhs_data,
             const db::item_type<RhsTag>& rhs_data, const TimeStepId& time_id,
             const TimeStepId& next_time_id, const size_t density) noexcept {
            interpolation_manager->insert_data(get(inertial_retarded_time),
                                               get(lhs_data).data(),
                                               get(rhs_data).data());
            const double this_step_time = time_id.substep_time().value();
            const double next_step_time = next_time_id.substep_time().value();
            for (size_t i = 0; i < density; ++i) {
              interpolation_manager->insert_target_time(
                  this_step_time + (next_step_time - this_step_time) *
                                       static_cast<double>(i) /
                                       static_cast<double>(density));
            }
          },
          db::get<Tags::InertialRetardedTime>(box),
          db::get<LhsTagArgument>(box), db::get<RhsTag>(box),
          db::get<::Tags::TimeStepId>(box),
          db::get<::Tags::Next<::Tags::TimeStepId>>(box),
          db::get<InitializationTags::ScriOutputDensity>(box));
    }
    return std::forward_as_tuple(std::move(box));
  }
};
/// \endcond
}  // namespace Actions
}  // namespace Cce
