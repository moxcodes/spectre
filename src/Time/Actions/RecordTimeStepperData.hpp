// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class TimeStepId;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

/// Records the variables and their time derivatives in the time stepper
/// history.
///
/// \note this is a free function version of `Actions::RecordTimeStepperData`.
/// This free function alternative permits the inclusion of the time step
/// procedure in the middle of another action.
template <typename System, typename VariablesTag = NoSuchType, typename DbTags>
void record_time_stepper_data(
    const gsl::not_null<db::DataBox<DbTags>*> box) noexcept {
  using variables_tag =
      tmpl::conditional_t<std::is_same_v<VariablesTag, NoSuchType>,
                          typename System::variables_tag, VariablesTag>;
  using dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>;
  using history_tag = Tags::HistoryEvolvedVariables<variables_tag>;

  db::mutate<history_tag>(
      box,
      [](const gsl::not_null<typename history_tag::type*> history,
         const TimeStepId& time_step_id,
         const typename variables_tag::type& vars,
         const typename dt_variables_tag::type& dt_vars) noexcept {
        history->insert(time_step_id, vars, dt_vars);
      },
      db::get<Tags::TimeStepId>(*box), db::get<variables_tag>(*box),
      db::get<dt_variables_tag>(*box));
}

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Records the variables and their time derivatives in the
/// time stepper history.
///
/// With `dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>`:
///
/// Uses:
/// - GlobalCache: nothing
/// - DataBox:
///   - variables_tag (either the provided `VariablesTag` or the
///   `system::variables_tag` if none is provided)
///   - dt_variables_tag
///   - Tags::HistoryEvolvedVariables<variables_tag>
///   - Tags::TimeStepId
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - Tags::HistoryEvolvedVariables<variables_tag>
template <typename VariablesTag = NoSuchType>
struct RecordTimeStepperData {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {  // NOLINT const
    record_time_stepper_data<typename Metavariables::system, VariablesTag>(
        make_not_null(&box));
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
