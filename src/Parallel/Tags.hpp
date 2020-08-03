// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "Options/Options.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
namespace OptionTags {
template <typename TriggerRegistrars>
struct GlobalSyncTrigger {
  using type = std::unique_ptr<Trigger<TriggerRegistrars>>;
  static constexpr OptionString help =
      "condition for which to trigger a global synchronization";
};

template <typename TriggerRegistrars>
struct TerminationTrigger {
  using type = std::unique_ptr<Trigger<TriggerRegistrars>>;
  static constexpr OptionString help =
      "condition for which to trigger algorithm termination";
};

template <typename Metavariables, typename PhaseConstant,
          typename TriggerRegistrars>
struct PhaseTrigger {
  using type = std::unique_ptr<Trigger<TriggerRegistrars>>;
  static constexpr OptionString help =
      "condition for which to request a jump to a particular phase";
  static std::string name() noexcept {
    return "PhaseTrigger" + Metavariables::phase_name(PhaseConstant::value);
  }
};
}  // namespace OptionTags

namespace Tags {
template <typename TriggerRegistrars>
struct GlobalSyncTrigger : db::SimpleTag {
  using type = std::unique_ptr<Trigger<TriggerRegistrars>>;
  using option_tags =
      tmpl::list<OptionTags::GlobalSyncTrigger<TriggerRegistrars>>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<Trigger<TriggerRegistrars>> create_from_options(
      const std::unique_ptr<Trigger<TriggerRegistrars>>& trigger) noexcept {
    return deserialize<type>(serialize<type>(trigger).data());
  }
};

template <typename TriggerRegistrars>
struct TerminationTrigger : db::SimpleTag {
  using type = std::unique_ptr<Trigger<TriggerRegistrars>>;
  using option_tags =
      tmpl::list<OptionTags::TerminationTrigger<TriggerRegistrars>>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<Trigger<TriggerRegistrars>> create_from_options(
      const std::unique_ptr<Trigger<TriggerRegistrars>>& trigger) noexcept {
    return deserialize<type>(serialize<type>(trigger).data());
  }
};

template <typename Metavariables, typename PhaseConstant,
          typename TriggerRegistrars>
struct PhaseTrigger : db::SimpleTag {
  using type = std::unique_ptr<Trigger<TriggerRegistrars>>;
  using option_tags =
      tmpl::list<OptionTags::PhaseTrigger<Metavariables, PhaseConstant,
                                          TriggerRegistrars>>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<Trigger<TriggerRegistrars>> create_from_options(
      const std::unique_ptr<Trigger<TriggerRegistrars>>& trigger) noexcept {
    return deserialize<type>(serialize<type>(trigger).data());
  }
};
}  // namespace Tags
}  // namespace Parallel
