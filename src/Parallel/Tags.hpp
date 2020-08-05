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
namespace Tags {
template <typename Metavariables>
struct GlobalSyncTrigger : db::SimpleTag {
  using type =
      std::unique_ptr<Trigger<typename Metavariables::global_sync_triggers>>;
};

template <typename Metavariables>
struct PauseTrigger : db::SimpleTag {
  using type = std::unique_ptr<Trigger<typename Metavariables::triggers>>;
};

template <typename Metavariables>
struct HaltTrigger : db::SimpleTag {
  using type = std::unique_ptr<Trigger<typename Metavariables::triggers>>;
};

template <typename Metavariables, typename PhaseConstant>
struct PhaseTrigger : db::SimpleTag {
  using type = std::unique_ptr<Trigger<typename Metavariables::triggers>>;
};
}  // namespace Tags

template <typename Metavariables>
struct AlgorithmControlTriggers {
  template <typename PhaseConstant>
  struct PhaseTrigger {
    using type = std::unique_ptr<Trigger<typename Metavariables::triggers>>;
    static constexpr OptionString help = {"Trigger for a global sync phase"};
    static std::string name() noexcept {
      return Metavariables::phase_name(PhaseConstant::value);
    };
    static type default_value() noexcept {
      return std::make_unique<Triggers::Not<typename Metavariables::triggers>>(
          std::make_unique<
              Triggers::Always<typename Metavariables::triggers>>());
    }
  };

  struct GlobalSyncTrigger {
    using type =
        std::unique_ptr<Trigger<typename Metavariables::global_sync_triggers>>;
    static constexpr OptionString help = {
        "Trigger for imposing a global sync and running sync phases."};
    static std::string name() noexcept { return "GlobalSync"; }
    static type default_value() noexcept {
      return std::make_unique<
          Triggers::Not<typename Metavariables::global_sync_triggers>>(
          std::make_unique<Triggers::Always<
              typename Metavariables::global_sync_triggers>>());
    }
  };

  struct PauseTrigger {
    using type = std::unique_ptr<Trigger<typename Metavariables::triggers>>;
    static constexpr OptionString help = {"Trigger for pausing the algorithm."};
    static std::string name() noexcept { return "Pause"; }
    static type default_value() noexcept {
      return std::make_unique<Triggers::Not<typename Metavariables::triggers>>(
          std::make_unique<
              Triggers::Always<typename Metavariables::triggers>>());
    }
  };

  struct HaltTrigger {
    using type = std::unique_ptr<Trigger<typename Metavariables::triggers>>;
    static constexpr OptionString help = {"Trigger for halting the algorithm."};
    static std::string name() noexcept { return "Halt"; }
    static type default_value() noexcept {
      return std::make_unique<Triggers::Not<typename Metavariables::triggers>>(
          std::make_unique<
              Triggers::Always<typename Metavariables::triggers>>());
    }
  };

  using options = tmpl::flatten<
      tmpl::list<GlobalSyncTrigger, PauseTrigger, HaltTrigger,
                 tmpl::transform<typename Metavariables::global_sync_phases,
                                 tmpl::bind<PhaseTrigger, tmpl::_1>>>>;

  static constexpr OptionString help = {
      "A collection of triggers for determining run-time algorithm control "
      "flow."};

  using trigger_tuple_type =
      tuples::tagged_tuple_from_typelist<tmpl::flatten<tmpl::list<
          Tags::GlobalSyncTrigger<Metavariables>,
          Tags::PauseTrigger<Metavariables>, Tags::HaltTrigger<Metavariables>,
          tmpl::transform<typename Metavariables::global_sync_phases,
                          tmpl::bind<Tags::PhaseTrigger,
                                     tmpl::pin<Metavariables>, tmpl::_1>>>>>;

  AlgorithmControlTriggers() noexcept = default;
  AlgorithmControlTriggers(const AlgorithmControlTriggers&) noexcept = default;
  AlgorithmControlTriggers(AlgorithmControlTriggers&&) noexcept = default;
  AlgorithmControlTriggers& operator=(
      const AlgorithmControlTriggers&) noexcept = default;
  AlgorithmControlTriggers& operator=(AlgorithmControlTriggers&&) noexcept =
      default;
  ~AlgorithmControlTriggers() noexcept = default;

  explicit AlgorithmControlTriggers(CkMigrateMessage* /*unused*/) noexcept {}

  AlgorithmControlTriggers(trigger_tuple_type input_triggers) noexcept
      : triggers{std::move(input_triggers)} {}

  template <typename... PhaseTriggers>
  AlgorithmControlTriggers(
      std::unique_ptr<Trigger<typename Metavariables::global_sync_triggers>>
          global_sync_trigger,
      std::unique_ptr<Trigger<typename Metavariables::triggers>> pause_trigger,
      std::unique_ptr<Trigger<typename Metavariables::triggers>> halt_trigger,
      std::unique_ptr<PhaseTriggers>... phase_triggers) noexcept {
    get<Tags::GlobalSyncTrigger<Metavariables>>(triggers) =
        std::move(global_sync_trigger);
    get<Tags::PauseTrigger<Metavariables>>(triggers) = std::move(pause_trigger);
    get<Tags::HaltTrigger<Metavariables>>(triggers) = std::move(halt_trigger);

    initialize_phase_triggersimpl(typename Metavariables::global_sync_phases{},
                                  std::move(phase_triggers)...);
  }

  void pup(PUP::er& p) { p | triggers; }  // NOLINT

  trigger_tuple_type triggers;

 private:
  template <typename... PhaseTriggers, typename... PhaseConstants>
  void initialize_phase_triggersimpl(
      tmpl::list<PhaseConstants...> /*meta*/,
      std::unique_ptr<PhaseTriggers>... phase_triggers) noexcept {
    EXPAND_PACK_LEFT_TO_RIGHT(
        [this](auto trigger, auto phase_constant_v) noexcept {
          using phase_constant = typename decltype(phase_constant_v)::type;
          get<Tags::PhaseTrigger<Metavariables, phase_constant>>(triggers) =
              std::move(trigger);
        }(std::move(phase_triggers), tmpl::type_<PhaseConstants>{}));
  }
};

namespace OptionTags {
template <typename Metavariables>
struct AlgorithmControlTriggers {
  using type = ::Parallel::AlgorithmControlTriggers<Metavariables>;
  static constexpr OptionString help =
      "triggers for determining the algorithm control-flow";
};
}  // namespace OptionTags

namespace Tags {
template <typename Metavariables>
struct AlgorithmControlTriggers {
  using type = ::Parallel::AlgorithmControlTriggers<Metavariables>;
  using option_tags =
      tmpl::list<OptionTags::AlgorithmControlTriggers<Metavariables>>;

  static constexpr bool pass_metavariables = false;
  static ::Parallel::AlgorithmControlTriggers<Metavariables>
  create_from_options(const ::Parallel::AlgorithmControlTriggers<Metavariables>&
                          triggers) noexcept {
    return deserialize<type>(serialize<type>(triggers).data());
  }
};
}  // namespace Tags
}  // namespace Parallel
