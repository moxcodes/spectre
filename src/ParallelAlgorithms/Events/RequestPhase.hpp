// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

namespace Events {
template <typename Metavariables, typename PhaseConstant,
          typename EventRegistrars>
class RequestPhase;

namespace Registrars {
template <typename Metavariables, typename PhaseConstant>
struct RequestPhase {
  template <typename RegistrarList>
  using f = Events::RequestPhase<Metavariables, PhaseConstant, RegistrarList>;
};
}  // namespace Registrars

template <typename Metavariables, typename PhaseConstant,
          typename EventRegistrars>
class RequestPhase : public Event<EventRegistrars> {
 public:
  explicit RequestPhase(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(RequestPhase);

  using options = tmpl::list<>;
  static constexpr OptionString help = {"Event for requesting phase"};

  static std::string name() noexcept {
    return "RequestPhase" + Metavariables::phase_name(PhaseConstant::value);
  }

  RequestPhase() = default;

  using argument_tags = tmpl::list<>;
  template <typename ArrayIndex, typename Component>
  void operator()(Parallel::ConstGlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const Component* const /*meta*/) const noexcept {
    auto array_element =
        Parallel::get_parallel_component<Component>(cache)[array_index]
            .ckLocal();
    array_element->request_sync_phase(PhaseConstant::value);
  }
};

template <typename Metavariables, typename PhaseConstant,
          typename EventRegistrars>
PUP::able::PUP_ID
    RequestPhase<Metavariables, PhaseConstant, EventRegistrars>::my_PUP_ID = 0;
}  // namespace Events
