// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Parallel/Info.hpp"
#include "Evolution/Systems/Cce/Boundary.hpp"

namespace Cce {

// component
template <class Metavariables>
struct WorltubeBoundary {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tag_list = tmpl::list<>;
  using metavariables = Metavariables;
  using worldtube_boundary_computation_steps =
      tmpl::list<BoundaryTrigonometricValues>;

      using phase_dependent_action_list = tmpl::list<>;
  using options = tmpl::list<>;

  static void initialize(Parallel::CProxy_ConstGlobalCache<
                         Metavariables>& /*global_cache*/) noexcept {}

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>&
          global_cache) noexcept {}
};


template <typename Tag, typename Metavariables>
struct RecieveDbTag {
  using temporal_id = typename Metavariables::temporal_id;
  using type = std::unordered_map<temporal_id, db::item_type<Tag>>;
};

struct has_all_of {
  constexpr bool value = /*TODO*/;
};


constexpr bool has_all_of_v = has_all_of::value;

struct has_none_of {
  constexpr bool value = /*TODO*/;
};

constexpr bool has_none_of_v = has_none_of::value;

// Actions
template <typename WorltubeComputation>
struct AddAndComputeWorldtubeValue {
  using inbox_tags = tmpl::list<>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<has_all_of_v<DbTags, WorldtubeComputation::argument_tags>>>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    /*perform the computation by just mutating the databox*/
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<has_none_of<DbTags, WorldtubeComputation::argument_tags>>>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    /* perform the computation by adding and mutating*/

  }

  // we will check that the interpolator has given the WorldtubeBoundary the
  // right stuff before allowing execution.
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalcache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {}
};


// removes the extra tags stuck in by the worldtube computation
template <typename WorldtubeComputationList>
struct CleanupWorldtubeValues {


};

}  // namespace Cce
