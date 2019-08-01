// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/CharacteristicExtractor.hpp"
#include "Evolution/Systems/Cce/InitializeWorldtubeBoundary.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"

namespace Cce {


namespace Actions {
struct BoundaryComputeAndSendToExtractor {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double time) noexcept {
    db::mutate<Tags::H5WorldtubeBoundaryDataManager<CubicInterpolator>,
               ::Tags::Variables<
                   typename Metavariables::cce_boundary_communication_tags>>(
        make_not_null(&box),
        [&time](const gsl::not_null<CceH5BoundaryDataManager<Interpolator>*>
                    boundary_data_manager,
                const gsl::not_null<Variables<
                    typename Metavariables::cce_boundary_communication_tags>*>
                    boundary_variables_to_populate) noexcept {
          boundary_data_manager->populate_hypersurface_boundary_data(
              *boundary_variables_to_populate, time);
        });
    send_boundary_data(
        Parallel::get_parallel_component<CharacteristicExtractor>(cache),
        db::get<::Tags::Variables<
            typename Metavariables::cce_boundary_communication_tags>>,
        time, typename Metavariables::cce_boundary_communication_tags{});
  }

 private:
  template <typename ProxyType, typename VariableTagList, typename... Tags>
  static void send_boundary_data(
      ProxyType& extractor_proxy,
      const Variables<VariableTagList>& boundary_quantities, const double time,
      tmpl::list<Tags...> /*meta*/) noexcept {
    Parallel::simple_action<Actions::ReceiveWorldtubeData<tmpl::list<Tags...>>>(
        extractor_proxy, time, get<Tags>(boundary_quantities)...);
  }
};
}  // namespace Actions

// component
template <class Metavariables>
struct H5WorldtubeBoundary {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tag_list = tmpl::list<>;
  using metavariables = Metavariables;
  using add_options_to_databox = typename Parallel::AddNoOptionsToDataBox;
  using initialize_action_list = tmpl::list<InitializeH5WorldtubeBoundary,
                                            Parallel::Actions::TerminatePhase>;

  using worldtube_boundary_computation_steps = tmpl::list<>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        initialize_action_list>,
                 Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Extraction,
                                        worldtube_boundary_computation_steps>>;

  using options = tmpl::list<>;

  static void initialize(Parallel::CProxy_ConstGlobalCache<
                         Metavariables>& /*global_cache*/) noexcept {}

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>&
          global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::Extraction) {
      Parallel::get_parallel_component<H5WorldtubeBoundary>(local_cache)
          .perform_algorithm();
    }
  }
};

}  // namespace Cce
