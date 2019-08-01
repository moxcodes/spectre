// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/ComputePreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/ComputeSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/Equations.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/InitializeCharacteristic.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/LinearSolve.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/Time.hpp"

namespace Cce {

template <class Metavariables>
struct CharacteristicExtractor;

namespace Actions {
struct BoundaryComputeAndSendToExtractor;
}

namespace Actions {
template <typename BondiTag>
struct CalculatePreSwshDerivativesForTag {};

template <typename BondiTag>
struct CalculateSwshDerivativesForTag {};

template <typename BondiTag>
struct FilterSwshVolumeQuantity {};

struct CalculateAndInterpolateNews {};

struct RecomputeAngularCoordinateFunctions {};

template <typename WorldtubeBoundaryComponent>
struct RequestBoundaryData {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf("requesting boundary data\n");
    Parallel::simple_action<Actions::BoundaryComputeAndSendToExtractor>(
        Parallel::get_parallel_component<WorldtubeBoundaryComponent>(cache),
        db::get<::Tags::Time>(box).value());
    return std::forward_as_tuple(std::move(box));
  }
};

struct BlockUntilBoundaryDataReceived {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    Parallel::printf("blocking for boundary data...\n");
    return db::get<Tags::BoundaryTime>(box) ==
           db::get<::Tags::Time>(box).value();
  }
};

struct PrecomputeGlobalCceDependencies {
  // template <typename DbTags, typename... InboxTags, typename Metavariables,
  // typename ArrayIndex, typename ActionList,
  // typename ParallelComponent>
  // static auto apply(db::DataBox<DbTags>& box,
  // const tuples::TaggedTuple<InboxTags...>& inboxes,
  // const Parallel::ConstGlobalCache<Metavariables>& cache,
  // const ArrayIndex& /*array_index*/,
  // const ActionList /*meta*/,
  // const ParallelComponent* const /*meta*/) noexcept {
  // }
};

template <typename TagList>
struct ReceiveWorldtubeData;

template <typename... BoundaryTags>
struct ReceiveWorldtubeData<tmpl::list<BoundaryTags...>> {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const double time,
      const db::item_type<BoundaryTags>&... boundary_data) noexcept {
    db::mutate<Tags::BoundaryTime, BoundaryTags...>(
        make_not_null(&box),
        [](const gsl::not_null<double*> databox_boundary_time,
           const gsl::not_null<
               db::item_type<BoundaryTags>*>... databox_boundary_data,
           const double& time_to_copy,
           const db::item_type<
               BoundaryTags>&... boundary_data_to_copy) noexcept {
          *databox_boundary_time = time_to_copy;
          expand_pack(set(databox_boundary_data, boundary_data_to_copy)...);
        },
        time, boundary_data...);
    // Parallel::get_parallel_component<CharacteristicExtractor>(cache)
        // .perform_algorithm();
  }

 private:
  template <typename T>
  static void set(const gsl::not_null<T*> to, const T& from) noexcept {
    *to = from;
  }
};
}  // namespace Actions

template <class Metavariables>
struct CharacteristicExtractor {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tag_list = tmpl::list<>;
  using metavariables = Metavariables;
  // TODO this should probably actually add LMax and the radial resolution
  // to the databox, and possibly other things?
  using add_options_to_databox = typename Parallel::AddNoOptionsToDataBox;

  // using initial_databox =
  //     db::compute_databox_type<typename InitializeCharacteristic::
  //                                  template return_tag_list<Metavariables>
  // >;

  using initialize_action_list =
      tmpl::list<Actions::InitializeCharacteristic,
                 Actions::RequestBoundaryData<
                     typename Metavariables::cce_boundary_component>,
                 Actions::BlockUntilBoundaryDataReceived,
                 Actions::PopulateCharacteristicInitialHypersurface,
                 Parallel::Actions::TerminatePhase>;

  template <typename BondiTag>
  using hypersurface_computation = tmpl::list<
      /*need something to check that the boundary data has been sent*/
      ::Actions::MutateApply<ComputeGaugeAdjustedBoundaryValue<BondiTag>>,
      /*      CalculatePreSwshDerivativesForTag<BondiTag>,*/
      /*      CalculateSwshDerivativesForTag<BondiTag>,*/
      tmpl::transform<integrand_terms_to_compute_for_bondi_variable<BondiTag>,
                      tmpl::bind<::Actions::MutateApply,
                                 tmpl::bind<ComputeBondiIntegrand, tmpl::_1>>>,
      ::Actions::MutateApply<
          RadialIntegrateBondi<Tags::EvolutionGaugeBoundaryValue, BondiTag>>,
      /*TODO we need to do something about options for this*/
      /*      FilterSwshVolumeQuantity<BondiTag>,*/
      /*Once we finish the U computation, we need to update all the
        time-dependent gauge stuff*/
      tmpl::conditional_t<
          cpp17::is_same_v<BondiTag, Tags::BondiU>,
          tmpl::list<
              ::Actions::MutateApply<GaugeUpdateU>,
              ::Actions::MutateApply<
                  ComputeGaugeAdjustedBoundaryValue<Tags::DuRDividedByR>>,
              ::Actions::MutateApply<PrecomputeCceDependencies<
                  Tags::EvolutionGaugeBoundaryValue, Tags::DuRDividedByR>>>,
          tmpl::list<>>>;

  // using extract_action_list = tmpl::flatten<tmpl::list<
  //     tmpl::transform<bondi_hypersurface_step_tags,
  //                     tmpl::bind<hypersurface_computation, tmpl::_1>>,
  //     /*      CalculateAndInterpolateNews,*/
  //     /*TODO observers...*/
  //     Actions::RecordTimeStepperDataSingleTensor<
  //         typename Metavariables::evolved_swsh_tag,
  //         typename Metavariables::evolved_swsh_dt_tag>,
  //     Actions::RecordTimeStepperData<
  //         typename Metavariables::evolved_coordinates_variables_tag>,
  //     // update J
  // Actions::UpdateUSingleTensor<typename Metavariables::evolved_swsh_tag,
  // typename Metavariables::evolved_swsh_dt_tag>,
  //     // update coordinate maps
  // Actions::UpdateU<
  // typename Metavariables::evolved_coordinates_variables_tag>,
  // /*      RecomputeAngularCoordinateFunctions,*/ Actions::AdvanceTime>>;
  using extract_action_list =
      tmpl::list<Actions::BlockUntilBoundaryDataReceived,
                 /*Actions::PrecomputeGlobalCceDependencies,*/
                 Actions::RequestBoundaryData<
                     typename Metavariables::cce_boundary_component>,
                 ::Actions::AdvanceTime>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        initialize_action_list>,
                 Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Extraction,
                                        extract_action_list>>;

  // TODO options changes in progress, decide what to do with the options
  // type alias when determined
  using options = tmpl::list<>;

  static void initialize(Parallel::CProxy_ConstGlobalCache<
                         Metavariables>& /*global_cache*/) noexcept {}

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>&
          global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::Extraction) {
      Parallel::get_parallel_component<CharacteristicExtractor>(local_cache)
          .perform_algorithm();
    }
  }
};

}  // namespace Cce
