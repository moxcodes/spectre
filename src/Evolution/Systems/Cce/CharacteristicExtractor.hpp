// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Actions/MutateApply.hpp"
#include "Evolution/Systems/Cce/ComputePreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/ComputeSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/Equations.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/InitializeCharacteristic.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/LinearSolve.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Parallel/Info.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/UpdateU.hpp"

namespace Cce {

// simple action
template <typename BondiTag>
struct CalculatePreSwshDerivativesForTag {};

// simple action
template <typename BondiTag>
struct CalculateSwshDerivativesForTag {};

template <typename BondiTag>
struct FilterSwshVolumeQuantity {};

struct CalculateAndInterpolateNews {};

struct RecoomputeAngularCoordinateFunctions {};

template <class Metavariables>
struct CharacteristicExtractor {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tag_list = tmpl::list<>;
  using metavariables = Metavariables;

  using initial_databox =
      db::compute_databox_type<typename InitializeCharacteristic::
                                   template return_tag_list<Metavariables>>;

  using initialize_action_list = tmpl::list<InitializeCharacteristic>;

  template <typename BondiTag>
  using hypersurface_computation = tmpl::list<
      /*need something to check that the boundary data has been sent*/
      db::Actions::MutateApply<ComputeGaugeAdjustedBoundaryValue<BondiTag>>,
      CalculatePreSwshDerivativesForTag<BondiTag>,
      CalculateSwshDerivativesForTag<BondiTag>,
      tmpl::transform<integrand_terms_to_compute_for_bondi_variable<BondiTag>,
                      tmpl::bind<db::Actions::MutateApply,
                                 ComputeBondiIntegrand<tmpl::_1>>>,
      db::Actions::MutateApply<
          RadialIntegrateBondi<Tags::EvolutionGaugeBoundaryValue, BondiTag>>,
      /*TODO we need to do something about options for this*/
      FilterSwshVolumeQuantity<BondiTag>,
      /*Once we finish the U computation, we need to update all the
        time-dependent gauge stuff*/
      tmpl::conditional_t<
          cpp17::is_same_v<BondiTag, Tags::U>,
          tmpl::list<
              db::Actions::MutateApply<GaugeUpdateU>,
              db::Actions::MutateApply<
                  ComputeGaugeAdjustedBoundaryValue<Tags::DuRDividedByR>>,
              db::Actions::MutateApply<PrecomputeCceDependencies<
                  Tags::EvolutionGaugeBoundaryValue, Tags::DuRDividedByR>>>,
          tmpl::list<>>>;

  using extract_action_list = tmpl::list<
      tmpl::transform<bondi_hypersurface_step_tags,
                      tmpl::bind<hypersurface_computation, tmpl::_1>>,
      CalculateAndInterpolateNews,
      /*TODO observers...*/
      Actions::RecordTimeStepperDataSingleTensor<
          typename Metavariables::evolved_swsh_tag,
          typename Metavariables::evolved_swsh_dt_tag>,
      Actions::RecordTimeStepperData<
          typename Metavariables::evolved_coordinates_variables_tag>,
      // update J
      Actions::UpdateUSingleTensor<typename Metavariables::evolved_swsh_tag,
                                   typename Metavariables::evolved_swsh_dt_tag>,
      // update coordinate maps
      Actions::UpdateU<
          typename Metavariables::evolved_coordinates_variables_tag>,
      RecomputeAngularCoordinateFunctions, Actions::AdvanceTime>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialize,
                                        initialize_action_list>,
                 Parallel::PhaseActions<
                     typename Metavariables::Phase,
                     Metavariables::Phase::Extract /*,extract_action_list*/>>;

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
