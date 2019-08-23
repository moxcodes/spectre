// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmSingleton.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "Evolution/EventsAndTriggers/EventsAndTriggers.hpp"
#include "Evolution/EventsAndTriggers/Tags.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/CharacteristicExtractor.hpp"
#include "Evolution/Systems/Cce/CharacteristicScri.hpp"
#include "Evolution/Systems/Cce/ObserveSwshModes.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/WorldtubeBoundary.hpp"
#include "Options/Options.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"

namespace Cce {
struct System {
  using variables_tag = ::Tags::Variables<tmpl::list<>>;
};
}  // namespace Cce

struct EvolutionMetavars {
  // TODO consider making all of the generalizable ones system tags?
  using system = Cce::System;

  using evolved_swsh_tag = Cce::Tags::BondiJ;
  using evolved_swsh_dt_tag = Cce::Tags::BondiH;
  using evolved_coordinates_variables_tag =
      Tags::Variables<tmpl::list<Cce::Tags::CauchyCartesianCoords,
                                 Cce::Tags::InertialRetardedTime>>;
  using cce_boundary_communication_tags =
      Cce::characteristic_worldtube_boundary_tags;

  using cce_gauge_boundary_tags = tmpl::flatten<tmpl::list<
      tmpl::transform<
          tmpl::list<Cce::Tags::BondiR, Cce::Tags::DuRDividedByR,
                     Cce::Tags::BondiJ, Cce::Tags::Dr<Cce::Tags::BondiJ>,
                     Cce::Tags::BondiBeta, Cce::Tags::BondiQ, Cce::Tags::BondiU,
                     Cce::Tags::BondiW, Cce::Tags::BondiH>,
          tmpl::bind<Cce::Tags::EvolutionGaugeBoundaryValue, tmpl::_1>>,
      Cce::Tags::U0, Cce::Tags::GaugeC, Cce::Tags::GaugeD,
      Cce::Tags::GaugeOmegaCD, Cce::Tags::Du<Cce::Tags::GaugeOmegaCD>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::GaugeOmegaCD,
                                       Spectral::Swsh::Tags::Eth>>>;

  using cce_scri_tags = tmpl::list<Cce::Tags::News>;
  using cce_integrand_tags = tmpl::flatten<tmpl::transform<
      Cce::bondi_hypersurface_step_tags,
      tmpl::bind<Cce::integrand_terms_to_compute_for_bondi_variable,
                 tmpl::_1>>>;
  using cce_integration_independent_tags = Cce::pre_computation_tags;
  using cce_temporary_equations_tags = tmpl::remove_duplicates<tmpl::flatten<
      tmpl::transform<cce_integrand_tags,
                      tmpl::bind<Cce::integrand_temporary_tags, tmpl::_1>>>>;
  using cce_pre_swsh_derivatives_tags = Cce::all_pre_swsh_derivative_tags;
  using cce_transform_buffer_tags = Cce::all_transform_buffer_tags;
  using cce_swsh_derivative_tags = Cce::all_swsh_derivative_tags;
  using cce_angular_coordinate_tags =
      tmpl::list<Cce::Tags::CauchyAngularCoords>;

  using cce_boundary_component = Cce::H5WorldtubeBoundary<EvolutionMetavars>;
  // TODO select between input interpolators
  template <typename ToObserve>
  using observation_type = typename db::item_type<
      Spectral::Swsh::Tags::SwshTransform<ToObserve>>::type::value_type;

  using output_cce_scri_scalars = tmpl::list<Cce::Tags::U0, Cce::Tags::News>;
  using output_cce_volume_scalars = tmpl::list<Cce::Tags::BondiJ>;

  // TODO expand with other outputs
  using observed_reduction_data_tags =
      tmpl::list<typename Cce::Events::reduction_data_noop_type<
          tmpl::transform<output_cce_scri_scalars,
                          tmpl::bind<observation_type, tmpl::_1>>>::tag>;

  using events = tmpl::list<
      Cce::Events::Registrars::ObserveBoundarySwshModes<output_cce_scri_scalars>
      /*, Events::Registrars::ObserveVolumeSwshModes*/>;
  using triggers = Triggers::time_triggers;

  using const_global_cache_tag_list =
      tmpl::list<OptionTags::TypedTimeStepper<TimeStepper>,
                 Cce::OptionTags::LMax, Cce::OptionTags::ObservationLMax,
                 Cce::OptionTags::NumberOfRadialPoints,
                 OptionTags::EventsAndTriggers<events, triggers>>;

  struct BoundaryObservationType {};
  struct InertialObservationType {};
  struct VolumeObservationType {};
  struct InterpolatedScriObservationType {};
  using swsh_boundary_observation_type = BoundaryObservationType;
  using swsh_inertial_scri_observation_type = InertialObservationType;
  using swsh_volume_observation_type = VolumeObservationType;
  using swsh_interpolation_observation_type = InterpolatedScriObservationType;

  using component_list =
      tmpl::list<observers::Observer<EvolutionMetavars>,
                 observers::ObserverWriter<EvolutionMetavars>,
                 cce_boundary_component,
                 Cce::CharacteristicExtractor<EvolutionMetavars>,
                 Cce::CharacteristicScri<EvolutionMetavars>>;

  static constexpr OptionString help{
      "Perform Cauchy Characteristic Extraction using .h5 input data.\n"
      "Uses regularity-preserving formulation."};

  enum class Phase { Initialization, RegisterWithObserver, Extraction, Exit };

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          EvolutionMetavars>& /*cache_proxy*/) noexcept {
    if (current_phase == Phase::Initialization) {
      return Phase::RegisterWithObserver;
    } else if (current_phase == Phase::RegisterWithObserver) {
      return Phase::Extraction;
    } else {
      return Phase::Exit;
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<Cce::Interpolator>,
    &Parallel::register_derived_classes_with_charm<
        Event<EvolutionMetavars::events>>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<EvolutionMetavars::triggers>>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
