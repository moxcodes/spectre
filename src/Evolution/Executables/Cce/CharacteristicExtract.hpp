// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmSingleton.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/CharacteristicExtractor.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/WorldtubeBoundary.hpp"
#include "Options/Options.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"

struct EvolutionMetavars {
  // TODO consider making these system tags?
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
                     Cce::Tags::BondiJ, Cce::Tags::BondiBeta, Cce::Tags::BondiQ,
                     Cce::Tags::BondiU, Cce::Tags::BondiW, Cce::Tags::BondiH>,
          tmpl::bind<Cce::Tags::EvolutionGaugeBoundaryValue, tmpl::_1>>,
      Cce::Tags::GaugeC, Cce::Tags::GaugeD, Cce::Tags::GaugeOmegaCD,
      Cce::Tags::Du<Cce::Tags::GaugeOmegaCD>,
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
  using cce_pre_swsh_derivavtives_tags = Cce::all_pre_swsh_derivative_tags;
  using cce_transform_buffer_tags = Cce::all_transform_buffer_tags;
  using cce_swsh_derivative_tags = Cce::all_swsh_derivative_tags;

  using cce_boundary_component = Cce::H5WorldtubeBoundary<EvolutionMetavars>;
  // TODO select between input interpolators

  using const_global_cache_tag_list =
      tmpl::list<OptionTags::TypedTimeStepper<TimeStepper>,
                 Cce::OptionTags::LMax, Cce::OptionTags::NumberOfRadialPoints>;

  using component_list =
      tmpl::list<cce_boundary_component,
                 Cce::CharacteristicExtractor<EvolutionMetavars>>;

  static constexpr OptionString help{
      "Perform Cauchy Characteristic Extraction using .h5 input data.\n"
      "Uses regularity-preserving formulation."};

  enum class Phase { Initialization, Extraction, Exit };

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          EvolutionMetavars>& /*cache_proxy*/) noexcept {
    return current_phase == Phase::Initialization ? Phase::Extraction
                                                  : Phase::Exit;
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &Parallel::register_derived_classes_with_charm<TimeStepper>};

static const std::vector<void (*)()> charm_init_proc_funcs{
  &enable_floating_point_exceptions};
