// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmSingleton.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Systems/Cce/CharacteristicExtractor.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Options/Options.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"

struct EvolutionMetavars {
  // TODO consider making these system tags?
  using evolved_swsh_tag = Tags::J;
  using evolved_swsh_dt_tag = Tags::H;
  using evolved_coordinates_variables_tag =
      Tags::Variables<Cce::Tags::CauchyCartesianCoords,
                      Cce::Tags::InertialRetardedTime>;

  using const_global_cache_tag_list =
      tmpl::list<OptionTags::TypedTimeStepper<TimeStepper>,
                 Cce::OptionTags::LMax, Cce::OptionTags::NumberOfRadialPoints>;

  using component_list = tmpl::list<Cce::CharacteristicExtractor>;

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
