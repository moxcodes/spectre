// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Executables/GeneralizedHarmonic/GeneralizedHarmonicBase.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "Options/Options.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"

/// \cond
namespace Frame {
// IWYU pragma: no_forward_declare MathFunction
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
/// \endcond

template <typename InitialData, typename BoundaryConditions,
          bool BjorhusExternalBoundary = false>
struct EvolutionMetavars
    : public GeneralizedHarmonicTemplateBase<EvolutionMetavars<
          InitialData, BoundaryConditions, BjorhusExternalBoundary>>,
      public virtual GeneralizedHarmonicDefaults {
  using events = typename GeneralizedHarmonicTemplateBase<
      EvolutionMetavars<InitialData, BoundaryConditions>>::events;

  // A tmpl::list of tags to be added to the GlobalCache by the
  // metavariables
  using const_global_cache_tags =
      typename GeneralizedHarmonicTemplateBase<EvolutionMetavars<
          InitialData, BoundaryConditions>>::const_global_cache_tags;

  using observed_reduction_data_tags =
      typename GeneralizedHarmonicTemplateBase<EvolutionMetavars<
          InitialData, BoundaryConditions>>::observed_reduction_data_tags;

  using component_list = typename GeneralizedHarmonicTemplateBase<
      EvolutionMetavars<InitialData, BoundaryConditions>>::component_list;

  static constexpr Options::String help{
      "Evolve a generalized harmonic system.\n"};
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &disable_openblas_multithreading,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &domain::creators::register_derived_with_charm,
    &GeneralizedHarmonic::ConstraintDamping::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::slab_choosers>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::step_choosers>>,
    &Parallel::register_derived_classes_with_charm<StepController>,
    &Parallel::register_derived_classes_with_charm<TimeSequence<double>>,
    &Parallel::register_derived_classes_with_charm<TimeSequence<std::uint64_t>>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
