// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "ApparentHorizons/ComputeItems.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsCharacteresticSpeeds.hpp"
#include "Evolution/Actions/AddMeshVelocityNonconservative.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/NumericInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/InitializeDampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/UpwindPenaltyCorrection.hpp"
#include "Evolution/TypeTraits.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/Actions/ReceiveVolumeData.hpp"
#include "IO/Importers/Actions/RegisterWithElementDataReader.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "Informer/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderSchemeLts.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/AddTemporalIdsToInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolate.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceivePoints.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/CollectDataForFluxes.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.hpp"
#include "ParallelAlgorithms/Events/ObserveVolumeIntegrals.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintGammas.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepChoosers/StepToTimes.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

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

struct GeneralizedHarmonicDefaults {
  static constexpr int volume_dim = 3;
  using frame = Frame::Inertial;
  using system = GeneralizedHarmonic::System<volume_dim>;
  static constexpr dg::Formulation dg_formulation =
      dg::Formulation::StrongInertial;
  static constexpr bool use_damped_harmonic_rollon = true;
  // Set override_cubic_functions_of_time to true to override the cubic
  // piecewise polynomial functions of time using
  // `read_spec_third_order_piecewise_polynomial()`
  static constexpr bool override_cubic_functions_of_time = false;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;

  using normal_dot_numerical_flux = Tags::NumericalFlux<
      GeneralizedHarmonic::UpwindPenaltyCorrection<volume_dim>>;

  using step_choosers_common =
      tmpl::list<StepChoosers::Registrars::Cfl<volume_dim, Frame::Inertial>,
                 StepChoosers::Registrars::Constant,
                 StepChoosers::Registrars::Increase>;
  using step_choosers_for_step_only =
      tmpl::list<StepChoosers::Registrars::PreventRapidIncrease>;
  using step_choosers_for_slab_only =
      tmpl::list<StepChoosers::Registrars::StepToTimes>;
  using step_choosers = std::conditional_t<
      local_time_stepping,
      tmpl::append<step_choosers_common, step_choosers_for_step_only>,
      tmpl::list<>>;
  using slab_choosers = std::conditional_t<
      local_time_stepping,
      tmpl::append<step_choosers_common, step_choosers_for_slab_only>,
      tmpl::append<step_choosers_common, step_choosers_for_step_only,
                   step_choosers_for_slab_only>>;

  using time_stepper_tag = Tags::TimeStepper<
      std::conditional_t<local_time_stepping, LtsTimeStepper, TimeStepper>>;
  using boundary_scheme = std::conditional_t<
      local_time_stepping,
      dg::FirstOrderScheme::FirstOrderSchemeLts<
          volume_dim, typename system::variables_tag,
          db::add_tag_prefix<::Tags::dt, typename system::variables_tag>,
          normal_dot_numerical_flux, Tags::TimeStepId, time_stepper_tag>,
      dg::FirstOrderScheme::FirstOrderScheme<
          volume_dim, typename system::variables_tag,
          db::add_tag_prefix<::Tags::dt, typename system::variables_tag>,
          normal_dot_numerical_flux, Tags::TimeStepId>>;

  using analytic_solution_fields = typename system::variables_tag::tags_list;

  struct AhA {
    using tags_to_observe =
        tmpl::list<StrahlkorperGr::Tags::AreaCompute<frame>>;
    using compute_items_on_source = tmpl::list<
        gr::Tags::SpatialMetricCompute<volume_dim, frame, DataVector>,
        ah::Tags::InverseSpatialMetricCompute<volume_dim, frame>,
        ah::Tags::ExtrinsicCurvatureCompute<volume_dim, frame>,
        ah::Tags::SpatialChristoffelSecondKindCompute<volume_dim, frame>>;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::SpatialMetric<volume_dim, frame, DataVector>,
                   gr::Tags::InverseSpatialMetric<volume_dim, frame>,
                   gr::Tags::ExtrinsicCurvature<volume_dim, frame>,
                   gr::Tags::SpatialChristoffelSecondKind<volume_dim, frame>>;
    using compute_items_on_target = tmpl::append<
        tmpl::list<StrahlkorperGr::Tags::AreaElementCompute<frame>>,
        tags_to_observe>;
    using compute_target_points =
        intrp::TargetPoints::ApparentHorizon<AhA, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::FindApparentHorizon<AhA, ::Frame::Inertial>;
    using post_horizon_find_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, AhA, AhA>;
  };

  using interpolation_target_tags = tmpl::list<AhA>;
  using interpolator_source_vars =
      tmpl::list<gr::Tags::SpacetimeMetric<volume_dim, frame>,
                 GeneralizedHarmonic::Tags::Pi<volume_dim, frame>,
                 GeneralizedHarmonic::Tags::Phi<volume_dim, frame>>;

  using triggers = Triggers::time_triggers;

  struct ObservationType {};
  using element_observation_type = ObservationType;

  enum class Phase {
    Initialization,
    RegisterWithElementDataReader,
    ImportInitialData,
    InitializeInitialDataDependentQuantities,
    InitializeTimeStepperHistory,
    Register,
    Evolve,
    Exit
  };

  using initialize_initial_data_dependent_quantities_actions = tmpl::list<
      ::Actions::SetupDataBox,
      GeneralizedHarmonic::gauges::Actions::InitializeDampedHarmonic<
          volume_dim, use_damped_harmonic_rollon>,
      GeneralizedHarmonic::Actions::InitializeConstraints<volume_dim>,
      Parallel::Actions::TerminatePhase>;
};

template <typename EvolutionMetavarsDerived>
struct GeneralizedHarmonicTemplateBase;

template <template <typename, typename, bool> class EvolutionMetavarsDerived,
          typename InitialData, typename BoundaryConditions,
          bool BjorhusExternalBoundary>
struct GeneralizedHarmonicTemplateBase<EvolutionMetavarsDerived<
    InitialData, BoundaryConditions, BjorhusExternalBoundary>>
    : public virtual GeneralizedHarmonicDefaults {
  using derived_metavars =
      EvolutionMetavarsDerived<InitialData, BoundaryConditions,
                               BjorhusExternalBoundary>;

  using initial_data = InitialData;
  using boundary_conditions = BoundaryConditions;
  // Only Dirichlet boundary conditions imposed by an analytic solution are
  // supported right now.
  using analytic_solution = tmpl::conditional_t<
      evolution::is_analytic_solution_v<initial_data>, initial_data,
      tmpl::conditional_t<
          evolution::is_analytic_solution_v<boundary_conditions>,
          boundary_conditions,
          GeneralizedHarmonic::Solutions::WrappedGr<
              gr::Solutions::KerrSchild>>>;
  using analytic_solution_tag = Tags::AnalyticSolution<analytic_solution>;
  using boundary_condition_tag = analytic_solution_tag;

  using observe_fields = tmpl::append<
      tmpl::push_back<
          analytic_solution_fields,
          ::Tags::PointwiseL2Norm<
              GeneralizedHarmonic::Tags::GaugeConstraint<volume_dim, frame>>,
          ::Tags::PointwiseL2Norm<GeneralizedHarmonic::Tags::
                                      ThreeIndexConstraint<volume_dim, frame>>>,
      std::conditional_t<volume_dim == 3,
                         tmpl::list<::Tags::PointwiseL2Norm<
                             GeneralizedHarmonic::Tags::FourIndexConstraint<
                                 volume_dim, frame>>>,
                         tmpl::list<>>>;

  using observation_events = tmpl::list<
      dg::Events::Registrars::ObserveErrorNorms<Tags::Time,
                                                analytic_solution_fields>,
      dg::Events::Registrars::ObserveFields<
          volume_dim, Tags::Time, observe_fields, analytic_solution_fields>,
      Events::Registrars::ObserveTimeStep<system>,
      Events::Registrars::ChangeSlabSize<slab_choosers>>;

  // Events include the observation events and finding the horizon
  using events = tmpl::push_back<
      observation_events,
      intrp::Events::Registrars::Interpolate<3, AhA, interpolator_source_vars>>;

  // A tmpl::list of tags to be added to the GlobalCache by the
  // metavariables
  using const_global_cache_tags = tmpl::conditional_t<
      evolution::is_analytic_solution_v<analytic_solution>,
      tmpl::list<analytic_solution_tag, normal_dot_numerical_flux,
                 time_stepper_tag, Tags::EventsAndTriggers<events, triggers>,
                 GeneralizedHarmonic::ConstraintDamping::Tags::
                     DampingFunctionGamma0<volume_dim, frame>,
                 GeneralizedHarmonic::ConstraintDamping::Tags::
                     DampingFunctionGamma1<volume_dim, frame>,
                 GeneralizedHarmonic::ConstraintDamping::Tags::
                     DampingFunctionGamma2<volume_dim, frame>>,
      tmpl::list<normal_dot_numerical_flux, time_stepper_tag,
                 Tags::EventsAndTriggers<events, triggers>,
                 GeneralizedHarmonic::ConstraintDamping::Tags::
                     DampingFunctionGamma0<volume_dim, frame>,
                 GeneralizedHarmonic::ConstraintDamping::Tags::
                     DampingFunctionGamma1<volume_dim, frame>,
                 GeneralizedHarmonic::ConstraintDamping::Tags::
                     DampingFunctionGamma2<volume_dim, frame>>>;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::push_back<typename Event<observation_events>::creatable_classes,
                      typename AhA::post_horizon_find_callback>>;

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          derived_metavars>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return evolution::is_numeric_initial_data_v<initial_data>
                   ? Phase::RegisterWithElementDataReader
                   : Phase::InitializeInitialDataDependentQuantities;
      case Phase::RegisterWithElementDataReader:
        return Phase::ImportInitialData;
      case Phase::ImportInitialData:
        return Phase::InitializeInitialDataDependentQuantities;
      case Phase::InitializeInitialDataDependentQuantities:
        return Phase::InitializeTimeStepperHistory;
      case Phase::InitializeTimeStepperHistory:
        return Phase::Register;
      case Phase::Register:
        return Phase::Evolve;
      case Phase::Evolve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }

  using step_actions = tmpl::list<
      evolution::dg::Actions::ComputeTimeDerivative<derived_metavars>,
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          domain::Tags::BoundaryDirectionsInterior<volume_dim>>,
      tmpl::conditional_t<
          BjorhusExternalBoundary, tmpl::list<>,
          tmpl::list<
              dg::Actions::ImposeDirichletBoundaryConditions<derived_metavars>,
              dg::Actions::CollectDataForFluxes<
                  boundary_scheme,
                  domain::Tags::BoundaryDirectionsInterior<volume_dim>>>>,
      dg::Actions::ReceiveDataForFluxes<boundary_scheme>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<tmpl::conditional_t<
                         BjorhusExternalBoundary,
                         tmpl::list<GeneralizedHarmonic::Actions::
                                        ImposeBjorhusBoundaryConditions<
                                            derived_metavars>>,
                         tmpl::list<>>,
                     Actions::RecordTimeStepperData<>,
                     Actions::MutateApply<boundary_scheme>>,
          tmpl::list<Actions::MutateApply<boundary_scheme>,
                     tmpl::conditional_t<
                         BjorhusExternalBoundary,
                         tmpl::list<GeneralizedHarmonic::Actions::
                                        ImposeBjorhusBoundaryConditions<
                                            derived_metavars>>,
                         tmpl::list<>>,
                     Actions::RecordTimeStepperData<>>>,
      Actions::UpdateU<>>;

  using initialization_actions = tmpl::list<
      Actions::SetupDataBox,
      Initialization::Actions::TimeAndTimeStep<derived_metavars>,
      evolution::dg::Initialization::Domain<volume_dim,
                                            override_cubic_functions_of_time>,
      Initialization::Actions::NonconservativeSystem<system>,
      std::conditional_t<
          evolution::is_numeric_initial_data_v<initial_data>, tmpl::list<>,
          evolution::Initialization::Actions::SetVariables<
              domain::Tags::Coordinates<volume_dim, Frame::Logical>>>,
      Initialization::Actions::TimeStepperHistory<derived_metavars>,
      GeneralizedHarmonic::Actions::InitializeGhAnd3Plus1Variables<volume_dim>,
      dg::Actions::InitializeInterfaces<
          system,
          dg::Initialization::slice_tags_to_face<
              typename system::variables_tag,
              gr::Tags::SpatialMetric<volume_dim, frame, DataVector>,
              typename gr::Tags::DetAndInverseSpatialMetricCompute<
                  volume_dim, frame, DataVector>::base,
              gr::Tags::Shift<volume_dim, frame, DataVector>,
              gr::Tags::Lapse<DataVector>,
              GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma0,
              GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
              GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2>,
          tmpl::conditional_t<
              BjorhusExternalBoundary,
              dg::Initialization::slice_tags_to_exterior<
                  typename system::variables_tag,
                  gr::Tags::SpatialMetric<volume_dim, frame, DataVector>,
                  typename gr::Tags::DetAndInverseSpatialMetricCompute<
                      volume_dim, frame, DataVector>::base,
                  gr::Tags::Shift<volume_dim, frame, DataVector>,
                  gr::Tags::Lapse<DataVector>,
                  GeneralizedHarmonic::ConstraintDamping::Tags::
                      ConstraintGamma0,
                  GeneralizedHarmonic::ConstraintDamping::Tags::
                      ConstraintGamma1,
                  GeneralizedHarmonic::ConstraintDamping::Tags::
                      ConstraintGamma2>,
              dg::Initialization::slice_tags_to_exterior<
                  gr::Tags::SpatialMetric<volume_dim, frame, DataVector>,
                  typename gr::Tags::DetAndInverseSpatialMetricCompute<
                      volume_dim, frame, DataVector>::base,
                  gr::Tags::Shift<volume_dim, frame, DataVector>,
                  gr::Tags::Lapse<DataVector>,
                  GeneralizedHarmonic::ConstraintDamping::Tags::
                      ConstraintGamma0,
                  GeneralizedHarmonic::ConstraintDamping::Tags::
                      ConstraintGamma1,
                  GeneralizedHarmonic::ConstraintDamping::Tags::
                      ConstraintGamma2>>,
          dg::Initialization::face_compute_tags<
              domain::Tags::BoundaryCoordinates<volume_dim, true>,
              GeneralizedHarmonic::CharacteristicFieldsCompute<volume_dim,
                                                               frame>>,
          dg::Initialization::exterior_compute_tags<
              GeneralizedHarmonic::CharacteristicFieldsCompute<volume_dim,
                                                               frame>>,
          !BjorhusExternalBoundary, true>,
      tmpl::conditional_t<evolution::is_analytic_solution_v<analytic_solution>,
                          Initialization::Actions::AddComputeTags<
                              tmpl::list<evolution::Tags::AnalyticCompute<
                                  volume_dim, analytic_solution_tag,
                                  analytic_solution_fields>>>,
                          tmpl::list<>>,
      dg::Actions::InitializeMortars<boundary_scheme, !BjorhusExternalBoundary>,
      Initialization::Actions::AddComputeTags<
          tmpl::list<evolution::Tags::AnalyticCompute<
              volume_dim, analytic_solution_tag, analytic_solution_fields>>>,
      Initialization::Actions::DiscontinuousGalerkin<derived_metavars>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using gh_dg_element_array = DgElementArray<
      derived_metavars,
      tmpl::flatten<tmpl::list<
          Parallel::PhaseActions<Phase, Phase::Initialization,
                                 initialization_actions>,
          tmpl::conditional_t<
              evolution::is_numeric_initial_data_v<initial_data>,
              tmpl::list<
                  Parallel::PhaseActions<
                      Phase, Phase::RegisterWithElementDataReader,
                      tmpl::list<
                          importers::Actions::RegisterWithElementDataReader,
                          Parallel::Actions::TerminatePhase>>,
                  Parallel::PhaseActions<
                      Phase, Phase::ImportInitialData,
                      tmpl::list<importers::Actions::ReadVolumeData<
                                     evolution::OptionTags::NumericInitialData,
                                     typename system::variables_tag::tags_list>,
                                 importers::Actions::ReceiveVolumeData<
                                     evolution::OptionTags::NumericInitialData,
                                     typename system::variables_tag::tags_list>,
                                 Parallel::Actions::TerminatePhase>>>,
              tmpl::list<>>,
          Parallel::PhaseActions<
              Phase, Phase::InitializeInitialDataDependentQuantities,
              initialize_initial_data_dependent_quantities_actions>,
          Parallel::PhaseActions<Phase, Phase::InitializeTimeStepperHistory,
                                 SelfStart::self_start_procedure<step_actions>>,
          Parallel::PhaseActions<
              Phase, Phase::Register,
              tmpl::list<intrp::Actions::RegisterElementWithInterpolator,
                         observers::Actions::RegisterEventsWithObservers,
                         Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Phase, Phase::Evolve,
              tmpl::list<Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                         step_actions, Actions::AdvanceTime>>>>>;
  using component_list = tmpl::flatten<tmpl::list<
      observers::Observer<derived_metavars>,
      observers::ObserverWriter<derived_metavars>,
      intrp::Interpolator<derived_metavars>,
      intrp::InterpolationTarget<derived_metavars, AhA>,
      std::conditional_t<evolution::is_numeric_initial_data_v<initial_data>,
                         importers::ElementDataReader<derived_metavars>,
                         tmpl::list<>>,
      gh_dg_element_array>>;
};
