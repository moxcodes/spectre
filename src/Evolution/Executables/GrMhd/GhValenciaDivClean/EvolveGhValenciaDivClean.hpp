// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "ApparentHorizons/Tags.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/Executables/GrMhd/GhValenciaDivClean/GhValenciaDivCleanBase.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/RegisterDerived.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

template <typename InitialData, typename... InterpolationTargetTags>
struct EvolutionMetavars
    : public virtual GhValenciaDivCleanDefaults,
      public GhValenciaDivCleanTemplateBase<
          EvolutionMetavars<InitialData, InterpolationTargetTags...>> {
  using const_global_cache_tags = typename GhValenciaDivCleanTemplateBase<
      EvolutionMetavars>::const_global_cache_tags;
  using observed_reduction_data_tags = typename GhValenciaDivCleanTemplateBase<
      EvolutionMetavars>::observed_reduction_data_tags;
  using component_list = typename GhValenciaDivCleanTemplateBase<
      EvolutionMetavars>::component_list;
  using factory_creation = typename GhValenciaDivCleanTemplateBase<
      EvolutionMetavars>::factory_creation;
  template <typename ParallelComponent>
  using registration_list = typename GhValenciaDivCleanTemplateBase<
      EvolutionMetavars>::template registration_list<ParallelComponent>;

  static constexpr Options::String help{
      "Evolve the Valencia formulation of the GRMHD system with divergence "
      "cleaning, coupled to a dynamic spacetime evolved with the Generalized "
      "Harmonic formulation\n"};
};

struct KerrHorizon {
  using tags_to_observe =
      tmpl::list<StrahlkorperTags::EuclideanSurfaceIntegralVectorCompute<
          hydro::Tags::MassFlux<DataVector, 3>, ::Frame::Inertial>>;
  using compute_items_on_source = tmpl::list<>;
  using vars_to_interpolate_to_target =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>>;
  using compute_items_on_target = tmpl::push_front<
      tags_to_observe,
      StrahlkorperTags::EuclideanAreaElementCompute<::Frame::Inertial>,
      hydro::Tags::MassFluxCompute<DataVector, 3, ::Frame::Inertial>>;
  using compute_target_points =
      intrp::TargetPoints::KerrHorizon<KerrHorizon, ::Frame::Inertial>;
  using post_interpolation_callback =
      intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, KerrHorizon,
                                                   KerrHorizon>;
  using interpolating_component =
      typename metavariables::dg_element_array_component;
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &grmhd::GhValenciaDivClean::BoundaryConditions::register_derived_with_charm,
    &grmhd::GhValenciaDivClean::BoundaryCorrections::
        register_derived_with_charm,
    &GeneralizedHarmonic::ConstraintDamping::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<TimeSequence<double>>,
    &Parallel::register_derived_classes_with_charm<TimeSequence<std::uint64_t>>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        PhaseChange<metavariables::phase_changes>>,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
