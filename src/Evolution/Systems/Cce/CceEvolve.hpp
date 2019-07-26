// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/ComputePreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/ComputeSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/Equations.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/InitializeCce.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/LinearSolve.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "Evolution/Systems/Cce/SerialEvolveHelpers.hpp"
#include "Evolution/Systems/Cce/WriteToH5.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Time/History.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"

namespace Cce {

// TODO rework tags

using boundary_value_tags = tmpl::list<
    Tags::BoundaryValue<Tags::BondiBeta>, Tags::BoundaryValue<Tags::BondiJ>,
    Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
    Tags::BoundaryValue<Tags::BondiQ>, Tags::BoundaryValue<Tags::BondiU>,
    Tags::BoundaryValue<Tags::Dr<Tags::BondiU>>,
    Tags::BoundaryValue<Tags::BondiW>, Tags::BoundaryValue<Tags::BondiH>,
    Tags::BoundaryValue<Tags::SpecH>>;

using gauge_transform_boundary_tags =
    tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
               Tags::EvolutionGaugeBoundaryValue<Tags::DuRDividedByR>,
               Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>,
               Tags::EvolutionGaugeBoundaryValue<Tags::Dr<Tags::BondiJ>>,
               Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>,
               Tags::EvolutionGaugeBoundaryValue<Tags::BondiQ>,
               Tags::EvolutionGaugeBoundaryValue<Tags::BondiU>,
               Tags::EvolutionGaugeBoundaryValue<Tags::BondiW>,
               Tags::EvolutionGaugeBoundaryValue<Tags::BondiH>, Tags::GaugeA,
               Tags::GaugeB, Tags::GaugeC, Tags::GaugeD, Tags::Du<Tags::GaugeA>,
               Tags::Du<Tags::GaugeB>, Tags::Du<Tags::GaugeC>,
               Tags::Du<Tags::GaugeD>, Tags::GaugeOmega, Tags::GaugeOmegaCD,
               Tags::Du<Tags::GaugeOmega>, Tags::Du<Tags::GaugeOmegaCD>,
               Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                Spectral::Swsh::Tags::Eth>,
               Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                Spectral::Swsh::Tags::Eth>,
               Tags::U0, Tags::RobinsonTrautmanW,
               Tags::Du<Tags::RobinsonTrautmanW>>;

using gauge_confirmation_scri_tags =
    tmpl::list<Tags::CauchyGaugeScriPlus<Tags::BondiBeta>,
               Tags::CauchyGaugeScriPlus<Tags::U0>>;

using gauge_confirmation_volume_tags =
    tmpl::list<Tags::CauchyGauge<Tags::BondiBeta>,
               Tags::CauchyGauge<Tags::BondiJ>, Tags::CauchyGauge<Tags::BondiU>,
               Tags::CauchyGauge<Tags::BondiQ>, Tags::CauchyGauge<Tags::BondiW>,
               Tags::CauchyGauge<Tags::SpecH>, Tags::CauchyGauge<Tags::BondiH>>;

using angular_coordinate_tags =
    tmpl::list<Tags::CauchyAngularCoords, Tags::DuCauchyAngularCoords,
               Tags::CauchyCartesianCoords, Tags::DuCauchyCartesianCoords,
               /* the following only used for gauge transform confirmation
                  routine*/
               Tags::InertialAngularCoords, Tags::DuInertialAngularCoords,
               Tags::InertialCartesianCoords, Tags::DuInertialCartesianCoords>;

using scri_tags = tmpl::list<Tags::CauchyGauge<Tags::News>, Tags::News,
                             Tags::NonInertialNews, Tags::InertialRetardedTime,
                             Tags::Du<Tags::InertialRetardedTime>>;

using all_boundary_tags =
    tmpl::append<boundary_value_tags, pre_computation_boundary_tags>;

using all_integrand_tags = tmpl::flatten<
    tmpl::list<integrand_terms_to_compute_for_bondi_variable<Tags::BondiBeta>,
               integrand_terms_to_compute_for_bondi_variable<Tags::BondiQ>,
               integrand_terms_to_compute_for_bondi_variable<Tags::BondiU>,
               integrand_terms_to_compute_for_bondi_variable<Tags::BondiW>,
               integrand_terms_to_compute_for_bondi_variable<Tags::BondiH>>>;

using all_temporary_equation_tags = tmpl::remove_duplicates<tmpl::append<
    ComputeBondiIntegrand<Tags::Integrand<Tags::BondiBeta>>::temporary_tags,
    ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiQ>>::temporary_tags,
    ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiQ>>::temporary_tags,
    ComputeBondiIntegrand<Tags::Integrand<Tags::BondiU>>::temporary_tags,
    ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiW>>::temporary_tags,
    ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiW>>::temporary_tags,
    ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiH>>::temporary_tags,
    ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiH>>::temporary_tags,
    ComputeBondiIntegrand<Tags::LinearFactor<Tags::BondiH>>::temporary_tags,
    ComputeBondiIntegrand<
        Tags::LinearFactorForConjugate<Tags::BondiH>>::temporary_tags>>;

template <typename BondiTag>
ComplexModalVector compute_mode_difference_at_scri(
    double time, std::string prefix, const ComplexModalVector& modes,
    size_t l_max);

void run_trial_cce(std::string input_filename,
                   std::string comparison_file_prefix, size_t simulation_l_max,
                   size_t comparison_l_max, size_t number_of_radial_points,
                   std::string output_file_suffix,
                   size_t rational_timestep_numerator,
                   size_t rational_timestep_denominator,
                   bool calculate_psi4_diagnostic, size_t l_filter_start,
                   double start_time = 0.0, double end_time = -1.0) noexcept;

void run_trial_regularity_preserving_cce(
    std::string input_filename, std::string comparison_file_prefix,
    size_t simulation_l_max, size_t comparison_l_max,
    size_t number_of_radial_points, std::string output_file_suffix,
    size_t rational_timestep_numerator, size_t rational_timestep_denominator,
    bool /*calculate_psi4_diagnostic*/, size_t l_filter_start,
    double start_time = 0.0, double end_time = -1.0,
    bool regularity_preserving = true) noexcept;

void test_regularity_preserving_cce_rt(
    std::string input_filename, size_t simulation_l_max,
    size_t comparison_l_max, size_t number_of_radial_points,
    std::string output_file_suffix, size_t rational_timestep_numerator,
    size_t rational_timestep_denominator, double end_time) noexcept;
}  // namespace Cce
