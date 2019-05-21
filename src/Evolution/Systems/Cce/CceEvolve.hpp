// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Equations.hpp"
#include "Evolution/Systems/Cce/LinearSolve.hpp"
#include "Evolution/Systems/Cce/NonInertialPsi4.hpp"
#include "Evolution/Systems/Cce/Precomputation.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Evolution/Systems/Cce/SerialEvolveHelpers.hpp"
#include "Evolution/Systems/Cce/WriteToH5.hpp"
#include "Time/History.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"

namespace Cce{

using boundary_variables_tags = tmpl::list<
    Tags::BoundaryValue<Tags::Beta>, Tags::BoundaryValue<Tags::J>,
    Tags::BoundaryValue<Tags::Dr<Tags::J>>, Tags::BoundaryValue<Tags::Q>,
    Tags::BoundaryValue<Tags::U>, Tags::BoundaryValue<Tags::W>,
    Tags::BoundaryValue<Tags::H>, Tags::BoundaryValue<Tags::SpecH>,
    Tags::BoundaryValue<Tags::R>, Tags::BoundaryValue<Tags::NullL<0>>,
    Tags::BoundaryValue<Tags::NullL<1>>, Tags::BoundaryValue<Tags::NullL<2>>,
    Tags::BoundaryValue<Tags::NullL<3>>,
    Tags::BoundaryValue<Tags::AngularDNullL<1, 0>>,
    Tags::BoundaryValue<Tags::AngularDNullL<1, 1>>,
    Tags::BoundaryValue<Tags::AngularDNullL<1, 2>>,
    Tags::BoundaryValue<Tags::AngularDNullL<1, 3>>,

    Tags::BoundaryValue<Tags::AngularDNullL<2, 0>>,
    Tags::BoundaryValue<Tags::AngularDNullL<2, 1>>,
    Tags::BoundaryValue<Tags::AngularDNullL<2, 2>>,
    Tags::BoundaryValue<Tags::AngularDNullL<2, 3>>,

    Tags::BoundaryValue<Tags::DLambdaNullMetric<2, 2>>,
    Tags::BoundaryValue<Tags::DLambdaNullMetric<2, 3>>,
    Tags::BoundaryValue<Tags::DLambdaNullMetric<3, 3>>,

    Tags::BoundaryValue<Tags::InverseAngularNullMetric<2, 2>>,
    Tags::BoundaryValue<Tags::InverseAngularNullMetric<2, 3>>,
    Tags::BoundaryValue<Tags::InverseAngularNullMetric<3, 3>>>;

using all_boundary_tags =
    tmpl::append<boundary_variables_tags, pre_computation_boundary_tags>;

using all_integrand_tags = tmpl::flatten<
    tmpl::list<integrand_terms_to_compute_for_bondi_variable<Tags::Beta>,
               integrand_terms_to_compute_for_bondi_variable<Tags::Q>,
               integrand_terms_to_compute_for_bondi_variable<Tags::U>,
               integrand_terms_to_compute_for_bondi_variable<Tags::W>,
               integrand_terms_to_compute_for_bondi_variable<Tags::H>>>;

using all_temporary_equation_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::list<
        tags_needed_for_integrand_computation<Tags::Integrand<Tags::Beta>,
                                              TagsCategory::Temporary>,
        tags_needed_for_integrand_computation<Tags::PoleOfIntegrand<Tags::Q>,
                                              TagsCategory::Temporary>,
        tags_needed_for_integrand_computation<Tags::RegularIntegrand<Tags::Q>,
                                              TagsCategory::Temporary>,
        tags_needed_for_integrand_computation<Tags::Integrand<Tags::U>,
                                              TagsCategory::Temporary>,
        tags_needed_for_integrand_computation<Tags::PoleOfIntegrand<Tags::W>,
                                              TagsCategory::Temporary>,
        tags_needed_for_integrand_computation<Tags::RegularIntegrand<Tags::W>,
                                              TagsCategory::Temporary>,
        tags_needed_for_integrand_computation<Tags::PoleOfIntegrand<Tags::H>,
                                              TagsCategory::Temporary>,
        tags_needed_for_integrand_computation<Tags::RegularIntegrand<Tags::H>,
                                              TagsCategory::Temporary>,
        tags_needed_for_integrand_computation<Tags::LinearFactor<Tags::H>,
                                              TagsCategory::Temporary>,
        tags_needed_for_integrand_computation<
            Tags::LinearFactorForConjugate<Tags::H>,
            TagsCategory::Temporary>>>>;

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
                   bool calculate_psi4_diagnostic,
                   size_t l_filter_start, double start_time = 0.0,
                   double end_time = -1.0) noexcept;
}  // namespace Cce
