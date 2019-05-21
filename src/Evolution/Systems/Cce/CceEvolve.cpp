// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/CceEvolve.hpp"

namespace Cce {

template <typename BondiTag>
ComplexModalVector compute_mode_difference_at_scri(
    double time, std::string prefix, const ComplexModalVector& modes,
    size_t l_max) {
  std::string filename;
  if (cpp17::is_same_v<BondiTag, Tags::Beta>) {
    filename = "betaScriNoninertial.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::U>) {
    filename = "UScriNoninertial.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::Q>) {
    filename = "QScriNoninertial.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::W>) {
    filename = "WScriNoninertial.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::J>) {
    filename = "JScriNoninertial.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::H>) {
    filename = "HScriNoninertial.h5";
  }

  ModeComparisonManager mode_compare(prefix + filename, l_max);
  return mode_compare.mode_difference(time, modes);
}

void run_trial_cce(std::string input_filename,
                   std::string comparison_file_prefix, size_t simulation_l_max,
                   size_t comparison_l_max, size_t number_of_radial_points,
                   std::string output_file_suffix,
                   size_t rational_timestep_numerator,
                   size_t rational_timestep_denominator,
                   bool calculate_psi4_diagnostic,
                   size_t l_filter_start, double start_time,
                   double end_time) noexcept {
  TimeSteppers::RungeKutta3 stepper{};
  TimeSteppers::History<ComplexDataVector, ComplexDataVector> history{};

  auto data_manager = CceBoundaryDataManager<CubicInterpolator, 500>{
      input_filename + ".h5", simulation_l_max};
  // auto data_manager = CceBoundaryDataManager<BarycentricInterpolator<10>,
  // 500>{ input_filename + ".h5", simulation_l_max};

  Variables<tmpl::append<sw_derivative_tags_to_compute_for<Tags::Beta>,
                         sw_derivative_tags_to_compute_for<Tags::Q>,
                         sw_derivative_tags_to_compute_for<Tags::U>,
                         sw_derivative_tags_to_compute_for<Tags::W>,
                         sw_derivative_tags_to_compute_for<Tags::H>>>
      test_creation{2};

  size_t l_max = data_manager.get_l_max();

  ModeRecorder recorder{input_filename + output_file_suffix + ".h5",
                        comparison_l_max, simulation_l_max};

  // TODO This is a bit inelegant
  std::vector<double> times{};
  if (comparison_file_prefix != "") {
    h5::H5File<h5::AccessType::ReadOnly> beta_comparison_file{
        comparison_file_prefix + "betaScriNoninertial.h5"};
    auto& mode00_data = beta_comparison_file.get<h5::Dat>("/Y_l0_m0");
    Matrix times_mat = mode00_data.get_data_subset(
        std::vector<size_t>{0}, 0, mode00_data.get_dimensions()[0]);
    times.resize(mode00_data.get_dimensions()[0]);
    for (size_t i = 0; i < times_mat.rows(); ++i) {
      times[i] = times_mat(i, 0);
    }
  }
  // create a variables for boundary data
  Variables<all_boundary_tags> boundary_variables{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  // for the pre-computation buffers
  Variables<pre_computation_boundary_coefficient_buffer_tags> boundary_buffers{
      2 * Spectral::Swsh::number_of_swsh_coefficients(l_max)};
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>>>
      radial_derivative_buffers{
          number_of_radial_points *
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  // For the intermediate computation buffers
  Variables<tmpl::remove_duplicates<
      tmpl::flatten<tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
          all_sw_derivative_tags,
          tmpl::bind<coefficient_buffer_tags_for_derivative_tag, tmpl::_1>>>>>>>
      angular_coefficient_buffers{
          2 * number_of_radial_points *
          Spectral::Swsh::number_of_swsh_coefficients(l_max)};

  // BondiAndSWCache Variables
  Variables<all_bondi_and_sw_cache_tags> bondi_and_sw_cache{
      number_of_radial_points *
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  Variables<all_sw_derivative_tags> sw_derivatives{
      number_of_radial_points *
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  Variables<pre_computation_tags> secondary{
      number_of_radial_points *
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  Variables<all_integrand_tags> integrands{
      number_of_radial_points *
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  Variables<all_temporary_equation_tags> temporary{
      number_of_radial_points *
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  std::deque<Variables<tmpl::append<
      ComputeNonInertial<Tags::BoundaryValue<Tags::Psi4>>::return_tags,
      ComputeNonInertial<Tags::BoundaryValue<Tags::Psi4>>::argument_tags>>>
      scri_plus_values;


  // in order to use the time architecture, we need to make a slab
  if (end_time == -1.0) {
    end_time =
        data_manager
            .get_time_buffer()[data_manager.get_time_buffer().size() - 1];
  }
  Slab only_slab{start_time, end_time};
  TimeDelta time_step{
      only_slab, Time::rational_t{
                     static_cast<int>(rational_timestep_numerator),
                     static_cast<int>(static_cast<int>(end_time - start_time) *
                                      rational_timestep_denominator)}};
  TimeId time{true, 0, Time{only_slab, Time::rational_t{0, 1}}};
  data_manager.populate_hypersurface_boundary_data(
      make_not_null(&boundary_variables), start_time);

  initialize_first_j_slice(
      make_not_null(&radial_derivative_buffers),
      make_not_null(&boundary_variables), make_not_null(&boundary_buffers),
      make_not_null(&bondi_and_sw_cache), make_not_null(&secondary), l_max);

  // In order to perform the time differentiation for the psi_4 diagnostic, we
  // need to keep track of the time values
  size_t max_time_list_size = 30;

  size_t max_scri_deque_size = 15;

  // we'll only update the time list if we get to a new, later time (so we
  // avoid storing out-of-order substeps.
  double latest_time = start_time - 1.0;

  std::deque<double> time_list;

  std::deque<ComplexDataVector> lambda_1_history;

  size_t step_counter = 0;

  while (data_manager.populate_hypersurface_boundary_data(
             make_not_null(&boundary_variables), time.time().value()) and
         time.time().value() < end_time) {
    step_counter++;
    precompute_shared_cce_quantities(
        make_not_null(&radial_derivative_buffers),
        make_not_null(&boundary_variables), make_not_null(&boundary_buffers),
        make_not_null(&bondi_and_sw_cache), make_not_null(&secondary), l_max);

    // Fix boundary condition for H to be like that of SpEC (there's a
    // coordinates on boundary vs coordinates in bulk problem so the definitions
    // do not quite align).
    ComplexDataVector boundary_dy_j{
        get(get<Tags::Dy<Tags::J>>(bondi_and_sw_cache)).data().data(),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    ComplexDataVector boundary_du_r_divided_by_r{
        get(get<Tags::DuRDividedByR>(secondary)).data().data(),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    get(get<Tags::BoundaryValue<Tags::H>>(boundary_variables)).data() =
        get(get<Tags::BoundaryValue<Tags::SpecH>>(boundary_variables)).data() +
        2.0 * boundary_du_r_divided_by_r * boundary_dy_j;

    tmpl::for_each<tmpl::list<Tags::Beta, Tags::Q, Tags::U, Tags::W,
                              Tags::H>>([&bondi_and_sw_cache, &sw_derivatives,
                                         &secondary, &temporary, &integrands,
                                         &boundary_variables, &history,
                                         &angular_coefficient_buffers, &l_max,
                                         &number_of_radial_points, &time,
                                         &start_time,
                                         &l_filter_start](auto x) {
      using bondi_tag = typename decltype(x)::type;

      compute_cce_bondi_values_and_sw_cache_for_tag<bondi_tag>(
          make_not_null(&bondi_and_sw_cache), sw_derivatives, secondary, l_max);

      compute_cce_sw_derivatives_for_tag<bondi_tag>(
          make_not_null(&bondi_and_sw_cache), make_not_null(&sw_derivatives),
          secondary, make_not_null(&angular_coefficient_buffers), l_max);

      tmpl::for_each<integrand_terms_to_compute_for_bondi_variable<bondi_tag>>(
          [&bondi_and_sw_cache, &sw_derivatives, &secondary, &temporary,
           &integrands, &number_of_radial_points, &l_max](auto y) {
            using bondi_integrand_tag = typename decltype(y)::type;

            ComputeBondiIntegrandFromVariables<bondi_integrand_tag>{}.evaluate(
                make_not_null(&integrands), make_not_null(&temporary),
                bondi_and_sw_cache, sw_derivatives, secondary);
          });
      RadialIntegrateBondi<bondi_tag>{}(make_not_null(&bondi_and_sw_cache),
                                        integrands, boundary_variables, l_max);
      filter_bondi_value<bondi_tag>(make_not_null(&bondi_and_sw_cache),
                                    number_of_radial_points, 4.0, l_max,
                                    l_filter_start);
    });

    if (time.time() == Time{only_slab, Time::rational_t{0, 1}}) {
      history.insert_initial(time.time(),
                             get(get<Tags::J>(bondi_and_sw_cache)).data(),
                             get(get<Tags::H>(bondi_and_sw_cache)).data());
    } else {
      ComplexDataVector deriv = get(get<Tags::H>(bondi_and_sw_cache)).data();
      history.insert(time.time(), get(get<Tags::J>(bondi_and_sw_cache)).data(),
                     std::move(deriv));
    }
    if (time.step_time().value() > latest_time && time.substep() == 0) {
      // perform a comparison of boundary values and scri+ values on each new
      // time advancement

      // only dump the comparison data if the current timestep is close to one
      // of the comparison points, otherwise dump every fourth timestep
      if (std::find_if(times.begin(), times.end(),
                       [&time](auto x) {
                         return abs(x - time.step_time().value()) < 1.0e-8;
                       }) != times.end() or
          (comparison_file_prefix == "" and step_counter % 12 == 0)) {
        tmpl::for_each<tmpl::list<
            Tags::Beta, Tags::Q, Tags::U, Tags::W, Tags::SpecH, Tags::J,
            Tags::NullL<0>, Tags::NullL<1>, Tags::NullL<2>, Tags::NullL<3>,
            Tags::R>>([&comparison_file_prefix, &boundary_variables,
                       &bondi_and_sw_cache, &sw_derivatives, &integrands,
                       &l_max, &recorder, &time, &number_of_radial_points,
                       &comparison_l_max](auto x) {
          using tag = typename decltype(x)::type;
          recorder.append_mode_data(
              "/" + tag::name() + "_boundary", time.step_time().value(),
              Spectral::Swsh::libsharp_to_standard_modes<tag::type::type::spin>(
                  Spectral::Swsh::swsh_transform<tag::type::type::spin>(
                      get(get<Tags::BoundaryValue<tag>>(boundary_variables))
                          .data(),
                      l_max),
                  l_max),
              comparison_l_max);
        });
        tmpl::for_each<tmpl::list<Tags::J, Tags::Beta, Tags::Q, Tags::U,
                                  Tags::W, Tags::H>>([&comparison_file_prefix,
                                                      &boundary_variables,
                                                      &bondi_and_sw_cache,
                                                      &l_max, &recorder, &time,
                                                      &number_of_radial_points,
                                                      &comparison_l_max](
                                                         auto x) {
          using tag = typename decltype(x)::type;
          ComplexDataVector scri_slice{
              get(get<tag>(bondi_and_sw_cache)).data().data() +
                  (number_of_radial_points - 1) *
                      Spectral::Swsh::number_of_swsh_collocation_points(l_max),
              Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
          recorder.append_mode_data(
              "/" + tag::name() + "_scri", time.step_time().value(),
              Spectral::Swsh::libsharp_to_standard_modes<tag::type::type::spin>(
                  Spectral::Swsh::swsh_transform<tag::type::type::spin>(
                      scri_slice, l_max),
                  l_max),
              comparison_l_max);
          if (comparison_file_prefix != "") {
            recorder.append_mode_data(
                "/" + tag::name() + "_scri_difference",
                time.step_time().value(),
                compute_mode_difference_at_scri<tag>(
                    time.step_time().value(), comparison_file_prefix,
                    Spectral::Swsh::libsharp_to_standard_modes<
                        tag::type::type::spin>(
                        Spectral::Swsh::swsh_transform<tag::type::type::spin>(
                            scri_slice, l_max),
                        l_max),
                    comparison_l_max),
                comparison_l_max);
          }
        });
      }

      // perform the computation for the psi_4 diagnostic
      latest_time = time.step_time().value();
      scri_plus_values.emplace(
          scri_plus_values.end(),
          Spectral::Swsh::number_of_swsh_collocation_points(l_max));
      auto& scri_plus_vars = scri_plus_values.back();

      // prepare the boundary values in the new variables
      tmpl::for_each<typename ComputeNonInertial<
          Tags::BoundaryValue<Tags::Psi4>>::scri_newman_penrose_tags>(
          [&scri_plus_vars, &boundary_variables, &bondi_and_sw_cache,
           &sw_derivatives, &secondary, &l_max](auto x) {
            using tag = typename decltype(x)::type;
            ComputeNonInertialFromVariables<tag>::evaluate(
                l_max, make_not_null(&scri_plus_vars), boundary_variables,
                bondi_and_sw_cache, sw_derivatives, secondary);
          });

      tmpl::for_each<typename ComputeNonInertial<Tags::BoundaryValue<
          Tags::Psi4>>::boundary_from_volume_bondi_and_sw_cache_tags>(
          [&scri_plus_vars, &bondi_and_sw_cache, &l_max](auto x) {
            using boundary_tag = typename decltype(x)::type;
            using tag = typename boundary_tag::tag;
            get(get<boundary_tag>(scri_plus_vars)).data() =
                surface_value_at_scri(get(get<tag>(bondi_and_sw_cache)).data(),
                                      l_max);
          });
      tmpl::for_each<typename ComputeNonInertial<Tags::BoundaryValue<
          Tags::Psi4>>::boundary_from_volume_sw_derivative_tags>(
          [&scri_plus_vars, &sw_derivatives, &l_max](auto x) {
            using boundary_tag = typename decltype(x)::type;
            using tag = typename boundary_tag::tag;
            get(get<boundary_tag>(scri_plus_vars)).data() =
                surface_value_at_scri(get(get<tag>(sw_derivatives)).data(),
                                      l_max);
          });
      tmpl::for_each<typename ComputeNonInertial<Tags::BoundaryValue<
          Tags::Psi4>>::boundary_from_volume_secondary_tags>(
          [&scri_plus_vars, &secondary, &l_max](auto x) {
            using boundary_tag = typename decltype(x)::type;
            using tag = typename boundary_tag::tag;
            get(get<boundary_tag>(scri_plus_vars)).data() =
                surface_value_at_scri(get(get<tag>(secondary)).data(), l_max);
          });

      if (scri_plus_values.size() > max_scri_deque_size) {
        scri_plus_values.pop_front();
      }
      time_list.push_back(time.step_time().value());
      lambda_1_history.push_back(
          get(get<Tags::BoundaryValue<Tags::Lambda1>>(scri_plus_vars)).data());

      if (time_list.size() >= max_time_list_size) {
        if (calculate_psi4_diagnostic) {
          // the time list is full, so we can extract the psi value associated
          // with the front of the scri_plus deque

          auto& past_scri_vars = scri_plus_values.front();
          double psi_4_time =
              time_list[time_list.size() - scri_plus_values.size()];
          compute_du_lambda_1(
              make_not_null(&get(
                  get<Tags::BoundaryValue<Tags::DuLambda1>>(past_scri_vars))),
              psi_4_time, lambda_1_history, time_list);

          ComputeNonInertialFromVariables<Tags::BoundaryValue<Tags::Psi4>>::
              evaluate(l_max, make_not_null(&past_scri_vars),
                       boundary_variables, bondi_and_sw_cache, sw_derivatives,
                       secondary);

          recorder.append_mode_data(
              "/psi4", psi_4_time,
              Spectral::Swsh::libsharp_to_standard_modes<-2>(
                  Spectral::Swsh::swsh_transform<-2>(
                      get(get<Tags::BoundaryValue<Tags::Psi4>>(past_scri_vars))
                          .data(),
                      l_max),
                  l_max),
              comparison_l_max);
        }
        time_list.pop_front();
        lambda_1_history.pop_front();
      }
      //
    }
    stepper.update_u(
        make_not_null(&get(get<Tags::J>(bondi_and_sw_cache)).data()),
        make_not_null(&history), time_step);

    size_t i = (number_of_radial_points - 1) *
               Spectral::Swsh::number_of_swsh_collocation_points(l_max);

    time = stepper.next_time_id(time, time_step);
    printf("text time: %f\n", time.time().value());
    // filter_bondi_value<Tags::J>(make_not_null(&bondi_and_sw_cache),
    // radial_filter_start, 4.0, l_max,
    // l_filter_start);
  }
}
}  // namespace Cce
