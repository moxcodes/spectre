// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/CceEvolve.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/RobinsonTrautman.hpp"
#include "Evolution/Systems/Cce/ScriPlusInterpolationManager.hpp"
#include "Evolution/Systems/Cce/ScriPlusValues.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"

namespace Cce {

template <typename BondiTag>
ComplexModalVector compute_mode_difference_at_scri(
    double time, std::string prefix, const ComplexModalVector& modes,
    size_t l_max) {
  std::string filename;
  if (cpp17::is_same_v<BondiTag, Tags::BondiBeta> or
      cpp17::is_same_v<BondiTag, Tags::CauchyGaugeScriPlus<Tags::BondiBeta>>) {
    filename = "betaScriNoninertial.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::BondiU> or
      cpp17::is_same_v<BondiTag, Tags::U0> or
      cpp17::is_same_v<BondiTag, Tags::CauchyGaugeScriPlus<Tags::U0>>) {
    filename = "UScriNoninertial.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::BondiQ> or
      cpp17::is_same_v<BondiTag, Tags::CauchyGaugeScriPlus<Tags::BondiQ>>) {
    filename = "QScriNoninertial.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::BondiW> or
      cpp17::is_same_v<BondiTag, Tags::CauchyGaugeScriPlus<Tags::BondiW>>) {
    filename = "WScriNoninertial.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::BondiJ> or
      cpp17::is_same_v<BondiTag, Tags::CauchyGaugeScriPlus<Tags::BondiJ>>) {
    filename = "JScriNoninertial.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::BondiH> or
      cpp17::is_same_v<BondiTag, Tags::CauchyGaugeScriPlus<Tags::BondiH>>) {
    filename = "HScriNoninertial.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::News>) {
    filename = "NewsNoninertial.h5";
    ComplexModalVector news_corrected_modes{modes.size()};
    for (int l = 0; l <= l_max; ++l) {
      for (int m = -l; m <= l; ++m) {
        news_corrected_modes[static_cast<size_t>(square(l) + l + m)] =
            ((m % 2) == 0 ? -1.0 : 1.0) *
            conj(modes[static_cast<size_t>(square(l) + l - m)]);
      }
    }
    ModeComparisonManager mode_compare(prefix + filename, l_max);
    return mode_compare.mode_difference(time, news_corrected_modes);
  }

  ModeComparisonManager mode_compare(prefix + filename, l_max);
  return mode_compare.mode_difference(time, modes);
}

template <typename BondiTag>
ComplexModalVector compute_mode_difference_at_bondi_r_200(
    double time, std::string prefix, const ComplexModalVector& modes,
    size_t l_max) {
  std::string filename;
  if (cpp17::is_same_v<BondiTag, Tags::BondiBeta> or
      cpp17::is_same_v<BondiTag, Tags::CauchyGauge<Tags::BondiBeta>>) {
    filename = "betaR200modes.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::BondiU> or
      cpp17::is_same_v<BondiTag, Tags::CauchyGauge<Tags::BondiU>>) {
    filename = "UR200modes.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::BondiQ> or
      cpp17::is_same_v<BondiTag, Tags::CauchyGauge<Tags::BondiQ>>) {
    filename = "QR200modes.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::BondiW> or
      cpp17::is_same_v<BondiTag, Tags::CauchyGauge<Tags::BondiW>>) {
    filename = "WR200modes.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::BondiJ> or
      cpp17::is_same_v<BondiTag, Tags::CauchyGauge<Tags::BondiJ>>) {
    filename = "JR200modes.h5";
  }
  if (cpp17::is_same_v<BondiTag, Tags::SpecH> or
      cpp17::is_same_v<BondiTag, Tags::CauchyGauge<Tags::SpecH>>) {
    filename = "HR200modes.h5";
  }

  ModeComparisonManager mode_compare(prefix + filename, l_max);
  return mode_compare.mode_difference(time, modes);
}

template <typename BondiTag, typename SwshVariablesTag,
          typename SwshBufferVariablesTag, typename PreSwshVariablesTag,
          typename DataBoxType>
void perform_hypersurface_computation(
    const gsl::not_null<DataBoxType*> box) noexcept {
  mutate_all_pre_swsh_derivatives_for_tag<BondiTag>(box);

  mutate_all_swsh_derivatives_for_tag<
      BondiTag, SwshVariablesTag, SwshBufferVariablesTag, PreSwshVariablesTag>(
      box);

  tmpl::for_each<integrand_terms_to_compute_for_bondi_variable<BondiTag>>(
      [&box](auto x) {
        using bondi_integrand_tag = typename decltype(x)::type;
        db::mutate_apply<ComputeBondiIntegrand<bondi_integrand_tag>>(box);
      });

  db::mutate_apply<RadialIntegrateBondi<Tags::BoundaryValue, BondiTag>>(box);
}

template <typename BondiTag, typename SwshVariablesTag,
          typename SwshBufferVariablesTag, typename PreSwshVariablesTag,
          typename DataBoxType>
void regularity_preserving_hypersurface_computation(
    const gsl::not_null<DataBoxType*> box,
    bool regularity_preserving = true) noexcept {
  // printf("boundary computation\n");
  if (regularity_preserving) {
    db::mutate_apply<ComputeGaugeAdjustedBoundaryValue<BondiTag>>(box);
  }
  // printf("pre swsh derivatives\n");
  mutate_all_pre_swsh_derivatives_for_tag<BondiTag>(box);

  // printf("swsh derivatives\n");
  mutate_all_swsh_derivatives_for_tag<
      BondiTag, SwshVariablesTag, SwshBufferVariablesTag, PreSwshVariablesTag>(
      box);

  // printf("integrand terms\n");
  tmpl::for_each<integrand_terms_to_compute_for_bondi_variable<BondiTag>>(
      [&box](auto x) {
        using bondi_integrand_tag = typename decltype(x)::type;
        db::mutate_apply<ComputeBondiIntegrand<bondi_integrand_tag>>(box);
      });

  // printf("integral\n");
  if (regularity_preserving) {
    db::mutate_apply<
        RadialIntegrateBondi<Tags::EvolutionGaugeBoundaryValue, BondiTag>>(box);
  } else {
    db::mutate_apply<RadialIntegrateBondi<Tags::BoundaryValue, BondiTag>>(box);
  }

  if (cpp17::is_same_v<BondiTag, Tags::BondiU>) {
    db::mutate_apply<GaugeUpdateU>(box, regularity_preserving);
    db::mutate_apply<GaugeUpdateDuXtildeOfX>(box);

    if (regularity_preserving) {
      db::mutate_apply<ComputeGaugeAdjustedBoundaryValue<Tags::DuRDividedByR>>(
          box);
      db::mutate_apply<PrecomputeCceDependencies<
          Tags::EvolutionGaugeBoundaryValue, Tags::DuRDividedByR>>(box);
    }
  }
}

template <typename BondiTag, typename SwshVariablesTag,
          typename SwshBufferVariablesTag, typename PreSwshVariablesTag,
          typename DataBoxType>
void regularity_preserving_robinson_trautman_hypersurface_computation(
    const gsl::not_null<DataBoxType*> box, double time) noexcept {
  // printf("boundary computation\n");
  db::mutate_apply<ComputeGaugeAdjustedBoundaryValue<BondiTag>>(box);
  // printf("pre swsh derivatives\n");
  mutate_all_pre_swsh_derivatives_for_tag<BondiTag>(box);

  // printf("swsh derivatives\n");
  mutate_all_swsh_derivatives_for_tag<
      BondiTag, SwshVariablesTag, SwshBufferVariablesTag, PreSwshVariablesTag>(
      box);

  // printf("integrand terms\n");
  tmpl::for_each<integrand_terms_to_compute_for_bondi_variable<BondiTag>>(
      [&box](auto x) {
        using bondi_integrand_tag = typename decltype(x)::type;
        db::mutate_apply<ComputeBondiIntegrand<bondi_integrand_tag>>(box);
      });

  // printf("integral\n");
  db::mutate_apply<
      RadialIntegrateBondi<Tags::EvolutionGaugeBoundaryValue, BondiTag>>(box);

  if (cpp17::is_same_v<BondiTag, Tags::BondiU>) {
    db::mutate<Tags::U0>(
        box,
        [&time](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
                    u_0,
                const size_t l_max) {
          Spectral::Swsh::SpinWeightedSphericalHarmonic swsh{1, 2, 2};
          for (const auto& collocation_point :
               Spectral::Swsh::precomputed_collocation<
                   Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max)) {
            // needs to be strictly positive.
            get(*u_0).data()[collocation_point.offset] =
                sin(5.0 * time) * 1.0e-3 *
                swsh.evaluate(collocation_point.theta, collocation_point.phi);
          }
        },
        db::get<Tags::LMax>(*box));

    db::mutate_apply<GaugeUpdateUManualTransform>(box);
    db::mutate_apply<GaugeUpdateDuXtildeOfX>(box);
    // db::mutate_apply<GaugeUpdateJacobianFromCoords<
    // Tags::GaugeA, Tags::GaugeB, Tags::CauchyAngularCoords,
    // Tags::InertialAngularCoords, Tags::DuInertialAngularCoords>>(box);
    // db::mutate_apply<GaugeUpdateJacobianFromCoords<
    // Tags::GaugeC, Tags::GaugeD, Tags::InertialAngularCoords,
    // Tags::CauchyAngularCoords, Tags::DuCauchyAngularCoords>>(box);
    db::mutate_apply<ComputeGaugeAdjustedBoundaryValue<Tags::DuRDividedByR>>(
        box);
    db::mutate_apply<PrecomputeCceDependencies<
        Tags::EvolutionGaugeBoundaryValue, Tags::DuRDividedByR>>(box);
  }
}

template <template <typename> class BoundaryPrefix, typename TagList,
          typename DataBoxType>
void record_boundary_values(const gsl::not_null<DataBoxType*> box,
                            const gsl::not_null<ModeRecorder*> recorder,
                            const double time, const size_t l_max,
                            const size_t comparison_l_max) noexcept {
  tmpl::for_each<TagList>([&box, &l_max, &recorder, &time,
                           &comparison_l_max](auto x) {
    using tag = typename decltype(x)::type;
    typename db::item_type<BoundaryPrefix<tag>>::type transform_buffer =
        get(db::get<BoundaryPrefix<tag>>(*box));
    recorder->append_mode_data("/" + tag::name() + "_boundary", time,
                               Spectral::Swsh::libsharp_to_goldberg_modes(
                                   Spectral::Swsh::swsh_transform(
                                       make_not_null(&transform_buffer), l_max),
                                   l_max)
                                   .data(),
                               comparison_l_max);
  });
}

template <typename TagList, typename DataBoxType>
void compare_and_record_scri_values(
    const gsl::not_null<DataBoxType*> box,
    const gsl::not_null<ModeRecorder*> recorder,
    const std::string comparison_file_prefix, const double time,
    const size_t l_max, const size_t comparison_l_max,
    const size_t number_of_radial_points) noexcept {
  tmpl::for_each<TagList>([&comparison_file_prefix, &box, &l_max, &recorder,
                           &time, &number_of_radial_points,
                           &comparison_l_max](auto x) {
    using tag = typename decltype(x)::type;
    typename db::item_type<tag>::type scri_slice;
    ComplexDataVector scri_slice_buffer = get(db::get<tag>(*box)).data();
    scri_slice.data() = ComplexDataVector{
        scri_slice_buffer.data() +
            (number_of_radial_points - 1) *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    auto scri_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
        Spectral::Swsh::swsh_transform(make_not_null(&scri_slice), l_max),
        l_max);
    recorder->append_mode_data("/" + tag::name() + "_scri", time,
                               scri_goldberg_modes.data(), comparison_l_max);

    if (comparison_file_prefix != "") {
      recorder->append_mode_data(
          "/" + tag::name() + "_scri_difference", time,
          compute_mode_difference_at_scri<tag>(time, comparison_file_prefix,
                                               scri_goldberg_modes.data(),
                                               comparison_l_max),
          comparison_l_max);
    }
  });
}

template <typename Tag>
void compare_and_record_scri_values(
    const gsl::not_null<ModeRecorder*> recorder,
    const ComplexDataVector& slice_data,
    const std::string comparison_file_prefix, const double time,
    const size_t l_max, const size_t comparison_l_max,
    const size_t number_of_radial_points) noexcept {
  typename db::item_type<Tag>::type scri_slice_buffer;
  scri_slice_buffer.data() = slice_data;
  auto scri_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
      Spectral::Swsh::swsh_transform(make_not_null(&scri_slice_buffer), l_max),
      l_max);
  recorder->append_mode_data("/" + Tag::name() + "_scri", time,
                             scri_goldberg_modes.data(), comparison_l_max);

  if (comparison_file_prefix != "") {
    recorder->append_mode_data(
        "/" + Tag::name() + "_scri_difference", time,
        compute_mode_difference_at_scri<Tag>(time, comparison_file_prefix,
                                             scri_goldberg_modes.data(),
                                             comparison_l_max),
        comparison_l_max);
  }
}

template <int spin>
void record_scri_output(
    const gsl::not_null<ModeRecorder*> recorder,
    const std::pair<ComplexDataVector, double>& interpolation_data,
    const std::string& record_tag, const size_t l_max,
    const size_t comparison_l_max) noexcept {
  SpinWeighted<ComplexDataVector, spin> scri_slice;
  scri_slice.data() = interpolation_data.first;

  auto scri_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
      Spectral::Swsh::swsh_transform(make_not_null(&scri_slice), l_max), l_max);
  recorder->append_mode_data("/" + record_tag + "_scri",
                             interpolation_data.second,
                             scri_goldberg_modes.data(), comparison_l_max);
}

template <typename TagList, typename DataBoxType>
void compare_and_record_r200_values(
    const gsl::not_null<DataBoxType*> box,
    const gsl::not_null<ModeRecorder*> recorder,
    const std::string comparison_file_prefix, const double time,
    const size_t l_max, const size_t comparison_l_max,
    const size_t /*number_of_radial_points*/) noexcept {
  tmpl::for_each<TagList>([&comparison_file_prefix, &box, &l_max, &recorder,
                           &time, &comparison_l_max](auto x) {
    using tag = typename decltype(x)::type;
    typename db::item_type<tag>::type r200_slice;
    ComplexDataVector r200_slice_buffer = get(db::get<tag>(*box)).data();
    r200_slice.data() = interpolate_to_bondi_r(
        get(db::get<tag>(*box)).data(), get(db::get<Tags::BondiR>(*box)).data(),
        200.0, l_max);

    auto r200_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
        Spectral::Swsh::swsh_transform(make_not_null(&r200_slice), l_max),
        l_max);
    recorder->append_mode_data("/" + tag::name() + "_r200", time,
                               r200_goldberg_modes.data(), comparison_l_max);

    if (comparison_file_prefix != "") {
      recorder->append_mode_data(
          "/" + tag::name() + "_r200_difference", time,
          compute_mode_difference_at_bondi_r_200<tag>(
              time, comparison_file_prefix, r200_goldberg_modes.data(),
              comparison_l_max),
          comparison_l_max);
    }
  });
}

template <typename TagList, typename DataBoxType>
void compare_and_record_r200_values_from_rp(
    const gsl::not_null<DataBoxType*> box,
    const gsl::not_null<ModeRecorder*> recorder,
    const std::string comparison_file_prefix, const double time,
    const size_t l_max, const size_t comparison_l_max,
    const size_t /*number_of_radial_points*/) noexcept {
  tmpl::for_each<TagList>([&comparison_file_prefix, &box, &l_max, &recorder,
                           &time, &comparison_l_max](auto x) {
    using tag = typename decltype(x)::type;
    typename db::item_type<tag>::type r200_slice;
    ComplexDataVector r200_slice_buffer = get(db::get<tag>(*box)).data();

    r200_slice.data() = interpolate_to_bondi_r(
        get(db::get<tag>(*box)).data(),
        get(db::get<Tags::BoundaryValue<Tags::BondiR>>(*box)).data(), 200.0,
        l_max);

    typename db::item_type<tag>::type r200_cauchy_gauge{r200_slice.size()};
    Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&r200_slice),
                                                  l_max, l_max - 4);

    auto r200_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
        Spectral::Swsh::swsh_transform(make_not_null(&r200_slice), l_max),
        l_max);
    recorder->append_mode_data("/" + tag::name() + "_r200", time,
                               r200_goldberg_modes.data(), comparison_l_max);

    if (comparison_file_prefix != "") {
      recorder->append_mode_data(
          "/" + tag::name() + "_r200_difference", time,
          compute_mode_difference_at_bondi_r_200<tag>(
              time, comparison_file_prefix, r200_goldberg_modes.data(),
              comparison_l_max),
          comparison_l_max);
    }
  });
}

template <typename Tag, typename DataBoxType>
void compare_and_record_r200_value_inertial(
    const gsl::not_null<DataBoxType*> box,
    const gsl::not_null<ModeRecorder*> recorder,
    const std::string comparison_file_prefix, std::string file_name,
    const double time, const size_t l_max, const size_t comparison_l_max,
    const size_t /*number_of_radial_points*/) noexcept {
  typename db::item_type<Tag>::type r200_slice;
  ComplexDataVector r200_slice_buffer = get(db::get<Tag>(*box)).data();

  r200_slice.data() = interpolate_to_bondi_r(
      get(db::get<Tag>(*box)).data(),
      get(db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>>(*box))
              .data() /
          get(db::get<Tags::GaugeOmegaCD>(*box)).data(),
      200.0, l_max);

  auto r200_goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
      Spectral::Swsh::swsh_transform(make_not_null(&r200_slice), l_max), l_max);
  recorder->append_mode_data("/" + Tag::name() + "_r200", time,
                             r200_goldberg_modes.data(), comparison_l_max);

  if (comparison_file_prefix != "") {
    ModeComparisonManager mode_compare(comparison_file_prefix + file_name,
                                       l_max);
    auto inertial_modes_from_file = CalculateInertialModes<Tag>::compute(
        box, mode_compare.get_comparison_modes(time),
        2.0 * get(db::get<Tags::BondiR>(*box)).data() / 200.0);
    recorder->append_mode_data(
        "/" + Tag::name() + "_r200_difference", time,
        r200_goldberg_modes.data() - inertial_modes_from_file,
        comparison_l_max);
  }
}

void run_trial_cce(std::string input_filename,
                   std::string comparison_file_prefix, size_t simulation_l_max,
                   size_t comparison_l_max, size_t number_of_radial_points,
                   std::string output_file_suffix,
                   size_t rational_timestep_numerator,
                   size_t rational_timestep_denominator,
                   bool /*calculate_psi4_diagnostic*/, size_t l_filter_start,
                   double start_time, double end_time) noexcept {
  TimeSteppers::RungeKutta3 stepper{};
  TimeSteppers::History<ComplexDataVector, ComplexDataVector> history{};

  // TODO: upgrade interpolator choice
  auto data_manager = CceH5BoundaryDataManager<CubicInterpolator>{
      input_filename + ".h5", simulation_l_max, 100};

  // this is where we will add another for the gauge adjustment
  using boundary_variables_tag = ::Tags::Variables<all_boundary_tags>;
  using integration_independent_variables_tag =
      ::Tags::Variables<pre_computation_tags>;
  using pre_swsh_derivatives_variables_tag = ::Tags::Variables<
      tmpl::append<all_pre_swsh_derivative_tags, tmpl::list<Tags::SpecH>>>;
  using transform_buffer_variables_tag =
      ::Tags::Variables<all_transform_buffer_tags>;
  using swsh_derivatives_variables_tag =
      ::Tags::Variables<all_swsh_derivative_tags>;
  using temporary_variables_tag =
      ::Tags::Variables<all_temporary_equation_tags>;
  using integrand_variables_tag = ::Tags::Variables<all_integrand_tags>;

  size_t l_max = data_manager.get_l_max();
  size_t boundary_size =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  size_t volume_size = boundary_size * number_of_radial_points;
  size_t transform_buffer_size =
      2 * number_of_radial_points *
      Spectral::Swsh::number_of_swsh_coefficients(l_max);

  auto box = db::create<db::AddSimpleTags<
      Tags::LMax, boundary_variables_tag, integration_independent_variables_tag,
      pre_swsh_derivatives_variables_tag, transform_buffer_variables_tag,
      swsh_derivatives_variables_tag, temporary_variables_tag,
      integrand_variables_tag>>(
      l_max, db::item_type<boundary_variables_tag>{boundary_size},
      db::item_type<integration_independent_variables_tag>{volume_size},
      db::item_type<pre_swsh_derivatives_variables_tag>{volume_size},
      db::item_type<transform_buffer_variables_tag>{transform_buffer_size},
      db::item_type<swsh_derivatives_variables_tag>{volume_size},
      db::item_type<temporary_variables_tag>{volume_size},
      db::item_type<integrand_variables_tag>{volume_size});

  ModeRecorder recorder{input_filename + output_file_suffix + ".h5",
                        comparison_l_max, simulation_l_max};

  // TODO This is a bit inelegant, used entirely for comparing directly to SpEC
  // data
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
  // end time of -1.0 is used to indicate that CCE should just run through all
  // available data
  if (end_time == -1.0) {
    end_time =
        data_manager
            .get_time_buffer()[data_manager.get_time_buffer().size() - 1];
  }
  // in order to use the time architecture, we need to make a slab
  Slab only_slab{start_time, end_time};
  TimeDelta time_step{
      only_slab,
      Time::rational_t{static_cast<int>(rational_timestep_numerator),
                       static_cast<int>(end_time - start_time) *
                           static_cast<int>(rational_timestep_denominator)}};
  TimeId time{true, 0, Time{only_slab, Time::rational_t{0, 1}}};
  bool data_still_available = false;
  db::mutate<boundary_variables_tag>(
      make_not_null(&box),
      [&start_time, &data_manager, &data_still_available](
          const gsl::not_null<db::item_type<boundary_variables_tag>*>
              boundary_variables) {
        data_still_available = data_manager.populate_hypersurface_boundary_data(
            boundary_variables, start_time);
      });
  db::mutate_apply<InitializeJ<Tags::BoundaryValue>>(make_not_null(&box));

  size_t step_counter = 0;

  // main loop
  while (data_still_available and time.time().value() < end_time) {
    step_counter++;

    mutate_all_precompute_cce_dependencies<Tags::BoundaryValue>(
        make_not_null(&box));

    // Fix boundary condition for H to be like that of SpEC (there's a
    // coordinates on boundary vs coordinates in bulk problem so the definitions
    // do not quite align).
    // TODO: investigate if this is still necessary. If it is, there is still a
    // problem.
    db::mutate_apply<ComputePreSwshDerivatives<Tags::Dy<Tags::BondiJ>>>(
        make_not_null(&box));
    db::mutate<Tags::BoundaryValue<Tags::BondiH>,
               Tags::BoundaryValue<Tags::DuRDividedByR>,
               Tags::Dy<Tags::BondiJ>>(
        make_not_null(&box),
        [&l_max](
            const gsl::not_null<
                db::item_type<Tags::BoundaryValue<Tags::BondiH>>*>
                boundary_h,
            const gsl::not_null<
                db::item_type<Tags::BoundaryValue<Tags::DuRDividedByR>>*>
                du_r_divided_by_r,
            const gsl::not_null<db::item_type<Tags::Dy<Tags::BondiJ>>*> dy_j,
            const db::item_type<Tags::BoundaryValue<Tags::SpecH>>& spec_h) {
          ComplexDataVector boundary_dy_j{
              get(*dy_j).data().data(),
              Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
          ComplexDataVector boundary_du_r_divided_by_r{
              get(*du_r_divided_by_r).data().data(),
              Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
          get(*boundary_h).data() =
              get(spec_h).data() +
              2.0 * boundary_du_r_divided_by_r * boundary_dy_j;
        },
        db::get<Tags::BoundaryValue<Tags::SpecH>>(box));

    tmpl::for_each<tmpl::list<Tags::BondiBeta, Tags::BondiQ, Tags::BondiU,
                              Tags::BondiW, Tags::BondiH>>([&box, &l_max,
                                                            &l_filter_start](
                                                               auto x) {
      using bondi_tag = typename decltype(x)::type;
      perform_hypersurface_computation<
          bondi_tag, swsh_derivatives_variables_tag,
          transform_buffer_variables_tag, pre_swsh_derivatives_variables_tag>(
          make_not_null(&box));

      db::mutate<bondi_tag>(
          make_not_null(&box),
          [&l_max, &l_filter_start](
              const gsl::not_null<db::item_type<bondi_tag>*> bondi_quantity) {
            Spectral::Swsh::filter_swsh_volume_quantity(
                make_not_null(&get(*bondi_quantity)), l_max, l_filter_start,
                108.0, 8);
          });
    });

    ComplexDataVector du_j = get(db::get<Tags::BondiH>(box)).data();
    history.insert(time.time(), get(db::get<Tags::BondiJ>(box)).data(),
                   std::move(du_j));

    db::mutate<Tags::SpecH>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<Tags::SpecH>*> spec_h,
           const db::item_type<Tags::BondiH>& h,
           const db::item_type<Tags::DuRDividedByR>& du_r_divided_by_r,
           const db::item_type<Tags::OneMinusY>& one_minus_y,
           const db::item_type<Tags::Dy<Tags::BondiJ>>& dy_j) {
          get(*spec_h) =
              get(h) - get(du_r_divided_by_r) * get(one_minus_y) * get(dy_j);
        },
        db::get<Tags::BondiH>(box), db::get<Tags::DuRDividedByR>(box),
        db::get<Tags::OneMinusY>(box), db::get<Tags::Dy<Tags::BondiJ>>(box));
    if (time.substep() == 0) {
      // perform a comparison of boundary values and scri+ values on each new
      // time advancement

      // only dump the comparison data if the current timestep is close to one
      // of the comparison points, otherwise dump every fourth timestep
      if (std::find_if(times.begin(), times.end(),
                       [&time](auto x) {
                         return abs(x - time.step_time().value()) < 1.0e-8;
                       }) != times.end() or
          (comparison_file_prefix == "" and step_counter % 12 == 0)) {
        record_boundary_values<
            Tags::BoundaryValue,
            tmpl::list<Tags::BondiBeta, Tags::BondiQ, Tags::BondiU,
                       Tags::BondiW, Tags::SpecH, Tags::BondiJ, Tags::BondiR>>(
            make_not_null(&box), make_not_null(&recorder),
            time.step_time().value(), l_max, comparison_l_max);

        compare_and_record_scri_values<
            tmpl::list<Tags::BondiJ, Tags::BondiBeta, Tags::BondiQ,
                       Tags::BondiU, Tags::BondiW, Tags::BondiH>>(
            make_not_null(&box), make_not_null(&recorder),
            comparison_file_prefix, time.step_time().value(), l_max,
            comparison_l_max, number_of_radial_points);

        // note: SpecH is available, but needs to be computed on its own for the
        // comparison to be useful.
        compare_and_record_r200_values<
            tmpl::list<Tags::BondiJ, Tags::BondiBeta, Tags::BondiQ,
                       Tags::BondiU, Tags::BondiW, Tags::SpecH>>(
            make_not_null(&box), make_not_null(&recorder),
            comparison_file_prefix, time.step_time().value(), l_max,
            comparison_l_max, number_of_radial_points);
      }
    }
    db::mutate<Tags::BondiJ>(
        make_not_null(&box),
        [&stepper, &time_step,
         &history](const gsl::not_null<db::item_type<Tags::BondiJ>*> j) {
          stepper.update_u(make_not_null(&get(*j).data()),
                           make_not_null(&history), time_step);
        });
    time = stepper.next_time_id(time, time_step);
    // printf("next time: %f\n", time.time().value());
    // get the worldtube data for the next time step.
    db::mutate<boundary_variables_tag>(
        make_not_null(&box),
        [&time, &data_manager, &data_still_available](
            const gsl::not_null<db::item_type<boundary_variables_tag>*>
                boundary_variables) {
          data_still_available =
              data_manager.populate_hypersurface_boundary_data(
                  boundary_variables, time.time().value());
        });
  }
}

// TODO figure out how to merge common functionality between these two functions
void run_trial_regularity_preserving_cce(
    std::string input_filename, std::string comparison_file_prefix,
    size_t simulation_l_max, size_t comparison_l_max,
    size_t number_of_radial_points, std::string output_file_suffix,
    size_t rational_timestep_numerator, size_t rational_timestep_denominator,
    bool /*calculate_psi4_diagnostic*/, size_t l_filter_start,
    double start_time, double end_time, bool regularity_preserving) noexcept {
  TimeSteppers::RungeKutta3 stepper{};
  TimeSteppers::History<ComplexDataVector, ComplexDataVector> j_history{};
  TimeSteppers::History<ComplexDataVector, ComplexDataVector> u_bondi_history{};
  TimeSteppers::History<DataVector, DataVector> x_history{};
  TimeSteppers::History<DataVector, DataVector> y_history{};
  TimeSteppers::History<DataVector, DataVector> z_history{};
  TimeSteppers::History<DataVector, DataVector> x_tilde_history{};
  TimeSteppers::History<DataVector, DataVector> y_tilde_history{};
  TimeSteppers::History<DataVector, DataVector> z_tilde_history{};
  TimeSteppers::History<ComplexDataVector, ComplexDataVector> gauge_c_history{};
  TimeSteppers::History<ComplexDataVector, ComplexDataVector> gauge_d_history{};

  // NOTE: this interoplator needs to be Cubic whenever comparisons with SpEC
  // are desired with tight agreement. Otherwise, the interpolation error will
  // dominate the difference.
  auto data_manager = CceH5BoundaryDataManager<
      CubicInterpolator /*BarycentricInterpolator<10>*/>{input_filename + ".h5",
                                                         simulation_l_max, 100};

  using boundary_variables_tag = ::Tags::Variables<all_boundary_tags>;
  using gauge_transform_boundary_variables_tag =
      ::Tags::Variables<gauge_transform_boundary_tags>;
  using angular_coordinate_variables_tag =
      ::Tags::Variables<angular_coordinate_tags>;
  using scri_variables_tag = ::Tags::Variables<scri_tags>;
  using gauge_confirmation_scri_variables_tag =
      ::Tags::Variables<gauge_confirmation_scri_tags>;
  using volume_gauge_confirmation_variables_tag =
      ::Tags::Variables<gauge_confirmation_volume_tags>;
  using integration_independent_variables_tag =
      ::Tags::Variables<pre_computation_tags>;
  using pre_swsh_derivatives_variables_tag = ::Tags::Variables<
      tmpl::append<all_pre_swsh_derivative_tags>>;
  using transform_buffer_variables_tag =
      ::Tags::Variables<all_transform_buffer_tags>;
  using swsh_derivatives_variables_tag =
      ::Tags::Variables<all_swsh_derivative_tags>;
  using temporary_variables_tag =
      ::Tags::Variables<all_temporary_equation_tags>;
  using integrand_variables_tag = ::Tags::Variables<all_integrand_tags>;

  size_t l_max = data_manager.get_l_max();
  size_t boundary_size =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  size_t volume_size = boundary_size * number_of_radial_points;
  size_t transform_buffer_size =
      2 * number_of_radial_points *
      Spectral::Swsh::number_of_swsh_coefficients(l_max);

  auto box = db::create<db::AddSimpleTags<
      Tags::LMax, boundary_variables_tag,
      gauge_transform_boundary_variables_tag, angular_coordinate_variables_tag,
      scri_variables_tag, volume_gauge_confirmation_variables_tag,
      gauge_confirmation_scri_variables_tag,
      integration_independent_variables_tag, pre_swsh_derivatives_variables_tag,
      transform_buffer_variables_tag, swsh_derivatives_variables_tag,
      temporary_variables_tag, integrand_variables_tag>>(
      l_max, db::item_type<boundary_variables_tag>{boundary_size},
      db::item_type<gauge_transform_boundary_variables_tag>{boundary_size},
      db::item_type<angular_coordinate_variables_tag>{boundary_size},
      db::item_type<scri_variables_tag>{boundary_size},
      db::item_type<volume_gauge_confirmation_variables_tag>{volume_size},
      db::item_type<gauge_confirmation_scri_variables_tag>{boundary_size},
      db::item_type<integration_independent_variables_tag>{volume_size},
      db::item_type<pre_swsh_derivatives_variables_tag>{volume_size},
      db::item_type<transform_buffer_variables_tag>{transform_buffer_size},
      db::item_type<swsh_derivatives_variables_tag>{volume_size},
      db::item_type<temporary_variables_tag>{volume_size},
      db::item_type<integrand_variables_tag>{volume_size});

  ModeRecorder recorder{input_filename + output_file_suffix + ".h5",
                        comparison_l_max, simulation_l_max};

  // TODO This is a bit inelegant, used entirely for comparing directly to SpEC
  // data
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
  // end time of -1.0 is used to indicate that CCE should just run through all
  // available data
  if (end_time == -1.0) {
    end_time =
        data_manager
            .get_time_buffer()[data_manager.get_time_buffer().size() - 1];
  }
  printf("time span: %f, %f\n", start_time, end_time);
  // in order to use the time architecture, we need to make a slab
  Slab only_slab{start_time, end_time};
  TimeDelta time_step{
      only_slab,
      Time::rational_t{static_cast<int>(rational_timestep_numerator),
                       static_cast<int>(end_time - start_time) *
                           static_cast<int>(rational_timestep_denominator)}};
  TimeId time{true, 0, Time{only_slab, Time::rational_t{0, 1}}};
  bool data_still_available = false;
  db::mutate<boundary_variables_tag>(
      make_not_null(&box),
      [&start_time, &data_manager, &data_still_available](
          const gsl::not_null<db::item_type<boundary_variables_tag>*>
              boundary_variables) {
        data_still_available = data_manager.populate_hypersurface_boundary_data(
            boundary_variables, start_time);
      });

  ScriPlusInterpolationManager<FlexibleBarycentricInterpolator,
                               ComplexDataVector>
      interpolation_manager{5, boundary_size};

  db::mutate_apply<InitializeJ<Tags::BoundaryValue>>(make_not_null(&box));
  db::mutate_apply<InitializeGauge>(make_not_null(&box));
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::InertialAngularCoords, Tags::InertialCartesianCoords>>(
      make_not_null(&box));
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      make_not_null(&box));

  db::mutate_apply<InitializeXtildeOfX>(make_not_null(&box));
  db::mutate_apply<GaugeUpdateJacobianFromCoords<Tags::GaugeA, Tags::GaugeB,
                                                 Tags::InertialCartesianCoords,
                                                 Tags::InertialAngularCoords>>(
      make_not_null(&box));
  db::mutate_apply<GaugeUpdateJacobianFromCoords<Tags::GaugeC, Tags::GaugeD,
                                                 Tags::CauchyCartesianCoords,
                                                 Tags::CauchyAngularCoords>>(
      make_not_null(&box));
  db::mutate_apply<GaugeUpdateOmega>(make_not_null(&box));
  db::mutate_apply<GaugeUpdateOmegaCD>(make_not_null(&box));
  db::mutate_apply<GaugeAdjustInitialJ>(make_not_null(&box));

  // TESTING
  // db::mutate_apply<ComputeGaugeAdjustedBoundaryValue<Tags::BondiBeta>>
  // (make_not_null(&box));
  // db::mutate_apply<ComputeGaugeAdjustedBoundaryValue<Tags::BondiJ>>
  // (make_not_null(&box));
  // db::mutate_apply<ComputeGaugeAdjustedBoundaryValue<
  // Tags::Dr<Tags::BondiJ>>>(
  // make_not_null(&box));
  // TESTING
  db::mutate_apply<InitializeScriPlusValue<Tags::InertialRetardedTime>>(
      make_not_null(&box), start_time);

  size_t step_counter = 0;
  printf("beginning loop\n");
  // main loop
  while (data_still_available and time.time().value() < end_time) {
    printf("time : %f\n", time.time().value());
    tmpl::for_each<compute_gauge_adjustments_setup_tags>([&box](auto x) {
      using tag = typename decltype(x)::type;
      db::mutate_apply<ComputeGaugeAdjustedBoundaryValue<tag>>(
          make_not_null(&box));
    });
    if (regularity_preserving) {
      mutate_all_precompute_cce_dependencies<Tags::EvolutionGaugeBoundaryValue>(
          make_not_null(&box));
    } else {
      mutate_all_precompute_cce_dependencies<Tags::BoundaryValue>(
          make_not_null(&box));

      db::mutate_apply<ComputePreSwshDerivatives<Tags::Dy<Tags::BondiJ>>>(
          make_not_null(&box));
      db::mutate<Tags::BoundaryValue<Tags::BondiH>,
                 Tags::BoundaryValue<Tags::DuRDividedByR>,
                 Tags::Dy<Tags::BondiJ>>(
          make_not_null(&box),
          [&l_max](
              const gsl::not_null<
                  db::item_type<Tags::BoundaryValue<Tags::BondiH>>*>
                  boundary_h,
              const gsl::not_null<
                  db::item_type<Tags::BoundaryValue<Tags::DuRDividedByR>>*>
                  du_r_divided_by_r,
              const gsl::not_null<db::item_type<Tags::Dy<Tags::BondiJ>>*> dy_j,
              const db::item_type<Tags::BoundaryValue<Tags::SpecH>>& spec_h) {
            ComplexDataVector boundary_dy_j{
                get(*dy_j).data().data(),
                Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
            ComplexDataVector boundary_du_r_divided_by_r{
                get(*du_r_divided_by_r).data().data(),
                Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
            get(*boundary_h).data() =
                get(spec_h).data() +
                2.0 * boundary_du_r_divided_by_r * boundary_dy_j;
          },
          db::get<Tags::BoundaryValue<Tags::SpecH>>(box));
    }

    tmpl::for_each<tmpl::list<Tags::BondiBeta, Tags::BondiQ, Tags::BondiU,
                              Tags::BondiW, Tags::BondiH>>(
        [&box, &l_max, &l_filter_start, &regularity_preserving](auto x) {
          using bondi_tag = typename decltype(x)::type;
          regularity_preserving_hypersurface_computation<
              bondi_tag, swsh_derivatives_variables_tag,
              transform_buffer_variables_tag,
              pre_swsh_derivatives_variables_tag>(make_not_null(&box),
                                                  regularity_preserving);

          // this isn't super elegant. consider alternatives
          // if (cpp17::is_same_v<bondi_tag, Tags::BondiU>) {
          // db::mutate<Tags::Du<Tags::GaugeA>, Tags::Du<Tags::GaugeB>,
          // Tags::Du<Tags::GaugeOmega>>(
          // make_not_null(&box),
          // [&l_max, &l_filter_start](
          // const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          // du_gauge_a,
          // const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          // du_gauge_b,
          // const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          // du_omega) {
          // Spectral::Swsh::filter_swsh_boundary_quantity(
          // make_not_null(&get(*du_gauge_a)), l_max, l_filter_start);
          // Spectral::Swsh::filter_swsh_boundary_quantity(
          // make_not_null(&get(*du_gauge_b)), l_max, l_filter_start);
          // Spectral::Swsh::filter_swsh_boundary_quantity(
          // make_not_null(&get(*du_omega)), l_max, l_filter_start);
          // });
          // }

          // db::mutate<bondi_tag>(
          //     make_not_null(&box),
          //     [&l_max,
          //      &l_filter_start](const
          //      gsl::not_null<db::item_type<bondi_tag>*>
          //                           bondi_quantity) {
          //       Spectral::Swsh::filter_swsh_volume_quantity(
          //           make_not_null(&get(*bondi_quantity)), l_max,
          //           l_filter_start, 108.0, 8);
          //     });
        });
    db::mutate_apply<
        CalculateScriPlusValue<Tags::Du<Tags::InertialRetardedTime>>>(
        make_not_null(&box));

    ComplexDataVector du_j = get(db::get<Tags::BondiH>(box)).data();
    j_history.insert(time.time(), get(db::get<Tags::BondiJ>(box)).data(),
                     std::move(du_j));

    DataVector du_x = get<0>(db::get<Tags::DuCauchyCartesianCoords>(box));
    x_history.insert(time.time(),
                     get<0>(db::get<Tags::CauchyCartesianCoords>(box)),
                     std::move(du_x));
    DataVector du_y = get<1>(db::get<Tags::DuCauchyCartesianCoords>(box));
    y_history.insert(time.time(),
                     get<1>(db::get<Tags::CauchyCartesianCoords>(box)),
                     std::move(du_y));
    DataVector du_z = get<2>(db::get<Tags::DuCauchyCartesianCoords>(box));
    z_history.insert(time.time(),
                     get<2>(db::get<Tags::CauchyCartesianCoords>(box)),
                     std::move(du_z));

    ComplexDataVector du_u_bondi =
        get(db::get<Tags::Du<Tags::InertialRetardedTime>>(box)).data();
    u_bondi_history.insert(time.time(),
                           get(db::get<Tags::InertialRetardedTime>(box)).data(),
                           std::move(du_u_bondi));

    ComplexDataVector du_gauge_c =
        get(db::get<Tags::Du<Tags::GaugeC>>(box)).data();
    gauge_c_history.insert(time.time(), get(db::get<Tags::GaugeC>(box)).data(),
                           std::move(du_gauge_c));
    ComplexDataVector du_gauge_d =
        get(db::get<Tags::Du<Tags::GaugeD>>(box)).data();
    gauge_d_history.insert(time.time(), get(db::get<Tags::GaugeD>(box)).data(),
                           std::move(du_gauge_d));

    DataVector du_x_tilde =
        get<0>(db::get<Tags::DuInertialCartesianCoords>(box));
    x_tilde_history.insert(time.time(),
                           get<0>(db::get<Tags::InertialCartesianCoords>(box)),
                           std::move(du_x_tilde));
    DataVector du_y_tilde =
        get<1>(db::get<Tags::DuInertialCartesianCoords>(box));
    y_tilde_history.insert(time.time(),
                           get<1>(db::get<Tags::InertialCartesianCoords>(box)),
                           std::move(du_y_tilde));
    DataVector du_z_tilde =
        get<2>(db::get<Tags::DuInertialCartesianCoords>(box));
    z_tilde_history.insert(time.time(),
                           get<2>(db::get<Tags::InertialCartesianCoords>(box)),
                           std::move(du_z_tilde));

    // compute the scri+ values in the Cauchy gauge to verify against the SpEC
    // computation that we're actually performing a reliable coordinate
    // transformation as we hope.

    tmpl::for_each<tmpl::list<Tags::BondiBeta, Tags::U0>>([&box](auto x) {
      using tag = typename decltype(x)::type;
      db::mutate_apply<CalculateScriPlusValue<Tags::CauchyGaugeScriPlus<tag>>>(
          make_not_null(&box));
    });

    // DEBUG output
    record_scri_output<0>(
        make_not_null(&recorder),
        std::make_pair(get(db::get<Tags::GaugeOmega>(box)).data(),
                       time.time().value()),
        "omega", l_max, comparison_l_max);

    db::mutate<Tags::SpecH>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<Tags::SpecH>*> spec_h,
           const db::item_type<Tags::BondiH>& h,
           const db::item_type<Tags::DuRDividedByR>& du_r_divided_by_r,
           const db::item_type<Tags::OneMinusY>& one_minus_y,
           const db::item_type<Tags::Dy<Tags::BondiJ>>& dy_j) {
          get(*spec_h) =
              get(h) - get(du_r_divided_by_r) * get(one_minus_y) * get(dy_j);
        },
        db::get<Tags::BondiH>(box), db::get<Tags::DuRDividedByR>(box),
        db::get<Tags::OneMinusY>(box), db::get<Tags::Dy<Tags::BondiJ>>(box));

    tmpl::for_each<tmpl::list<Tags::BondiBeta, Tags::BondiJ, Tags::BondiW,
                              Tags::BondiU, Tags::SpecH>>([&box](auto x) {
      using tag = typename decltype(x)::type;
      db::mutate_apply<CalculateCauchyGauge<Tags::CauchyGauge<tag>>>(
          make_not_null(&box));
    });

    // DEBUG output

    db::mutate_apply<CalculateScriPlusValue<Tags::News>>(make_not_null(&box));
    printf("computing news\n");
    if (regularity_preserving) {
      tnsr::i<DataVector, 2> x_of_x_tilde_identity{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
      const auto& collocation = Spectral::Swsh::precomputed_collocation<
          Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
      for (const auto& collocation_point : collocation) {
        get<0>(x_of_x_tilde_identity)[collocation_point.offset] =
            collocation_point.theta;
        get<1>(x_of_x_tilde_identity)[collocation_point.offset] =
            collocation_point.phi;
      }
      Scalar<SpinWeighted<ComplexDataVector, 2>> gauge_c_identity;
      get(gauge_c_identity).data() = ComplexDataVector{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max), 0.0};
      Scalar<SpinWeighted<ComplexDataVector, 0>> gauge_d_identity;
      get(gauge_d_identity).data() = ComplexDataVector{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max), 2.0};
      Scalar<SpinWeighted<ComplexDataVector, 0>> gauge_omega_identity;
      get(gauge_omega_identity).data() = ComplexDataVector{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max), 1.0};
      Scalar<SpinWeighted<ComplexDataVector, 1>> eth_gauge_omega_identity;
      get(eth_gauge_omega_identity).data() = ComplexDataVector{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max), 0.0};
      Scalar<SpinWeighted<ComplexDataVector, 0>> du_gauge_omega_identity;
      get(du_gauge_omega_identity).data() = ComplexDataVector{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max), 0.0};

      db::mutate<Tags::NonInertialNews, Tags::BondiJ, Tags::BondiBeta,
                 Tags::SpecH, Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
                 Tags::BondiU>(make_not_null(&box), calculate_non_inertial_news,
                               gauge_c_identity, gauge_d_identity,
                               gauge_omega_identity, du_gauge_omega_identity,
                               eth_gauge_omega_identity, x_of_x_tilde_identity,
                               x_of_x_tilde_identity, l_max, false);
    } else {
      db::mutate<Tags::NonInertialNews, Tags::BondiJ, Tags::BondiBeta,
                 Tags::SpecH, Tags::BoundaryValue<Tags::BondiR>, Tags::BondiU>(
          make_not_null(&box), calculate_non_inertial_news,
          db::get<Tags::GaugeC>(box), db::get<Tags::GaugeD>(box),
          db::get<Tags::GaugeOmegaCD>(box),
          db::get<Tags::Du<Tags::GaugeOmegaCD>>(box),
          db::get<Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                   Spectral::Swsh::Tags::Eth>>(
              box),
          db::get<Tags::CauchyAngularCoords>(box),
          db::get<Tags::InertialAngularCoords>(box), l_max, false);
    }

    if (time.substep() == 0) {
      auto news = db::get<Tags::News>(box);
      interpolation_manager.insert_data(
          real(get(db::get<Tags::InertialRetardedTime>(box)).data()),
          get(db::get<Tags::News>(box)).data());
      double mean_time = 0.0;
      for (const auto& val :
           real(get(db::get<Tags::InertialRetardedTime>(box)).data())) {
        mean_time += val;
      }
      mean_time /= boundary_size;
      // printf("mean time : %f, step counter : %zu\n", mean_time,
      // step_counter); if (step_counter > 2) {
      interpolation_manager.insert_target_time(time.time().value());
      // }
      while (interpolation_manager.first_time_is_ready_to_interpolate()) {
        auto interpolation =
            interpolation_manager.interpolate_and_pop_first_time();
        // DEBUG output
        // record_scri_output<2>(make_not_null(&recorder), interpolation,
        // "News", l_max, comparison_l_max);
        if (std::find_if(times.begin(), times.end(),
                         [&interpolation](auto x) {
                           return abs(x - interpolation.second) < 1.0e-8;
                         }) != times.end() or
            (comparison_file_prefix == "" and step_counter % 12 == 0)) {
          compare_and_record_scri_values<Tags::News>(
              make_not_null(&recorder), interpolation.first,
              comparison_file_prefix, interpolation.second, l_max,
              comparison_l_max, 1);
        }
      }
    }

    if (time.substep() == 0) {
      // perform a comparison of boundary values and scri+ values on each new
      // time advancement

      // only dump the comparison data if the current timestep is close to one
      // of the comparison points, otherwise dump every fourth timestep
      if (std::find_if(times.begin(), times.end(),
                       [&time](auto x) {
                         return abs(x - time.step_time().value()) < 1.0e-8;
                       }) != times.end() or
          (comparison_file_prefix == "" and step_counter % 12 == 0)) {
        record_boundary_values<
            Tags::BoundaryValue,
            tmpl::list<Tags::BondiBeta, Tags::BondiQ, Tags::BondiU,
                       Tags::BondiW, Tags::SpecH, Tags::BondiJ, Tags::BondiR>>(
            make_not_null(&box), make_not_null(&recorder),
            time.step_time().value(), l_max, comparison_l_max);

        compare_and_record_scri_values<
            tmpl::list<Tags::CauchyGaugeScriPlus<Tags::BondiBeta>,
                       Tags::CauchyGaugeScriPlus<Tags::U0>>>(
            make_not_null(&box), make_not_null(&recorder),
            comparison_file_prefix, time.step_time().value(), l_max,
            comparison_l_max, 1);

        compare_and_record_scri_values<tmpl::list<Tags::NonInertialNews>>(
            make_not_null(&box), make_not_null(&recorder), "",
            time.step_time().value(), l_max, comparison_l_max, 1);

        if (regularity_preserving) {
          SpinWeighted<ComplexDataVector, 2> news_difference =
              get(db::get<Tags::NonInertialNews>(box)) -
              get(db::get<Tags::News>(box));
          recorder.append_mode_data(
              "/News_vs_NonInertial", time.step_time().value(),
              Spectral::Swsh::libsharp_to_goldberg_modes(
                  Spectral::Swsh::swsh_transform(
                      make_not_null(&news_difference), l_max),
                  l_max)
                  .data(),
              comparison_l_max);
        }

        compare_and_record_scri_values<
            tmpl::list<Tags::BondiJ, Tags::BondiBeta, Tags::BondiQ,
                       Tags::BondiU, Tags::BondiW, Tags::BondiH>>(
            make_not_null(&box), make_not_null(&recorder),
            comparison_file_prefix, time.step_time().value(), l_max,
            comparison_l_max, number_of_radial_points);

        // note: SpecH is available, but needs to be computed on its own for
        // the comparison to be useful.

        compare_and_record_r200_value_inertial<Tags::BondiJ>(
            make_not_null(&box), make_not_null(&recorder),
            comparison_file_prefix, "JR200modes.h5", time.step_time().value(),
            l_max, comparison_l_max, number_of_radial_points);

        compare_and_record_r200_values_from_rp<tmpl::list<
            Tags::CauchyGauge<Tags::BondiJ>, Tags::CauchyGauge<Tags::BondiBeta>,
            Tags::CauchyGauge<Tags::BondiW>, Tags::CauchyGauge<Tags::BondiU>,
            Tags::CauchyGauge<Tags::SpecH>
            /*,Tags::CauchyGauge<Tags::BondiQ>,
              Tags::CauchyGauge<Tags::BondiU>*/>>(
            make_not_null(&box), make_not_null(&recorder),
            comparison_file_prefix, time.step_time().value(), l_max,
            comparison_l_max, number_of_radial_points);

        // compare_and_record_r200_values<tmpl::list<Tags::BondiJ,
        // Tags::BondiBeta>>( make_not_null(&box), make_not_null(&recorder),
        // comparison_file_prefix, time.step_time().value(), l_max,
        // comparison_l_max, number_of_radial_points);
      }
    }
    db::mutate<Tags::BondiJ, Tags::CauchyCartesianCoords,
               Tags::InertialCartesianCoords, Tags::InertialRetardedTime,
               Tags::GaugeC, Tags::GaugeD>(
        make_not_null(&box),
        [&stepper, &time_step, &j_history, &x_history, &y_history, &z_history,
         &x_tilde_history, &y_tilde_history, &z_tilde_history, &u_bondi_history,
         &gauge_c_history, &gauge_d_history](
            const gsl::not_null<db::item_type<Tags::BondiJ>*> j,
            const gsl::not_null<db::item_type<Tags::CauchyCartesianCoords>*>
                x_of_x_tilde,
            const gsl::not_null<db::item_type<Tags::InertialCartesianCoords>*>
                x_tilde_of_x,
            const gsl::not_null<db::item_type<Tags::InertialRetardedTime>*>
                u_bondi,
            const gsl::not_null<db::item_type<Tags::GaugeC>*> gauge_c,
            const gsl::not_null<db::item_type<Tags::GaugeD>*> gauge_d) {
          stepper.update_u(make_not_null(&get(*j).data()),
                           make_not_null(&j_history), time_step);

          stepper.update_u(make_not_null(&get<0>(*x_of_x_tilde)),
                           make_not_null(&x_history), time_step);
          stepper.update_u(make_not_null(&get<1>(*x_of_x_tilde)),
                           make_not_null(&y_history), time_step);
          stepper.update_u(make_not_null(&get<2>(*x_of_x_tilde)),
                           make_not_null(&z_history), time_step);

          stepper.update_u(make_not_null(&get<0>(*x_tilde_of_x)),
                           make_not_null(&x_tilde_history), time_step);
          stepper.update_u(make_not_null(&get<1>(*x_tilde_of_x)),
                           make_not_null(&y_tilde_history), time_step);
          stepper.update_u(make_not_null(&get<2>(*x_tilde_of_x)),
                           make_not_null(&z_tilde_history), time_step);

          stepper.update_u(make_not_null(&get(*u_bondi).data()),
                           make_not_null(&u_bondi_history), time_step);

          stepper.update_u(make_not_null(&get(*gauge_c).data()),
                           make_not_null(&gauge_c_history), time_step);
          stepper.update_u(make_not_null(&get(*gauge_d).data()),
                           make_not_null(&gauge_d_history), time_step);
        });
    time = stepper.next_time_id(time, time_step);
    // printf("next time: %f\n", time.time().value());
    db::mutate_apply<GaugeUpdateAngularFromCartesian<
        Tags::InertialAngularCoords, Tags::InertialCartesianCoords>>(
        make_not_null(&box));
    db::mutate_apply<GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
        make_not_null(&box));

    db::mutate_apply<GaugeUpdateJacobianFromCoords<
        Tags::GaugeA, Tags::GaugeB, Tags::InertialCartesianCoords,
        Tags::InertialAngularCoords>>(make_not_null(&box));
    db::mutate_apply<GaugeUpdateJacobianFromCoords<Tags::GaugeC, Tags::GaugeD,
                                                   Tags::CauchyCartesianCoords,
                                                   Tags::CauchyAngularCoords>>(
        make_not_null(&box));
    db::mutate_apply<GaugeUpdateOmega>(make_not_null(&box));
    db::mutate_apply<GaugeUpdateOmegaCD>(make_not_null(&box));
    // db::mutate<Tags::GaugeOmega>(
    // make_not_null(&box),
    // [&l_max, &l_filter_start](
    // const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
    // omega) {
    // Spectral::Swsh::filter_swsh_boundary_quantity(
    // make_not_null(&get(*omega)), l_max, l_filter_start);
    // });

    // get the worldtube data for the next time step.
    db::mutate<boundary_variables_tag>(
        make_not_null(&box),
        [&time, &data_manager, &data_still_available](
            const gsl::not_null<db::item_type<boundary_variables_tag>*>
                boundary_variables) {
          data_still_available =
              data_manager.populate_hypersurface_boundary_data(
                  boundary_variables, time.time().value());
        });
  }
  // clean up the last few interpolation points we didn't get enough data to
  // finish. The last few points will have worse accuracy, but probably still
  // worth recording.
  // while (interpolation_manager.target_times_size() != 0) {
  // auto interpolation =
  // interpolation_manager.interpolate_and_pop_first_time();
  // record_scri_output<2>(make_not_null(&recorder), interpolation, "News",
  // l_max, comparison_l_max);
  // }
}

template <typename Tag, typename DataBoxType>
void record_l2_error_with_cauchy_gauge(
    const gsl::not_null<DataBoxType*> box,
    const gsl::not_null<ModeRecorder*> recorder, const double time,
    const size_t l_max, const size_t comparison_l_max,
    const size_t number_of_radial_points) noexcept {
  ComplexDataVector cauchy_buffer =
      get(db::get<Tags::CauchyGauge<Tag>>(*box)).data();
  ComplexDataVector inertial_buffer = get(db::get<Tag>(*box)).data();

  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  ComplexModalVector l2_error{square(l_max + 1), 0.0};
  Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>> cauchy_view;
  SpinWeighted<ComplexDataVector, Tag::type::type::spin> inertial_view;
  for (size_t i = 0; i < number_of_radial_points; ++i) {
    get(cauchy_view).data() =
        ComplexDataVector{cauchy_buffer.data() + i * number_of_angular_points,
                          number_of_angular_points};
    inertial_view.data() =
        ComplexDataVector{inertial_buffer.data() + i * number_of_angular_points,
                          number_of_angular_points};

    auto inertial_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
        Spectral::Swsh::swsh_transform(make_not_null(&inertial_view), l_max),
        l_max);
    ComplexModalVector transformed_cauchy_modes;

    // if (cpp17::is_same_v<Tag, Tags::BondiJ>) {
    transformed_cauchy_modes = CalculateInertialModes<Tag>::compute(
        box, make_not_null(&cauchy_view),
        ComplexDataVector{number_of_angular_points, 1.0});
    // } else {
    // TEST for testing, the other scalars do not have a volume gauge
    // transformation routine, so this will only work if in the noninertial
    // gauge

    // transformed_cauchy_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
    // Spectral::Swsh::swsh_transform(
    // make_not_null(&get(cauchy_view)), l_max),
    // l_max)
    // .data();
    // }
    SpinWeighted<ComplexModalVector, Tag::type::type::spin> mode_difference;
    mode_difference.data() = inertial_modes.data() - transformed_cauchy_modes;

    // printf("mode differences %zu\n", i);
    // for (size_t j = 0; j < mode_difference.size(); ++j) {
    // printf("(%e, %e)\n", real(mode_difference.data()[j]),
    // imag(mode_difference.data()[j]));
    // }
    // printf("\n");
    l2_error += mode_difference.data();
  }
  l2_error /= number_of_radial_points;

  recorder->append_mode_data("/" + Tag::name() + "_l2_difference", time,
                             l2_error, comparison_l_max);
}

// TODO work on the names of these functions...
// template <typename Tag, typename DataBoxType>
// void calculate_and_record_l2_error_of_cauchy_from_inertial(
// ) noexcept {

// }

// TODO: implement a factor for the robinson-trautman time stepping and a
// distinct resolution l_max.
void test_regularity_preserving_cce_rt(
    std::string input_filename, size_t simulation_l_max,
    size_t comparison_l_max, size_t number_of_radial_points,
    std::string output_file_suffix, size_t rational_timestep_numerator,
    size_t rational_timestep_denominator, double end_time) noexcept {
  TimeSteppers::RungeKutta3 stepper{};
  TimeSteppers::History<ComplexDataVector, ComplexDataVector> j_history{};
  TimeSteppers::History<ComplexDataVector, ComplexDataVector> rt_w_history{};
  TimeSteppers::History<ComplexDataVector, ComplexDataVector> omega_history{};

  TimeSteppers::History<DataVector, DataVector> x_history{};
  TimeSteppers::History<DataVector, DataVector> y_history{};
  TimeSteppers::History<DataVector, DataVector> z_history{};

  TimeSteppers::History<DataVector, DataVector> x_tilde_history{};
  TimeSteppers::History<DataVector, DataVector> y_tilde_history{};
  TimeSteppers::History<DataVector, DataVector> z_tilde_history{};

  using boundary_variables_tag = ::Tags::Variables<all_boundary_tags>;
  using gauge_transform_boundary_variables_tag =
      ::Tags::Variables<gauge_transform_boundary_tags>;
  using angular_coordinate_variables_tag =
      ::Tags::Variables<angular_coordinate_tags>;
  using scri_variables_tag = ::Tags::Variables<scri_tags>;
  using gauge_confirmation_scri_variables_tag =
      ::Tags::Variables<gauge_confirmation_scri_tags>;
  using volume_gauge_confirmation_variables_tag =
      ::Tags::Variables<gauge_confirmation_volume_tags>;
  using integration_independent_variables_tag =
      ::Tags::Variables<pre_computation_tags>;
  using pre_swsh_derivatives_variables_tag = ::Tags::Variables<
      tmpl::append<all_pre_swsh_derivative_tags, tmpl::list<Tags::SpecH>>>;
  using transform_buffer_variables_tag =
      ::Tags::Variables<all_transform_buffer_tags>;
  using swsh_derivatives_variables_tag =
      ::Tags::Variables<all_swsh_derivative_tags>;
  using temporary_variables_tag =
      ::Tags::Variables<all_temporary_equation_tags>;
  using integrand_variables_tag = ::Tags::Variables<all_integrand_tags>;

  size_t l_max = simulation_l_max;
  size_t l_filter_start = l_max - 2;
  size_t boundary_size =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  size_t volume_size = boundary_size * number_of_radial_points;
  size_t transform_buffer_size =
      2 * number_of_radial_points *
      Spectral::Swsh::number_of_swsh_coefficients(l_max);

  auto box = db::create<db::AddSimpleTags<
      Tags::LMax, boundary_variables_tag,
      gauge_transform_boundary_variables_tag, angular_coordinate_variables_tag,
      scri_variables_tag, volume_gauge_confirmation_variables_tag,
      gauge_confirmation_scri_variables_tag,
      integration_independent_variables_tag, pre_swsh_derivatives_variables_tag,
      transform_buffer_variables_tag, swsh_derivatives_variables_tag,
      temporary_variables_tag, integrand_variables_tag>>(
      l_max, db::item_type<boundary_variables_tag>{boundary_size},
      db::item_type<gauge_transform_boundary_variables_tag>{boundary_size},
      db::item_type<angular_coordinate_variables_tag>{boundary_size},
      db::item_type<scri_variables_tag>{boundary_size},
      db::item_type<volume_gauge_confirmation_variables_tag>{volume_size},
      db::item_type<gauge_confirmation_scri_variables_tag>{boundary_size},
      db::item_type<integration_independent_variables_tag>{volume_size},
      db::item_type<pre_swsh_derivatives_variables_tag>{volume_size},
      db::item_type<transform_buffer_variables_tag>{transform_buffer_size},
      db::item_type<swsh_derivatives_variables_tag>{volume_size},
      db::item_type<temporary_variables_tag>{volume_size},
      db::item_type<integrand_variables_tag>{volume_size});

  ModeRecorder recorder{input_filename + output_file_suffix + ".h5",
                        comparison_l_max, simulation_l_max};

  // in order to use the time architecture, we need to make a slab
  Slab only_slab{0.0, end_time};
  TimeDelta time_step{
      only_slab,
      Time::rational_t{static_cast<int>(rational_timestep_numerator),
                       static_cast<int>(end_time) *
                           static_cast<int>(rational_timestep_denominator)}};

  TimeId time{true, 0, Time{only_slab, Time::rational_t{0, 1}}};

  db::mutate_apply<InitializeRobinsonTrautman>(make_not_null(&box));

  tmpl::for_each<tmpl::list<
      Tags::BoundaryValue<Tags::BondiR>,
      Tags::BoundaryValue<Tags::DuRDividedByR>,
      Tags::BoundaryValue<Tags::BondiJ>,
      Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
      Tags::BoundaryValue<Tags::Dr<Tags::BondiU>>,
      Tags::BoundaryValue<Tags::BondiBeta>, Tags::BoundaryValue<Tags::BondiU>,
      Tags::BoundaryValue<Tags::BondiQ>, Tags::BoundaryValue<Tags::BondiW>,
      Tags::BoundaryValue<Tags::BondiH>, Tags::BoundaryValue<Tags::SpecH>>>(
      [&box](auto x) {
        using tag = typename decltype(x)::type;
        db::mutate_apply<CalculateRobinsonTrautman<tag>>(make_not_null(&box));
      });

  ScriPlusInterpolationManager<FlexibleBarycentricInterpolator,
                               ComplexDataVector>
      interpolation_manager{1, boundary_size};

  db::mutate_apply<InitializeJ<Tags::BoundaryValue>>(make_not_null(&box));
  db::mutate_apply<InitializeGauge>(make_not_null(&box));
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::InertialAngularCoords, Tags::InertialCartesianCoords>>(
      make_not_null(&box));
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      make_not_null(&box));

  db::mutate_apply<InitializeXtildeOfX>(make_not_null(&box));
  db::mutate_apply<GaugeUpdateJacobianFromCoords<Tags::GaugeA, Tags::GaugeB,
                                                 Tags::InertialCartesianCoords,
                                                 Tags::InertialAngularCoords>>(
      make_not_null(&box));
  db::mutate_apply<GaugeUpdateJacobianFromCoords<Tags::GaugeC, Tags::GaugeD,
                                                 Tags::CauchyCartesianCoords,
                                                 Tags::CauchyAngularCoords>>(
      make_not_null(&box));
  db::mutate_apply<GaugeUpdateOmega>(make_not_null(&box));
  db::mutate_apply<GaugeUpdateOmegaCD>(make_not_null(&box));
  db::mutate_apply<GaugeAdjustInitialJ>(make_not_null(&box));

  db::mutate_apply<InitializeScriPlusValue<Tags::InertialRetardedTime>>(
      make_not_null(&box), 0.0);

  size_t step_counter = 0;

  // main loop
  while (time.time().value() < end_time) {
    step_counter++;
    printf("starting step %zu\n", step_counter);
    tmpl::for_each<compute_gauge_adjustments_setup_tags>([&box](auto x) {
      using tag = typename decltype(x)::type;
      db::mutate_apply<ComputeGaugeAdjustedBoundaryValue<tag>>(
          make_not_null(&box));
    });
    mutate_all_precompute_cce_dependencies<Tags::EvolutionGaugeBoundaryValue>(
        make_not_null(&box));

    tmpl::for_each<tmpl::list<Tags::BondiBeta, Tags::BondiQ, Tags::BondiU,
                              Tags::BondiW, Tags::BondiH>>([&box, &l_max,
                                                            &l_filter_start,
                                                            &time](auto x) {
      using bondi_tag = typename decltype(x)::type;
      // printf("before computation for : %s\n", bondi_tag::name().c_str());
      regularity_preserving_robinson_trautman_hypersurface_computation<
          bondi_tag, swsh_derivatives_variables_tag,
          transform_buffer_variables_tag, pre_swsh_derivatives_variables_tag>(
          make_not_null(&box), time.time().value());

      // db::mutate<bondi_tag>(
      // make_not_null(&box),
      // [&l_max,
      // &l_filter_start](const gsl::not_null<db::item_type<bondi_tag>*>
      // bondi_quantity) {
      // Spectral::Swsh::filter_swsh_volume_quantity(
      // make_not_null(&get(*bondi_quantity)), l_max, l_filter_start,
      // 108.0, 8);
      // });
    });
    // printf("calculating time derivatives\n");
    db::mutate_apply<
        CalculateScriPlusValue<Tags::Du<Tags::InertialRetardedTime>>>(
        make_not_null(&box));

    db::mutate_apply<
        CalculateRobinsonTrautman<Tags::Du<Tags::RobinsonTrautmanW>>>(
        make_not_null(&box));

    // printf("recording time derivatives\n");
    ComplexDataVector du_j = get(db::get<Tags::BondiH>(box)).data();
    j_history.insert(time.time(), get(db::get<Tags::BondiJ>(box)).data(),
                     std::move(du_j));

    ComplexDataVector du_omega_cd =
        get(db::get<Tags::Du<Tags::GaugeOmegaCD>>(box)).data();
    omega_history.insert(time.time(),
                         get(db::get<Tags::GaugeOmegaCD>(box)).data(),
                         std::move(du_omega_cd));

    ComplexDataVector du_rt_w =
        get(db::get<Tags::Du<Tags::RobinsonTrautmanW>>(box)).data();
    rt_w_history.insert(time.time(),
                        get(db::get<Tags::RobinsonTrautmanW>(box)).data(),
                        std::move(du_rt_w));

    DataVector du_x = get<0>(db::get<Tags::DuCauchyCartesianCoords>(box));
    x_history.insert(time.time(),
                     get<0>(db::get<Tags::CauchyCartesianCoords>(box)),
                     std::move(du_x));
    DataVector du_y = get<1>(db::get<Tags::DuCauchyCartesianCoords>(box));
    y_history.insert(time.time(),
                     get<1>(db::get<Tags::CauchyCartesianCoords>(box)),
                     std::move(du_y));
    DataVector du_z = get<2>(db::get<Tags::DuCauchyCartesianCoords>(box));
    z_history.insert(time.time(),
                     get<2>(db::get<Tags::CauchyCartesianCoords>(box)),
                     std::move(du_z));

    DataVector du_x_tilde =
        get<0>(db::get<Tags::DuInertialCartesianCoords>(box));
    x_tilde_history.insert(time.time(),
                           get<0>(db::get<Tags::InertialCartesianCoords>(box)),
                           std::move(du_x_tilde));
    DataVector du_y_tilde =
        get<1>(db::get<Tags::DuInertialCartesianCoords>(box));
    y_tilde_history.insert(time.time(),
                           get<1>(db::get<Tags::InertialCartesianCoords>(box)),
                           std::move(du_y_tilde));
    DataVector du_z_tilde =
        get<2>(db::get<Tags::DuInertialCartesianCoords>(box));
    z_tilde_history.insert(time.time(),
                           get<2>(db::get<Tags::InertialCartesianCoords>(box)),
                           std::move(du_z_tilde));

    // compute the scri+ values in the Cauchy gauge to verify against the SpEC
    // computation that we're actually performing a reliable coordinate
    // transformation as we hope.
    db::mutate<Tags::SpecH>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<Tags::SpecH>*> spec_h,
           const db::item_type<Tags::BondiH>& h,
           const db::item_type<Tags::DuRDividedByR>& du_r_divided_by_r,
           const db::item_type<Tags::OneMinusY>& one_minus_y,
           const db::item_type<Tags::Dy<Tags::BondiJ>>& dy_j) {
          get(*spec_h) =
              get(h) - get(du_r_divided_by_r) * get(one_minus_y) * get(dy_j);
        },
        db::get<Tags::BondiH>(box), db::get<Tags::DuRDividedByR>(box),
        db::get<Tags::OneMinusY>(box), db::get<Tags::Dy<Tags::BondiJ>>(box));

    db::mutate_apply<CalculateRobinsonTrautman<Tags::CauchyGauge<Tags::News>>>(
        make_not_null(&box));

    // note gauge arguments are 'backwards', because the 'cauchy' gauge for RT
    // is actually the inertial gauge, and we deliberately warp it for the
    // evolution
    db::mutate<Tags::News, Tags::BondiJ, Tags::BondiBeta, Tags::SpecH,
               Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>, Tags::BondiU>(
        make_not_null(&box), calculate_non_inertial_news,
        db::get<Tags::GaugeA>(box), db::get<Tags::GaugeB>(box),
        db::get<Tags::GaugeOmega>(box),
        db::get<Tags::Du<Tags::GaugeOmega>>(box),
        db::get<Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                 Spectral::Swsh::Tags::Eth>>(
            box),
        db::get<Tags::InertialAngularCoords>(box),
        db::get<Tags::CauchyAngularCoords>(box), l_max, false);

    tmpl::for_each<tmpl::list<
        Tags::CauchyGauge<Tags::BondiJ>, Tags::CauchyGauge<Tags::BondiBeta>,
        Tags::CauchyGauge<Tags::BondiU>, Tags::CauchyGauge<Tags::BondiQ>,
        Tags::CauchyGauge<Tags::BondiW>, Tags::CauchyGauge<Tags::BondiH>>>(
        [&box](auto x) {
          using tag = typename decltype(x)::type;
          db::mutate_apply<CalculateRobinsonTrautman<tag>>(make_not_null(&box));
        });

    // tmpl::for_each<tmpl::list<Tags::BondiBeta>>([&box](auto x) {
    // using tag = typename decltype(x)::type;
    // db::mutate_apply<CalculateCauchyGauge<Tags::CauchyGauge<tag>>>(
    // make_not_null(&box));
    // });

    // printf("recording output\n");
    if (time.substep() == 0) {
      // perform a comparison of boundary values and scri+ values on each new
      // time advancement

      // only dump the comparison data if the current timestep is close to one
      // of the comparison points, otherwise dump every fourth timestep
      // printf("recording boundary values\n");
      record_boundary_values<
          Tags::EvolutionGaugeBoundaryValue,
          tmpl::list<Tags::BondiBeta, Tags::BondiQ, Tags::BondiU, Tags::BondiW,
                     Tags::BondiJ, Tags::BondiR>>(
          make_not_null(&box), make_not_null(&recorder),
          time.step_time().value(), l_max, comparison_l_max);

      // printf("recording scri values\n");
      compare_and_record_scri_values<
          tmpl::list<Tags::BondiJ, Tags::BondiBeta, Tags::BondiQ, Tags::BondiU,
                     Tags::BondiW, Tags::BondiH>>(
          make_not_null(&box), make_not_null(&recorder), "",
          time.step_time().value(), l_max, comparison_l_max,
          number_of_radial_points);

      // record the news difference as its own special output.
      SpinWeighted<ComplexDataVector, 2> news_difference =
          get(db::get<Tags::CauchyGauge<Tags::News>>(box)) -
          get(db::get<Tags::News>(box));
      recorder.append_mode_data("/News_vs_RT", time.step_time().value(),
                                Spectral::Swsh::libsharp_to_goldberg_modes(
                                    Spectral::Swsh::swsh_transform(
                                        make_not_null(&news_difference), l_max),
                                    l_max)
                                    .data(),
                                comparison_l_max);

      compare_and_record_scri_values<
          tmpl::list<Tags::News, Tags::CauchyGauge<Tags::News>>>(
          make_not_null(&box), make_not_null(&recorder), "",
          time.step_time().value(), l_max, comparison_l_max, 1);

      // printf("recording l2 error with cauchy gauge values\n");
      record_l2_error_with_cauchy_gauge<Tags::BondiJ>(
          make_not_null(&box), make_not_null(&recorder),
          time.step_time().value(), l_max, comparison_l_max,
          number_of_radial_points);

      // printf("recording r200 values\n");
      compare_and_record_r200_values<
          tmpl::list<Tags::BondiJ, Tags::BondiBeta, Tags::BondiQ, Tags::BondiU,
                     Tags::BondiH>>(make_not_null(&box),
                                    make_not_null(&recorder), "",
                                    time.step_time().value(), l_max,
                                    comparison_l_max, number_of_radial_points);
      // }
    }
    // printf("stepping\n");
    db::mutate<Tags::BondiJ, Tags::CauchyCartesianCoords,
               Tags::InertialCartesianCoords, Tags::RobinsonTrautmanW,
               Tags::GaugeOmegaCD>(
        make_not_null(&box),
        [
          &stepper, &time_step, &j_history, &rt_w_history, &x_history,
          &y_history, &z_history, &x_tilde_history, &y_tilde_history,
          &z_tilde_history, &omega_history
        ](const gsl::not_null<db::item_type<Tags::BondiJ>*> j,
          const gsl::not_null<db::item_type<Tags::CauchyCartesianCoords>*>
              x_of_x_tilde,
          const gsl::not_null<db::item_type<Tags::InertialCartesianCoords>*>
              x_tilde_of_x,
          const gsl::not_null<db::item_type<Tags::RobinsonTrautmanW>*> rt_w,
          const gsl::not_null<db::item_type<Tags::GaugeOmegaCD>*>
              omega_cd) noexcept {
          // printf("stepping j\n");
          stepper.update_u(make_not_null(&get(*j).data()),
                           make_not_null(&j_history), time_step);

          // printf("stepping x\n");
          stepper.update_u(make_not_null(&get<0>(*x_of_x_tilde)),
                           make_not_null(&x_history), time_step);
          stepper.update_u(make_not_null(&get<1>(*x_of_x_tilde)),
                           make_not_null(&y_history), time_step);
          stepper.update_u(make_not_null(&get<2>(*x_of_x_tilde)),
                           make_not_null(&z_history), time_step);

          stepper.update_u(make_not_null(&get<0>(*x_tilde_of_x)),
                           make_not_null(&x_tilde_history), time_step);
          stepper.update_u(make_not_null(&get<1>(*x_tilde_of_x)),
                           make_not_null(&y_tilde_history), time_step);
          stepper.update_u(make_not_null(&get<2>(*x_tilde_of_x)),
                           make_not_null(&z_tilde_history), time_step);

          // stepper.update_u(make_not_null(&get(*omega_cd).data()),
          // make_not_null(&omega_history), time_step);
          // printf("stepping rt_w\n");
          stepper.update_u(make_not_null(&get(*rt_w).data()),
                           make_not_null(&rt_w_history), time_step);
        });
    time = stepper.next_time_id(time, time_step);
    // printf("next time: %f\n", time.time().value());
    db::mutate_apply<GaugeUpdateAngularFromCartesian<
        Tags::InertialAngularCoords, Tags::InertialCartesianCoords>>(
        make_not_null(&box));
    db::mutate_apply<GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
        make_not_null(&box));

    db::mutate_apply<GaugeUpdateJacobianFromCoords<
        Tags::GaugeA, Tags::GaugeB, Tags::InertialCartesianCoords,
        Tags::InertialAngularCoords>>(make_not_null(&box));
    db::mutate_apply<GaugeUpdateJacobianFromCoords<Tags::GaugeC, Tags::GaugeD,
                                                   Tags::CauchyCartesianCoords,
                                                   Tags::CauchyAngularCoords>>(
        make_not_null(&box));
    db::mutate_apply<GaugeUpdateOmega>(make_not_null(&box));
    db::mutate_apply<GaugeUpdateOmegaCD>(make_not_null(&box));
    // db::mutate<Tags::GaugeOmega>(
    // make_not_null(&box),
    // [&l_max, &l_filter_start](
    // const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
    // omega) {
    // Spectral::Swsh::filter_swsh_boundary_quantity(
    // make_not_null(&get(*omega)), l_max, l_filter_start);
    // });

    // get the worldtube data for the next time step.
    tmpl::for_each<tmpl::list<
        Tags::BoundaryValue<Tags::BondiR>,
        Tags::BoundaryValue<Tags::DuRDividedByR>,
        Tags::BoundaryValue<Tags::BondiJ>,
        Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
        Tags::BoundaryValue<Tags::Dr<Tags::BondiU>>,
        Tags::BoundaryValue<Tags::BondiBeta>, Tags::BoundaryValue<Tags::BondiU>,
        Tags::BoundaryValue<Tags::BondiQ>, Tags::BoundaryValue<Tags::BondiW>,
        Tags::BoundaryValue<Tags::BondiH>>>([&box](auto x) {
      using tag = typename decltype(x)::type;
      db::mutate_apply<CalculateRobinsonTrautman<tag>>(make_not_null(&box));
    });
    step_counter++;
  }
}
}  // namespace Cce
