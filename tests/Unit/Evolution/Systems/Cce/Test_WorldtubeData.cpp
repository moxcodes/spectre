// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/ReducedWorldtubeModeRecorder.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "Helpers/Evolution/Systems/Cce/WriteToWorldtubeH5.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {

class DummyBufferUpdater
    : public WorldtubeBufferUpdater<cce_metric_input_tags> {
 public:
  DummyBufferUpdater(DataVector time_buffer,
                     const gr::Solutions::KerrSchild& solution,
                     const std::optional<double> extraction_radius,
                     const double coordinate_amplitude,
                     const double coordinate_frequency, const size_t l_max,
                     const bool apply_normalization_bug = false,
                     const bool has_version_history = true) noexcept
      : time_buffer_{std::move(time_buffer)},
        solution_{solution},
        extraction_radius_{extraction_radius},
        coordinate_amplitude_{coordinate_amplitude},
        coordinate_frequency_{coordinate_frequency},
        l_max_{l_max},
        apply_normalization_bug_{apply_normalization_bug},
        has_version_history_{has_version_history} {}

  WRAPPED_PUPable_decl_template(DummyBufferUpdater);  // NOLINT

  explicit DummyBufferUpdater(CkMigrateMessage* /*unused*/) noexcept
      : extraction_radius_{1.0},
        coordinate_amplitude_{0.0},
        coordinate_frequency_{0.0},
        l_max_{0} {}

  double update_buffers_for_time(
      const gsl::not_null<Variables<cce_metric_input_tags>*> buffers,
      const gsl::not_null<size_t*> time_span_start,
      const gsl::not_null<size_t*> time_span_end, const double time,
      const size_t /*l_max*/, const size_t interpolator_length,
      const size_t buffer_depth) const noexcept override {
    if (*time_span_end > interpolator_length and
        time_buffer_[*time_span_end - interpolator_length + 1] > time) {
      // the next time an update will be required
      return time_buffer_[*time_span_end - interpolator_length + 1];
    }
    // find the time spans that are needed
    auto new_span_pair = detail::create_span_for_time_value(
        time, buffer_depth, interpolator_length, 0, time_buffer_.size(),
        time_buffer_);
    *time_span_start = new_span_pair.first;
    *time_span_end = new_span_pair.second;

    const size_t goldberg_size = square(l_max_ + 1);
    tnsr::ii<ComplexModalVector, 3> spatial_metric_coefficients{goldberg_size};
    tnsr::ii<ComplexModalVector, 3> dt_spatial_metric_coefficients{
        goldberg_size};
    tnsr::ii<ComplexModalVector, 3> dr_spatial_metric_coefficients{
        goldberg_size};
    tnsr::I<ComplexModalVector, 3> shift_coefficients{goldberg_size};
    tnsr::I<ComplexModalVector, 3> dt_shift_coefficients{goldberg_size};
    tnsr::I<ComplexModalVector, 3> dr_shift_coefficients{goldberg_size};
    Scalar<ComplexModalVector> lapse_coefficients{goldberg_size};
    Scalar<ComplexModalVector> dt_lapse_coefficients{goldberg_size};
    Scalar<ComplexModalVector> dr_lapse_coefficients{goldberg_size};
    for (size_t time_index = 0; time_index < *time_span_end - *time_span_start;
         ++time_index) {
      TestHelpers::create_fake_time_varying_modal_data(
          make_not_null(&spatial_metric_coefficients),
          make_not_null(&dt_spatial_metric_coefficients),
          make_not_null(&dr_spatial_metric_coefficients),
          make_not_null(&shift_coefficients),
          make_not_null(&dt_shift_coefficients),
          make_not_null(&dr_shift_coefficients),
          make_not_null(&lapse_coefficients),
          make_not_null(&dt_lapse_coefficients),
          make_not_null(&dr_lapse_coefficients), solution_,
          extraction_radius_.value_or(default_extraction_radius_),
          coordinate_amplitude_, coordinate_frequency_,
          time_buffer_[time_index + *time_span_start], l_max_, true,
          apply_normalization_bug_);

      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<Tags::detail::SpatialMetric>(*buffers)),
          spatial_metric_coefficients, time_index,
          *time_span_end - *time_span_start);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(
              &get<Tags::detail::Dr<Tags::detail::SpatialMetric>>(*buffers)),
          dr_spatial_metric_coefficients, time_index,
          *time_span_end - *time_span_start);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(
              &get<::Tags::dt<Tags::detail::SpatialMetric>>(*buffers)),
          dt_spatial_metric_coefficients, time_index,
          *time_span_end - *time_span_start);

      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<Tags::detail::Shift>(*buffers)),
          shift_coefficients, time_index, *time_span_end - *time_span_start);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<Tags::detail::Dr<Tags::detail::Shift>>(*buffers)),
          dr_shift_coefficients, time_index, *time_span_end - *time_span_start);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<::Tags::dt<Tags::detail::Shift>>(*buffers)),
          dt_shift_coefficients, time_index, *time_span_end - *time_span_start);

      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<Tags::detail::Lapse>(*buffers)),
          lapse_coefficients, time_index, *time_span_end - *time_span_start);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<Tags::detail::Dr<Tags::detail::Lapse>>(*buffers)),
          dr_lapse_coefficients, time_index, *time_span_end - *time_span_start);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<::Tags::dt<Tags::detail::Lapse>>(*buffers)),
          dt_lapse_coefficients, time_index, *time_span_end - *time_span_start);
    }
    return time_buffer_[*time_span_end - interpolator_length + 1];
  }

  std::unique_ptr<WorldtubeBufferUpdater<cce_metric_input_tags>> get_clone()
      const noexcept override {
    return std::make_unique<DummyBufferUpdater>(*this);
  }

  bool time_is_outside_range(const double time) const noexcept override {
    return time < time_buffer_[0] or
           time > time_buffer_[time_buffer_.size() - 1];
  }

  size_t get_l_max() const noexcept override { return l_max_; }

  double get_extraction_radius() const noexcept override {
    return extraction_radius_.value_or(default_extraction_radius_);
  }

  bool has_version_history() const noexcept override {
    return has_version_history_;
  }

  DataVector& get_time_buffer() noexcept override { return time_buffer_; }

  void pup(PUP::er& p) noexcept override {
    p | time_buffer_;
    p | solution_;
    p | extraction_radius_;
    p | coordinate_amplitude_;
    p | coordinate_frequency_;
    p | default_extraction_radius_;
    p | l_max_;
    p | apply_normalization_bug_;
    p | has_version_history_;
  }

 private:
  template <typename... Structure>
  void update_tensor_buffer_with_tensor_at_time_index(
      const gsl::not_null<Tensor<ComplexModalVector, Structure...>*>
          tensor_buffer,
      const Tensor<ComplexModalVector, Structure...>& tensor_at_time,
      const size_t time_index, const size_t time_span_extent) const noexcept {
    for (size_t i = 0; i < tensor_at_time.size(); ++i) {
      for (size_t k = 0; k < tensor_at_time[i].size(); ++k) {
        (*tensor_buffer)[i][time_index + k * time_span_extent] =
            tensor_at_time[i][k];
      }
    }
  }

  DataVector time_buffer_;
  gr::Solutions::KerrSchild solution_;
  std::optional<double> extraction_radius_;
  double default_extraction_radius_ = 100.0;
  double coordinate_amplitude_;
  double coordinate_frequency_;
  size_t l_max_;
  bool apply_normalization_bug_ = false;
  bool has_version_history_ = true;
};

class ReducedDummyBufferUpdater
    : public WorldtubeBufferUpdater<cce_bondi_input_tags> {
 public:
  ReducedDummyBufferUpdater(DataVector time_buffer,
                            const gr::Solutions::KerrSchild& solution,
                            const std::optional<double> extraction_radius,
                            const double coordinate_amplitude,
                            const double coordinate_frequency,
                            const size_t l_max,
                            const bool /*unused*/ = false) noexcept
      : time_buffer_{std::move(time_buffer)},
        solution_{solution},
        extraction_radius_{extraction_radius},
        coordinate_amplitude_{coordinate_amplitude},
        coordinate_frequency_{coordinate_frequency},
        l_max_{l_max} {}

  WRAPPED_PUPable_decl_template(ReducedDummyBufferUpdater);  // NOLINT

  explicit ReducedDummyBufferUpdater(CkMigrateMessage* /*unused*/) noexcept {}

  double update_buffers_for_time(
      const gsl::not_null<Variables<cce_bondi_input_tags>*> buffers,
      const gsl::not_null<size_t*> time_span_start,
      const gsl::not_null<size_t*> time_span_end, const double time,
      const size_t l_max, const size_t interpolator_length,
      const size_t buffer_depth) const noexcept override {
    if (*time_span_end > interpolator_length and
        time_buffer_[*time_span_end - interpolator_length + 1] > time) {
      // the next time an update will be required
      return time_buffer_[*time_span_end - interpolator_length + 1];
    }
    // find the time spans that are needed
    auto new_span_pair = detail::create_span_for_time_value(
        time, buffer_depth, interpolator_length, 0, time_buffer_.size(),
        time_buffer_);
    *time_span_start = new_span_pair.first;
    *time_span_end = new_span_pair.second;

    const size_t libsharp_size =
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max);
    tnsr::ii<ComplexModalVector, 3> spatial_metric_coefficients{libsharp_size};
    tnsr::ii<ComplexModalVector, 3> dt_spatial_metric_coefficients{
        libsharp_size};
    tnsr::ii<ComplexModalVector, 3> dr_spatial_metric_coefficients{
        libsharp_size};
    tnsr::I<ComplexModalVector, 3> shift_coefficients{libsharp_size};
    tnsr::I<ComplexModalVector, 3> dt_shift_coefficients{libsharp_size};
    tnsr::I<ComplexModalVector, 3> dr_shift_coefficients{libsharp_size};
    Scalar<ComplexModalVector> lapse_coefficients{libsharp_size};
    Scalar<ComplexModalVector> dt_lapse_coefficients{libsharp_size};
    Scalar<ComplexModalVector> dr_lapse_coefficients{libsharp_size};

    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
        boundary_variables{number_of_angular_points};

    for (size_t time_index = 0; time_index < *time_span_end - *time_span_start;
         ++time_index) {
      TestHelpers::create_fake_time_varying_modal_data(
          make_not_null(&spatial_metric_coefficients),
          make_not_null(&dt_spatial_metric_coefficients),
          make_not_null(&dr_spatial_metric_coefficients),
          make_not_null(&shift_coefficients),
          make_not_null(&dt_shift_coefficients),
          make_not_null(&dr_shift_coefficients),
          make_not_null(&lapse_coefficients),
          make_not_null(&dt_lapse_coefficients),
          make_not_null(&dr_lapse_coefficients), solution_,
          extraction_radius_.value_or(default_extraction_radius_),
          coordinate_amplitude_, coordinate_frequency_,
          time_buffer_[time_index + *time_span_start], l_max_, false);

      Cce::create_bondi_boundary_data(
          make_not_null(&boundary_variables), spatial_metric_coefficients,
          dt_spatial_metric_coefficients, dr_spatial_metric_coefficients,
          shift_coefficients, dt_shift_coefficients, dr_shift_coefficients,
          lapse_coefficients, dt_lapse_coefficients, dr_lapse_coefficients,
          extraction_radius_.value_or(default_extraction_radius_), l_max);
      tmpl::for_each<tmpl::transform<
          cce_bondi_input_tags, tmpl::bind<db::remove_tag_prefix, tmpl::_1>>>(
          [this, &boundary_variables, &buffers, &time_index, &time_span_end,
           &time_span_start, &l_max](auto tag_v) noexcept {
            using tag = typename decltype(tag_v)::type;
            this->update_buffer_with_scalar_at_time_index(
                make_not_null(
                    &get<Spectral::Swsh::Tags::SwshTransform<tag>>(*buffers)),
                Spectral::Swsh::libsharp_to_goldberg_modes(
                    Spectral::Swsh::swsh_transform(
                        l_max, 1,
                        get(get<Tags::BoundaryValue<tag>>(boundary_variables))),
                    l_max),
                time_index, *time_span_end - *time_span_start);
          });
    }
    return time_buffer_[*time_span_end - interpolator_length + 1];
  }
  std::unique_ptr<WorldtubeBufferUpdater<cce_bondi_input_tags>> get_clone()
      const noexcept override {
    return std::make_unique<ReducedDummyBufferUpdater>(*this);
  }

  bool time_is_outside_range(const double time) const noexcept override {
    return time < time_buffer_[0] or
           time > time_buffer_[time_buffer_.size() - 1];
  }

  size_t get_l_max() const noexcept override { return l_max_; }

  double get_extraction_radius() const noexcept override {
    return extraction_radius_.value_or(default_extraction_radius_);
  }

  DataVector& get_time_buffer() noexcept override { return time_buffer_; }

  bool has_version_history() const noexcept override {
    return true;
  }

  void pup(PUP::er& p) noexcept override {
    p | time_buffer_;
    p | solution_;
    p | extraction_radius_;
    p | coordinate_amplitude_;
    p | coordinate_frequency_;
    p | l_max_;
  }

 private:
  template <int Spin>
  void update_buffer_with_scalar_at_time_index(
      const gsl::not_null<Scalar<SpinWeighted<ComplexModalVector, Spin>>*>
          scalar_buffer,
      const SpinWeighted<ComplexModalVector, Spin>& spin_weighted_at_time,
      const size_t time_index, const size_t time_span_extent) const noexcept {
    for (size_t k = 0; k < spin_weighted_at_time.size(); ++k) {
      get(*scalar_buffer).data()[time_index + k * time_span_extent] =
          spin_weighted_at_time.data()[k];
    }
  }

  DataVector time_buffer_;
  gr::Solutions::KerrSchild solution_;
  std::optional<double> extraction_radius_;
  double default_extraction_radius_ = 100.0;
  double coordinate_amplitude_ = 0.0;
  double coordinate_frequency_ = 0.0;
  size_t l_max_ = 0;
};

PUP::able::PUP_ID Cce::DummyBufferUpdater::my_PUP_ID = 0;
PUP::able::PUP_ID Cce::ReducedDummyBufferUpdater::my_PUP_ID = 0;

namespace {

template <typename DataManager, typename DummyUpdater, typename Generator>
void test_data_manager_with_dummy_buffer_updater(
    const gsl::not_null<Generator*> gen,
    const bool apply_normalization_bug = false, const bool is_spec_input = true,
    const std::optional<double> extraction_radius = std::nullopt) noexcept {
  // note that the default_extraction_radius is what will be reported
  // from the buffer updater when the extraction_radius is the default
  // `std::nullopt`.
  const double default_extraction_radius = 100.0;
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  gr::Solutions::KerrSchild solution{mass, spin, center};


  // acceptable parameters for the fake sinusoid variation in the input
  // parameters
  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);

  const size_t buffer_size = 4;
  const size_t l_max = 8;

  DataVector time_buffer{30};
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    time_buffer[i] = target_time - 1.55 + 0.1 * i;
  }

  DataManager boundary_data_manager;
  if constexpr (std::is_same_v<DataManager, MetricWorldtubeDataManager>) {
    if (not apply_normalization_bug) {
      boundary_data_manager = DataManager{
          std::make_unique<DummyUpdater>(
              time_buffer, solution, extraction_radius, amplitude, frequency,
              l_max, false, is_spec_input),
          l_max, buffer_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u),
          is_spec_input};
    } else {
      boundary_data_manager = DataManager{
          std::make_unique<DummyUpdater>(time_buffer, solution,
                                         extraction_radius, amplitude,
                                         frequency, l_max, true, false),
          l_max, buffer_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u),
          is_spec_input};
    }
  } else {
    // avoid compiler warnings in the case where the normalization bug booleans
    // aren't used.
    (void)apply_normalization_bug;
    (void)is_spec_input;
    boundary_data_manager = DataManager{
        std::make_unique<DummyUpdater>(time_buffer, solution, extraction_radius,
                                       amplitude, frequency, l_max, false),
        l_max, buffer_size,
        std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u)};
  }
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
      expected_boundary_variables{number_of_angular_points};
  Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
      interpolated_boundary_variables{number_of_angular_points};

  boundary_data_manager.populate_hypersurface_boundary_data(
      make_not_null(&interpolated_boundary_variables), target_time);

  // populate the expected variables with the result from the analytic modes
  // passed to the boundary data computation.
  const size_t libsharp_size =
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max);
  tnsr::ii<ComplexModalVector, 3> spatial_metric_coefficients{libsharp_size};
  tnsr::ii<ComplexModalVector, 3> dt_spatial_metric_coefficients{libsharp_size};
  tnsr::ii<ComplexModalVector, 3> dr_spatial_metric_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> shift_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> dt_shift_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> dr_shift_coefficients{libsharp_size};
  Scalar<ComplexModalVector> lapse_coefficients{libsharp_size};
  Scalar<ComplexModalVector> dt_lapse_coefficients{libsharp_size};
  Scalar<ComplexModalVector> dr_lapse_coefficients{libsharp_size};
  TestHelpers::create_fake_time_varying_modal_data(
      make_not_null(&spatial_metric_coefficients),
      make_not_null(&dt_spatial_metric_coefficients),
      make_not_null(&dr_spatial_metric_coefficients),
      make_not_null(&shift_coefficients), make_not_null(&dt_shift_coefficients),
      make_not_null(&dr_shift_coefficients), make_not_null(&lapse_coefficients),
      make_not_null(&dt_lapse_coefficients),
      make_not_null(&dr_lapse_coefficients), solution,
      extraction_radius.value_or(default_extraction_radius), amplitude,
      frequency, target_time, l_max, false);

  create_bondi_boundary_data(
      make_not_null(&expected_boundary_variables), spatial_metric_coefficients,
      dt_spatial_metric_coefficients, dr_spatial_metric_coefficients,
      shift_coefficients, dt_shift_coefficients, dr_shift_coefficients,
      lapse_coefficients, dt_lapse_coefficients, dr_lapse_coefficients,
      extraction_radius.value_or(default_extraction_radius), l_max);
  Approx angular_derivative_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  tmpl::for_each<
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>(
      [&expected_boundary_variables, &interpolated_boundary_variables,
       &angular_derivative_approx](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        INFO(db::tag_name<tag>());
        const auto& test_lhs = get<tag>(expected_boundary_variables);
        const auto& test_rhs = get<tag>(interpolated_boundary_variables);
        CHECK_ITERABLE_CUSTOM_APPROX(test_lhs, test_rhs,
                                     angular_derivative_approx);
      });
}

template <typename Generator>
void test_spec_worldtube_buffer_updater(
    const gsl::not_null<Generator*> gen,
    const bool extraction_radius_in_filename) noexcept {
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  gr::Solutions::KerrSchild solution{mass, spin, center};

  const double extraction_radius = 100.0;

  // acceptable parameters for the fake sinusoid variation in the input
  // parameters
  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);

  const size_t buffer_size = 4;
  const size_t interpolator_length = 2;
  const size_t l_max = 8;

  Variables<cce_metric_input_tags> coefficients_buffers_from_file{
      (buffer_size + 2 * interpolator_length) * square(l_max + 1)};
  Variables<cce_metric_input_tags> expected_coefficients_buffers{
      (buffer_size + 2 * interpolator_length) * square(l_max + 1)};
  const std::string filename = extraction_radius_in_filename
                                   ? "BoundaryDataH5Test_CceR0100.h5"
                                   : "BoundaryDataH5Test.h5";
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  TestHelpers::write_test_file(solution, filename, target_time,
                               extraction_radius, frequency, amplitude, l_max);

  // request an appropriate buffer
  auto buffer_updater =
      extraction_radius_in_filename
          ? MetricWorldtubeH5BufferUpdater{filename}
          : MetricWorldtubeH5BufferUpdater{filename, extraction_radius};
  auto serialized_and_deserialized_updater =
      serialize_and_deserialize(buffer_updater);
  size_t time_span_start = 0;
  size_t time_span_end = 0;
  buffer_updater.update_buffers_for_time(
      make_not_null(&coefficients_buffers_from_file),
      make_not_null(&time_span_start), make_not_null(&time_span_end),
      target_time, l_max, interpolator_length, buffer_size);

  Variables<cce_metric_input_tags> coefficients_buffers_from_serialized{
      (buffer_size + 2 * interpolator_length) * square(l_max + 1)};
  size_t time_span_start_from_serialized = 0;
  size_t time_span_end_from_serialized = 0;
  serialized_and_deserialized_updater.update_buffers_for_time(
      make_not_null(&coefficients_buffers_from_serialized),
      make_not_null(&time_span_start_from_serialized),
      make_not_null(&time_span_end_from_serialized), target_time, l_max,
      interpolator_length, buffer_size);

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  time_span_start = 0;
  time_span_end = 0;
  const auto& time_buffer = buffer_updater.get_time_buffer();
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    CHECK(time_buffer[i] == approx(target_time - 1.5 + 0.1 * i));
  }
  const auto& time_buffer_from_serialized =
      serialized_and_deserialized_updater.get_time_buffer();
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    CHECK(time_buffer_from_serialized[i] ==
          approx(target_time - 1.5 + 0.1 * i));
  }

  const DummyBufferUpdater dummy_buffer_updater{
      time_buffer, solution, extraction_radius, amplitude, frequency, l_max};
  dummy_buffer_updater.update_buffers_for_time(
      make_not_null(&expected_coefficients_buffers),
      make_not_null(&time_span_start), make_not_null(&time_span_end),
      target_time, l_max, interpolator_length, buffer_size);
  // check that the data in the buffer matches the expected analytic data.
  tmpl::for_each<cce_metric_input_tags>(
      [&expected_coefficients_buffers, &coefficients_buffers_from_file,
       &coefficients_buffers_from_serialized](auto tag_v) noexcept {
        using tag = typename decltype(tag_v)::type;
        INFO(db::tag_name<tag>());
        const auto& test_lhs = get<tag>(expected_coefficients_buffers);
        const auto& test_rhs = get<tag>(coefficients_buffers_from_file);
        CHECK_ITERABLE_APPROX(test_lhs, test_rhs);
        const auto& test_rhs_from_serialized =
            get<tag>(coefficients_buffers_from_serialized);
        CHECK_ITERABLE_APPROX(test_lhs, test_rhs_from_serialized);
      });
  CHECK(buffer_updater.get_extraction_radius() == 100.0);
}

template <typename Generator>
void test_reduced_spec_worldtube_buffer_updater(
    const gsl::not_null<Generator*> gen,
    const bool extraction_radius_in_filename) noexcept {
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  gr::Solutions::KerrSchild solution{mass, spin, center};

  const double extraction_radius = 100.0;

  // acceptable parameters for the fake sinusoid variation in the input
  // parameters
  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);

  const size_t buffer_size = 4;
  const size_t interpolator_length = 3;
  const size_t file_l_max = 8;
  const size_t computation_l_max = 10;

  Variables<cce_bondi_input_tags> coefficients_buffers_from_file{
      (buffer_size + 2 * interpolator_length) * square(computation_l_max + 1)};
  Variables<cce_bondi_input_tags> expected_coefficients_buffers{
      (buffer_size + 2 * interpolator_length) * square(computation_l_max + 1)};
  size_t libsharp_size =
      Spectral::Swsh::size_of_libsharp_coefficient_vector(file_l_max);
  tnsr::ii<ComplexModalVector, 3> spatial_metric_coefficients{libsharp_size};
  tnsr::ii<ComplexModalVector, 3> dt_spatial_metric_coefficients{libsharp_size};
  tnsr::ii<ComplexModalVector, 3> dr_spatial_metric_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> shift_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> dt_shift_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> dr_shift_coefficients{libsharp_size};
  Scalar<ComplexModalVector> lapse_coefficients{libsharp_size};
  Scalar<ComplexModalVector> dt_lapse_coefficients{libsharp_size};
  Scalar<ComplexModalVector> dr_lapse_coefficients{libsharp_size};

  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(file_l_max);
  Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
      boundary_data_variables{number_of_angular_points};

  // write times to file for several steps before and after the target time
  const std::string filename = extraction_radius_in_filename
      ? "BoundaryDataH5Test_CceR0100.h5"
      : "BoundaryDataH5Test.h5";
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  ComplexModalVector output_goldberg_mode_buffer{square(file_l_max + 1)};
  ComplexModalVector output_libsharp_mode_buffer{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(file_l_max)};

  // scoped to close the file
  {
    Cce::ReducedWorldtubeModeRecorder recorder{filename};
    for (size_t t = 0; t < 20; ++t) {
      const double time = 0.01 * t + target_time - 0.1;
      TestHelpers::create_fake_time_varying_modal_data(
          make_not_null(&spatial_metric_coefficients),
          make_not_null(&dt_spatial_metric_coefficients),
          make_not_null(&dr_spatial_metric_coefficients),
          make_not_null(&shift_coefficients),
          make_not_null(&dt_shift_coefficients),
          make_not_null(&dr_shift_coefficients),
          make_not_null(&lapse_coefficients),
          make_not_null(&dt_lapse_coefficients),
          make_not_null(&dr_lapse_coefficients), solution, extraction_radius,
          amplitude, frequency, time, file_l_max, false);

      create_bondi_boundary_data(
          make_not_null(&boundary_data_variables), spatial_metric_coefficients,
          dt_spatial_metric_coefficients, dr_spatial_metric_coefficients,
          shift_coefficients, dt_shift_coefficients, dr_shift_coefficients,
          lapse_coefficients, dt_lapse_coefficients, dr_lapse_coefficients,
          extraction_radius, file_l_max);

      using reduced_boundary_tags = tmpl::list<
          Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>,
          Cce::Tags::BoundaryValue<Cce::Tags::BondiU>,
          Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>,
          Cce::Tags::BoundaryValue<Cce::Tags::BondiW>,
          Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>,
          Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>,
          Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiJ>>,
          Cce::Tags::BoundaryValue<Cce::Tags::BondiR>,
          Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>>;

      // loop over the tags that we want to dump.
      tmpl::for_each<reduced_boundary_tags>(
          [&recorder, &boundary_data_variables, &output_goldberg_mode_buffer,
           &output_libsharp_mode_buffer, &file_l_max, &time](auto tag_v) {
            using tag = typename decltype(tag_v)::type;
            SpinWeighted<ComplexModalVector, tag::type::type::spin>
                spin_weighted_libsharp_view;
            spin_weighted_libsharp_view.set_data_ref(
                output_libsharp_mode_buffer.data(),
                output_libsharp_mode_buffer.size());
            Spectral::Swsh::swsh_transform(
                file_l_max, 1, make_not_null(&spin_weighted_libsharp_view),
                get(get<tag>(boundary_data_variables)));

            SpinWeighted<ComplexModalVector, tag::type::type::spin>
                spin_weighted_goldberg_view;
            spin_weighted_goldberg_view.set_data_ref(
                output_goldberg_mode_buffer.data(),
                output_goldberg_mode_buffer.size());
            Spectral::Swsh::libsharp_to_goldberg_modes(
                make_not_null(&spin_weighted_goldberg_view),
                spin_weighted_libsharp_view, file_l_max);

            recorder.append_worldtube_mode_data(
                "/" + dataset_label_for_tag<tag>(), time,
                output_goldberg_mode_buffer, file_l_max,
                tag::type::type::spin == 0);
          });
    }
  }
  // request an appropriate buffer
  auto buffer_updater =
      extraction_radius_in_filename
          ? BondiWorldtubeH5BufferUpdater{filename}
          : BondiWorldtubeH5BufferUpdater{filename, extraction_radius};
  auto serialized_and_deserialized_updater =
      serialize_and_deserialize(buffer_updater);
  size_t time_span_start = 0;
  size_t time_span_end = 0;
  buffer_updater.update_buffers_for_time(
      make_not_null(&coefficients_buffers_from_file),
      make_not_null(&time_span_start), make_not_null(&time_span_end),
      target_time, computation_l_max, interpolator_length, buffer_size);

  Variables<cce_bondi_input_tags> coefficients_buffers_from_serialized{
      (buffer_size + 2 * interpolator_length) * square(computation_l_max + 1)};
  size_t time_span_start_from_serialized = 0;
  size_t time_span_end_from_serialized = 0;
  serialized_and_deserialized_updater.update_buffers_for_time(
      make_not_null(&coefficients_buffers_from_serialized),
      make_not_null(&time_span_start_from_serialized),
      make_not_null(&time_span_end_from_serialized), target_time,
      computation_l_max, interpolator_length, buffer_size);

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  time_span_start = 0;
  time_span_end = 0;
  const auto& time_buffer = buffer_updater.get_time_buffer();
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    CHECK(time_buffer[i] == approx(target_time - 0.1 + 0.01 * i));
  }
  const auto& time_buffer_from_serialized =
      serialized_and_deserialized_updater.get_time_buffer();
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    CHECK(time_buffer_from_serialized[i] ==
          approx(target_time - 0.1 + 0.01 * i));
  }

  const ReducedDummyBufferUpdater dummy_buffer_updater{
      time_buffer, solution,  extraction_radius,
      amplitude,   frequency, computation_l_max};
  dummy_buffer_updater.update_buffers_for_time(
      make_not_null(&expected_coefficients_buffers),
      make_not_null(&time_span_start), make_not_null(&time_span_end),
      target_time, computation_l_max, interpolator_length, buffer_size);

  // this approximation needs to be comparatively loose because it is comparing
  // modes, which tend to have the error set by the scale of the original
  // collocation errors (so, the dominant modes), rather than the scale of the
  // individual mode being examined.
  Approx modal_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
          .scale(1.0);

  // check that the data in the buffer matches the expected analytic data.
  tmpl::for_each<cce_bondi_input_tags>(
      [&expected_coefficients_buffers, &coefficients_buffers_from_file,
       &coefficients_buffers_from_serialized, &modal_approx](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        INFO(db::tag_name<tag>());
        const auto& test_lhs = get<tag>(expected_coefficients_buffers);
        const auto& test_rhs = get<tag>(coefficients_buffers_from_file);
        CHECK_ITERABLE_CUSTOM_APPROX(test_lhs, test_rhs, modal_approx);
        const auto& test_rhs_from_serialized =
            get<tag>(coefficients_buffers_from_serialized);
        CHECK_ITERABLE_CUSTOM_APPROX(test_lhs, test_rhs_from_serialized,
                                     modal_approx);
      });
  CHECK(buffer_updater.get_extraction_radius() == 100.0);
}
}  // namespace

// An increased timeout because this test seems to have high variance in
// duration. It usually finishes within ~3 seconds. The high variance may be due
// to the comparatively high magnitude of disk operations in this test.
// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.ReadBoundaryDataH5",
                  "[Unit][Cce]") {
  Parallel::register_derived_classes_with_charm<
      Cce::WorldtubeBufferUpdater<cce_metric_input_tags>>();
  Parallel::register_derived_classes_with_charm<
      Cce::WorldtubeBufferUpdater<cce_bondi_input_tags>>();
  Parallel::register_derived_classes_with_charm<Cce::WorldtubeDataManager>();
  Parallel::register_derived_classes_with_charm<intrp::SpanInterpolator>();
  MAKE_GENERATOR(gen);
  {
    INFO("Testing buffer updaters");
    test_spec_worldtube_buffer_updater(make_not_null(&gen), true);
    test_spec_worldtube_buffer_updater(make_not_null(&gen), false);
    test_reduced_spec_worldtube_buffer_updater(make_not_null(&gen), true);
    test_reduced_spec_worldtube_buffer_updater(make_not_null(&gen), false);
  }
  {
    INFO("Testing data managers");
    test_data_manager_with_dummy_buffer_updater<MetricWorldtubeDataManager,
                                                DummyBufferUpdater>(
        make_not_null(&gen));
    // with normalization bug applied:
    test_data_manager_with_dummy_buffer_updater<MetricWorldtubeDataManager,
                                                DummyBufferUpdater>(
        make_not_null(&gen), true, true);
    test_data_manager_with_dummy_buffer_updater<MetricWorldtubeDataManager,
                                                DummyBufferUpdater>(
        make_not_null(&gen), false, true);
    test_data_manager_with_dummy_buffer_updater<MetricWorldtubeDataManager,
                                                DummyBufferUpdater>(
        make_not_null(&gen), false, false);
    // check the case for an explicitly provided extraction radius.
    test_data_manager_with_dummy_buffer_updater<MetricWorldtubeDataManager,
                                                DummyBufferUpdater>(
        make_not_null(&gen), false, false, 200.0);
    test_data_manager_with_dummy_buffer_updater<BondiWorldtubeDataManager,
                                                ReducedDummyBufferUpdater>(
        make_not_null(&gen));
  }
}
}  // namespace Cce
