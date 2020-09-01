// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/SpecBoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
MetricWorldtubeDataManager::MetricWorldtubeDataManager(
    std::unique_ptr<WorldtubeBufferUpdater<cce_metric_input_tags>>
        buffer_updater,
    const size_t l_max, const size_t buffer_depth,
    std::unique_ptr<intrp::SpanInterpolator> interpolator) noexcept
    : buffer_updater_{std::move(buffer_updater)},
      l_max_{l_max},
      interpolated_coefficients_{
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)},
      buffer_depth_{buffer_depth},
      interpolator_{std::move(interpolator)} {
  if (UNLIKELY(buffer_updater_->get_time_buffer().size() <
               2 * interpolator_->required_number_of_points_before_and_after() +
                   buffer_depth)) {
    ERROR(
        "The specified buffer updater doesn't have enough time points to "
        "supply the requested interpolation buffer. This almost certainly "
        "indicates that the corresponding file hasn't been created properly, "
        "but might indicate that the `buffer_depth` template parameter is "
        "too large or the specified Interpolator requests too many points");
  }

  const size_t size_of_buffer =
      square(l_max + 1) *
      (buffer_depth +
       2 * interpolator_->required_number_of_points_before_and_after());
  coefficients_buffers_ = Variables<cce_metric_input_tags>{size_of_buffer};
}

bool MetricWorldtubeDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
    const double time) const noexcept {
  if (buffer_updater_->time_is_outside_range(time)) {
    return false;
  }
  buffer_updater_->update_buffers_for_time(
      make_not_null(&coefficients_buffers_), make_not_null(&time_span_start_),
      make_not_null(&time_span_end_), time, l_max_,
      interpolator_->required_number_of_points_before_and_after(),
      buffer_depth_);
  const auto interpolation_time_span = detail::create_span_for_time_value(
      time, 0, interpolator_->required_number_of_points_before_and_after(),
      time_span_start_, time_span_end_, buffer_updater_->get_time_buffer());

  // search through and find the two interpolation points the time point is
  // between. If we can, put the range for the interpolation centered on the
  // desired point. If that can't be done (near the start or the end of the
  // simulation), make the range terminated at the start or end of the cached
  // data and extending for the desired range in the other direction.
  const size_t buffer_span_size = time_span_end_ - time_span_start_;
  const size_t interpolation_span_size =
      interpolation_time_span.second - interpolation_time_span.first;

  const DataVector time_points{
      buffer_updater_->get_time_buffer().data() + interpolation_time_span.first,
      interpolation_span_size};

  auto interpolate_from_column = [&time, &time_points, &buffer_span_size,
                                  &interpolation_time_span,
                                  &interpolation_span_size, this](
                                     auto data, const size_t column) noexcept {
    auto interp_val = interpolator_->interpolate(
        gsl::span<const double>(time_points.data(), time_points.size()),
        gsl::span<const std::complex<double>>(
            data + column * buffer_span_size +
                (interpolation_time_span.first - time_span_start_),
            interpolation_span_size),
        time);
    return interp_val;
  };

  // the ComplexModalVectors should be provided from the buffer_updater_ in
  // 'Goldberg' format, so we iterate over modes and convert to libsharp
  // format.

  // we'll just use this buffer to reference into the actual data to satisfy
  // the swsh interface requirement that the spin-weight be labelled with
  // `SpinWeighted`
  SpinWeighted<ComplexModalVector, 0> spin_weighted_buffer;
  for (const auto& libsharp_mode :
       Spectral::Swsh::cached_coefficients_metadata(l_max_)) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        tmpl::for_each<tmpl::list<Tags::detail::SpatialMetric,
                                  Tags::detail::Dr<Tags::detail::SpatialMetric>,
                                  ::Tags::dt<Tags::detail::SpatialMetric>>>(
            [this, &i, &j, &libsharp_mode, &interpolate_from_column,
             &spin_weighted_buffer](auto tag_v) noexcept {
              using tag = typename decltype(tag_v)::type;
              spin_weighted_buffer.set_data_ref(
                  get<tag>(interpolated_coefficients_).get(i, j).data(),
                  Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_));
              Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
                  libsharp_mode, make_not_null(&spin_weighted_buffer), 0,
                  interpolate_from_column(
                      get<tag>(coefficients_buffers_).get(i, j).data(),
                      Spectral::Swsh::goldberg_mode_index(
                          l_max_, libsharp_mode.l,
                          static_cast<int>(libsharp_mode.m))),
                  interpolate_from_column(
                      get<tag>(coefficients_buffers_).get(i, j).data(),
                      Spectral::Swsh::goldberg_mode_index(
                          l_max_, libsharp_mode.l,
                          -static_cast<int>(libsharp_mode.m))));
            });
      }
      tmpl::for_each<
          tmpl::list<Tags::detail::Shift, Tags::detail::Dr<Tags::detail::Shift>,
                     ::Tags::dt<Tags::detail::Shift>>>(
          [this, &i, &libsharp_mode, &interpolate_from_column,
           &spin_weighted_buffer](auto tag_v) noexcept {
            using tag = typename decltype(tag_v)::type;
            spin_weighted_buffer.set_data_ref(
                get<tag>(interpolated_coefficients_).get(i).data(),
                Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_));
            Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
                libsharp_mode, make_not_null(&spin_weighted_buffer), 0,
                interpolate_from_column(
                    get<tag>(coefficients_buffers_).get(i).data(),
                    Spectral::Swsh::goldberg_mode_index(
                        l_max_, libsharp_mode.l,
                        static_cast<int>(libsharp_mode.m))),
                interpolate_from_column(
                    get<tag>(coefficients_buffers_).get(i).data(),
                    Spectral::Swsh::goldberg_mode_index(
                        l_max_, libsharp_mode.l,
                        -static_cast<int>(libsharp_mode.m))));
          });
    }
    tmpl::for_each<
        tmpl::list<Tags::detail::Lapse, Tags::detail::Dr<Tags::detail::Lapse>,
                   ::Tags::dt<Tags::detail::Lapse>>>([this, &libsharp_mode,
                                                      &interpolate_from_column,
                                                      &spin_weighted_buffer](
                                                         auto tag_v) noexcept {
      using tag = typename decltype(tag_v)::type;
      spin_weighted_buffer.set_data_ref(
          get(get<tag>(interpolated_coefficients_)).data(),
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_));
      Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
          libsharp_mode, make_not_null(&spin_weighted_buffer), 0,
          interpolate_from_column(
              get(get<tag>(coefficients_buffers_)).data(),
              Spectral::Swsh::goldberg_mode_index(
                  l_max_, libsharp_mode.l, static_cast<int>(libsharp_mode.m))),
          interpolate_from_column(get(get<tag>(coefficients_buffers_)).data(),
                                  Spectral::Swsh::goldberg_mode_index(
                                      l_max_, libsharp_mode.l,
                                      -static_cast<int>(libsharp_mode.m))));
    });
  }
  // At this point, we have a collection of 9 tensors of libsharp
  // coefficients. This is what the boundary data calculation utility takes
  // as an input, so we now hand off the control flow to the boundary and
  // gauge transform utility
  if (buffer_updater_->radial_derivatives_need_renormalization()) {
    create_bondi_boundary_data_from_unnormalized_spec_modes(
        boundary_data_variables,
        get<Tags::detail::SpatialMetric>(interpolated_coefficients_),
        get<::Tags::dt<Tags::detail::SpatialMetric>>(
            interpolated_coefficients_),
        get<Tags::detail::Dr<Tags::detail::SpatialMetric>>(
            interpolated_coefficients_),
        get<Tags::detail::Shift>(interpolated_coefficients_),
        get<::Tags::dt<Tags::detail::Shift>>(interpolated_coefficients_),
        get<Tags::detail::Dr<Tags::detail::Shift>>(interpolated_coefficients_),
        get<Tags::detail::Lapse>(interpolated_coefficients_),
        get<::Tags::dt<Tags::detail::Lapse>>(interpolated_coefficients_),
        get<Tags::detail::Dr<Tags::detail::Lapse>>(interpolated_coefficients_),
        buffer_updater_->get_extraction_radius(), l_max_);
  } else {
    create_bondi_boundary_data(
        boundary_data_variables,
        get<Tags::detail::SpatialMetric>(interpolated_coefficients_),
        get<::Tags::dt<Tags::detail::SpatialMetric>>(
            interpolated_coefficients_),
        get<Tags::detail::Dr<Tags::detail::SpatialMetric>>(
            interpolated_coefficients_),
        get<Tags::detail::Shift>(interpolated_coefficients_),
        get<::Tags::dt<Tags::detail::Shift>>(interpolated_coefficients_),
        get<Tags::detail::Dr<Tags::detail::Shift>>(interpolated_coefficients_),
        get<Tags::detail::Lapse>(interpolated_coefficients_),
        get<::Tags::dt<Tags::detail::Lapse>>(interpolated_coefficients_),
        get<Tags::detail::Dr<Tags::detail::Lapse>>(interpolated_coefficients_),
        buffer_updater_->get_extraction_radius(), l_max_);
  }
  return true;
}

std::unique_ptr<WorldtubeDataManager> MetricWorldtubeDataManager::get_clone()
    const noexcept {
  return std::make_unique<MetricWorldtubeDataManager>(
      buffer_updater_->get_clone(), l_max_, buffer_depth_,
      interpolator_->get_clone());
}

std::pair<size_t, size_t> MetricWorldtubeDataManager::get_time_span()
    const noexcept {
  return std::make_pair(time_span_start_, time_span_end_);
}

void MetricWorldtubeDataManager::pup(PUP::er& p) noexcept {
  p | buffer_updater_;
  p | time_span_start_;
  p | time_span_end_;
  p | l_max_;
  p | buffer_depth_;
  p | interpolator_;
  if (p.isUnpacking()) {
    time_span_start_ = 0;
    time_span_end_ = 0;
    const size_t size_of_buffer =
        square(l_max_ + 1) *
        (buffer_depth_ +
         2 * interpolator_->required_number_of_points_before_and_after());
    coefficients_buffers_ = Variables<cce_metric_input_tags>{size_of_buffer};
    interpolated_coefficients_ = Variables<cce_metric_input_tags>{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_)};
  }
}

BondiWorldtubeDataManager::BondiWorldtubeDataManager(
    std::unique_ptr<WorldtubeBufferUpdater<cce_bondi_input_tags>>
        buffer_updater,
    const size_t l_max, const size_t buffer_depth,
    std::unique_ptr<intrp::SpanInterpolator> interpolator) noexcept
    : buffer_updater_{std::move(buffer_updater)},
      l_max_{l_max},
      interpolated_coefficients_{
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)},
      buffer_depth_{buffer_depth},
      interpolator_{std::move(interpolator)} {
  if (UNLIKELY(buffer_updater_->get_time_buffer().size() <
               2 * interpolator_->required_number_of_points_before_and_after() +
                   buffer_depth)) {
    ERROR(
        "The specified buffer updater doesn't have enough time points to "
        "supply the requested interpolation buffer. This almost certainly "
        "indicates that the corresponding file hasn't been created properly, "
        "but might indicate that the `buffer_depth` template parameter is "
        "too large or the specified SpanInterpolator requests too many "
        "points");
  }
  coefficients_buffers_ = Variables<cce_bondi_input_tags>{
      square(l_max + 1) *
      (buffer_depth +
       2 * interpolator_->required_number_of_points_before_and_after())};
}

bool BondiWorldtubeDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
    const double time) const noexcept {
  if (buffer_updater_->time_is_outside_range(time)) {
    return false;
  }
  buffer_updater_->update_buffers_for_time(
      make_not_null(&coefficients_buffers_), make_not_null(&time_span_start_),
      make_not_null(&time_span_end_), time, l_max_,
      interpolator_->required_number_of_points_before_and_after(),
      buffer_depth_);
  auto interpolation_time_span = detail::create_span_for_time_value(
      time, 0, interpolator_->required_number_of_points_before_and_after(),
      time_span_start_, time_span_end_, buffer_updater_->get_time_buffer());

  // search through and find the two interpolation points the time point is
  // between. If we can, put the range for the interpolation centered on the
  // desired point. If that can't be done (near the start or the end of the
  // simulation), make the range terminated at the start or end of the cached
  // data and extending for the desired range in the other direction.
  const size_t buffer_span_size = time_span_end_ - time_span_start_;
  const size_t interpolation_span_size =
      interpolation_time_span.second - interpolation_time_span.first;

  DataVector time_points{
      buffer_updater_->get_time_buffer().data() + interpolation_time_span.first,
      interpolation_span_size};

  auto interpolate_from_column =
      [&time, &time_points, &buffer_span_size, &interpolation_time_span,
       &interpolation_span_size, this](auto data, size_t column) {
        const auto interp_val = interpolator_->interpolate(
            gsl::span<const double>(time_points.data(), time_points.size()),
            gsl::span<const std::complex<double>>(
                data + column * (buffer_span_size) +
                    (interpolation_time_span.first - time_span_start_),
                interpolation_span_size),
            time);
        return interp_val;
      };

  // the ComplexModalVectors should be provided from the buffer_updater_ in
  // 'Goldberg' format, so we iterate over modes and convert to libsharp
  // format.
  for (const auto& libsharp_mode :
       Spectral::Swsh::cached_coefficients_metadata(l_max_)) {
    tmpl::for_each<cce_bondi_input_tags>(
        [this, &libsharp_mode, &interpolate_from_column](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
              libsharp_mode,
              make_not_null(&get(get<tag>(interpolated_coefficients_))), 0,
              interpolate_from_column(
                  get(get<tag>(coefficients_buffers_)).data().data(),
                  Spectral::Swsh::goldberg_mode_index(
                      l_max_, libsharp_mode.l,
                      static_cast<int>(libsharp_mode.m))),
              interpolate_from_column(
                  get(get<tag>(coefficients_buffers_)).data().data(),
                  Spectral::Swsh::goldberg_mode_index(
                      l_max_, libsharp_mode.l,
                      -static_cast<int>(libsharp_mode.m))));
        });
  }
  // just inverse transform the 'direct' tags
  tmpl::for_each<tmpl::transform<cce_bondi_input_tags,
                                 tmpl::bind<db::remove_tag_prefix, tmpl::_1>>>(
      [this, &boundary_data_variables](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        Spectral::Swsh::inverse_swsh_transform(
            l_max_, 1,
            make_not_null(
                &get(get<Tags::BoundaryValue<tag>>(*boundary_data_variables))),
            get(get<Spectral::Swsh::Tags::SwshTransform<tag>>(
                interpolated_coefficients_)));
      });
  const auto& du_r = get(get<Tags::BoundaryValue<Tags::Du<Tags::BondiR>>>(
      *boundary_data_variables));
  const auto& bondi_r =
      get(get<Tags::BoundaryValue<Tags::BondiR>>(*boundary_data_variables));

  get(get<Tags::BoundaryValue<Tags::DuRDividedByR>>(*boundary_data_variables)) =
      du_r / bondi_r;

  // there's only a couple of tags desired by the core computation that aren't
  // stored in the 'reduced' format, so we perform the remaining computation
  // in-line here.
  const auto& du_bondi_j = get(get<Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>>(
      *boundary_data_variables));
  const auto& dr_bondi_j = get(get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(
      *boundary_data_variables));
  get(get<Tags::BoundaryValue<Tags::BondiH>>(*boundary_data_variables)) =
      du_bondi_j + du_r * dr_bondi_j;

  const auto& bondi_j =
      get(get<Tags::BoundaryValue<Tags::BondiJ>>(*boundary_data_variables));
  const auto& bondi_beta =
      get(get<Tags::BoundaryValue<Tags::BondiBeta>>(*boundary_data_variables));
  const auto& bondi_q =
      get(get<Tags::BoundaryValue<Tags::BondiQ>>(*boundary_data_variables));
  const auto& bondi_k = sqrt(1.0 + bondi_j * conj(bondi_j));
  get(get<Tags::BoundaryValue<Tags::Dr<Tags::BondiU>>>(
      *boundary_data_variables)) =
      exp(2.0 * bondi_beta.data()) / square(bondi_r.data()) *
      (bondi_k.data() * bondi_q.data() - bondi_j.data() * conj(bondi_q.data()));
  return true;
}

std::unique_ptr<WorldtubeDataManager> BondiWorldtubeDataManager::get_clone()
    const noexcept {
  return std::make_unique<BondiWorldtubeDataManager>(
      buffer_updater_->get_clone(), l_max_, buffer_depth_,
      interpolator_->get_clone());
}

std::pair<size_t, size_t> BondiWorldtubeDataManager::get_time_span()
    const noexcept {
  return std::make_pair(time_span_start_, time_span_end_);
}

void BondiWorldtubeDataManager::pup(PUP::er& p) noexcept {
  p | buffer_updater_;
  p | time_span_start_;
  p | time_span_end_;
  p | l_max_;
  p | buffer_depth_;
  p | interpolator_;
  if (p.isUnpacking()) {
    time_span_start_ = 0;
    time_span_end_ = 0;
    const size_t size_of_buffer =
        square(l_max_ + 1) *
        (buffer_depth_ +
         2 * interpolator_->required_number_of_points_before_and_after());
    coefficients_buffers_ = Variables<cce_bondi_input_tags>{size_of_buffer};
    interpolated_coefficients_ = Variables<cce_bondi_input_tags>{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_)};
  }
}

PnWorldtubeDataManager::PnWorldtubeDataManager(
    std::unique_ptr<ModeSetBoundaryH5BufferUpdater> buffer_updater,
    const size_t l_max, const size_t buffer_depth,
    const double extraction_radius,
    std::unique_ptr<intrp::SpanInterpolator> interpolator) noexcept
    : buffer_updater_{std::move(buffer_updater)},
      l_max_{l_max},
      extraction_radius_{extraction_radius},
      interpolated_j_coefficients_{
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)},
      interpolated_h_coefficients_{
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)},
      buffer_depth_{buffer_depth},
      interpolator_{std::move(interpolator)} {
  if (UNLIKELY(buffer_updater_->get_time_buffer().size() <
               2 * interpolator_->required_number_of_points_before_and_after() +
                   buffer_depth)) {
    ERROR(
        "The specified buffer updater doesn't have enough time points to "
        "supply the requested interpolation buffer. This almost certainly "
        "indicates that the corresponding file hasn't been created properly, "
        "but might indicate that the `buffer_depth` template parameter is "
        "too large or the specified SpanInterpolator requests too many "
        "points");
  }
  coefficients_buffers_ = ComplexModalVector{
      square(l_max + 1) *
      (buffer_depth +
       2 * interpolator_->required_number_of_points_before_and_after())};
}

void PnWorldtubeDataManager::pup(PUP::er& p) noexcept {
  p | buffer_updater_;
  p | time_span_start_;
  p | time_span_end_;
  p | l_max_;
  p | extraction_radius_;
  p | buffer_depth_;
  p | interpolator_;
  if (p.isUnpacking()) {
    time_span_start_ = 0;
    time_span_end_ = 0;
    const size_t size_of_buffer =
        square(l_max_ + 1) *
        (buffer_depth_ +
         2 * interpolator_->required_number_of_points_before_and_after());
    coefficients_buffers_ = ComplexModalVector{size_of_buffer};
    interpolated_j_coefficients_ = ComplexModalVector{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_)};
    interpolated_h_coefficients_ = ComplexModalVector{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_)};
  }
}

std::unique_ptr<WorldtubeDataManager> PnWorldtubeDataManager::get_clone()
    const noexcept {
  return std::make_unique<PnWorldtubeDataManager>(
      buffer_updater_->get_clone(), l_max_, buffer_depth_, extraction_radius_,
      interpolator_->get_clone());
}

bool PnWorldtubeDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
    const double time) const noexcept {
  if (buffer_updater_->time_is_outside_range(time)) {
    return false;
  }
  buffer_updater_->update_buffer_for_time(
      make_not_null(&coefficients_buffers_), make_not_null(&time_span_start_),
      make_not_null(&time_span_end_), time, l_max_,
      interpolator_->required_number_of_points_before_and_after(),
      buffer_depth_);
  auto interpolation_time_span = detail::create_span_for_time_value(
      time, 0, interpolator_->required_number_of_points_before_and_after(),
      time_span_start_, time_span_end_, buffer_updater_->get_time_buffer());

  // search through and find the two interpolation points the time point is
  // between. If we can, put the range for the interpolation centered on the
  // desired point. If that can't be done (near the start or the end of the
  // simulation), make the range terminated at the start or end of the cached
  // data and extending for the desired range in the other direction.
  const size_t buffer_span_size = time_span_end_ - time_span_start_;
  const size_t interpolation_span_size =
      interpolation_time_span.second - interpolation_time_span.first;

  DataVector time_points{
      buffer_updater_->get_time_buffer().data() + interpolation_time_span.first,
      interpolation_span_size};

  auto interpolate_from_column =
      [&time_points, &buffer_span_size, &interpolation_time_span,
       &interpolation_span_size,
       this](auto data, size_t column, const double local_time) {
        const auto interp_val = interpolator_->interpolate(
            gsl::span<const double>(time_points.data(), time_points.size()),
            gsl::span<const std::complex<double>>(
                data + column * (buffer_span_size) +
                    (interpolation_time_span.first - time_span_start_),
                interpolation_span_size),
            local_time);
        return interp_val;
      };

  // optimization note: consider buffers for these as well. They are small
  // allocations, but could probably be performed at construction
  const DataVector time_collocation_points =
      Spectral::collocation_points<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto>(
          interpolation_span_size);

  ComplexModalVector plus_m_time_collocation_values{interpolation_span_size};
  ComplexModalVector minus_m_time_collocation_values{interpolation_span_size};

  const auto affine_transform_time_to_GL =
      [&time_points,
       &interpolation_span_size](const double local_time) noexcept {
        return (local_time - time_points[0]) /
                   (time_points[interpolation_span_size - 1] - time_points[0]) -
               1.0;
      };
  const auto affine_transform_GL_to_time =
      [&time_points,
       &interpolation_span_size](const double collocation_point) noexcept {
        return (collocation_point + 1.0) *
                   (time_points[interpolation_span_size - 1] - time_points[0]) +
               time_points[0];
      };

  const Matrix interpolate_derivative_matrix =
      Spectral::interpolation_matrix<Spectral::Basis::Legendre,
                                     Spectral::Quadrature::GaussLobatto>(
          interpolation_span_size,
          DataVector{1, affine_transform_time_to_GL(time)}) *
      Spectral::differentiation_matrix<Spectral::Basis::Legendre,
                                       Spectral::Quadrature::GaussLobatto>(
          interpolation_span_size);

  // the ComplexModalVectors should be provided from the buffer_updater_ in
  // 'Goldberg' format, so we iterate over modes and convert to libsharp
  // format.
  for (const auto& libsharp_mode :
       Spectral::Swsh::cached_coefficients_metadata(l_max_)) {
    // optimization note: consider a larger buffer and apply_matrices
    for (size_t i = 0; i < time_collocation_points.size(); ++i) {
      plus_m_time_collocation_values[i] = interpolate_from_column(
          coefficients_buffers_.data(),
          Spectral::Swsh::goldberg_mode_index(
              l_max_, libsharp_mode.l, static_cast<int>(libsharp_mode.m)),
          affine_transform_GL_to_time(time_collocation_points[i]));
      minus_m_time_collocation_values[i] = interpolate_from_column(
          coefficients_buffers_.data(),
          Spectral::Swsh::goldberg_mode_index(
              l_max_, libsharp_mode.l, -static_cast<int>(libsharp_mode.m)),
          affine_transform_GL_to_time(time_collocation_points[i]));
    }
    const auto plus_m_h_mode =
        (interpolate_derivative_matrix * plus_m_time_collocation_values)[0] /
        (time_points[interpolation_span_size - 1] - time_points[0]);
    const auto minus_m_h_mode =
        (interpolate_derivative_matrix * plus_m_time_collocation_values)[0] /
        (time_points[interpolation_span_size - 1] - time_points[0]);

    SpinWeighted<ComplexModalVector, 2> spin_2_view;
    spin_2_view.set_data_ref(interpolated_h_coefficients_.data(),
                             interpolated_h_coefficients_.size());

    Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
        libsharp_mode, make_not_null(&spin_2_view), 0, plus_m_h_mode,
        minus_m_h_mode);

    spin_2_view.set_data_ref(interpolated_j_coefficients_.data(),
                             interpolated_j_coefficients_.size());
    Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
        libsharp_mode, make_not_null(&spin_2_view), 0,
        interpolate_from_column(
            coefficients_buffers_.data(),
            Spectral::Swsh::goldberg_mode_index(
                l_max_, libsharp_mode.l, static_cast<int>(libsharp_mode.m)),
            time),
        interpolate_from_column(
            coefficients_buffers_.data(),
            Spectral::Swsh::goldberg_mode_index(
                l_max_, libsharp_mode.l, -static_cast<int>(libsharp_mode.m)),
            time));
  }

  const size_t size = Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  Variables<tmpl::list<Tags::detail::CosPhi, Tags::detail::CosTheta,
                       Tags::detail::SinPhi, Tags::detail::SinTheta,
                       Tags::detail::CartesianCoordinates,
                       Tags::detail::CartesianToSphericalJacobian,
                       Tags::detail::InverseCartesianToSphericalJacobian>>
      computation_variables{size};

  auto& cos_phi = get<Tags::detail::CosPhi>(computation_variables);
  auto& cos_theta = get<Tags::detail::CosTheta>(computation_variables);
  auto& sin_phi = get<Tags::detail::SinPhi>(computation_variables);
  auto& sin_theta = get<Tags::detail::SinTheta>(computation_variables);
  trigonometric_functions_on_swsh_collocation(
      make_not_null(&cos_phi), make_not_null(&cos_theta),
      make_not_null(&sin_phi), make_not_null(&sin_theta), l_max);

  auto& cartesian_coords =
      get<Tags::detail::CartesianCoordinates>(computation_variables);
  auto& cartesian_to_spherical_jacobian =
      get<Tags::detail::CartesianToSphericalJacobian>(computation_variables);
  auto& inverse_cartesian_to_spherical_jacobian =
      get<Tags::detail::InverseCartesianToSphericalJacobian>(
          computation_variables);
  cartesian_to_spherical_coordinates_and_jacobians(
      make_not_null(&cartesian_coords),
      make_not_null(&cartesian_to_spherical_jacobian),
      make_not_null(&inverse_cartesian_to_spherical_jacobian), cos_phi,
      cos_theta, sin_phi, sin_theta, extraction_radius);


  // The (pfaffian) components of the angular part are:
  //     [Re(J)   Im(J)]
  // r^2 [Im(J)  -Re(J)]

  SpinWeighted<ComplexModalVector, 2> spin_weighted_view;
  spin_weighted_view.set_data_ref(interpolated_j_coefficients_.data(),
                                  interpolated_j_coefficients_.size());
  // using J as a temporary cache even though we don't really want this as J
  Spectral::Swsh::inverse_swsh_transform(
      l_max_, 1,
      make_not_null(&get(
          get<Tags::BoundaryValue<Tags::BondiJ>>(*boundary_data_variables))),
      spin_weighted_view);

  get(get<Tags::BoundaryValue<Tags::BondiJ>>(*boundary_data_variables)) =
      get(get<Tags::BoundaryValue<Tags::BondiJ>>(*boundary_data_variables)) /
      extraction_radius_;

  spin_weighted_view.set_data_ref(interpolated_h_coefficients_.data(),
                                  interpolated_h_coefficients_.size());
  Spectral::Swsh::inverse_swsh_transform(
      l_max_, 1,
      make_not_null(&get(
          get<Tags::BoundaryValue<Tags::BondiH>>(*boundary_data_variables))),
      spin_weighted_view);
  get(get<Tags::BoundaryValue<Tags::BondiH>>(*boundary_data_variables)) =
      get(get<Tags::BoundaryValue<Tags::BondiH>>(*boundary_data_variables)) /
      extraction_radius_;

  if (metric_collocation_.number_of_independent_components() != size) {
    metric_collocation_ = Variables<cce_metric_collocation_tags>{size};
  }

  for(size_t i = 0; i <3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      const auto& j_collocation =
          get(get<Tags::BoundaryValue<Tags::BondiJ>>(*boundary_data_variables))
              .data();
      const auto& h_collocation =
          get(get<Tags::BoundaryValue<Tags::BondiH>>(*boundary_data_variables))
              .data();
      // TODO modes instead.
      get<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>(
          metric_collocation_)
          .get(i, j) =
          square(inverse_cartesian_to_spherical_jacobian.get(i, 0)) *
              real(j_collocation) +
          2.0 * inverse_cartesian_to_spherical_jacobian.get(i, 0) *
              inverse_cartesian_to_spherical_jacobian.get(i, 1) *
              imag(j_collocation) -
          square(inverse_cartesian_to_spherical_jacobian.get(i, 1)) *
              real(j_collocation);
      get<Tags::Dr<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>>(
          metric_collocation_)
          .get(i, j) =
          -get<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>(
               metric_collocation_)
               .get(i, j) /
          extraction_radius_;
      get<::Tags::dt<
          gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>>(
          metric_collocation_)
          .get(i, j) =
          square(inverse_cartesian_to_spherical_jacobian.get(i, 0)) *
              real(h_collocation) +
          2.0 * inverse_cartesian_to_spherical_jacobian.get(i, 0) *
              inverse_cartesian_to_spherical_jacobian.get(i, 1) *
              imag(h_collocation) -
          square(inverse_cartesian_to_spherical_jacobian.get(i, 1)) *
              real(h_collocation);
    }
    get<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>(metric_collocation_)
        .get(i) = 0.0;
    get<Tags::Dr<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>(
        metric_collocation_)
        .get(i) = 0.0;
    get<::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>(
        metric_collocation_)
        .get(i) = 0.0;
  }
  // this will need to be modified in the future pending Keefe's new worldtube
  // data -- that'll be scalar so I think we don't need to do as much caching
  get<gr::Tags::Lapse<3, ::Frame::Inertial, DataVector>>(metric_collocation_)
      .get(i) = -1.0;
  get<Tags::Dr<gr::Tags::Lapse<3, ::Frame::Inertial, DataVector>>>(
      metric_collocation_)
      .get(i) = 0.0;
  get<::Tags::dt<gr::Tags::Lapse<3, ::Frame::Inertial, DataVector>>>(
      metric_collocation_)
      .get(i) = 0.0;

  create_bondi_boundary_data(
      boundary_data_variables,
      get<Tags::detail::SpatialMetric>(interpolated_coefficients_),
      get<::Tags::dt<Tags::detail::SpatialMetric>>(interpolated_coefficients_),
      get<Tags::detail::Dr<Tags::detail::SpatialMetric>>(
          interpolated_coefficients_),
      get<Tags::detail::Shift>(interpolated_coefficients_),
      get<::Tags::dt<Tags::detail::Shift>>(interpolated_coefficients_),
      get<Tags::detail::Dr<Tags::detail::Shift>>(interpolated_coefficients_),
      get<Tags::detail::Lapse>(interpolated_coefficients_),
      get<::Tags::dt<Tags::detail::Lapse>>(interpolated_coefficients_),
      get<Tags::detail::Dr<Tags::detail::Lapse>>(interpolated_coefficients_),
      buffer_updater_->get_extraction_radius(), l_max_);

  return true;
}

/// \cond
PUP::able::PUP_ID MetricWorldtubeDataManager::my_PUP_ID = 0;
PUP::able::PUP_ID BondiWorldtubeDataManager::my_PUP_ID = 0;
PUP::able::PUP_ID PnWorldtubeDataManager::my_PUP_ID = 0;
/// \endcond
}  // namespace Cce
