// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include <boost/iterator/zip_iterator.hpp>
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#include "ApparentHorizons/SpherepackIterator.hpp"
#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Version.hpp"

#include "Utilities/TmplDebugging.hpp"

namespace Cce {

// For the file-reading, we'll need several utilities to get it from the file in
// the formats used into something we can actually start doing physics with.

// - Times
//   - Ideally, we want to have an nth-order interpolator for the time-series
//   data
//    Input: a data structure with the values at times nearby the target time,
//    and a list of corresponding times
//    Output: a ComplexModalVector for the desired value interpolated to the
//    desired points

//   - We also need a manager function for these times.
//    Input: a data structure with values at several times, an associated Dat
//    object, and an associated 'current time'
//    Output: the adjusted data structure with new values and times, and the
//    next time at which it will need to fetch more data. Should simply output
//    the 'next time' if it doesn't need to fetch yet.
//    There's a bit of a balance to be struck between how much to store and how
//    long between reads. A first pass will just have a template parameter for
//    how many file time steps should be available for interpolation, and it
//    just figures out how much data it needs
//
//  - Data handling
//    - Once we have the interpolated data, we need a utility that can take the
//    results and put them into tensors.
//    Input: a ComplexModalVector with coefficients, and a tensor with indices
//    of where to put it
//    Output: the tensor updated with the desired component
//
//  - This should all be aggregated into a big object
//    Constructed from: a string for the cce filename
//    Member variables: extraction radius
//                      several arrays of .dat objects
//                      several arrays of stored time blocks
//    Destructor: closes the file
//    Member functions:
//      populate_hypersurface_data(Variables*, double time)
//      (other ones above)
//
// - Also want a generic interpolator object
//   - abstact base class

// Abstract base class for interpolators
class Interpolator {
  static constexpr size_t required_number_of_points_before_and_after = 0;
  static double interpolate(const DataVector& points, const DataVector& values,
                            const double target) noexcept;
};

// iterpolates linearly
class LinearInterpolator : Interpolator {
 public:
  static constexpr size_t required_number_of_points_before_and_after = 1;
  static double interpolate(const DataVector& points, const DataVector& values,
                            const double target) {
    return values[0] + (values[1] - values[0]) / (points[1] - points[0]) *
                           (target - points[0]);
  }
};

// copied from SpEC for diagnostics
class CubicInterpolator : Interpolator {
 public:
  static constexpr size_t required_number_of_points_before_and_after = 2;
  static double interpolate(const DataVector& points, const DataVector& values,
                            const double T) {
    double t0 = points[0];
    double t1 = points[1];
    double t2 = points[2];
    double t3 = points[3];

    double d0 = values[0];
    double d1 = values[1];
    double d2 = values[2];
    double d3 = values[3];

    return (-((T - t2) *
              (d3 * (T - t0) * (T - t1) * (t0 - t1) * (t0 - t2) * (t1 - t2) +
               (d1 * (T - t0) * (t0 - t2) * (t0 - t3) -
                d0 * (T - t1) * (t1 - t2) * (t1 - t3)) *
                   (T - t3) * (t2 - t3))) +
            d2 * (T - t0) * (T - t1) * (t0 - t1) * (T - t3) * (t0 - t3) *
                (t1 - t3)) /
           ((t0 - t1) * (t0 - t2) * (t1 - t2) * (t0 - t3) * (t1 - t3) *
            (t2 - t3));
  }
};

// TODO: explore the possibility of moving this number to runtime and maybe
// making it non-fixed?

// A barycentric interpolator with a fixed, compile-time number of points. This
// allows for an easier time ensuring the right amount of data is loaded in the
// data manager
template <size_t N>
class BarycentricInterpolator : Interpolator {
 public:
  static constexpr size_t required_number_of_points_before_and_after = N;
  static double interpolate(const DataVector& points, const DataVector& values,
                            const double target) {
    if (UNLIKELY(points.size() < 2 * N)) {
      ERROR("provided independent values for interpolation too small.");
    }
    if (UNLIKELY(values.size() < 2 * N)) {
      ERROR("provided dependent values for interpolation too small.");
    }
    boost::math::barycentric_rational<double> interpolant(
        points.data(), values.data(), 2 * N, N);
    return interpolant(target);
  }
};

namespace InputTags {
struct SpatialMetric {
  using type = tnsr::ii</*ModalVector*/ DataVector, 3>;
  static std::string name() noexcept { return "SpatialMetric"; };
};

struct Shift {
  using type = tnsr::I</*ModalVector*/ DataVector, 3>;
  static std::string name() noexcept { return "Shift"; };
};

struct Lapse {
  using type = Scalar</*ModalVector*/ DataVector>;
  static std::string name() noexcept { return "Lapse"; };
};

template <typename Tag>
struct Dr {
  using type = typename Tag::type;
  static std::string name() noexcept { return "Dr(" + Tag::name() + ")"; }
};

template <typename Tag>
struct Dt {
  using type = typename Tag::type;
  static std::string name() noexcept { return "Dt(" + Tag::name() + ")"; }
};

template <typename Tag>
struct DataSet {
  using type = std::string;
  static std::string name() noexcept { return "DataSet(" + Tag::name() + ")"; }
};
}  // namespace InputTags



using cce_input_tags =
    tmpl::list<InputTags::SpatialMetric,
               InputTags::Dr<InputTags::SpatialMetric>,
               InputTags::Dt<InputTags::SpatialMetric>, InputTags::Shift,
               InputTags::Dr<InputTags::Shift>, InputTags::Dt<InputTags::Shift>,
               InputTags::Lapse, InputTags::Dr<InputTags::Lapse>,
               InputTags::Dt<InputTags::Lapse>>;

// Caches a target amount of data provided by the Cauchy simulation, and when
// ready provides the interpolated data to a running characteristic evolution
template <class Interp>
class CceCauchyBoundaryDataManager {
 public:
  CceCauchyBoundaryDataManager(const double extraction_radius,
                               const size_t cce_l_max,
                               const size_t spherepack_l_max,
                               const size_t target_buffer_pad) noexcept
      : extraction_radius_{extraction_radius},
        spherical_harmonic_{spherepack_l_max, spherepack_l_max},
        l_max_{cce_l_max},
        spherepack_l_max_{spherepack_l_max},
        time_span_start_{0},
        time_span_end_{0},
        target_buffer_pad_{target_buffer_pad} {}
  // This currently assumes that it will receive data in ascending time order
  template <typename TagList>
  void append_cauchy_worldtube_data_to_buffer(
      const Variables<TagList>& boundary_coefficients,
      const double time) noexcept {
    time_buffer_.push_back(time);
    Variables<cce_input_tags> to_add{boundary_coefficients.size()};
    tmpl::for_each<cce_input_tags>([&to_add, &boundary_coefficients](auto x) {
      using tag = typename decltype(x)::type;
      get<tag>(to_add) = get<tag>(boundary_coefficients);
    });
    coefficients_buffers_.push_back(std::move(to_add));
  }

  // This decides based on the target buffer size whether the manager has
  // obtained enough data to provide the extraction a good worldtube boundary
  // data set. This will always return false when the time is outside the
  // current buffer range.
  bool ready_to_provide_hypersurface_boundary_data(
      const double time, const bool no_more_steps) noexcept {
    if(time < time_buffer_.front() or time > time_buffer_.back()) {
      return false;
    }
    size_t i = 0;
    for(auto time_in_buffer : time_buffer_) {
      if(time < time_in_buffer) {
        break;
      }
      ++i;
    }
    if (not no_more_steps and (time_buffer_.size() - i) < target_buffer_pad) {
      return false;
    }
    return true;
  }

  // the boundary data is assumed to be passed in as ylmspherepack coefficients.
  // in this version, the calling code is expected to check
  // `ready_to_provide_hypersurface_boundary_data`. If this function is called
  // when not ready, the interpolator will error.
  template <typename TagList>
  void populate_hypersurface_boundary_data(
      const gsl::not_null<Variables<TagList>*> boundary_data,
      const double time) noexcept {
    // check for fencepost errors here...
    auto interpolation_time_begin = time_buffer_.begin();
    auto interpolation_time_mid = time_buffer_.begin();
    auto interpolation_time_end = time_buffer_.begin();
    auto interpolation_coefficients_begin = time_buffer_.begin();
    size_t size_so_far = 0;
    while(*interpolation_time_mid < time) {
      if (size_so_far > 2 * target_buffer_pad) {
        ++interpolation_time_begin;
        ++iterpolation_coefficients_begin;
      }
      ++size_so_far;
      ++interpolation_time_mid;
      ++interpolation_time_end;
    }
    size_t start_of_buffer_to_mid = size_so_far;
    size_t mid_to_end = 0;
    while (mid_to_end + min(start_of_buffer_to_mid, target_buffer_pad) <
               2 * target_buffer_pad and
           interpolation_time_end != time_buffer_.end()) {
      if(size_so_far > 2 * target_buffer_pad) {
        ++interpolation_time_begin;
        ++interpolation_coefficients_begin;
      }
      ++size_so_far;
      ++interpolation_time_end;
    }

    DataVector interpolation_times{2 * target_buffer_pad};
    size_t index = 0;
    for(const auto& time_in_buffer : time_buffer_) {
      interpolation_times[index] = time_in_buffer;
      ++index;
    }

    for (size_t offset = 0; offset < boundary_data.number_of_grid_points();
         ++offset) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          tmpl::for_each<tmpl::list<InputTags::SpatialMetric,
                                    InputTags::Dr<InputTags::SpatialMetric>,
                                    InputTags::Dt<InputTags::SpatialMetric>>>(
              [this, &i, &j, &offset, &interpolation_times](auto x) {
                using tag = typename decltype(x)::type;
                DataVector interpolation_data{2 * target_buffer_pad};
                size_t index = 0;
                for (const auto& variables_in_buffer : coefficients_buffer_) {
                  interpolation_data[index] =
                      get<tag>(variables_in_buffer).get(i, j)[offset];
                  ++index;
                }
                get<tag>(interpolated_coefficients).get(i, j)[offset] =
                    Interp::interpolate(interpolation_times, interpolation_data,
                                        time);
              });
        }
        tmpl::for_each<
            tmpl::list<InputTags::Shift, InputTags::Dr<InputTags::Shift>,
                       InputTags::Dt<InputTags::Shift>>>(
            [this, &i, &ylm_iter, &ylm_spherepack_prefactor,
             &interpolate_from_column, &column](auto x) {
              using tag = typename decltype(x)::type;
              DataVector interpolation_data{2 * target_buffer_pad};
              size_t index = 0;
              for (const auto& variables_in_buffer : coefficients_buffer_) {
                interpolation_data[index] =
                    get<tag>(variables_in_buffer).get(i)[offset];
                ++index;
              }
              get<tag>(interpolated_coefficients).get(i)[offset] =
                  Interp::interpolate(interpolation_times, interpolation_data,
                                      time);
            });
      }

      tmpl::for_each<
          tmpl::list<InputTags::Lapse, InputTags::Dr<InputTags::Lapse>,
                     InputTags::Dt<InputTags::Lapse>>>(
          [this, &ylm_iter, &ylm_spherepack_prefactor, &interpolate_from_column,
           &column](auto x) {
            using tag = typename decltype(x)::type;
            DataVector interpolation_data{2 * target_buffer_pad};
            size_t index = 0;
            for (const auto& variables_in_buffer : coefficients_buffer_) {
              interpolation_data[index] =
                  get(get<tag>(variables_in_buffer))[offset];
              ++index;
            }
            get(get<tag>(interpolated_coefficients))[offset] =
                Interp::interpolate(interpolation_times, interpolation_data,
                                    time);
          });
    }

    create_bondi_boundary_data_from_cauchy(
        boundary_data,
        get<InputTags::SpatialMetric>(interpolated_coefficients_),
        get<InputTags::Dt<InputTags::SpatialMetric>>(
            interpolated_coefficients_),
        get<InputTags::Dr<InputTags::SpatialMetric>>(
            interpolated_coefficients_),
        get<InputTags::Shift>(interpolated_coefficients_),
        get<InputTags::Dt<InputTags::Shift>>(interpolated_coefficients_),
        get<InputTags::Dr<InputTags::Shift>>(interpolated_coefficients_),
        get<InputTags::Lapse>(interpolated_coefficients_),
        get<InputTags::Dt<InputTags::Lapse>>(interpolated_coefficients_),
        get<InputTags::Dr<InputTags::Lapse>>(interpolated_coefficients_),
        extraction_radius_, l_max_, spherical_harmonic_,
        radial_derivatives_need_renormalization_);

  }

  size_t get_l_max() const noexcept { return l_max_; }

  size_t get_spherepack_l_max() const noexcept { return spherepack_l_max_; }

  const deque<double>& get_time_buffer() const noexcept { return time_buffer_; }

  std::pair<size_t, size_t> get_time_span() const noexcept {
    return std::make_pair(time_span_start_, time_span_end_);
  }

 private:
  deque<double> time_buffer_;
  size_t target_buffer_pad_;

  YlmSpherepack spherical_harmonic_;

  // These buffers are just kept around to avoid allocations; they're
  // updated every time a time is requested
  Variables<cce_input_tags> interpolated_coefficients_;

  deque<Variables<cce_input_tags>> coefficients_buffers_;

  // currently assumed to be constant
  double extraction_radius_;
  size_t l_max_;
  size_t spherepack_l_max_;
};

/// takes data as needed from a specified H5 file.
template <class Interp>
class CceH5BoundaryDataManager {
 public:
  template <typename... T>
  std::string dataset_name_for_component(std::string base_name,
                                         const T... indices) noexcept {
    auto add_index = [&base_name](size_t index) { base_name += ('x' + index); };
    EXPAND_PACK_LEFT_TO_RIGHT(add_index(indices));
    return base_name;
  }

  // we have to explicitly initialized the spherical harmonic object, but it
  // will be immediately overwritten in the true constructor with a better
  // version with the spherical harmonic l_max determined by reading in the
  // file.
  CceBoundaryDataManager(std::string cce_data_filename, size_t l_max) noexcept
      : cce_data_file_{cce_data_filename},
        spherical_harmonic_{2, 2},
        l_max_{l_max},
        time_span_start_{0},
        time_span_end_{0} {
    get<InputTags::DataSet<InputTags::SpatialMetric>>(dataset_names_) = "/g";
    get<InputTags::DataSet<InputTags::Dr<InputTags::SpatialMetric>>>(
        dataset_names_) = "/Drg";
    get<InputTags::DataSet<InputTags::Dt<InputTags::SpatialMetric>>>(
        dataset_names_) = "/Dtg";

    get<InputTags::DataSet<InputTags::Shift>>(dataset_names_) = "/Shift";
    get<InputTags::DataSet<InputTags::Dr<InputTags::Shift>>>(dataset_names_) =
        "/DrShift";
    get<InputTags::DataSet<InputTags::Dt<InputTags::Shift>>>(dataset_names_) =
        "/DtShift";

    get<InputTags::DataSet<InputTags::Lapse>>(dataset_names_) = "/Lapse";
    get<InputTags::DataSet<InputTags::Dr<InputTags::Lapse>>>(dataset_names_) =
        "/DrLapse";
    get<InputTags::DataSet<InputTags::Dt<InputTags::Lapse>>>(dataset_names_) =
        "/DtLapse";

    // We assume that the filename has the extraction radius between the first
    // occurrence of 'R' and the first occurrence of '.'
    const size_t r_pos = cce_data_filename.find("R");
    const size_t dot_pos = cce_data_filename.find(".");
    const std::string text_radius =
        cce_data_filename.substr(r_pos + 1, dot_pos - r_pos - 1);
    extraction_radius_ = stod(text_radius);
    // auto& lapse_data =
    // cce_data_file_.get<h5::Dat>(dataset_name_for_component(
    // get<InputTags::DataSet<InputTags::Lapse>>(dataset_names_)));
    auto& lapse_data = cce_data_file_.get<h5::Dat>("/Lapse");

    auto data_table_dimensions = lapse_data.get_dimensions();
    if (UNLIKELY(data_table_dimensions[0] <
                 2 * Interp::required_number_of_points_before_and_after +
                     buffer_depth)) {
      ERROR(
          "The specified file doesn't have enough time points to supply the "
          "requested interpolation buffer. This almost certainly indicates "
          "that the file hasn't been created properly, but might indicate that "
          "the `buffer_depth` template parameter is too large or the "
          "Interpolator specified requests too many points");
    }
    Matrix time_matrix = lapse_data.get_data_subset(std::vector<size_t>{0}, 0,
                                                    data_table_dimensions[0]);
    time_buffer_ = DataVector{data_table_dimensions[0]};
    for (size_t i = 0; i < data_table_dimensions[0]; ++i) {
      time_buffer_[i] = time_matrix(i, 0);
    }
    spherepack_l_max_ = sqrt(data_table_dimensions[1] / 2) - 1;
    cce_data_file_.close_current_object();
    size_t number_of_coefficients =
        SpherepackIterator(spherepack_l_max_, spherepack_l_max_)
            .spherepack_array_size();
    interpolated_coefficients_ =
        Variables<cce_input_tags>{number_of_coefficients};

    radial_derivatives_need_renormalization_ =
        not cce_data_file_.exists<h5::Version>("/VersionHist");

    size_t size_of_buffer =
        number_of_coefficients *
        (buffer_depth + 2 * Interp::required_number_of_points_before_and_after);
    coefficients_buffers_ = Variables<cce_input_tags>{size_of_buffer};
    // create and store a YlmSpherepack object to pass into the boundary data
    // generation
    // TODO: generate a pointer, so we don't have to bother with default
    // constructing a throw-away.
    spherical_harmonic_ = YlmSpherepack{spherepack_l_max_, spherepack_l_max_};
  }

  template <typename TagList>
  bool populate_hypersurface_boundary_data(
      gsl::not_null<Variables<TagList>*> boundary_data, double time) noexcept {
    if (time_is_outside_range(time)) {
      return false;
    }
    update_buffers_for_time(time);
    auto interpolation_time_span =
        create_span_for_time_value(time, 0, time_span_start_, time_span_end_);
    // search through and find the two interpolation points the time point is
    // between. If we can, put the range for the interpolation centered on the
    // desired point. If that can't be done (near the start or the end of the
    // simulation), make the range terminated at the start or end of the cached
    // data and extending for the desired range in the other direction.
    size_t buffer_span_size = time_span_end_ - time_span_start_;
    size_t interpolation_span_size =
        interpolation_time_span.second - interpolation_time_span.first;

    DataVector time_points{time_buffer_.data() + interpolation_time_span.first,
                           interpolation_span_size};

    auto interpolate_from_column =
        [&time, &time_points, &buffer_span_size, &interpolation_time_span,
         &interpolation_span_size, this](auto data, size_t column) {
          auto sample =
              DataVector{data + column * (buffer_span_size) +
                             (interpolation_time_span.first - time_span_start_),
                         interpolation_span_size};
          auto interp_val = Interp::interpolate(
              time_points,
              DataVector{data + column * (buffer_span_size) +
                             (interpolation_time_span.first - time_span_start_),
                         interpolation_span_size},
              time);
          return interp_val;
        };

    tmpl::for_each<cce_input_tags>([this](auto x) {
      using tag = typename decltype(x)::type;
      std::for_each(boost::make_zip_iterator(boost::make_tuple(
                        get<tag>(coefficients_buffers_).begin(),
                        get<tag>(interpolated_coefficients_).begin())),
                    boost::make_zip_iterator(boost::make_tuple(
                        get<tag>(coefficients_buffers_).end(),
                        get<tag>(interpolated_coefficients_).end())),
                    [](auto pair) { pair.template get<1>() = 0.0; });
    });

    double sqrt_2_over_pi = sqrt(2.0 / M_PI);
    for (SpherepackIterator ylm_iter{spherepack_l_max_, spherepack_l_max_};
         ylm_iter; ++ylm_iter) {
      size_t column =
          2 * (square(ylm_iter.l()) + (-ylm_iter.m() + ylm_iter.l())) +
          (ylm_iter.coefficient_array() ==
                   SpherepackIterator::CoefficientArray::a
               ? 0
               : 1);
      auto ylm_spherepack_prefactor =
          ((ylm_iter.m() % 2) == 0 ? 1.0 : -1.0) * sqrt_2_over_pi;
      // TODO: use a std::for_each and some boost zip iterator
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          tmpl::for_each<tmpl::list<InputTags::SpatialMetric,
                                    InputTags::Dr<InputTags::SpatialMetric>,
                                    InputTags::Dt<InputTags::SpatialMetric>>>(
              [this, &i, &j, &ylm_iter, &ylm_spherepack_prefactor,
               &interpolate_from_column, &column](auto x) {
                using tag = typename decltype(x)::type;
                get<tag>(interpolated_coefficients_).get(i, j)[ylm_iter()] =
                    ylm_spherepack_prefactor *
                    interpolate_from_column(
                        get<tag>(coefficients_buffers_).get(i, j).data(),
                        column);
              });
        }
        tmpl::for_each<
            tmpl::list<InputTags::Shift, InputTags::Dr<InputTags::Shift>,
                       InputTags::Dt<InputTags::Shift>>>(
            [this, &i, &ylm_iter, &ylm_spherepack_prefactor,
             &interpolate_from_column, &column](auto x) {
              using tag = typename decltype(x)::type;
              get<tag>(interpolated_coefficients_).get(i)[ylm_iter()] =
                  ylm_spherepack_prefactor *
                  interpolate_from_column(
                      get<tag>(coefficients_buffers_).get(i).data(), column);
            });
      }

      tmpl::for_each<
          tmpl::list<InputTags::Lapse, InputTags::Dr<InputTags::Lapse>,
                     InputTags::Dt<InputTags::Lapse>>>(
          [this, &ylm_iter, &ylm_spherepack_prefactor, &interpolate_from_column,
           &column](auto x) {
            using tag = typename decltype(x)::type;
            get(get<tag>(interpolated_coefficients_))[ylm_iter()] =
                ylm_spherepack_prefactor *
                interpolate_from_column(
                    get(get<tag>(coefficients_buffers_)).data(), column);
          });
    }

    // At this point, we have a collection of 9 tensors of YlmSpherepack
    // coefficients. This is what the boundary data calculation utility takes
    // as an input, so we now hand off the control flow to the boundary and
    // gauge transform utility

    create_bondi_boundary_data_from_cauchy(
        boundary_data,
        get<InputTags::SpatialMetric>(interpolated_coefficients_),
        get<InputTags::Dt<InputTags::SpatialMetric>>(
            interpolated_coefficients_),
        get<InputTags::Dr<InputTags::SpatialMetric>>(
            interpolated_coefficients_),
        get<InputTags::Shift>(interpolated_coefficients_),
        get<InputTags::Dt<InputTags::Shift>>(interpolated_coefficients_),
        get<InputTags::Dr<InputTags::Shift>>(interpolated_coefficients_),
        get<InputTags::Lapse>(interpolated_coefficients_),
        get<InputTags::Dt<InputTags::Lapse>>(interpolated_coefficients_),
        get<InputTags::Dr<InputTags::Lapse>>(interpolated_coefficients_),
        extraction_radius_, l_max_, spherical_harmonic_,
        radial_derivatives_need_renormalization_);

    return true;
  }

  double update_buffers_for_time(double time) noexcept {
    if (time_span_end_ > Interp::required_number_of_points_before_and_after and
        time_buffer_[time_span_end_ -
                     Interp::required_number_of_points_before_and_after + 1] >
            time) {
      // the next time an update will be required
      return time_buffer_[time_span_end_ -
                          Interp::required_number_of_points_before_and_after +
                          1];
    }
    // find the time spans that are needed
    auto new_span_pair =
        create_span_for_time_value(time, buffer_depth, 0, time_buffer_.size());
    time_span_start_ = new_span_pair.first;
    time_span_end_ = new_span_pair.second;
    // load the desired time spans into the buffers
    // g
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        tmpl::for_each<tmpl::list<InputTags::SpatialMetric,
                                  InputTags::Dr<InputTags::SpatialMetric>,
                                  InputTags::Dt<InputTags::SpatialMetric>>>(
            [this, &i, &j](auto x) {
              using tag = typename decltype(x)::type;
              auto& dat =
                  cce_data_file_.get<h5::Dat>(dataset_name_for_component(
                      get<InputTags::DataSet<tag>>(dataset_names_), i, j));
              update_buffer(
                  make_not_null(&get<tag>(coefficients_buffers_).get(i, j)),
                  cce_data_file_.get<h5::Dat>(dataset_name_for_component(
                      get<InputTags::DataSet<tag>>(dataset_names_), i, j)));
              cce_data_file_.close_current_object();
            });
      }
      // shift
      tmpl::for_each<
          tmpl::list<InputTags::Shift, InputTags::Dr<InputTags::Shift>,
                     InputTags::Dt<InputTags::Shift>>>([this, &i](auto x) {
        using tag = typename decltype(x)::type;
        update_buffer(make_not_null(&get<tag>(coefficients_buffers_).get(i)),
                      cce_data_file_.get<h5::Dat>(dataset_name_for_component(
                          get<InputTags::DataSet<tag>>(dataset_names_), i)));
        cce_data_file_.close_current_object();
      });
    }
    // lapse
    tmpl::for_each<tmpl::list<InputTags::Lapse, InputTags::Dr<InputTags::Lapse>,
                              InputTags::Dt<InputTags::Lapse>>>([this](auto x) {
      using tag = typename decltype(x)::type;
      update_buffer(make_not_null(&get(get<tag>(coefficients_buffers_))),
                    cce_data_file_.get<h5::Dat>(dataset_name_for_component(
                        get<InputTags::DataSet<tag>>(dataset_names_))));
      cce_data_file_.close_current_object();
    });
    // the next time an update will be required
    return time_buffer_[time_span_end_ -
                        Interp::required_number_of_points_before_and_after + 1];
  }

  std::pair<size_t, size_t> create_span_for_time_value(
      const double time, const size_t pad, const size_t lower_bound,
      const size_t upper_bound) const noexcept {
    size_t range_start = lower_bound;
    size_t range_end = upper_bound;
    while (range_end - range_start > 1) {
      if (time_buffer_[(range_start + range_end) / 2] < time) {
        range_start = (range_start + range_end) / 2;
      } else {
        range_end = (range_start + range_end) / 2;
      }
    }
    // always keep the difference between start and end the same, even when
    // the interpolations starts to get worse
    size_t span_start = lower_bound;
    size_t span_end =
        std::min(Interp::required_number_of_points_before_and_after * 2 + pad +
                     lower_bound,
                 upper_bound);
    if (range_end + Interp::required_number_of_points_before_and_after + pad >
        upper_bound) {
      span_start = std::max(
          upper_bound -
              (Interp::required_number_of_points_before_and_after * 2 + pad),
          lower_bound);
      span_end = upper_bound;
    } else if (range_start >
               lower_bound +
                   Interp::required_number_of_points_before_and_after - 1) {
      span_start =
          range_start - Interp::required_number_of_points_before_and_after + 1;

      span_end =
          range_end + Interp::required_number_of_points_before_and_after + pad;
    }

    return std::make_pair(span_start, span_end);
  }

  bool time_is_outside_range(const double time) const noexcept {
    return time < time_buffer_[0] or
           time > time_buffer_[time_buffer_.size() - 1];
  }

  size_t get_l_max() const noexcept { return l_max_; }

  size_t get_spherepack_l_max() const noexcept { return spherepack_l_max_; }

  const DataVector& get_time_buffer() const noexcept { return time_buffer_; }

  std::pair<size_t, size_t> get_time_span() const noexcept {
    return std::make_pair(time_span_start_, time_span_end_);
  }

 private:
  void update_buffer(
      const gsl::not_null</*ModalVector*/ DataVector*> buffer_to_update,
      const h5::Dat& read_data) noexcept {
    size_t number_of_columns = read_data.get_dimensions()[1];
    if (UNLIKELY(buffer_to_update->size() !=
                 (time_span_end_ - time_span_start_) *
                     (number_of_columns - 1))) {
      ERROR("Incorrect storage size for the data to be loaded in.");
    }
    std::vector<size_t> cols(number_of_columns - 1);
    std::iota(cols.begin(), cols.end(), 1);
    Matrix data_matrix = read_data.get_data_subset(
        cols, time_span_start_, time_span_end_ - time_span_start_);
    for (size_t i = 0;
         i < (time_span_end_ - time_span_start_) * (number_of_columns - 1);
         ++i) {
      (*buffer_to_update)[i] =
          data_matrix(i % (time_span_end_ - time_span_start_),
                      i / (time_span_end_ - time_span_start_));
    }
  }

  h5::H5File<h5::AccessType::ReadOnly> cce_data_file_;

  // stores all the times in the input file
  DataVector time_buffer_;

  bool radial_derivatives_need_renormalization_;

  size_t time_span_start_;
  size_t time_span_end_;

  YlmSpherepack spherical_harmonic_;

  // These buffers are just kept around to avoid allocations; they're
  // updated every time a time is requested
  Variables<cce_input_tags> interpolated_coefficients_;

  // note: buffers store data in an 'time-varies-fastest' manner
  Variables<cce_input_tags> coefficients_buffers_;

  double extraction_radius_;
  size_t l_max_;
  size_t spherepack_l_max_;

  tuples::tagged_tuple_from_typelist<
      db::wrap_tags_in<InputTags::DataSet, cce_input_tags>>
      dataset_names_;
};
}  // namespace Cce
