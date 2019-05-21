// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <class Interp>
CceH5BoundaryDataManager::CceH5BoundaryDataManager(
    const std::string cce_data_filename, const size_t l_max,
    const size_t buffer_depth) noexcept
    : cce_data_file_{cce_data_filename},
      spherical_harmonic_{2, 2},
      l_max_{l_max},
      time_span_start_{0},
      time_span_end_{0},
      buffer_depth_{buffer_depth} {
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
                   buffer_depth_)) {
    ERROR(
        "The specified file doesn't have enough time points to supply the "
        "requested interpolation buffer. This almost certainly indicates "
        "that the file hasn't been created properly, but might indicate that "
        "the `buffer_depth` parameter is too large or the "
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
      (buffer_depth_ + 2 * Interp::required_number_of_points_before_and_after);
  coefficients_buffers_ = Variables<cce_input_tags>{size_of_buffer};
  // create and store a YlmSpherepack object to pass into the boundary data
  // generation
  spherical_harmonic_ = YlmSpherepack{spherepack_l_max_, spherepack_l_max_};
}


template <class Interp>
double CceH5BoundaryDataManager::update_buffers_for_time(double time) noexcept {
  if (time_span_end_ > Interp::required_number_of_points_before_and_after and
      time_buffer_[time_span_end_ -
                   Interp::required_number_of_points_before_and_after + 1] >
          time) {
    // the next time an update will be required
    return time_buffer_[time_span_end_ -
                        Interp::required_number_of_points_before_and_after + 1];
  }
  // find the time spans that are needed
  auto new_span_pair =
      create_span_for_time_value(time, buffer_depth_, 0, time_buffer_.size());
  time_span_start_ = new_span_pair.first;
  time_span_end_ = new_span_pair.second;
  // load the desired time spans into the buffers
  // g
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      tmpl::for_each<tmpl::list<
          InputTags::SpatialMetric, InputTags::Dr<InputTags::SpatialMetric>,
          InputTags::Dt<InputTags::SpatialMetric>>>([this, &i, &j](auto x) {
        using tag = typename decltype(x)::type;
        auto& dat = cce_data_file_.get<h5::Dat>(dataset_name_for_component(
            get<InputTags::DataSet<tag>>(dataset_names_), i, j));
        update_buffer(make_not_null(&get<tag>(coefficients_buffers_).get(i, j)),
                      cce_data_file_.get<h5::Dat>(dataset_name_for_component(
                          get<InputTags::DataSet<tag>>(dataset_names_), i, j)));
        cce_data_file_.close_current_object();
      });
    }
    // shift
    tmpl::for_each<tmpl::list<InputTags::Shift, InputTags::Dr<InputTags::Shift>,
                              InputTags::Dt<InputTags::Shift>>>(
        [this, &i](auto x) {
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

template <class Interp>
std::pair<size_t, size_t> CceH5BoundaryDataManager::create_span_for_time_value(
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
             lower_bound + Interp::required_number_of_points_before_and_after -
                 1) {
    span_start =
        range_start - Interp::required_number_of_points_before_and_after + 1;

    span_end =
        range_end + Interp::required_number_of_points_before_and_after + pad;
  }

  return std::make_pair(span_start, span_end);
}

template <class Interp>
void CceH5BoundaryDataManager::update_buffer(
    const gsl::not_null<DataVector*> buffer_to_update,
    const h5::Dat& read_data) noexcept {
  size_t number_of_columns = read_data.get_dimensions()[1];
  if (UNLIKELY(buffer_to_update->size() !=
               (time_span_end_ - time_span_start_) * (number_of_columns - 1))) {
    ERROR("Incorrect storage size for the data to be loaded in.");
  }
  std::vector<size_t> cols(number_of_columns - 1);
  std::iota(cols.begin(), cols.end(), 1);
  Matrix data_matrix = read_data.get_data_subset(
      cols, time_span_start_, time_span_end_ - time_span_start_);
  for (size_t i = 0;
       i < (time_span_end_ - time_span_start_) * (number_of_columns - 1); ++i) {
    (*buffer_to_update)[i] =
        data_matrix(i % (time_span_end_ - time_span_start_),
                    i / (time_span_end_ - time_span_start_));
  }
}

#define GET_INTERP(data) BOOST_PP_TUPLE_ELEM(0, data);

#define H5_BOUNDARY_MANAGER(r, data)                                        \
  template CceH5BoundaryDataManager<GET_INTERP(                             \
      data)>::CceH5BoundaryDataManager(const std::string cce_data_filename, \
                                       const size_t l_max,                  \
                                       const size_t buffer_depth) noexcept; \
  template double                                                           \
  CceH5BoundaryDataManager<GET_INTERP(data)>::update_buffers_for_time(      \
      double time) noexcept;                                                \
  template std::pair<size_t, size_t>                                        \
  CceH5BoundaryDataManager<GET_INTERP(data)>::create_span_for_time_value(   \
      const double time, const size_t pad, const size_t lower_bound,        \
      const size_t upper_bound) const noexcept;                             \
  template void CceH5BoundaryDataManager<GET_INTERP(data)>::update_buffer(  \
      const gsl::not_null<DataVector*> buffer_to_update,                    \
      const h5::Dat& read_data) noexcept;
