// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveGhWorldtubeData.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Observer/WriteSimpleData.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp {
namespace callbacks {

/// \brief post_interpolation_callback that calls Cce::ReceiveGhWorldTubeData
///
/// Uses:
/// - Metavariables
///   - `temporal_id`
/// - DataBox:
///   - `::gr::Tags::SpacetimeMetric<3,Frame::Inertial>`
///   - `::GeneralizedHarmonic::Tags::Pi<3,Frame::Inertial>`
///   - `::GeneralizedHarmonic::Tags::Phi<3,Frame::Inertial>`
///
/// This is an InterpolationTargetTag::post_interpolation_callback;
/// see InterpolationTarget for a description of InterpolationTargetTag.
template <typename CceEvolutionComponent>
struct SendGhWorldtubeData {
  using observation_types = tmpl::list<>;
  template <typename DbTags, typename Metavariables>
  static void apply(
      const db::DataBox<DbTags>& box,
      Parallel::GlobalCache<Metavariables>& cache,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    auto& cce_gh_boundary_component = Parallel::get_parallel_component<
        Cce::GhWorldtubeBoundary<Metavariables>>(cache);
    Parallel::simple_action<
        typename Cce::Actions::ReceiveGhWorldtubeData<CceEvolutionComponent>>(
        cce_gh_boundary_component, temporal_id,
        db::get<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(box),
        db::get<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(box),
        db::get<::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>(box),
        db::get<::Tags::dt<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>>(
            box),
        db::get<
            ::Tags::dt<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>>(
            box),
        db::get<
            ::Tags::dt<::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>>(
            box));
  }
};

struct DumpGhWorldtubeData {
  using observation_types = tmpl::list<>;
  template <typename DbTags, typename Metavariables>
  static void apply(
      const db::DataBox<DbTags>& box,
      Parallel::GlobalCache<Metavariables>& cache,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    const size_t tensor_component_size =
        get<0, 0>(db::get<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(box))
            .size();
    const double time = temporal_id.substep_time().value();
    std::vector<double> tensor_component_buffer(tensor_component_size + 1);
    const auto tensor_component_char = [](size_t index) noexcept {
      return index == 0 ? 't'
                        : static_cast<char>(static_cast<int>('w') + index);
    };
    std::string dataset;
    auto observer_proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(
        cache)[static_cast<size_t>(Parallel::my_node())];
    std::vector<std::string> tensor_component_legend(tensor_component_size + 1);
    const size_t l_max = (sqrt(8 * tensor_component_size + 1) - 3) / 4;
    size_t legend_index = 0;
    for (size_t i = 0;
         i < Spectral::Swsh::number_of_swsh_theta_collocation_points(l_max);
         ++i) {
      for (size_t j = 0;
           j < Spectral::Swsh::number_of_swsh_phi_collocation_points(l_max);
           ++j) {
        tensor_component_legend[legend_index] = MakeString{} << "(" << i << ", "
                                                             << j << ")";
        ++legend_index;
      }
    }
    tensor_component_buffer[0] = temporal_id.substep_time().value();
    // write the tensor components
    for (size_t a = 0; a < 4; ++a) {
      for (size_t b = a; b < 4; ++b) {
        DumpGhWorldtubeData::write_vector(
            make_not_null(&tensor_component_buffer),
            db::get<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(box).get(
                a, b),
            time, observer_proxy,
            MakeString{} << "/Psi/" << tensor_component_char(a)
                         << tensor_component_char(b),
            tensor_component_legend, tensor_component_size);

        DumpGhWorldtubeData::write_vector(
            make_not_null(&tensor_component_buffer),
            db::get<
                ::Tags::dt<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>>(
                box)
                .get(a, b),
            time, observer_proxy,
            MakeString{} << "/DtPsi/" << tensor_component_char(a)
                         << tensor_component_char(b),
            tensor_component_legend, tensor_component_size);

        DumpGhWorldtubeData::write_vector(
            make_not_null(&tensor_component_buffer),
            db::get<::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>(box)
                .get(a, b),
            time, observer_proxy,
            MakeString{} << "/Pi/" << tensor_component_char(a)
                         << tensor_component_char(b),
            tensor_component_legend, tensor_component_size);

        DumpGhWorldtubeData::write_vector(
            make_not_null(&tensor_component_buffer),
            db::get<::Tags::dt<
                ::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>>(box)
                .get(a, b),
            time, observer_proxy,
            MakeString{} << "/DtPi/" << tensor_component_char(a)
                         << tensor_component_char(b),
            tensor_component_legend, tensor_component_size);

        for (size_t i = 0; i < 3; ++i) {
          DumpGhWorldtubeData::write_vector(
              make_not_null(&tensor_component_buffer),
              db::get<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(box)
                  .get(i, a, b),
              time, observer_proxy,
              MakeString{} << "/Phi/" << tensor_component_char(i + 1)
                           << tensor_component_char(a)
                           << tensor_component_char(b),
              tensor_component_legend, tensor_component_size);

          DumpGhWorldtubeData::write_vector(
              make_not_null(&tensor_component_buffer),
              db::get<::Tags::dt<
                  ::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>>(box)
                  .get(i, a, b),
              time, observer_proxy,
              MakeString{} << "/DtPhi/" << tensor_component_char(i + 1)
                           << tensor_component_char(a)
                           << tensor_component_char(b),
              tensor_component_legend, tensor_component_size);
        }
      }
    }

    // also compute the 'traditional' worldtube data to dump.

    Variables<tmpl::list<
        gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>,
        ::Tags::dt<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>,
        ::Tags::deriv<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>,
                      tmpl::size_t<3>, ::Frame::Inertial>,
        gr::Tags::InverseSpatialMetric<3, ::Frame::Inertial, DataVector>,
        gr::Tags::Shift<3, ::Frame::Inertial, DataVector>,
        ::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>,
        ::Tags::deriv<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>,
                      tmpl::size_t<3>, ::Frame::Inertial>,
        Cce::Tags::Dr<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>,
        Cce::Tags::Dr<
            gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>,
        gr::Tags::Lapse<DataVector>, ::Tags::dt<gr::Tags::Lapse<DataVector>>,
        ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                      ::Frame::Inertial>,
        Cce::Tags::Dr<gr::Tags::Lapse<DataVector>>,
        gr::Tags::SpacetimeNormalVector<3, ::Frame::Inertial, DataVector>,
        // for the detail function called at the end
        gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>,
        ::Tags::dt<gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>,
        gr::Tags::InverseSpacetimeMetric<3, Frame::Inertial, DataVector>>>
        computation_variables{tensor_component_size};

    auto& spatial_metric =
        get<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>(
            computation_variables);
    gr::spatial_metric(
        make_not_null(&spatial_metric),
        db::get<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(box));

    auto& inverse_spatial_metric =
        get<gr::Tags::InverseSpatialMetric<3, ::Frame::Inertial, DataVector>>(
            computation_variables);
    inverse_spatial_metric = determinant_and_inverse(spatial_metric).second;

    auto& shift = get<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>(
        computation_variables);
    gr::shift(make_not_null(&shift),
              db::get<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(box),
              inverse_spatial_metric);

    auto& lapse = get<gr::Tags::Lapse<DataVector>>(computation_variables);
    gr::lapse(make_not_null(&lapse), shift,
              db::get<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(box));

    auto& dt_spacetime_metric = get<::Tags::dt<
        gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>>(
        computation_variables);

    GeneralizedHarmonic::time_derivative_of_spacetime_metric(
        make_not_null(&dt_spacetime_metric), lapse, shift,
        db::get<::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>(box),
        db::get<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(box));

    auto& spacetime_unit_normal =
        get<gr::Tags::SpacetimeNormalVector<3, ::Frame::Inertial, DataVector>>(
            computation_variables);
    gr::spacetime_normal_vector(make_not_null(&spacetime_unit_normal), lapse,
                                shift);
    auto& dt_lapse =
        get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(computation_variables);
    GeneralizedHarmonic::time_deriv_of_lapse(
        make_not_null(&dt_lapse), lapse, shift, spacetime_unit_normal,
        db::get<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(box),
        db::get<::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>(box));
    auto& dt_shift =
        get<::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>(
            computation_variables);
    GeneralizedHarmonic::time_deriv_of_shift(
        make_not_null(&dt_shift), lapse, shift, inverse_spatial_metric,
        spacetime_unit_normal,
        db::get<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(box),
        db::get<::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>(box));

    auto& d_spatial_metric = get<
        ::Tags::deriv<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>,
                      tmpl::size_t<3>, ::Frame::Inertial>>(
        computation_variables);
    GeneralizedHarmonic::deriv_spatial_metric(
        make_not_null(&d_spatial_metric),
        db::get<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(box));

    auto& d_lapse =
        get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                          ::Frame::Inertial>>(computation_variables);
    GeneralizedHarmonic::spatial_deriv_of_lapse(
        make_not_null(&d_lapse), lapse, spacetime_unit_normal,
        db::get<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(box));

    auto& inverse_spacetime_metric =
        get<gr::Tags::InverseSpacetimeMetric<3, Frame::Inertial, DataVector>>(
            computation_variables);
    gr::inverse_spacetime_metric(make_not_null(&inverse_spacetime_metric),
                                 lapse, shift, inverse_spatial_metric);

    auto& d_shift =
        get<::Tags::deriv<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>,
                          tmpl::size_t<3>, ::Frame::Inertial>>(
            computation_variables);
    GeneralizedHarmonic::spatial_deriv_of_shift(
        make_not_null(&d_shift), lapse, inverse_spacetime_metric,
        spacetime_unit_normal,
        db::get<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(box));

    auto& dr_spatial_metric = get<Cce::Tags::Dr<
        gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>>(
        computation_variables);

    auto& dr_lapse =
        get<Cce::Tags::Dr<gr::Tags::Lapse<DataVector>>>(computation_variables);

    auto& dr_shift =
        get<Cce::Tags::Dr<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>(
            computation_variables);

    tnsr::I<DataVector, 3> collocation_coordinates{tensor_component_size};
    for (const auto& collocation_point :
         Spectral::Swsh::cached_collocation_metadata<
             Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max)) {
      get<0>(collocation_coordinates)[collocation_point.offset] =
          sin(collocation_point.theta) * cos(collocation_point.phi);
      get<1>(collocation_coordinates)[collocation_point.offset] =
          sin(collocation_point.theta) * sin(collocation_point.phi);
      get<2>(collocation_coordinates)[collocation_point.offset] =
          cos(collocation_point.theta);
    }

    get(dr_lapse) = (get<0>(collocation_coordinates) * get<0>(d_lapse) +
                     get<1>(collocation_coordinates) * get<1>(d_lapse) +
                     get<2>(collocation_coordinates) * get<2>(d_lapse));
    for (size_t i = 0; i < 3; ++i) {
      dr_shift.get(i) = (get<0>(collocation_coordinates) * d_shift.get(0, i) +
                         get<1>(collocation_coordinates) * d_shift.get(1, i) +
                         get<2>(collocation_coordinates) * d_shift.get(2, i));
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        dr_spatial_metric.get(i, j) =
            (get<0>(collocation_coordinates) * d_spatial_metric.get(0, i, j) +
             get<1>(collocation_coordinates) * d_spatial_metric.get(1, i, j) +
             get<2>(collocation_coordinates) * d_spatial_metric.get(2, i, j));
      }
    }

    std::vector<std::string> coefficients_legend;
    coefficients_legend.emplace_back("time");
    for (int i = 0; i <= static_cast<int>(l_max); ++i) {
      for (int j = -i; j <= i; ++j) {
        coefficients_legend.push_back(MakeString{} << "Real Y_" << i << ","
                                                   << j);
        coefficients_legend.push_back(MakeString{} << "Imag Y_" << i << ","
                                                   << j);
      }
    }

    write_tensor_component_coefficients(get(lapse), time, observer_proxy,
                                        "/Lapse", coefficients_legend,
                                        l_max);
    write_tensor_component_coefficients(get(dt_lapse), time, observer_proxy,
                                        "/DtLapse", coefficients_legend,
                                        l_max);
    write_tensor_component_coefficients(get(dr_lapse), time, observer_proxy,
                                        "/DrLapse", coefficients_legend,
                                        l_max);
    for (size_t i = 0; i < 3; ++i) {
      write_tensor_component_coefficients(
          shift.get(i), time, observer_proxy,
          MakeString{} << "/Shift" << tensor_component_char(i + 1),
          coefficients_legend, l_max);
      write_tensor_component_coefficients(
          dt_shift.get(i), time, observer_proxy,
          MakeString{} << "/DtShift" << tensor_component_char(i + 1),
          coefficients_legend, l_max);
      write_tensor_component_coefficients(
          dr_shift.get(i), time, observer_proxy,
          MakeString{} << "/DrShift" << tensor_component_char(i + 1),
          coefficients_legend, l_max);

      for (size_t j = i; j < 3; ++j) {
        write_tensor_component_coefficients(
            spatial_metric.get(i, j), time, observer_proxy,
            MakeString{} << "/g" << tensor_component_char(i + 1)
                         << tensor_component_char(j + 1),
            coefficients_legend, l_max);
        write_tensor_component_coefficients(
            dt_spacetime_metric.get(i + 1, j + 1), time, observer_proxy,
            MakeString{} << "/Dtg" << tensor_component_char(i + 1)
                         << tensor_component_char(j + 1),
            coefficients_legend, l_max);
        write_tensor_component_coefficients(
            dr_spatial_metric.get(i, j), time, observer_proxy,
            MakeString{} << "/Drg" << tensor_component_char(i + 1)
                         << tensor_component_char(j + 1),
            coefficients_legend, l_max);
      }
    }
  }

 private:
  template <typename ObserverProxy>
  static void write_tensor_component_coefficients(
      const DataVector& data_to_write, const double time,
      ObserverProxy& observer_proxy, const std::string& dataset_name,
      const std::vector<std::string>& legend, size_t l_max) noexcept {
    // this has a few allocations that could be aggregated with a buffer passed
    // in.
    std::vector<double> coefficients_vector(1 + 2 * square(l_max + 1));
    coefficients_vector[0] = time;
    SpinWeighted<ComplexDataVector, 0> to_transform{data_to_write.size()};
    for(size_t i = 0; i < to_transform.size(); ++i) {
      to_transform.data()[i] = data_to_write[i];
    }
    SpinWeighted<ComplexModalVector, 0> goldberg_modes(square(l_max + 1));
    Spectral::Swsh::libsharp_to_goldberg_modes(
        make_not_null(&goldberg_modes),
        Spectral::Swsh::swsh_transform(l_max, 1, to_transform), l_max);
    for (size_t i = 0; i < square(l_max + 1); ++i) {
      coefficients_vector[2 * i + 1] = real(goldberg_modes.data()[i]);
      coefficients_vector[2 * i + 2] = imag(goldberg_modes.data()[i]);
    }

    Parallel::threaded_action<observers::ThreadedActions::WriteSimpleData>(
        observer_proxy, legend, coefficients_vector, dataset_name);
  }

  template <typename ObserverProxy>
  static void write_vector(
      const gsl::not_null<std::vector<double>*> data_buffer,
      const DataVector& data_to_write, const double time,
      ObserverProxy& observer_proxy, const std::string& dataset_name,
      const std::vector<std::string>& legend,
      const size_t tensor_component_size) noexcept {
    (*data_buffer)[0] = time;
    for (size_t i = 0; i < tensor_component_size; ++i) {
      (*data_buffer)[i + 1] = data_to_write[i];
    }
    Parallel::threaded_action<observers::ThreadedActions::WriteSimpleData>(
        observer_proxy, legend, *data_buffer, dataset_name);
  }
};
}  // namespace callbacks
}  // namespace intrp
