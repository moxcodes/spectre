// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/Systems/Cce/InitializeCce.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Parallel/Info.hpp"
#include "Time/Tags.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Rational.hpp"

namespace Cce {
namespace Tags {

struct BoundaryTime : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "BoundaryTime"; }
};
}  // namespace Tags

namespace Actions {

struct PopulateCharacteristicInitialHypersurface {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // J on the first hypersurface
    db::mutate_apply<InitializeJ<Tags::BoundaryValue>>(make_not_null(&box));
    // Gauge quantities
    db::mutate_apply<InitializeGauge>(make_not_null(&box));
    db::mutate_apply<GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
        make_not_null(&box));
    db::mutate_apply<GaugeUpdateJacobianFromCoords<Tags::GaugeC, Tags::GaugeD,
                                                   Tags::CauchyCartesianCoords,
                                                   Tags::CauchyAngularCoords>>(
        make_not_null(&box));
    db::mutate_apply<GaugeUpdateOmegaCD>(make_not_null(&box));
    return std::forward_as_tuple(std::move(box));
  }
};

struct InitializeCharacteristic {
  /* Options are being updated. The correct way to store options will be to put
   * the option tag list in the initialization action (here), and they'll be
   * automatically extracted and parsed from any action placed in the special
   * "Initialization" phase*/
  // TODO
  static constexpr double start_time = 1000.0;
  static constexpr double end_time_input =
      std::numeric_limits<double>::quiet_NaN();
  static constexpr double target_step_size = 1.0;

  using interpolator = CubicInterpolator;
  static std::string input_filename() noexcept { return "CceR0100.h5"; }
  // \TODO

  template <typename Metavariables>
  struct EvolutionTags {
    using coordinate_variables_tag =
        typename Metavariables::evolved_coordinates_variables_tag;
    using dt_coordinate_variables_tag =
        db::add_tag_prefix<::Tags::dt, coordinate_variables_tag>;
    using evolution_simple_tags = db::AddSimpleTags<
        ::Tags::TimeId, ::Tags::Next<::Tags::TimeId>, ::Tags::TimeStep,
        ::Tags::HistoryEvolvedVariables<coordinate_variables_tag,
                                        dt_coordinate_variables_tag>,
        ::Tags::HistoryEvolvedTensor<
            typename Metavariables::evolved_swsh_tag,
            typename Metavariables::evolved_swsh_dt_tag>>;
    using evolution_compute_tags = db::AddComputeTags<::Tags::Time>;

    template <typename TagList>
    static auto initialize(
        db::DataBox<TagList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const double initial_time_value, const double /*end_time_value*/,
        const double step_size) noexcept {
      // currently hard-coded to fixed step size
      const Slab single_step_slab{initial_time_value,
                                  initial_time_value + step_size};
      const Time initial_time = single_step_slab.start();
      const TimeDelta fixed_time_step =
          TimeDelta{single_step_slab, Rational{1, 1}};
      const TimeId initial_time_id{true, 0, initial_time};

      const auto& time_stepper =
          Parallel::get<::OptionTags::TimeStepper>(cache);
      const TimeId second_time_id =
          time_stepper.next_time_id(initial_time_id, fixed_time_step);

      typename db::item_type<::Tags::HistoryEvolvedVariables<
          coordinate_variables_tag, dt_coordinate_variables_tag>>
          coordinate_history;

      typename db::item_type<::Tags::HistoryEvolvedTensor<
          typename Metavariables::evolved_swsh_tag,
          typename Metavariables::evolved_swsh_dt_tag>>
          swsh_history;

      return db::create_from<db::RemoveTags<>, evolution_simple_tags,
                             evolution_compute_tags>(
          std::move(box), initial_time_id, second_time_id, fixed_time_step,
          std::move(coordinate_history), std::move(swsh_history));
    }
  };

  template <typename Metavariables>
  struct CharacteristicTags {
    using boundary_value_variables_tag = ::Tags::Variables<
        tmpl::append<typename Metavariables::cce_boundary_communication_tags,
                     typename Metavariables::cce_gauge_boundary_tags>>;

    using scri_variables_tag =
        ::Tags::Variables<typename Metavariables::cce_scri_tags>;
    using volume_variables_tag = ::Tags::Variables<
        tmpl::append<typename Metavariables::cce_integrand_tags,
                     typename Metavariables::cce_integration_independent_tags,
                     typename Metavariables::cce_temporary_equations_tags>>;
    using pre_swsh_derivatives_variables_tag = ::Tags::Variables<
        typename Metavariables::cce_pre_swsh_derivavtives_tags>;
    using transform_buffer_variables_tag =
        ::Tags::Variables<typename Metavariables::cce_transform_buffer_tags>;
    using swsh_derivative_variables_tag =
        ::Tags::Variables<typename Metavariables::cce_swsh_derivative_tags>;

    using coordinate_variables_tag =
        typename Metavariables::evolved_coordinates_variables_tag;
    using dt_coordinate_variables_tag =
        db::add_tag_prefix<::Tags::dt, coordinate_variables_tag>;

    template <typename TagList>
    static auto initialize(
        db::DataBox<TagList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
        const size_t l_max, const size_t number_of_radial_points) noexcept {
      const size_t boundary_size =
          Spectral::Swsh::number_of_swsh_collocation_points(l_max);
      const size_t volume_size = boundary_size * number_of_radial_points;
      const size_t transform_buffer_size =
          2 * number_of_radial_points *
          Spectral::Swsh::number_of_swsh_coefficients(l_max);

      return db::create_from<
          db::RemoveTags<>,
          db::AddSimpleTags<
              boundary_value_variables_tag, coordinate_variables_tag,
              dt_coordinate_variables_tag, scri_variables_tag,
              volume_variables_tag, pre_swsh_derivatives_variables_tag,
              transform_buffer_variables_tag, swsh_derivative_variables_tag>,
          db::AddComputeTags<>>(
          std::move(box),
          db::item_type<boundary_value_variables_tag>{boundary_size},
          db::item_type<coordinate_variables_tag>{boundary_size},
          db::item_type<dt_coordinate_variables_tag>{boundary_size},
          db::item_type<scri_variables_tag>{boundary_size},
          db::item_type<volume_variables_tag>{volume_size},
          db::item_type<pre_swsh_derivatives_variables_tag>{volume_size},
          db::item_type<transform_buffer_variables_tag>{transform_buffer_size},
          db::item_type<swsh_derivative_variables_tag>{volume_size});
    }
  };

  template <class Metavariables>
  using return_tag_list = tmpl::append<
      typename EvolutionTags<Metavariables>::evolution_simple_tags,
      typename EvolutionTags<Metavariables>::evolution_compute_tags>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // Evolution quantity initialization
    double end_time = end_time_input;
    const size_t l_max = Parallel::get<OptionTags::LMax>(cache);
    const size_t number_of_radial_points =
        Parallel::get<OptionTags::NumberOfRadialPoints>(cache);

    if (std::isnan(end_time)) {
      CceH5BoundaryDataManager<interpolator> h5_boundary_data_manager{
          input_filename() + ".h5", l_max, 100};
      const auto& time_buffer = h5_boundary_data_manager.get_time_buffer();
      end_time = time_buffer[time_buffer.size() - 1];
    }
    // clear out the box first
    auto initialize_box =
        db::create<db::AddSimpleTags<Tags::LMax, Tags::BoundaryTime>,
                   db::AddComputeTags<>>(
            l_max, std::numeric_limits<double>::quiet_NaN());
    auto evolution_box = EvolutionTags<Metavariables>::initialize(
        std::move(initialize_box), cache, start_time, end_time,
        target_step_size);
    auto characteristic_evolution_box =
        CharacteristicTags<Metavariables>::initialize(
            std::move(evolution_box), cache, l_max, number_of_radial_points);
    return std::make_tuple(std::move(characteristic_evolution_box));
  }
};
}  // namespace Actions
}  // namespace Cce
