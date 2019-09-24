// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TensorData.hpp"
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "Evolution/Systems/Cce/CharacteristicExtractor.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace Events {

template <typename ToObserve, typename EventRegistrars>
class ObserveBoundarySwshModes;

namespace Registrars {
template <typename ToObserve>
struct ObserveBoundarySwshModes {
  template <typename RegistrarList>
  using f = ::Cce::Events::ObserveBoundarySwshModes<ToObserve, RegistrarList>;
};
}  // namespace Registrars

template <typename T>
struct reduction_data_noop_type;

template <typename T>
struct reduction_data_noop_names {
  static std::string name() noexcept { return "SwshReductionData"; }
  using type =
      std::unordered_map<observers::ObservationId, std::vector<std::string>>;
  using data_tag = reduction_data_noop_type<T>;
};

template <typename... Ts>
struct reduction_data_noop_type<tmpl::list<Ts...>> {
  // reduction data only used because that's what the observerwriter can write
  // to disk.
  using type = Parallel::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<Ts, funcl::AssertEqual<>>...>;
  using tag = ::observers::Tags::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<Ts, funcl::AssertEqual<>>...>;

  using names_tag = reduction_data_noop_names<tmpl::list<Ts...>>;
};

template <typename ToObserve,
          typename EventRegistrars =
              tmpl::list<Registrars::ObserveBoundarySwshModes<ToObserve>>>
class ObserveBoundarySwshModes;  // IWYU pragma: keep

template <typename... ToObserve, typename EventRegistrars>
class ObserveBoundarySwshModes<tmpl::list<ToObserve...>, EventRegistrars>
    : public Event<EventRegistrars> {
  /// \cond
  explicit ObserveBoundarySwshModes(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveBoundarySwshModes);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;

  static constexpr OptionString help =
      "Observe a spin-weighted scalar provided on a single spherical shell.\n"
      "\n"
      "Writes the simulation time, and in ascending l and m order, swsh \n "
      "modes of the solution.";

  ObserveBoundarySwshModes() = default;

  using argument_tags = tmpl::list<::Tags::TimeId, ToObserve...>;

  template <typename Metavariables, typename ParallelComponent,
            typename ArrayIndex>
  void operator()(const TimeId& time_id,
                  const db::item_type<ToObserve>&... boundary_swsh_scalars,
                  Parallel::ConstGlobalCache<Metavariables>& cache,
                  const ArrayIndex& /*array_index*/,
                  const ParallelComponent* const /*meta*/) const noexcept {
    const size_t observation_l_max =
        Parallel::get<Tags::ObservationLMax>(cache);
    const size_t l_max = Parallel::get<Spectral::Swsh::Tags::LMax>(cache);

    // kick off the observation for the same time at scri+, because current
    // volume outputs have to be synchronized.
    tmpl::for_each<
        typename Metavariables::scri_values_to_observe>([&cache,
                                                         &time_id](auto tag_v) {
      using tag = typename decltype(tag_v)::type;
      Parallel::simple_action<Actions::AddTargetInterpolationTime<tag>>(
          Parallel::get_parallel_component<CharacteristicScri<Metavariables>>(
              cache),
          time_id.time().value());
    });

    std::vector<std::string> file_legend;
    file_legend.push_back("times, sim lmax: " + std::to_string(l_max));
    for (int i = 0; i <= static_cast<int>(observation_l_max); ++i) {
      for (int j = -i; j <= i; ++j) {
        file_legend.push_back("Real Y_" + std::to_string(i) + "," +
                              std::to_string(j));
        file_legend.push_back("Imag Y_" + std::to_string(i) + "," +
                              std::to_string(j));
      }
    }

    std::vector<TensorComponent> swsh_scalars_data;

    swsh_scalars_data.reserve(sizeof...(ToObserve));

    const auto append_goldberg_swsh_mode =
        [ this, &observation_l_max, &l_max, &swsh_scalars_data ](
            const auto tag_v, const auto& boundary_swsh_scalar) noexcept {
      using tag = typename decltype(tag_v)::type;
      typename db::item_type<tag>::type boundary_swsh_copy =
          get(boundary_swsh_scalar);
      ComplexModalVector goldberg_modes =
          Spectral::Swsh::libsharp_to_goldberg_modes(
              Spectral::Swsh::swsh_transform(l_max, 1, boundary_swsh_copy),
              l_max)
              .data();
      // suspicious reinterpret cast for making a series of doubles, because
      // that's needed for the h5
      DataVector goldberg_subset{
          reinterpret_cast<double*>(goldberg_modes.data()),
          2 * square(observation_l_max + 1)};
      swsh_scalars_data.emplace_back("Extract0/" + tag::name(),
                                     goldberg_subset);
    };

    EXPAND_PACK_LEFT_TO_RIGHT(append_goldberg_swsh_mode(
        tmpl::type_<ToObserve>{}, boundary_swsh_scalars));

    auto& observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();

    // auto reduction_data = typename reduction_data_noop_type<
    // tmpl::list<typename db::item_type<Spectral::Swsh::Tags::SwshTransform<
    // ToObserve>>::type::value_type...>>::type{
    // time.value(), goldberg_swsh_mode_subset(tmpl::type_<ToObserve>{},
    // boundary_swsh_scalars)...};
    Parallel::printf("writing boundary data at %f\n", time_id.time().value());
    // TEST
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        observer,
        observers::ObservationId(
            time_id.time().value(),
            typename Metavariables::swsh_boundary_observation_type{}),
        std::string{"/swsh_boundary_data"},
        observers::ArrayComponentId{
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<int>(0)},
        std::move(swsh_scalars_data),
        Index<1>(2 * square(observation_l_max + 1)));

    // Parallel::threaded_action<observers
    // ::ThreadedActions::WriteReductionData>(
    // observer[0],
    // observers::ObservationId(
    // time.value(),
    // typename Metavariables::swsh_boundary_observation_type{}),
    // std::string{"/swsh_boundary_data"}, file_legend, reduction_data);
  }
};

/// \cond
template <typename... ToObserve, typename EventRegistrars>
PUP::able::PUP_ID ObserveBoundarySwshModes<tmpl::list<ToObserve...>,
                                           EventRegistrars>::my_PUP_ID = 0;
// \endcond

}  // namespace Events
}  // namespace Cce
