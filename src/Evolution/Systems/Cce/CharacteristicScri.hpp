// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/InitializeScri.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ScriPlusInterpolationManager.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

namespace Actions {

// For now, these actions are pretty geared towards the news. I'm not sure how
// easy they'll be to generalize or whether we'll need an entirely new set of
// actions for other quantities.

struct ObserveInertialNews {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    std::pair<ComplexDataVector, double> interpolation;
    db::mutate<Tags::InterpolationManager<ComplexDataVector, Tags::News>>(
        make_not_null(&box),
        [&interpolation](const gsl::not_null<
                         ScriPlusInterpolationManager<ComplexDataVector>*>
                             interpolation_manager) {
          interpolation =
              interpolation_manager->interpolate_and_pop_first_time();
        });
    // swsh transform
    const size_t l_max = Parallel::get<OptionTags::LMax>(cache);
    const size_t observation_l_max =
        Parallel::get<OptionTags::ObservationLMax>(cache);
    auto to_transform = SpinWeighted<ComplexDataVector, 2>{interpolation.first};
    ComplexModalVector goldberg_modes =
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(l_max, 1, to_transform), l_max)
            .data();
    DataVector goldberg_mode_subset{
        reinterpret_cast<double*>(goldberg_modes.data()),
        2 * square(observation_l_max + 1)};

    auto& observer_proxy =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();

    std::vector<TensorComponent> data_to_send_to_observer;
    data_to_send_to_observer.emplace_back("Scri0/News", goldberg_mode_subset);

    printf("writing inertial news at %f:\n", interpolation.second);

    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        observer_proxy,
        observers::ObservationId(
            interpolation.second,
            typename Metavariables::swsh_inertial_scri_observation_type{}),
        std::string{"/cce_scri_data"},
        // what could go wrong?
        observers::ArrayComponentId{
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<int>(0)},
        std::move(data_to_send_to_observer),
        Index<1>(2 * square(observation_l_max + 1)));
    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    return db::get<Tags::InterpolationManager<ComplexDataVector, Tags::News>>(
               box)
        .first_time_is_ready_to_interpolate();
  }
};

struct ReceiveNonInertialNews {
  template <
      typename ParallelComponent, typename... DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<
          tmpl::list<DbTags...>,
          Tags::InterpolationManager<ComplexDataVector, Tags::News>>> = nullptr>
  static void apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const Scalar<DataVector>& inertial_retarded_time,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& news) noexcept {
    db::mutate<Tags::InterpolationManager<ComplexDataVector, Tags::News>>(
        make_not_null(&box),
        [&inertial_retarded_time,
         &news](const gsl::not_null<
                ScriPlusInterpolationManager<ComplexDataVector>*>
                    interpolation_manager) {
          interpolation_manager->insert_data(get(inertial_retarded_time),
                                            get(news).data());
        });
    // continue observing if there's points that are now ready
    Parallel::get_parallel_component<CharacteristicScri<Metavariables>>(cache)
        .perform_algorithm();
  }
};

struct AddTargetInterpolationTime {
  template <
      typename ParallelComponent, typename... DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<
          tmpl::list<DbTags...>,
          Tags::InterpolationManager<ComplexDataVector, Tags::News>>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double time) noexcept {
    db::mutate<Tags::InterpolationManager<ComplexDataVector, Tags::News>>(
        make_not_null(&box),
        [&time](const gsl::not_null<
                ScriPlusInterpolationManager<ComplexDataVector>*>
                    interpolation_manager) {
          interpolation_manager->insert_target_time(time);
        });
    // continue observing if there's points that are now ready
    Parallel::get_parallel_component<CharacteristicScri<Metavariables>>(cache)
        .perform_algorithm();
  }
};
}  // namespace Actions

// component for scri+ interpolation and calculation
template <class Metavariables>
struct CharacteristicScri {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tag_list = tmpl::list<>;
  using metavariables = Metavariables;

  using add_options_to_databox = typename Parallel::AddNoOptionsToDataBox;

  struct RegistrationHelper {
    template <typename ParallelComponent, typename DbTagsList,
              typename ArrayIndex>
    static std::pair<observers::TypeOfObservation, observers::ObservationId>
    register_info(const db::DataBox<DbTagsList>& /*box*/,
                  const ArrayIndex& /*array_index*/) noexcept {
      observers::ObservationId fake_initial_observation_id{
          0., typename Metavariables::swsh_inertial_scri_observation_type{}};

      Parallel::printf("debug array id check %zu, %s\n",
                       observers::ArrayComponentId{
                           std::add_pointer_t<ParallelComponent>{nullptr},
                           Parallel::ArrayIndex<int>(0)}
                           .component_id(),
                       pretty_type::get_name<ParallelComponent>().c_str());
      return {observers::TypeOfObservation::ReductionAndVolume,
              fake_initial_observation_id};
    }
  };

  using initialize_action_list =
      tmpl::list<Actions::InitializeCharacteristicScri,
                 Parallel::Actions::TerminatePhase>;

  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;


  using registration_action_list = tmpl::list<
      ::observers::Actions::RegisterSingletonWithObservers<RegistrationHelper>,
      Parallel::Actions::TerminatePhase>;

  using extract_action_list = tmpl::list<Actions::ObserveInertialNews>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::RegisterWithObserver,
                             registration_action_list>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Extraction,
                             extract_action_list>>;
  using options = tmpl::list<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>&
          global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::RegisterWithObserver or
        next_phase == Metavariables::Phase::Extraction) {
      Parallel::get_parallel_component<CharacteristicScri<Metavariables>>(
          local_cache)
          .start_phase(next_phase);
    }
  }
};
}  // namespace Cce
