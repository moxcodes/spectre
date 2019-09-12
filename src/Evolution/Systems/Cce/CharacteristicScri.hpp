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
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

namespace {
template <typename Tag>
struct ScriOutput {
  static std::string name() noexcept { return Tag::name(); }
};
template <typename Tag>
struct ScriOutput<Tags::ScriPlus<Tag>> {
  static std::string name() noexcept {
    return pretty_type::short_name<Tag>();
  }
};
template <>
struct ScriOutput<::Tags::Multiplies<
  Tags::Du<Tags::TimeIntegral<Tags::ScriPlus<Tags::Psi4>>>,
  Tags::ScriPlusFactor<Tags::Psi4>>> {
  static std::string name() noexcept { return "Psi4"; }
};
}  // namespace

namespace Actions {

template <typename TagList>
struct ObserveInertialInterpolated {
  // specialized names for common scri+ observation values

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    std::vector<TensorComponent> data_to_send_to_observer;
    const size_t observation_l_max =
        Parallel::get<Tags::ObservationLMax>(cache);
    double interpolated_time = std::numeric_limits<double>::quiet_NaN();
    tmpl::for_each<TagList>([&box, &cache, &data_to_send_to_observer,
                             &interpolated_time,
                             &observation_l_max](auto tag_v) {
      using tag = typename decltype(tag_v)::type;
      std::pair<ComplexDataVector, double> interpolation;
      db::mutate<Tags::InterpolationManager<ComplexDataVector, tag>>(
          make_not_null(&box),
          [&interpolation](
              const gsl::not_null<
                  ScriPlusInterpolationManager<ComplexDataVector, tag>*>
                  interpolation_manager) {
            interpolation =
                interpolation_manager->interpolate_and_pop_first_time();
          });
      // swsh transform
      const size_t l_max = Parallel::get<Tags::LMax>(cache);
      auto to_transform =
          SpinWeighted<ComplexDataVector, 2>{interpolation.first};
      ComplexModalVector goldberg_modes =
          Spectral::Swsh::libsharp_to_goldberg_modes(
              Spectral::Swsh::swsh_transform(l_max, 1, to_transform), l_max)
              .data();
      DataVector goldberg_mode_subset{
          reinterpret_cast<double*>(goldberg_modes.data()),
          2 * square(observation_l_max + 1)};

      data_to_send_to_observer.emplace_back("Scri/" + ScriOutput<tag>::name(),
                                            goldberg_mode_subset);
      if (isnan(interpolated_time)) {
        interpolated_time = interpolation.second;
      } else {
        ASSERT(
            interpolated_time == interpolation.second,
            "All interpolation results to observe are not time-synchronized");
      }
    });
    auto& observer_proxy =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();

    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        observer_proxy,
        observers::ObservationId(
            interpolated_time,
            typename Metavariables::swsh_boundary_observation_type{}),
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
    // we just check the first. All should be synchronized, and if they are not
    // it will generate a runtime error in the observation
    return db::get<Tags::InterpolationManager<ComplexDataVector,
                                              tmpl::front<TagList>>>(box)
        .first_time_is_ready_to_interpolate();
  }
};

template <typename Tag>
struct ReceiveNonInertial {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<
                tmpl::list<DbTags...>,
                Tags::InterpolationManager<ComplexDataVector, Tag>>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const Scalar<DataVector>& inertial_retarded_time,
                    const db::item_type<Tag>& received_data) noexcept {
    db::mutate<Tags::InterpolationManager<ComplexDataVector, Tag>>(
        make_not_null(&box),
        [&inertial_retarded_time,
         &received_data](const gsl::not_null<
                         ScriPlusInterpolationManager<ComplexDataVector, Tag>*>
                             interpolation_manager) {
          interpolation_manager->insert_data(get(inertial_retarded_time),
                                             get(received_data).data());
        });
    // continue observing if there are any points that are now ready
    Parallel::get_parallel_component<CharacteristicScri<Metavariables>>(cache)
        .perform_algorithm();
  }
};

template <typename MultipliesLhs, typename MultipliesRhs>
struct ReceiveNonInertial<::Tags::Multiplies<MultipliesLhs, MultipliesRhs>> {
  template <
      typename ParallelComponent, typename... DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<
          tmpl::list<DbTags...>,
          Tags::InterpolationManager<
              ComplexDataVector,
              ::Tags::Multiplies<MultipliesLhs, MultipliesRhs>>>> = nullptr>
  static void apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const Scalar<DataVector>& inertial_retarded_time,
      const db::item_type<MultipliesLhs>& received_data_lhs,
      const db::item_type<MultipliesRhs>& received_data_rhs) noexcept {
    db::mutate<Tags::InterpolationManager<
        ComplexDataVector, ::Tags::Multiplies<MultipliesLhs, MultipliesRhs>>>(
        make_not_null(&box),
        [&inertial_retarded_time, &received_data_lhs, &received_data_rhs](
            const gsl::not_null<ScriPlusInterpolationManager<
                ComplexDataVector,
                ::Tags::Multiplies<MultipliesLhs, MultipliesRhs>>*>
                interpolation_manager) {
          interpolation_manager->insert_data(get(inertial_retarded_time),
                                             get(received_data_lhs).data(),
                                             get(received_data_rhs).data());
        });
    // continue observing if there are any points that are now ready
    Parallel::get_parallel_component<CharacteristicScri<Metavariables>>(cache)
        .perform_algorithm();
  }
};

template <typename Tag>
struct AddTargetInterpolationTime {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<
                tmpl::list<DbTags...>,
                Tags::InterpolationManager<ComplexDataVector, Tag>>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double time) noexcept {
    db::mutate<Tags::InterpolationManager<ComplexDataVector, Tag>>(
        make_not_null(&box),
        [&time](const gsl::not_null<
                ScriPlusInterpolationManager<ComplexDataVector, Tag>*>
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
  using metavariables = Metavariables;

  struct RegistrationHelper {
    template <typename ParallelComponent, typename DbTagsList,
              typename ArrayIndex>
    static std::pair<observers::TypeOfObservation, observers::ObservationId>
    register_info(const db::DataBox<DbTagsList>& /*box*/,
                  const ArrayIndex& /*array_index*/) noexcept {
      observers::ObservationId fake_initial_observation_id{
          0., typename Metavariables::swsh_boundary_observation_type{}};

      return {observers::TypeOfObservation::ReductionAndVolume,
              fake_initial_observation_id};
    }
  };

  using initialize_action_list =
      tmpl::list<Actions::InitializeCharacteristicScri,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using registration_action_list = tmpl::list<
      ::observers::Actions::RegisterSingletonWithObservers<RegistrationHelper>,
      Parallel::Actions::TerminatePhase>;

  using extract_action_list = tmpl::list<Actions::ObserveInertialInterpolated<
      typename Metavariables::scri_values_to_observe>>;

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

  using const_global_cache_tag_list = tmpl::list<Tags::ScriInterpolationPoints>;
  // Parallel::get_const_global_cache_tags_from_pdal<
  // phase_dependent_action_list>;

  static void initialize(Parallel::CProxy_ConstGlobalCache<
                         Metavariables>& /*global_cache*/) noexcept {}

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
