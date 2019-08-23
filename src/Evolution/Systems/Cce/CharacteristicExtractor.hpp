// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Evolution/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/Systems/Cce/ComputePreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/ComputeSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/Equations.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/InitializeCharacteristic.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/LinearSolve.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Evolution/Systems/Cce/ScriPlusValues.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/RegisterObservers.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/Time.hpp"

#include "Utilities/TmplDebugging.hpp"

namespace Cce {

template <class Metavariables>
struct CharacteristicExtractor;

template <class Metavariables>
struct CharacteristicScri;

namespace Actions {
struct BoundaryComputeAndSendToExtractor;
struct ReceiveNonInertialNews;
struct AddTargetInterpolationTime;
}

template <typename CollectionTagList, typename SearchTag>
struct is_member;

template <typename... CollectionTags, typename SearchTag>
struct is_member<tmpl::list<CollectionTags...>, SearchTag> {
  static constexpr bool value =
      tmpl2::flat_any_v<cpp17::is_same_v<SearchTag, CollectionTags>...>;
};

template <typename CollectionTagList, typename SearchTag>
static constexpr bool is_member_v =
    is_member<CollectionTagList, SearchTag>::value;

namespace Actions {
template <typename BondiTag>
struct CalculateIntegrandInputsForTag {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // Parallel::printf("starting hypersurface computation for %s\n",
                     // BondiTag::name());
    mutate_all_pre_swsh_derivatives_for_tag<BondiTag>(make_not_null(&box));
    mutate_all_swsh_derivatives_for_tag<BondiTag>(make_not_null(&box));
    return std::forward_as_tuple(std::move(box));
  }
};

template <typename BondiTag>
struct FilterSwshVolumeQuantity {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const size_t l_max = db::get<Spectral::Swsh::Tags::LMax>(box);
    const size_t l_filter_start = l_max - 2;
    db::mutate<BondiTag>(
        make_not_null(&box),
        [&l_max, &l_filter_start](
            const gsl::not_null<db::item_type<BondiTag>*> bondi_quantity) {
          Spectral::Swsh::filter_swsh_volume_quantity(
              make_not_null(&get(*bondi_quantity)), l_max, l_filter_start,
              108.0, 8);
        });
    return std::forward_as_tuple(std::move(box));
  }
};

struct SendNewsToScri {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if (db::get<::Tags::TimeId>(box).substep() == 0) {
      Parallel::simple_action<Actions::ReceiveNonInertialNews>(
          Parallel::get_parallel_component<CharacteristicScri<Metavariables>>(
              cache),
          db::get<Tags::InertialRetardedTime>(box), db::get<Tags::News>(box));
    }
    return std::forward_as_tuple(std::move(box));
  }
};

struct ExitIfEndTimeReached {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf("Time : %f, End: %f\n", db::get<::Tags::Time>(box).value(),
                     db::get<Tags::EndTime>(box));
    return std::tuple<db::DataBox<DbTags>&&, bool>(
        std::move(box),
        db::get<::Tags::Time>(box).value() >= db::get<Tags::EndTime>(box));
  }
};

template <typename WorldtubeBoundaryComponent>
struct RequestBoundaryData {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    Parallel::simple_action<Actions::BoundaryComputeAndSendToExtractor>(
        Parallel::get_parallel_component<WorldtubeBoundaryComponent>(cache),
        db::get<::Tags::Time>(box).value());
    return std::forward_as_tuple(std::move(box));
  }
};

template <typename WorldtubeBoundaryComponent>
struct RequestNextBoundaryData {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // TODO pass timeids instead to avoid comparing doubles
    Parallel::simple_action<Actions::BoundaryComputeAndSendToExtractor>(
        Parallel::get_parallel_component<WorldtubeBoundaryComponent>(cache),
        db::get<::Tags::Next<::Tags::TimeId>>(box).time().value());
    return std::forward_as_tuple(std::move(box));
  }
};

struct BlockUntilBoundaryDataReceived {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    return db::get<Tags::BoundaryTime>(box) ==
           db::get<::Tags::Time>(box).value();
  }
};

struct PrecomputeGlobalCceDependencies {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // Parallel::print;f("global dependencies\n");
    tmpl::for_each<compute_gauge_adjustments_setup_tags>([&box](auto tag_v) {
      using tag = typename decltype(tag_v)::type;
      db::mutate_apply<ComputeGaugeAdjustedBoundaryValue<tag>>(
          make_not_null(&box));
    });
    mutate_all_precompute_cce_dependencies<Tags::EvolutionGaugeBoundaryValue>(
        make_not_null(&box));
    return std::forward_as_tuple(std::move(box));
  }
};

template <typename TagList>
struct ReceiveWorldtubeData;

template <typename... BoundaryTags>
struct ReceiveWorldtubeData<tmpl::list<BoundaryTags...>> {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl2::flat_all_v<
                is_member_v<tmpl::list<DbTags...>, BoundaryTags>...>> = nullptr>
  static void apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const double time,
      const db::item_type<BoundaryTags>&... boundary_data) noexcept {
    // Parallel::printf("receiving worldtube data...\n");
    db::mutate<Tags::BoundaryTime, BoundaryTags...>(
        make_not_null(&box),
        [](const gsl::not_null<double*> databox_boundary_time,
           const gsl::not_null<
               db::item_type<BoundaryTags>*>... databox_boundary_data,
           const double& time_to_copy,
           const db::item_type<
               BoundaryTags>&... boundary_data_to_copy) noexcept {
          *databox_boundary_time = time_to_copy;
          EXPAND_PACK_LEFT_TO_RIGHT(
              set(databox_boundary_data, boundary_data_to_copy));
        },
        time, boundary_data...);
    Parallel::get_parallel_component<CharacteristicExtractor<Metavariables>>(
        cache)
        .perform_algorithm();
    // Parallel::printf("done\n");
  }

 private:
  template <typename T>
  static void set(const gsl::not_null<T*> to, const T& from) noexcept {
    *to = from;
  }
};
}  // namespace Actions

template <class Metavariables>
struct CharacteristicExtractor {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;

  using add_options_to_databox = typename Parallel::AddNoOptionsToDataBox;

  struct RegistrationHelper {
    template <typename ParallelComponent, typename DbTagsList,
              typename ArrayIndex>
    static std::pair<observers::TypeOfObservation, observers::ObservationId>
    register_info(const db::DataBox<DbTagsList>& /*box*/,
                  const ArrayIndex& /*array_index*/) noexcept {
      observers::ObservationId fake_initial_observation_id{
          0., typename Metavariables::swsh_boundary_observation_type{}};
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
      tmpl::list<Actions::InitializeCharacteristic,
                 Actions::RequestBoundaryData<
                     typename Metavariables::cce_boundary_component>,
                 Actions::BlockUntilBoundaryDataReceived,
                 Actions::PopulateCharacteristicInitialHypersurface,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  template <typename BondiTag>
  using hypersurface_computation = tmpl::list<
      /*need something to check that the boundary data has been sent*/
      ::Actions::MutateApply<ComputeGaugeAdjustedBoundaryValue<BondiTag>>,
      Actions::CalculateIntegrandInputsForTag<BondiTag>,
      tmpl::transform<integrand_terms_to_compute_for_bondi_variable<BondiTag>,
                      tmpl::bind<::Actions::MutateApply,
                                 tmpl::bind<ComputeBondiIntegrand, tmpl::_1>>>,
      ::Actions::MutateApply<
          RadialIntegrateBondi<Tags::EvolutionGaugeBoundaryValue, BondiTag>>,
      /*Once we finish the U computation, we need to update all the
        time-dependent gauge stuff*/
      tmpl::conditional_t<
          cpp17::is_same_v<BondiTag, Tags::BondiU>,
          tmpl::list<
              ::Actions::MutateApply<GaugeUpdateU>,
              ::Actions::MutateApply<
                  ComputeGaugeAdjustedBoundaryValue<Tags::DuRDividedByR>>,
              ::Actions::MutateApply<PrecomputeCceDependencies<
                  Tags::EvolutionGaugeBoundaryValue, Tags::DuRDividedByR>>>,
          tmpl::list<>>>;

  using extract_action_list = tmpl::flatten<tmpl::list<
      Actions::BlockUntilBoundaryDataReceived,
      Actions::PrecomputeGlobalCceDependencies,
      tmpl::transform<bondi_hypersurface_step_tags,
                      tmpl::bind<hypersurface_computation, tmpl::_1>>,
      /*TODO we need to do something about options for this*/
      Actions::FilterSwshVolumeQuantity<Tags::BondiH>,
      // once we're done integrating, we may as well ask for the next step's
      // boundary data
      Actions::RequestNextBoundaryData<
          typename Metavariables::cce_boundary_component>,
      ::Actions::MutateApply<
          CalculateScriPlusValue<::Tags::dt<Tags::InertialRetardedTime>>>,
      ::Actions::MutateApply<ComputePreSwshDerivatives<Cce::Tags::SpecH>>,
      ::Actions::MutateApply<CalculateScriPlusValue<Tags::News>>,
      Actions::SendNewsToScri,
      ::Actions::RecordTimeStepperData<
          typename Metavariables::evolved_coordinates_variables_tag>,
      ::Actions::RecordTimeStepperDataSingleTensor<
          typename Metavariables::evolved_swsh_tag,
          typename Metavariables::evolved_swsh_dt_tag>,
      ::Actions::UpdateU<
          typename Metavariables::evolved_coordinates_variables_tag>,
      ::Actions::UpdateUSingleTensor<
          typename Metavariables::evolved_swsh_tag,
          typename Metavariables::evolved_swsh_dt_tag>,
      ::Actions::MutateApply<GaugeUpdateAngularFromCartesian<
          Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>,
      ::Actions::MutateApply<GaugeUpdateJacobianFromCoords<
          Tags::GaugeC, Tags::GaugeD, Tags::CauchyCartesianCoords,
          Tags::CauchyAngularCoords>>,
      ::Actions::MutateApply<GaugeUpdateOmegaCD>, ::Actions::AdvanceTime,
      ::Actions::RunEventsAndTriggers, Actions::ExitIfEndTimeReached>>;

  using registration_action_list = tmpl::list<
      // ::observers::Actions::RegisterSingletonWithObserverWriter<
      // RegistrationHelper>,
      ::observers::Actions::RegisterSingletonWithObservers<RegistrationHelper>,
      Parallel::Actions::TerminatePhase>;

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

  using const_global_cache_tag_list =
      Parallel::get_const_global_cache_tags_from_pdal<
          phase_dependent_action_list>;

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
      Parallel::get_parallel_component<CharacteristicExtractor<Metavariables>>(
          local_cache)
          .start_phase(next_phase);
    }
  }
};
}  // namespace Cce
