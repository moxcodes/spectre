// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/HasReceivedFromAllMortars.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Lb {
namespace Actions {

template <size_t Dim>
struct SendDataToNeighbors {
  static constexpr size_t volume_dim = Dim;

  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementId<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if(db::get<Tags::StepNumber>(box) >=
       db::get<Tags::NumberOfSteps>(box)){
      return std::forward_as_tuple(std::move(box));
    }
    const auto& element = db::get<domain::Tags::Element<volume_dim>>(box);
    auto& receiver_proxy =
        Parallel::get_parallel_component<LoadBalancingTestArray<Metavariables>>(
            cache);
    auto& internal_storage = db::get<Tags::InternalStorage>(box);
    size_t i = 0;
    // just send the first collection of vectors to the neighbors
    for (const auto& direction_and_neighbors : element.neighbors()) {
      for (const auto& neighbor : direction_and_neighbors.second) {
        std::vector<DataVector> bundled_data(
            db::get<Tags::CommunicationSize>(box));
        for (auto& vector_entry : bundled_data) {
          vector_entry = internal_storage[i % internal_storage.size()];
          ++i;
        }
        const auto& direction = direction_and_neighbors.first;
        const auto& orientation = direction_and_neighbors.second.orientation();
        const auto direction_from_neighbor = orientation(direction.opposite());

        // if this isn't enforced, sometimes when running on few cores, the
        // element will mistakenly start back up the algorithm on an element
        // that's still in Initialization, which will cause incomrehensible
        // errors
        if (db::get<Tags::StepNumber>(box) > 1) {
          Parallel::receive_data<ReceiveTags::LoadBalancingCommunication<Dim>>(
              receiver_proxy[neighbor], db::get<Tags::StepNumber>(box),
              std::make_pair(dg::MortarId<volume_dim>{direction_from_neighbor,
                                                      element.id()},
                             std::move(bundled_data)),
              true);
        } else {
          Parallel::receive_data<ReceiveTags::LoadBalancingCommunication<Dim>>(
              receiver_proxy[neighbor], db::get<Tags::StepNumber>(box),
              std::make_pair(dg::MortarId<volume_dim>{direction_from_neighbor,
                                                      element.id()},
                             std::move(bundled_data)));
        }
      }
    }
    return std::forward_as_tuple(std::move(box));
  }
};

template <size_t Dim>
struct ReceiveDataFromNeighbors {
  using inbox_tags = tmpl::list<ReceiveTags::LoadBalancingCommunication<Dim>>;
  static constexpr size_t volume_dim = Dim;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (UNLIKELY(
            get<domain::Tags::Element<volume_dim>>(box).number_of_neighbors() ==
                0 or
            db::get<Tags::StepNumber>(box) == 0)) {
      return {std::move(box)};
    }

    const auto& step = db::get<Tags::StepNumber>(box) - 1;
    auto& inbox =
        tuples::get<ReceiveTags::LoadBalancingCommunication<Dim>>(inboxes)
            .find(step)
            ->second;
    db::mutate<Tags::NeighborData<Dim>>(
        make_not_null(&box),
        [&inbox](const gsl::not_null<FixedHashMap<
                     maximum_number_of_neighbors(Dim), dg::MortarId<Dim>,
                     std::vector<DataVector>, boost::hash<dg::MortarId<Dim>>>*>
                     neighbor_data) noexcept {
          for (auto& received_mortar_data : inbox) {
            neighbor_data->at(received_mortar_data.first) =
                std::move(received_mortar_data.second);
          }
        });
    tuples::get<ReceiveTags::LoadBalancingCommunication<Dim>>(inboxes).erase(
        step);
    return {std::move(box)};
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    return db::get<Tags::StepNumber>(box) == 0 or
           dg::has_received_from_all_mortars<
               ReceiveTags::LoadBalancingCommunication<Dim>>(
               db::get<Tags::StepNumber>(box) - 1,
               get<domain::Tags::Element<volume_dim>>(box), inboxes);
  }
};

}  // namespace Actions
}  // namespace Lb
