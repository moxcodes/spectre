// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/MaxNumberOfNeighbors.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Options/Options.hpp"
#include "Parallel/InboxInserters.hpp"

namespace Lb {
namespace OptionTags {

struct TestLoadBalancing {
  static constexpr OptionString help{
      "Options for the load-balancer test array"};
};

struct ExecutionLoad {
  using type = size_t;
  static constexpr OptionString help{
      "Number of times to iterate the test load function"};
  using group = TestLoadBalancing;
};

struct InternalStorageSize {
  using type = size_t;
  static constexpr OptionString help{"Multiplier for internal storage cost"};
  using group = TestLoadBalancing;
};

struct CommunicationSize {
  using type = size_t;
  static constexpr OptionString help{
      "Multiplier for amount of data to send between elements"};
  using group = TestLoadBalancing;
};

struct NumberOfSteps {
  using type = size_t;
  static constexpr OptionString help{"number of steps to run"};
  using group = TestLoadBalancing;
};
}  // namespace OptionTags

namespace Tags {

struct ExecutionLoad : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::ExecutionLoad>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t execution_load) noexcept {
    return execution_load;
  }
};

struct InternalStorageSize : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::InternalStorageSize>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(
      const size_t internal_storage_size) noexcept {
    return internal_storage_size;
  }
};

struct CommunicationSize : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::CommunicationSize>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(
      const size_t communication_size) noexcept {
    return communication_size;
  }
};

struct NumberOfSteps : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::NumberOfSteps>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t number_of_steps) noexcept {
    return number_of_steps;
  }
};

struct StepNumber :db::SimpleTag {
  using type = size_t;
};

template <size_t Dim>
struct NeighborData : db::SimpleTag {
  using type =
      FixedHashMap<maximum_number_of_neighbors(Dim), dg::MortarId<Dim>,
                   std::vector<DataVector>, boost::hash<dg::MortarId<Dim>>>;
};

struct InternalStorage : db::SimpleTag {
  using type = std::vector<DataVector>;
};
}  // namespace Tags

namespace ReceiveTags {

template <size_t Dim>
struct LoadBalancingCommunication
    : Parallel::InboxInserters::Map<LoadBalancingCommunication<Dim>> {
  using temporal_id = size_t;
  using type = std::unordered_map<
      temporal_id,
      FixedHashMap<maximum_number_of_neighbors(Dim), dg::MortarId<Dim>,
                   std::vector<DataVector>, boost::hash<dg::MortarId<Dim>>>>;
};
}  // namespace ReceiveTags
}  // namespace Lb
