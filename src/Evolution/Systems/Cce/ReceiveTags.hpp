// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/System.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Time/TimeStepId.hpp"

namespace Cce {
namespace ReceiveTags {

/// A receive tag for the data sent to the CCE evolution component from the CCE
/// boundary component
template <typename CommunicationTagList>
struct BoundaryData
    : Parallel::InboxInserters::Value<BoundaryData<CommunicationTagList>> {
  using temporal_id = TimeStepId;
  using type = std::unordered_map<temporal_id, Variables<CommunicationTagList>>;
};

struct JHypersurfaceData : Parallel::InboxInserters::Value<JHypersurfaceData> {
  using temporal_id = size_t;
  using type = std::unordered_map<
      size_t,
      tuples::TaggedTuple<Tags::BondiJ, Tags::InertialRetardedTime,
                          Tags::LMax<InitializationRun>,
                          Tags::NumberOfRadialPoints<InitializationRun>>>;
};

}  // namespace ReceiveTags
}  // namespace Cce
