// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "Parallel/InboxInserters.hpp"

namespace Parallel {
namespace ReceiveTags {
struct IsRestartingFromCheckpointSync
    : Parallel::InboxInserters::Value<IsRestartingFromCheckpointSync> {
  using temporal_id = size_t; // unused
  using type = std::unordered_map<temporal_id, bool>;
};
}  // namespace ReceiveTags

namespace Tags {
struct IsSyncingForCheckpoint : db::SimpleTag {
  using type = bool;
};
}
}  // namespace Parallel
