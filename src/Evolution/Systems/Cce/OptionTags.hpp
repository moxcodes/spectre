// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"

namespace Cce {
namespace OptionTags{

struct LMax {
  using type = size_t;
  static constexpr OptionString help{
      "maximum l value for spin-weighted spherical harmonics, determines "
      "angular resolution."};
};

struct NumberOfRadialPoints {
  using type = size_t;
  static constexpr OptionString help{
      "Number of radial grid points in the spherical domain"};
};
}  // namespace OptionTags
}  // namespace Cce
