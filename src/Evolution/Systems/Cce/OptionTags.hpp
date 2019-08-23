// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"
#include "ReadBoundaryDataH5.hpp"

namespace Cce {
namespace OptionTags {

struct LMax {
  using type = size_t;
  static constexpr OptionString help{
      "maximum l value for spin-weighted spherical harmonics"};
};

struct ObservationLMax {
  using type = size_t;
  static constexpr OptionString help{"maximum l value for swsh output"};
};

struct NumberOfRadialPoints {
  using type = size_t;
  static constexpr OptionString help{
      "Number of radial grid points in the spherical domain"};
};

struct EndTime {
  using type = double;
  static constexpr OptionString help{"End time for the Cce Evolution."};
  static double default_value() noexcept {
    return std::numeric_limits<double>::quiet_NaN();
  }
};

struct StartTime {
  using type = double;
  static constexpr OptionString help{"Start time for the Cce Evolution."};
};

struct TargetStepSize {
  using type = double;
  static constexpr OptionString help{"Target time step size for Cce Evolution"};
};

struct BoundaryDataFilename {
  using type = std::string;
  static constexpr OptionString help{
      "h5 file to read the wordltube data from."};
};

struct H5LookaheadPoints {
  using type = size_t;
  static constexpr OptionString help{
      "Number of points to cache on each file read operation."};
  static size_t default_value() noexcept { return 200; }
};

struct H5Interpolator {
  static std::string name() noexcept { return "H5Interpolator"; }
  using type = std::unique_ptr<Interpolator>;
  static constexpr OptionString help{
      "The interpolator for imported h5 worldtube data."};
  // default values are hard for the factory mechanism :(
  // static std::unique_ptr<Interpolator> default_value() noexcept {
  // return std::unique_ptr<Interpolator>{new FixedBarycentricInterpolator{5}};
  // }
};

}  // namespace OptionTags
}  // namespace Cce
