// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterpolationStrategies.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Parallel/Printf.hpp"

namespace Cce {
namespace OptionTags {

/// %Option group
struct Cce {
  static constexpr Options::String help = {
      "Options for the Cce evolution system"};
};

/// %Option group
struct Filtering {
  static constexpr Options::String help = {"Options for the filtering in Cce"};
  using group = Cce;
};

struct LMax {
  using type = size_t;
  static constexpr Options::String help{
      "Maximum l value for spin-weighted spherical harmonics"};
  using group = Cce;
};

struct FilterLMax {
  using type = size_t;
  static constexpr Options::String help{"l mode cutoff for angular filtering"};
  using group = Filtering;
};

struct RadialFilterAlpha {
  using type = double;
  static constexpr Options::String help{
      "alpha parameter in exponential radial filter"};
  using group = Filtering;
};

struct RadialFilterHalfPower {
  using type = size_t;
  static constexpr Options::String help{
      "Half-power of the exponential radial filter argument"};
  using group = Filtering;
};

struct ObservationLMax {
  using type = size_t;
  static constexpr Options::String help{"Maximum l value for swsh output"};
  using group = Cce;
};

struct NumberOfRadialPoints {
  using type = size_t;
  static constexpr Options::String help{
      "Number of radial grid points in the spherical domain"};
  using group = Cce;
};

struct ExtractionRadius {
  using type = double;
  static constexpr Options::String help{"Extraction radius of the CCE system."};
  using group = Cce;
};

struct StandaloneExtractionRadius {
  static std::string name() noexcept { return "ExtractionRadius"; }
  using type = Options::Auto<double>;

  static constexpr Options::String help{
    "Extraction radius of the CCE system for a standalone run. This may be "
    "set to \"Auto\" to infer the radius from the filename (often used for "
    "SpEC worldtube data). This option is unused if `H5IsBondiData` is "
    "`true`, and should be \"Auto\" for such runs."};
  using group = Cce;
};

struct EndTime {
  using type = Options::Auto<double>;
  static constexpr Options::String help{"End time for the Cce Evolution."};
  static type default_value() noexcept { return {}; }
  using group = Cce;
};

struct StartTime {
  using type = Options::Auto<double>;
  static constexpr Options::String help{
      "Cce Start time (default to earliest possible time)."};
  static type default_value() noexcept { return {}; }
  using group = Cce;
};

struct TargetStepSize {
  using type = double;
  static constexpr Options::String help{
      "Target time step size for Cce Evolution"};
  using group = Cce;
};

struct BoundaryDataFilename {
  using type = std::string;
  static constexpr Options::String help{
      "H5 file to read the wordltube data from."};
  using group = Cce;
};

struct H5LookaheadTimes {
  using type = size_t;
  static constexpr Options::String help{
      "Number of times steps from the h5 to cache each read."};
  static size_t default_value() noexcept { return 200; }
  using group = Cce;
};

struct H5Interpolator {
  using type = std::unique_ptr<intrp::SpanInterpolator>;
  static constexpr Options::String help{
      "The interpolator for imported h5 worldtube data."};
  using group = Cce;
};

struct H5IsBondiData {
  using type = bool;
  static constexpr Options::String help{
      "true for boundary data in Bondi form, false for metric data. Metric "
      "data is more readily available from Cauchy simulations, so historically "
      "has been the typical format provided by SpEC simulations. Bondi data is "
      "much more efficient for storage size and performance, but both must be "
      "supported for compatibility with current CCE data sources."};
  static bool default_value() noexcept { return false; }
  using group = Cce;
};

struct FixSpecNormalization {
  using type = bool;
  static constexpr Options::String help{
      "Set to true if corrections for SpEC data impurities should be applied "
      "automatically based on the `VersionHist.ver` data set in the H5. "
      "Typically, this should be set to true if the metric data is created "
      "from SpEC, and false otherwise."};
  using group = Cce;
};

struct GhInterfaceManager {
  using type = std::unique_ptr<InterfaceManagers::GhInterfaceManager>;
  static constexpr Options::String help{
      "Class to manage worldtube data from a GH system."};
  using group = Cce;
};

struct ScriInterpolationOrder {
  static std::string name() noexcept { return "ScriInterpOrder"; }
  using type = size_t;
  static constexpr Options::String help{
      "Order of time interpolation at scri+."};
  static size_t default_value() noexcept { return 5; }
  using group = Cce;
};

struct ScriOutputDensity {
  using type = size_t;
  static constexpr Options::String help{
      "Number of scri output points per timestep."};
  static size_t default_value() noexcept { return 1; }
  using group = Cce;
};

struct InitializeJ {
  using type = std::unique_ptr<::Cce::InitializeJ::InitializeJ>;
  static constexpr Options::String help{
      "The initialization for the first hypersurface for J"};
  using group = Cce;
};

}  // namespace OptionTags

namespace InitializationTags {
struct ScriInterpolationOrder : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::ScriInterpolationOrder>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(
      const size_t scri_plus_interpolation_order) noexcept {
    return scri_plus_interpolation_order;
  }
};

struct TargetStepSize : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::TargetStepSize>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double target_step_size) noexcept {
    return target_step_size;
  }
};

struct ExtractionRadius : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::ExtractionRadius>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double extraction_radius) noexcept {
    return extraction_radius;
  }
};

struct ScriOutputDensity : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::ScriOutputDensity>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t scri_output_density) noexcept {
    return scri_output_density;
  }
};
}  // namespace InitializationTags

namespace Tags {
/// A tag that constructs a `MetricWorldtubeDataManager` from options
struct H5WorldtubeBoundaryDataManager : db::SimpleTag {
  using type = std::unique_ptr<WorldtubeDataManager>;
  using option_tags =
      tmpl::list<OptionTags::LMax, OptionTags::BoundaryDataFilename,
                 OptionTags::H5LookaheadTimes, OptionTags::H5Interpolator,
                 OptionTags::H5IsBondiData, OptionTags::FixSpecNormalization,
                 OptionTags::StandaloneExtractionRadius>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const size_t l_max, const std::string& filename,
      const size_t number_of_lookahead_times,
      const std::unique_ptr<intrp::SpanInterpolator>& interpolator,
      const bool h5_is_bondi_data, const bool fix_spec_normalization,
      const std::optional<double> extraction_radius) noexcept {
    if (h5_is_bondi_data) {
      if (static_cast<bool>(extraction_radius)) {
        Parallel::printf(
            "Warning: Option ExtractionRadius is set to a specific value and "
            "H5IsBondiData is set to `true` -- the ExtractionRadius will not "
            "be used, because all radius information is specified in the input "
            "file for the Bondi worldtube data format. It is recommended to "
            "set `ExtractionRadius` to `\"Auto\"` to make the input file "
            "clearer.");
      }
      return std::make_unique<BondiWorldtubeDataManager>(
          std::make_unique<BondiWorldtubeH5BufferUpdater>(filename,
                                                          extraction_radius),
          l_max, number_of_lookahead_times, interpolator->get_clone());
    } else {
      return std::make_unique<MetricWorldtubeDataManager>(
          std::make_unique<MetricWorldtubeH5BufferUpdater>(filename,
                                                           extraction_radius),
          l_max, number_of_lookahead_times, interpolator->get_clone(),
          fix_spec_normalization);
    }
  }
};

struct LMax : db::SimpleTag, Spectral::Swsh::Tags::LMaxBase {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::LMax>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t l_max) noexcept {
    return l_max;
  }
};

struct NumberOfRadialPoints : db::SimpleTag,
                              Spectral::Swsh::Tags::NumberOfRadialPointsBase {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::NumberOfRadialPoints>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(
      const size_t number_of_radial_points) noexcept {
    return number_of_radial_points;
  }
};

struct ObservationLMax : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::ObservationLMax>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t observation_l_max) noexcept {
    return observation_l_max;
  }
};

struct FilterLMax : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::FilterLMax>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t filter_l_max) noexcept {
    return filter_l_max;
  }
};

struct RadialFilterAlpha : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::RadialFilterAlpha>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double radial_filter_alpha) noexcept {
    return radial_filter_alpha;
  }
};

struct RadialFilterHalfPower : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::RadialFilterHalfPower>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(
      const size_t radial_filter_half_power) noexcept {
    return radial_filter_half_power;
  }
};

/// \brief Represents the start time of a bounded CCE evolution, determined
/// either from option specification or from the file
///
/// \details If no start time is specified in the input file (so the option
/// `OptionTags::StartTime` is set to "Auto"), this will find the start time
/// from the provided H5 file. If `OptionTags::StartTime` takes any other value,
/// it will be used directly as the start time for the CCE evolution instead.
struct StartTimeFromFile : Tags::StartTime, db::SimpleTag {
  using type = double;
  using option_tags =
      tmpl::list<OptionTags::StartTime, OptionTags::BoundaryDataFilename,
                 OptionTags::H5IsBondiData>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const std::optional<double> start_time,
                                    const std::string& filename,
                                    const bool is_bondi_data) noexcept {
    if (start_time) {
      return *start_time;
    }
    if (is_bondi_data) {
      BondiWorldtubeH5BufferUpdater h5_boundary_updater{filename};
      const auto& time_buffer = h5_boundary_updater.get_time_buffer();
      return time_buffer[0];
    } else {
      MetricWorldtubeH5BufferUpdater h5_boundary_updater{filename};
      const auto& time_buffer = h5_boundary_updater.get_time_buffer();
      return time_buffer[0];
    }
  }
};

/// \brief Represents the start time of a bounded CCE evolution that must be
/// supplied in the input file (for e.g. analytic tests).
struct SpecifiedStartTime : Tags::StartTime, db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::StartTime>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double start_time) noexcept {
    return start_time;
  }
};

/// \brief Represents the final time of a bounded CCE evolution, determined
/// either from option specification or from the file
///
/// \details If no end time is specified in the input file (so the option
/// `OptionTags::EndTime` is set to "Auto"), this will find the end time
/// from the provided H5 file. If `OptionTags::EndTime` takes any other value,
/// it will be used directly as the final time for the CCE evolution instead.
struct EndTimeFromFile : Tags::EndTime, db::SimpleTag {
  using type = double;
  using option_tags =
      tmpl::list<OptionTags::EndTime, OptionTags::BoundaryDataFilename,
                 OptionTags::H5IsBondiData>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const std::optional<double> end_time,
                                    const std::string& filename,
                                    const bool is_bondi_data) {
    if (end_time) {
      return *end_time;
    }
    if (is_bondi_data) {
      BondiWorldtubeH5BufferUpdater h5_boundary_updater{filename};
      const auto& time_buffer = h5_boundary_updater.get_time_buffer();
        return time_buffer[time_buffer.size() - 1];
    } else {
      MetricWorldtubeH5BufferUpdater h5_boundary_updater{filename};
      const auto& time_buffer = h5_boundary_updater.get_time_buffer();
      return time_buffer[time_buffer.size() - 1];
    }
  }
};

/// \brief Represents the final time of a CCE evolution that should just proceed
/// until it receives no more boundary data and becomes quiescent.
struct NoEndTime : Tags::EndTime, db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options() noexcept {
    return std::numeric_limits<double>::infinity();
  }
};

/// \brief Represents the final time of a bounded CCE evolution that must be
/// supplied in the input file (for e.g. analytic tests).
struct SpecifiedEndTime : Tags::EndTime, db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::EndTime>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double end_time) noexcept {
    return end_time;
  }
};

struct GhInterfaceManager : db::SimpleTag {
  using type = std::unique_ptr<InterfaceManagers::GhInterfaceManager>;
  using option_tags = tmpl::list<OptionTags::GhInterfaceManager>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<InterfaceManagers::GhInterfaceManager>
  create_from_options(
      const std::unique_ptr<InterfaceManagers::GhInterfaceManager>&
          interface_manager) noexcept {
    return interface_manager->get_clone();
  }
};

/// \brief Intended for use in the const global cache to communicate to the
/// sending elements when they should be sending worldtube data for CCE to the
/// interpolator.
///
/// \details This tag is not specifiable by independent options in the yaml, and
/// instead is entirely determined by the choice of interface manager, which
/// sets by virtual member function the interpolation strategy that is
/// compatible with the interface manager. The choice to extract this
/// information at option-parsing is to avoid needing to pass any information
/// from the interpolation manager that is typically stored in the
/// `WorldtubeBoundary` component \ref DataBoxGroup to the components that
/// provide data for CCE.
struct InterfaceManagerInterpolationStrategy : db::SimpleTag {
  using type = InterfaceManagers::InterpolationStrategy;
  using option_tags = tmpl::list<OptionTags::GhInterfaceManager>;

  static constexpr bool pass_metavariables = false;
  static InterfaceManagers::InterpolationStrategy
  create_from_options(
      const std::unique_ptr<InterfaceManagers::GhInterfaceManager>&
          interface_manager) noexcept {
    return interface_manager->get_interpolation_strategy();
  }
};

struct InitializeJ : db::SimpleTag {
  using type = std::unique_ptr<::Cce::InitializeJ::InitializeJ>;
  using option_tags = tmpl::list<OptionTags::InitializeJ>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<::Cce::InitializeJ::InitializeJ> create_from_options(
      const std::unique_ptr<::Cce::InitializeJ::InitializeJ>&
          initialize_j) noexcept {
    return initialize_j->get_clone();
  }
};

}  // namespace Tags
}  // namespace Cce
