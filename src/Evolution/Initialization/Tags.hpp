// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Options/Options.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Initialization {
namespace OptionTags {
struct FuncOfTimeFromFile {
  static constexpr OptionString help{
      "Options for reading a FunctionOfTime from an H5 file"};
};

struct FuncOfTimeFile {
  using type = std::string;
  static constexpr OptionString help{"Path to an H5 Dat file"};
  using group = FuncOfTimeFromFile;
};

struct FuncOfTimeSetNames {
  using type = std::vector<std::array<std::string, 2>>;
  static constexpr OptionString help{
      "String pairs mapping datasets to arbitrary names"};
  using group = FuncOfTimeFromFile;
};
}  // namespace OptionTags

/// \ingroup InitializationGroup
/// \brief %Tags used during initialization of parallel components.
namespace Tags {
/// \ingroup ControlSystemGroup
/// \brief The path to an H5 Dat file containing function-of-time data to read.
struct FuncOfTimeFile : db::SimpleTag {
  static std::string name() noexcept { return "FuncOfTimeFile"; }
  using type = std::string;
  using option_tags =
      tmpl::list<::Initialization::OptionTags::FuncOfTimeFile>;
  template <typename Metavariables>
  static std::string create_from_options(
      const std::string& function_of_time_file) noexcept {
    return function_of_time_file;
  }
};

/// \ingroup ControlSystemGroup
/// \brief List of pairs of strings mapping datasets to unordered-map keys
///
/// The first string in each pair names a dataset in an H5 file, and the
/// second string in each pair is the key that will be used to index
/// the `FunctionOfTime` in an unordered map after reading it.
struct FuncOfTimeSetNames : db::SimpleTag {
  static std::string name() noexcept { return "FuncOfTimeSetNames"; }
  using type = std::vector<std::array<std::string, 2>>;
  using option_tags =
      tmpl::list<::Initialization::OptionTags::FuncOfTimeSetNames>;
  template <typename Metavariables>
  static std::vector<std::array<std::string, 2>> create_from_options(
      const std::vector<std::array<std::string, 2>>& dataset_names) noexcept {
    return dataset_names;
  }
};

struct InitialTime : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<::OptionTags::InitialTime>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double initial_time) noexcept {
    return initial_time;
  }
};

struct InitialTimeDelta : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<::OptionTags::InitialTimeStep>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double initial_time_step) noexcept {
    return initial_time_step;
  }
};

template <bool UsingLocalTimeStepping>
struct InitialSlabSize : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<::OptionTags::InitialSlabSize>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double initial_slab_size) noexcept {
    return initial_slab_size;
  }
};

template <>
struct InitialSlabSize<false> : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<::OptionTags::InitialTimeStep>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double initial_time_step) noexcept {
    return std::abs(initial_time_step);
  }
};
}  // namespace Tags
}  // namespace Initialization
