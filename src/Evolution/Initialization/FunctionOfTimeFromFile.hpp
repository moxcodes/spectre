// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Initialization {
namespace Actions {
/// \ingroup InitializationGroup
/// \brief Initialize a `FunctionsOfTime` from dat files in an
/// H5 file.
///
/// Uses:
///  - DataBox:
///    - `Initialization::Tags::FuncOfTimeFile`
///    - `Initialization::Tags::FuncOfTimeSetNames`
///
/// DataBox changes:
/// - Adds:
///   - A `FunctionsOfTime` containing a `FunctionOfTime` for each dat
///   file read from the H5 file
///
/// - Removes: nothing
/// - Modifies: nothing
struct FunctionOfTimeFromFile {
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& file_name{
        db::get<Initialization::Tags::FuncOfTimeFile>(box)};

    // Each element of dataset_names is a std::array<std::string, 2>, where
    // element 0 is the dataset name in the h5 file, and element 1 is the
    // name used in the std::unordered_map that will store the read-in functions
    // of time
    const auto& dataset_names{
        db::get<Initialization::Tags::FuncOfTimeSetNames>(box)};

    // Currently, only support order 3 piecewise polynomials
    // This could be generalized later, if needed
    constexpr size_t max_deriv{3};

    std::unordered_map<std::string,
                       domain::FunctionsOfTime::PiecewisePolynomial<max_deriv>>
        functions_of_time;

    h5::H5File<h5::AccessType::ReadOnly> file(file_name);
    for (auto dataset_name : dataset_names) {
      const auto& dataset = dataset_name[0];
      const auto& name = dataset_name[1];
      const auto& dat_file = file.get<h5::Dat>("/" + dataset);
      const auto& dat_data = dat_file.get_data();

      // Check that the data in the file uses deriv order 3
      // Column 3 of the file contains the derivative order
      const size_t dat_max_deriv = dat_data(0, 3);
      ASSERT(dat_max_deriv == max_deriv,
             "Deriv order in " << file_name << " should be " << max_deriv
                               << ", not " << dat_max_deriv);

      // Get the initial time ('time of last update') from the file
      // and the values of the function and its derivatives at that time
      const double start_time = dat_data(0, 1);

      // Currently, assume the same number of components are used
      // at each time. This could be generalized if needed
      const size_t number_of_components = dat_data(0, 2);

      std::array<DataVector, max_deriv + 1> initial_coefficients;
      for (size_t i = 0; i < max_deriv + 1; ++i) {
        gsl::at(initial_coefficients, i) = DataVector(number_of_components);
        for (size_t a = 0; a < number_of_components; ++a) {
          // Columns in the file have the following form::
          // 0 = time
          // 1 = time of last update
          // 2 = number of components
          // 3 = maximum derivative order
          // 4 = version
          // 5, ... 5 + maximum derivative order: first component and derivs
          // ... (sets of coefficients for the next component and derivs)
          // ...
          gsl::at(initial_coefficients, i)[a] =
              dat_data(0, 5 + (max_deriv + 1) * a + i);
        }
      }
      functions_of_time[name] = domain::FunctionsOfTime::PiecewisePolynomial<3>(
          start_time, initial_coefficients);

      // Loop over the remaining times, updating the function of time
      DataVector highest_derivative(number_of_components);
      for (size_t row = 1; row < dat_data.rows(); ++row) {
        for (size_t a = 0; a < number_of_components; ++a) {
          highest_derivative[a] =
              dat_data(row, 4 + (max_deriv + 1) * a + max_deriv);
        }
        functions_of_time[name].update(dat_data(row, 1), highest_derivative);
      }
    }

    // Create an unordered map of the same type as the FunctionsOfTime tag,
    // and then add this to the databox
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time_for_databox;
    for (auto dataset_name : dataset_names) {
      const auto& name = dataset_name[1];
      functions_of_time_for_databox[name] = std::make_unique<
          domain::FunctionsOfTime::PiecewisePolynomial<max_deriv>>(
          std::move(functions_of_time[name]));
    }
    return std::make_tuple(::Initialization::merge_into_databox<
                           FunctionOfTimeFromFile,
                           db::AddSimpleTags<::domain::Tags::FunctionsOfTime>>(
        std::move(box), std::move(functions_of_time_for_databox)));
  }
};
}  // namespace Actions
}  // namespace Initialization
