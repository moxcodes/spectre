// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

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
#include "Utilities/Requires.hpp"
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
///   - `::domain::Tags::FunctionsOfTime`. This tag contains a `FunctionOfTime`
///   for each dat file read from the H5 file
///
/// - Removes: nothing
/// - Modifies: nothing
///
///
/// Columns in the file to be read must have the following form:
///   - 0 = time
///   - 1 = time of last update
///   - 2 = number of components
///   - 3 = maximum derivative order
///   - 4 = version
///   - 5 = function
///   - 6 = d/dt (function)
///   - 7 = d^2/dt^2 (function)
///   - 8 = d^3/dt^3 (function)
///
/// If the function has more than one component, columns 5-8 give
/// the first component and its derivatives, columns 9-12 give the second
/// component and its derivatives, etc.
///
struct FunctionOfTimeFromFile {
  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<
          db::tag_is_retrievable_v<Initialization::Tags::FuncOfTimeFile,
                                   db::DataBox<DbTagsList>> and
          db::tag_is_retrievable_v<Initialization::Tags::FuncOfTimeSetNames,
                                   db::DataBox<DbTagsList>>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& file_name{
        db::get<Initialization::Tags::FuncOfTimeFile>(box)};

    // Each element of dataset_names has two elements:
    // element 0 is the dataset name in the h5 file,
    // and element 1 is the name used in the std::unordered_map that
    // will store the corresponding read-in function of time
    const std::vector<std::array<std::string, 2>>& dataset_names{
        db::get<Initialization::Tags::FuncOfTimeSetNames>(box)};

    // Currently, only support order 3 piecewise polynomials
    // This could be generalized later, but the SpEC functions of time
    // that we will read in with this action always use max_deriv = 3
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
      for (size_t deriv_order = 0; deriv_order < max_deriv + 1; ++deriv_order) {
        gsl::at(initial_coefficients, deriv_order) =
            DataVector(number_of_components);
        for (size_t component = 0; component < number_of_components;
             ++component) {
          gsl::at(initial_coefficients, deriv_order)[component] =
              dat_data(0, 5 + (max_deriv + 1) * component + deriv_order);
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
