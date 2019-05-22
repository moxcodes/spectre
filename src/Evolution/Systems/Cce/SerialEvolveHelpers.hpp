// Distributed under the MIT License.
// See LICENSE.txt for details.

//// This file is only for my own practice, locally.
//// If it appears in a PR, please tell me that I have
//// made a version control error and that it should be removed.

#pragma once

#include "DataStructures/Variables.hpp"

#include <cstdio>

#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Equations.hpp"
#include "Evolution/Systems/Cce/LinearSolve.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "Time/History.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"

namespace Cce {

struct ModeComparisonManager {
 public:
  std::string dataset_name_for_mode(size_t l, int m) noexcept {
    return "/Y_l" + std::to_string(l) + "_m" + std::to_string(m);
  }
  ModeComparisonManager(std::string filename, size_t l_max) noexcept
      : l_max_{l_max}, mode_data_file_{filename} {
    auto& mode00_data =
        mode_data_file_.get<h5::Dat>(dataset_name_for_mode(0, 0));
    Matrix time_matrix =
        mode00_data.get_data_subset(std::vector<size_t>{0}, 0, 1);
    time_ = time_matrix(0, 0);
    modes_ = ComplexModalVector{square(l_max + 1)};
    for (int l = 0; l <= static_cast<int>(l_max); ++l) {
      for (int m = -l; m <= l; ++m) {
        auto& mode_data = mode_data_file_.get<h5::Dat>(
            dataset_name_for_mode(static_cast<size_t>(l), m));
        Matrix data_matrix =
            mode_data.get_data_subset(std::vector<size_t>{1, 2}, 0, 1);
        modes_[static_cast<size_t>(static_cast<int>(square(l) + l) + m)] =
            std::complex<double>(data_matrix(0, 0), data_matrix(0, 1));
      }
    }
  }

  template <typename T>
  ComplexModalVector mode_difference(const double time,
                                     const T& vector_to_compare) noexcept {
    // TODO for now, just find the nearest time. In future, we might want to
    // adapt this to perform interpolation. However, because that task is very
    // similar to CceBoundaryDataManager, we should come up with a unified
    // interface to avoid code duplication.
    auto& mode00_data =
        mode_data_file_.get<h5::Dat>(dataset_name_for_mode(0, 0));

    Matrix time_set = mode00_data.get_data_subset(
        std::vector<size_t>{0}, 0, mode00_data.get_dimensions()[0]);
    // TODO binary search
    size_t closest_time = 0;
    for(size_t i = 0; i < time_set.rows(); ++i) {
      if (abs(time_set(i, 0) - time) < abs(time - time_set(closest_time, 0))) {
        closest_time = i;
      }
    }

    auto mode_difference_at_time = ComplexModalVector{square(l_max_ + 1), 0.0};
    for (int l = 0; l <= static_cast<int>(l_max_); ++l) {
      for (int m = -l; m <= l; ++m) {
        auto& mode_data = mode_data_file_.get<h5::Dat>(
            dataset_name_for_mode(static_cast<size_t>(l), m));
        Matrix data_matrix = mode_data.get_data_subset(
            std::vector<size_t>{1, 2}, closest_time, 1);
        mode_difference_at_time[static_cast<size_t>(
            static_cast<int>(square(l) + l) + m)] =
            std::complex<double>(data_matrix(0, 0), data_matrix(0, 1)) -
            vector_to_compare[static_cast<size_t>(
                static_cast<int>(square(l) + l) + m)];
      }
    }
    return mode_difference_at_time;
  }

  double get_time() noexcept { return time_; }

 private:
  size_t l_max_;
  double time_;
  ComplexModalVector modes_;
  h5::H5File<h5::AccessType::ReadOnly> mode_data_file_;
};
}  // namespace Cce
