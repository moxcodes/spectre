// Distributed under the MIT License.
// See LICENSE.txt for details.

//// This file is only for my own practice, locally.
//// If it appears in a PR, please tell me that I have
//// made a version control error and that it should be removed.

#pragma once

#include "DataStructures/ComplexModalVector.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Dat.hpp"


struct ModeRecorder {
 public:
  ModeRecorder(std::string filename, size_t l_max, size_t info_l_max) noexcept
      : output_file_{filename} {
    // TODO: write the simulation l_max in a better place. The current hdf5
    // utilities don't offer much help so this is an acceptable stop-gap.
    file_legend_.push_back("times, sim lmax: " + std::to_string(info_l_max));
    for(int i = 0; i <= l_max; ++i){
      for(int j = -i; j <= i; ++j) {
        file_legend_.push_back("Real Y_" + std::to_string(i) + "," +
                              std::to_string(j));
        file_legend_.push_back("Imag Y_" + std::to_string(i) + "," +
                              std::to_string(j));
      }
    }

  }

  void append_mode_data(std::string path, double time, ComplexModalVector modes,
                        size_t l_max) {
    auto& output_mode_dataset =
        output_file_.try_insert<h5::Dat>(path, file_legend_, 0);
    size_t output_size = square(l_max + 1);
    std::vector<double> data_to_write(2 * output_size + 1);
    data_to_write[0] = time;
    for(size_t i = 0; i < output_size; ++i) {
      data_to_write[2 * i + 1] = real(modes[i]);
      data_to_write[2 * i + 2] = imag(modes[i]);
    }
    output_mode_dataset.append(data_to_write);
    output_file_.close_current_object();
  }

 private:
  h5::H5File<h5::AccessType::ReadWrite> output_file_;
  std::vector<std::string> file_legend_;
};
