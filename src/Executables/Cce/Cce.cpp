// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <iostream>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/Cce/CceEvolve.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

int main(int argc, char** argv) {
  boost::program_options::positional_options_description pos_desc;
  pos_desc.add("input_file", 1).add("output_suffix", 1);
  boost::program_options::options_description desc("Options");

  desc.add_options()("help", "show this help message")(
      "l_max", boost::program_options::value<size_t>()->default_value(16),
      "maximum l angular resolution")(
      "output_l_max", boost::program_options::value<size_t>()->default_value(8),
      "maximum l angular resolution to export to the h5")(
      "comparison_file_prefix",
      boost::program_options::value<std::string>()->default_value(""),
      "directory of SpEC CCE comparison data")(
      "rres", boost::program_options::value<size_t>()->default_value(20),
      "number of radial spectral collocation points")(
      "timestep_denominator",
      boost::program_options::value<size_t>()->default_value(2),
      "denominator to be used in construction the rational, fixed timestep "
      "size")("timestep_numerator",
              boost::program_options::value<size_t>()->default_value(1),
              "numerator to be used in construction of the rational, fixed "
              "timestep size")(
      "psi4", boost::program_options::value<bool>()->default_value(false))(
      "filter_l", boost::program_options::value<int>()->default_value(-1))(
      "start_t", boost::program_options::value<double>()->default_value(0.0),
      "time at which to start the simulation. Must be between the start and "
      "end times for the input data")(
      "end_t", boost::program_options::value<double>()->default_value(-1.0),
      "time at which to end the simulation. If unspecified, the simulation "
      "will run until the final time of the input data.")(
      "input_file", boost::program_options::value<std::string>()->required(),
      "input filename (without the .h5 extension)")(
      "output_suffix", boost::program_options::value<std::string>()->required(),
      "suffix for output filename. The full output filename will be the "
      "concatenation of the input filename followed by the specified suffix");

  boost::program_options::variables_map vars;
  // boost::program_options::store(
      // boost::program_options::parse_command_line(argc, argv, desc), vars);

  // boost::program_options::variables_map positional_vars;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .positional(pos_desc)
          .options(desc)
          .run(),
      vars);

  if(vars.count("help")) {
    printf("Usage: cce_extract [options] input_file output_suffix\n");
    desc.print(std::cout);
    return 1;
  }

  size_t l_filter = vars["l_max"].as<size_t>() - 1;
  if(vars["filter_l"].as<int>() !=  -1) {
    l_filter = static_cast<size_t>(vars["filter_l"].as<int>());
  }

  Cce::run_trial_cce(
      vars["input_file"].as<std::string>(),
      vars["comparison_file_prefix"].as<std::string>(),
      vars["l_max"].as<size_t>(), vars["output_l_max"].as<size_t>(),
      vars["rres"].as<size_t>(), vars["output_suffix"].as<std::string>(),
      vars["timestep_numerator"].as<size_t>(),
      vars["timestep_denominator"].as<size_t>(), vars["psi4"].as<bool>(),
      l_filter, vars["start_t"].as<double>(), vars["end_t"].as<double>());
}
