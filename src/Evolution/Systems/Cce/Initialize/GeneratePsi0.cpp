// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/GeneratePsi0.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <boost/math/differentiation/finite_difference.hpp>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Transpose.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace InitializeJ {

void read_in_worldtube_data(
    const gsl::not_null<
    std::vector<SpinWeighted<ComplexDataVector, 2>>*> j_container,
    const gsl::not_null<
    std::vector<SpinWeighted<ComplexDataVector, 2>>*> dr_j_container,
    const gsl::not_null<
    std::vector<SpinWeighted<ComplexDataVector, 0>>*> r_container,
    const gsl::not_null<size_t*> l_max,
    const string files,
    const int target_idx,
    const double target_time) noexcept {

  //read in j, dr_j, and r from worldtubes
  Variables<tmpl::list<
    //j
    ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexModalVector>,
                         std::integral_constant<int, 2>>,
    //dr_j
    ::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexModalVector>,
                          std::integral_constant<int, 2>>,
    //r
    ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexModalVector>,
                          std::integral_constant<int, 0>>>>
    computation_buffers{
       Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  auto& j =
    get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexModalVector>,
        std::integral_constant<int, 2>>>(computation_buffers));
  auto& dr_j =
    get(get<::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexModalVector>,
        std::integral_constant<int, 2>>>(computation_buffers));
  auto& r =
    get(get<::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexModalVector>,
        std::integral_constant<int, 0>>>(computation_buffers));

  ReducedSpecWorldtubeH5BuggerUpdater target_buffer_updater{files[target_idx]};
  const double target_radius = target_buffer_updater.get_extraction_radius();
  for(filename in files) {
    ReducedSpecWorldtubeH5BufferUpdater buffer_updater{filename};
    l_max = buffer_updater.get_l_max();
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);

    ReducedWorldtubeDataManager DataManager{
        buffer_updater, l_max, 100,
        intrp::BarycentricRationalSpanInterpolator interpolator{10_st, 10_st}};
    Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
        variables{number_of_angular_points};
    const double ext_radius = buffer_updater.get_extraction_radius()
    const double corrected_time = (ext_radius - target_radius) + target_time;
    DataManager.populate_hypersurface_boundary_data(
        make_not_null(&variables), corrected_time);

    get(*j).data() = get<Tags::BondiJ>(variables);
    get(*dr_j).data() = get<Tags::BondiDrJ>(variables);
    get(*r).data() = get<Tags::BondiR>(variables);

    //convert the j to libsharp convention
    SpinWeighted<ComplexModalVector, 2> target_j_transform{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
    Spectral::Swsh::goldberg_to_libsharp_modes(
        make_not_null(&target_j_transform), get(j), l_max);
    SpinWeighted<ComplexDataVector, 2> target_j =
        Spectral::Swsh::inverse_swsh_transform(
            l_max, 1, target_j_transform);
    //convert the dr_j to libsharp convention
    SpinWeighted<ComplexModalVector, 2> target_dr_j_transform{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
    Spectral::Swsh::goldberg_to_libsharp_modes(
        make_not_null(&target_dr_j_transform), get(dr_bondi_j), l_max);
    SpinWeighted<ComplexDataVector, 2> target_dr_j =
        Spectral::Swsh::inverse_swsh_transform(
            l_max, 1, target_dr_j_transform);
    //convert r to libsharp convention
    SpinWeighted<ComplexModalVector, 0> target_r_transform{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
    Spectral::Swsh::goldberg_to_libsharp_modes(
        make_not_null(&target_r_transform),get(bondi_r), l_max);
    SpinWeighted<ComplexDataVector, 0> target_r =
        Spectral::Swsh::inverse_swsh_transform(
            l_max, number_of_radial_points, target_r_transform);

    //fill the containers of the complex data vectors
    j_container.push_back(target_j);
    dr_j_container.push_back(target_dr_j);
    r_container.push_back(target_r);
  }
}

void second_derivative_of_j_from_worldtubes(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dr_dr_j,
    std::vector<SpinWeighted<ComplexDataVector, 2>>& j,
    std::vector<SpinWeighted<ComplexDataVector, 2>>& dr_j,
    std::vector<SpinWeighted<ComplexDataVector, 0>>& r,
    const size_t l_max) noexcept {

  get(*dr_dr_j).data() = trans(get(*dr_dr_j).data());
  for (const auto& mode :
         Spectral::Swsh::cached_coefficients_metadata(l_max)) {
    std::vector<double> dr_j_values_re;
    std::vector<double> r_values_re;
    std::vector<double> dr_j_values_im;
    std::vector<double> r_values_im;

    //tranpose the variables to cluster radii together
    //and evaluate at the relevant libsharp mode (real part and imag part)
    //this can't be the right way to do this...
    for (int i=0; i<j.size(); i++) {
      dr_j_values_re.push_back(
         trans(dr_j[i].data()).data()[mode.transform_of_real_part_offset]);
      r_values_re.push_back(
         trans(r[i].data()).data()[mode.transform_of_real_part_offset]);
      dr_j_values_im.push_back(
         trans(dr_j[i].data()).data()[mode.transform_of_imag_part_offset]);
      r_values_im.push_back(
         trans(r[i].data()).data()[mode.transform_of_imag_part_offset]);
    }
    gsl::span<const std::complex<double>> span_dr_j_re(
       dr_j_values_re, dr_j.size());
    gsl::span<const std::complex<double>> span_dr_j_im(
       dr_j_values_im, dr_j.size());
    gsl::span<const double> span_r_re(r_values_re, r.size());
    gsl::span<const double> span_r_im(r_values_im, r.size());

    intrp::BarycentricRationalSpanInterpolator interpolator{10_st, 10_st};
    //what should the target_points be? just the same points?
    auto derivative(double function) {
      return [=](double r) {return boost::math::differentiation::
          finite_difference_derivative(function, r); };
    }
    //i guess we still need to evaluate at the target radius?
    //worried i'm just making a mess of things...
    auto dr_dr_j_value_re = derivative(
      interpolator.interpolate(span_dr_j_re, span_r_re, span_re.data()));
    auto dr_dr_j_value_im = derivative(
      interpolator.interpolate(span_dr_j_im, span_r_im, span_im.data()));
    get(*dr_dr_j).data()[mode.transform_of_real_part_offset] = dr_dr_j_value_re;
    get(*dr_dr_j).data()[mode.transform_of_imag_part_offset] = dr_dr_j_value_im;
  }
  //transpose back
  get(*dr_dr_j).data() = trans(get(*dr_dr_j).data());
}

std::unique_ptr<InitializeJ> GeneratePsi0::get_clone() const noexcept {
  return std::make_unique<GeneratePsi0>();
}

void GeneratePsi0::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const string files,
    const int target_idx,
    const double target_time) const noexcept {

  std::vector<SpinWeighted<ComplexDataVector, 2>>& j_container;
  std::vector<SpinWeighted<ComplexDataVector, 2>>& dr_j_container;
  std::vector<SpinWeighted<ComplexDataVector, 0>>& r_container;
  read_in_worldtube_data(j_container, dr_j_container, r_container, l_max,
                         files, target_idx, target_time);

  const Scalar<SpinWeighted<ComplexDataVector, 2>>*> dr_dr_j;
  second_derivative_of_j_from_worldtubes(
      dr_dr_j, dr_j_container, r_container, l_max);
}

void GeneratePsi0::pup(PUP::er& /*p*/) noexcept {}

/// \cond
//PUP::able::PUP_ID GeneratePsi0::my_PUP_ID = 0;
/// \endcond
} // namespace InitializeJ
} // namespace Cce
