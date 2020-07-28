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
    const string files,
    const double t0) noexcept {

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
  for(filename in files) {
    ReducedSpecWorldtubeH5BufferUpdater buffer_updater{filename};
    size_t l_max = buffer_updater.get_extraction_radius();
    //get(*j).data() = buffer_updater
  }
}

void second_derivative_of_j_from_worldtubes(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dr_dr_j,
    const Scalar<SpinWeighted<ComplexModalVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexModalVector, 2>>& dr_bondi_j,
    const Scalar<SpinWeighted<ComplexModalVector, 0>>& bondi_r,
    const size_t l_max) noexcept {

  const size_t number_of_angular_points =
    Spectral::Swsh::number_of_swsh_collocation_points(l_max);

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
  //transpose inputs
  ComplexDataVector dr_bondi_j_transpose = trans(dr_bondi_j.data());
  ComplexDataVector bondi_r_tranpose = trans(bondi_r.data());

  for (const auto& libsharp_mode :
         Spectral::Swsh::cached_coefficients_metadata(l_max)) {
    //???
  }
  //use BarycentricRationalSpanInterpolator to do interpolation
  intrp::BarycentricRationalSpanInterpolator interpolator{10_st, 10_st};
  const auto dr_j_interpolated = interpolator.interpolate(
      gsl::span<const double>(bondi_r.data(), bondi_r.size()),
      gsl::span<const std::complex<double>>(
      dr_bondi_j.data(), dr_bondi_j.size()),
      *bondi_r.data());
  dr_dr_j = boost::math::differentiation::
      finite_difference_derivative(dr_j_interpolated, bondi_r.data());
}

void first_derivative_of_beta(
    const ComplexDataVector& bondi_j,
    const ComplexDataVector& dr_bondi_j,
    const DataVector& bondi_r) noexcept {}

void compute_psi0(
    const ComplexDataVector& bondi_j,
    const ComplexDataVector& dr_bondi_j,
    const ComplexDataVector& dr_dr_bondi_j,
    const ComplexDataVector& dr_beta,
    const DataVector& bondi_r) noexcept {}

std::unique_ptr<InitializeJ> GeneratePsi0::get_clone() const noexcept {
  return std::make_unique<GeneratePsi0>();
}

void GeneratePsi0::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const size_t l_max,
    const size_t number_of_concentric_worldtubes) const noexcept {

  read_in_worldtube_data()

    second_derivative_of_j_from_worldtubes()
  }

void GeneratePsi0::pup(PUP::er& /*p*/) noexcept {}

/// \cond
//PUP::able::PUP_ID GeneratePsi0::my_PUP_ID = 0;
/// \endcond
} // namespace InitializeJ
} // namespace Cce
