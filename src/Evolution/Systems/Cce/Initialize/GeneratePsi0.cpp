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
    //--//
    // JM: I think it will be easier for the next steps if you just take these
    // containers as `SpinWeighted<ComplexDataVector, N>` and store the
    // consecutive radii in big memory blocks
    //--//
    const gsl::not_null<
    std::vector<SpinWeighted<ComplexDataVector, 2>>*> j_container,
    const gsl::not_null<
    std::vector<SpinWeighted<ComplexDataVector, 2>>*> dr_j_container,
    const gsl::not_null<
    std::vector<SpinWeighted<ComplexDataVector, 0>>*> r_container,
    //--//
    // JM: For this one, you should just take it by argument, because we want to
    // have the calling utilities ask for a particular l_max for the evolution,
    // which might be different from the file l_max (so we also should have
    // different variables names)
    //--//
    const gsl::not_null<size_t*> l_max,
    //--//
    // JM: This is the right thing to do; taking the files as an argument, but
    // you'll  probably want to take a std::vector<std::string> to make the
    // iteration below easier.
    //--//
    const string files,
    const int target_idx,
    const double target_time) noexcept {
  //--//
  // JM: I think we can actually skip this temporary storage variable and just
  // go right to stashing the data in the pass-by-pointer vectors
  //--//
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
  //--//
  // JM: I think we'll actually need a counting index for referring into the big
  // blocks of data, so I think the best thing to do will be to just make this a
  // traditional for loop : `for(size_t i = 0; i < files.size(); ++i)` (note
  // that I'm suggesting that `files` will be a `std::vector<std::string>`).
  // Though, for reference you can do a range-based for in c++, the syntax is
  // just a little different: `for(auto& filename : files)`.
  //--//
  for(filename in files) {
    ReducedSpecWorldtubeH5BufferUpdater buffer_updater{filename};
    //--//
    // JM: The l_max we'll want for these uses will probably just be the one
    // passed in to the function arguments. The l_max of the file will tend to
    // be lower, but the `populate_hypersurface_data` routine will handle
    // correcting that mismatch.
    //--//
    l_max = buffer_updater.get_l_max();
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);

    //--//
    // JM: (just a note about what we've chosen for spectre style) We've decided
    // to prefer the snake case for local variable names, so this should be
    // `data_manager` to be consistent
    //--//
    ReducedWorldtubeDataManager DataManager{
        buffer_updater, l_max, 100,
        intrp::BarycentricRationalSpanInterpolator interpolator{10_st, 10_st}};
    //--//
    // JM: I think because the l_max we want is the global l_max passed into the
    // function, you can move this variables declaration to before the for loop
    // and then it only has to be allocated once and can be reused on subsequent
    // calls to `populate_hypersurface_boundary_data`
    //--//
    Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
        variables{number_of_angular_points};
    const double ext_radius = buffer_updater.get_extraction_radius()
    const double corrected_time = (ext_radius - target_radius) + target_time;
    DataManager.populate_hypersurface_boundary_data(
        make_not_null(&variables), corrected_time);

    //--//
    // JM: So this is where we'll have to use some tricks to reference into the
    // memory blocks -- the j, dr_j, and r (you can probably just use the ones
    // passed in by pointer) should hold N concentric shells of
    // angular collocation data, so here we'll want to create a view to refer
    // into that big block. You can create a 'view' into a spectre vector class
    // via:
    // `ComplexDataVector angular_view{get(*j).data() + offset, size};`,
    // then if you assign to `angular_view`, it will update values in the
    // `j` data structure, starting at `offset`. I think I have code that
    // does a similar thing in `InverseCubic.cpp`.
    //--//
    get(*j).data() = get<Tags::BondiJ>(variables);
    get(*dr_j).data() = get<Tags::BondiDrJ>(variables);
    get(*r).data() = get<Tags::BondiR>(variables);

    //--//
    // JM: Now that we're using the `populate_hypersurface_boundary_data`
    // utility, you won't have to worry about doing these transforms --
    // that function returns collocation values rather than modes.
    //--//
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
  //--//
  // JM: You have the right idea that we'll want a transpose here, but it's
  // the buffers we'll want to use to perform the interpolation that need to be
  // transposed (to move from an angles-varies fastest memory layout to a
  // radius-varies-fastest one)
  // The function will be called as `dr_dr_j = transpose(dr_dr_j,
  //   number_of_angular_points, number_of_radial_points)`,
  // where `number_of_angular_points` can be found with
  // `Spectral::Swsh::number_of_swsh_collocation_points` and
  // `number_of_radial_points` should be `dr_j.size() /
  //  number_of_angular_points` (provided `dr_j` is in the 'one big
  //  chunk of memory' format I suggested in other places)
  //--//
  get(*dr_dr_j).data() = trans(get(*dr_dr_j).data());
  //--//
  // JM: The loop here will then just be over the set of angular collocation
  // points (though you won't have to access the coordinates explicitly) --
  // I think you should be able to just use a for loop from 0 to
  // `number_of_angular_points`
  //--//
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
    //--//
    // JM: Once you've transposed the data, you can just reference the spans
    // into the big memory blocks `dr_j` and `r`, by passing the
    // `dr_j.data().data() + offset` (getting the raw pointer), where
    // the offset should be something like `number_of_radial_points * i`
    // where i is the loop counter, and the size for the span should be
    // `number_of_radial_points`. It's a little subtle from reading the
    // header file, but `BarycentricRationalSpanInterpolator` actually
    // just works gracefully when you pass in a span of complex values.
    //--//
    gsl::span<const std::complex<double>> span_dr_j_re(
       dr_j_values_re, dr_j.size());
    gsl::span<const std::complex<double>> span_dr_j_im(
       dr_j_values_im, dr_j.size());
    gsl::span<const double> span_r_re(r_values_re, r.size());
    gsl::span<const double> span_r_im(r_values_im, r.size());
    intrp::BarycentricRationalSpanInterpolator interpolator{10_st, 10_st};
    //--//
    // JM: This part will be a little complicated because I built the
    // interpolator with slightly different use-cases in mind. What
    // we'll need to do is create a lambda that evaluates the
    // interpolation at an arbitrary point. The syntax for declaring
    // that will be:
    // ```
    //     auto interpolated_dr_j =
    //         [&bondi_r_span, &dr_j_span, &interpolator](const double r)
    //          noexcept {
    //           return interpolator.interpolate(bondi_r_span, dr_j_span, r);
    //         }
    // ```
    // then you'll want to use that function as the argument to the
    // boost::math::differentiation::finite_difference_derivative.
    // the other argument will just be the single element of the `r`
    // currently being considered, so if loop index is `i` and the
    // index of which radius we want to evaluate the derivative at
    // is `target_idx`, the value to pass into the second argument
    // of the boost function will be `r.data()[target_idx +
    //    i * number_of_angular_points]` and store the result in
    // the appropriate location of the (single shell) dr_dr_j data.
    //--//
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
    //--//
    // JM: Similar to the comment on the above free function, I think
    // we'll want a `std::vector<std::string>`, though here I actually
    // think we'll want to take them as input to the constructor for the
    // `GeneratePsi0` object so that it can easily be specified by options,
    // then you'll just have to store it in a member variable and in this
    // function, you'll just retrieve the member variable rather than
    // taking this argument. The same probably also holds for the index.
    //--//
    const string files,
    const int target_idx,
    const double target_time) const noexcept {

  //--//
  // JM: To be consistent with the suggestion for the above free function
  // taking just big blocks of memory versions, these should be initialized
  // as `SpinWeighted<ComplexDataVector, N>` (not references), and initialized
  // with size (number of file strings) * (number of angular collocations for
  // input l_max) -- you'll probably also need to take the l_max as an argument.
  //--//
  std::vector<SpinWeighted<ComplexDataVector, 2>>& j_container;
  std::vector<SpinWeighted<ComplexDataVector, 2>>& dr_j_container;
  std::vector<SpinWeighted<ComplexDataVector, 0>>& r_container;
  read_in_worldtube_data(j_container, dr_j_container, r_container, l_max,
                         files, target_idx, target_time);

  //--//
  // JM: This one should probably be constructed with the same type:
  // `SpinWeighted<ComplexDataVector, 2>`, but only with size for
  // a single shell of angular collocation points
  //--//
  const Scalar<SpinWeighted<ComplexDataVector, 2>>*> dr_dr_j;
  second_derivative_of_j_from_worldtubes(
      dr_dr_j, dr_j_container, r_container, l_max);
  //--//
  // JM: After the above function call, you'll have the single-shell data
  // for dr_dr_j, which is much of the way to the inputs for the psi0
  // utility in `NewmanPenrose.hpp`. There's some jacobian factors that
  // you'll need to include for transforming from the dr to dy, but at the
  // inner worldtube where you're calculating this, \partial_y = R * \partial_r,
  // so I think the conversions should be quick enough that you won't have
  // to write a separate function for calculating the psi0.
  //--//
}

void GeneratePsi0::pup(PUP::er& /*p*/) noexcept {}

/// \cond
//PUP::able::PUP_ID GeneratePsi0::my_PUP_ID = 0;
/// \endcond
} // namespace InitializeJ
} // namespace Cce
