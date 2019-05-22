// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {

namespace detail {
template <typename DerivKind>
void angular_derivative_of_r_divided_by_r_impl(
    const gsl::not_null<
        SpinWeighted<ComplexDataVector,
                     Spectral::Swsh::Tags::derivative_spin_weight<DerivKind>>*>
        d_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& boundary_r,
    const size_t l_max) noexcept {
  size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  size_t number_of_radial_points =
      d_r_divided_by_r->size() / number_of_angular_points;
  // first set the first angular view
  SpinWeighted<ComplexDataVector,
               Spectral::Swsh::Tags::derivative_spin_weight<DerivKind>>
      d_r_divided_by_r_boundary;
  d_r_divided_by_r_boundary.data() = ComplexDataVector{
      d_r_divided_by_r->data().data(), number_of_angular_points};
  // optimization note: no buffers have been used for this derivative, and
  // passing in a buffer could limit allocations, including the explicit buffer
  // used here.
  SpinWeighted<ComplexDataVector, 0> r_buffer = boundary_r;
  Spectral::Swsh::swsh_derivatives<tmpl::list<DerivKind>>(
      l_max, 1, make_not_null(&d_r_divided_by_r_boundary), r_buffer);
  d_r_divided_by_r_boundary.data() /= boundary_r.data();
  // all of the angular shells after the innermost one
  ComplexDataVector d_r_divided_by_r_tail_shells{
      d_r_divided_by_r->data().data() + number_of_angular_points,
      (number_of_radial_points - 1) * number_of_angular_points};
  repeat(make_not_null(&d_r_divided_by_r_tail_shells),
         d_r_divided_by_r_boundary.data(), number_of_radial_points - 1);
}
}  // namespace detail

namespace detail {
// explicit  template instantiations
template void
angular_derivative_of_r_divided_by_r_impl<Spectral::Swsh::Tags::Eth>(
    const gsl::not_null<SpinWeighted<
        ComplexDataVector, Spectral::Swsh::Tags::derivative_spin_weight<
                               Spectral::Swsh::Tags::Eth>>*>
        d_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& boundary_r,
    const size_t l_max) noexcept;

template void
angular_derivative_of_r_divided_by_r_impl<Spectral::Swsh::Tags::EthEth>(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> d_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& boundary_r,
    const size_t l_max) noexcept;

template void
angular_derivative_of_r_divided_by_r_impl<Spectral::Swsh::Tags::EthEthbar>(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> d_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& boundary_r,
    const size_t l_max) noexcept;
}  // namespace detail
}  // namespace Cce
