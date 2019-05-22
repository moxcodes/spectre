// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"

namespace Cce {

template <typename Tag>
struct CalculateScriPlusValue;

template <>
struct CalculateScriPlusValue<Tags::News> {
  using argument_tags = tmpl::list<Tags::H, Tags::Beta, Tags::LMax>;
  using return_tags = tmpl::list<Tags::News>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> news,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& h,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& beta,
      const size_t l_max) noexcept {
    Scalar<SpinWeighted<ComplexDataVector, 2>> dy_h{get(h).size()};

    ComputePreSwshDerivatives<Tags::Dy<Tags::H>>(make_not_null(&dy_h), h,
                                                 l_max);
    auto dy_h_at_scri = ComplexDataVector{
        get(dy_h).data().data() +
            (number_of_radial_points - 1) *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    SpinWeighted<ComplexDataVector, 0> beta_at_scri;
    beta_at_scri.data() = ComplexDataVector{
        get(beta).data().data() +
            (number_of_radial_points - 1) *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    // in other contexts, it is worth worrying about whether these are at fixed
    // numerical radius or fixed Bondi radius, but those are equivalent at
    // scri+, so don't worry about it.
    auto eth_beta_at_scri =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&beta_at_scri), l_max);
    auto eth_eth_beta_at_scri =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::EthEth>(
            make_not_null(&beta_at_scri), l_max);

    // additional phase factor delta set to zero.
    get(*news).data() *=
        (0.5 * exp(-2.0 * beta_at_scri.data()) + eth_eth_beta_at_scri +
         2.0 * square(eth_beta_at_scri));
  }
};

template <>
struct CalculateScriPlusValue<Tags::Du<Tags::InertialRetardedTime>> {

  using argument_tags = tmpl::list<Tags::Exp2Beta>;
  using return_tags = tmpl::list<Tags::Du<Tags::InertialRetardedTime>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          du_inertial_time,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp2beta) noexcept {
    ComplexDataVector buffer = get(exp2beta).data();
    get(*du_intertial_time).data() = ComplexDataVector{
        buffer.data() + buffer.size() - get(*du_intertial_time).size(),
        get(*du_intertial_time).size()};
}
};
}  // namespace Cce
