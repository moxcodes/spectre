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
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta,
      const size_t l_max) noexcept {
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points = get(h).size() / number_of_angular_points;
    Scalar<SpinWeighted<ComplexDataVector, 2>> dy_h{get(h).size()};

    ComputePreSwshDerivatives<Tags::Dy<Tags::H>>::apply(make_not_null(&dy_h), h,
                                                        l_max);
    auto dy_h_at_scri = ComplexDataVector{
        get(dy_h).data().data() +
            (number_of_radial_points - 1) *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    ComplexDataVector beta_buffer = get(beta).data();
    SpinWeighted<ComplexDataVector, 0> beta_at_scri;
    beta_at_scri.data() = ComplexDataVector{
        beta_buffer.data() +
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
    get(*news).data() =
        (0.5 * exp(-2.0 * beta_at_scri.data()) * dy_h_at_scri +
         eth_eth_beta_at_scri.data() + 2.0 * square(eth_beta_at_scri.data()));
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
    get(*du_inertial_time).data() = ComplexDataVector{
        buffer.data() + buffer.size() - get(*du_inertial_time).size(),
        get(*du_inertial_time).size()};
}
};

template <typename Tag>
struct InitializeScriPlusValue;

template <>
struct InitializeScriPlusValue<Tags::InertialRetardedTime> {
  using argument_tags = tmpl::list<>;
  using return_tags = tmpl::list<Tags::InertialRetardedTime>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          inertial_time,
      const double& initial_time) noexcept {
    // this is arbitrary, has to do with choosing a BMS frame. We choose one
    // that has a particular specified initial time value.
    get(*inertial_time).data() = initial_time;
  }
};

}  // namespace Cce
