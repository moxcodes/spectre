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
  using argument_tags =
      tmpl::list<Tags::H, Tags::Dy<Tags::J>, Tags::Beta,
                 Tags::EvolutionGaugeBoundaryValue<Tags::R>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::DuRDividedByR>,
                 Tags::LMax>;
  using return_tags = tmpl::list<Tags::News, Tags::U0>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> news,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u_0,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& h,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r_divided_by_r,
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

    ComplexDataVector dy_j_buffer = get(dy_j).data();
    SpinWeighted<ComplexDataVector, 2> dy_j_at_scri;
    dy_j_at_scri.data() = ComplexDataVector{
        dy_j_buffer.data() +
            (number_of_radial_points - 1) *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    auto ethbar_u0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&get(*u_0)), l_max);
    SpinWeighted<ComplexDataVector, 0> r_buffer = get(r);
    SpinWeighted<ComplexDataVector, 1> u0bar_dy_j =
        conj(get(*u_0)) * dy_j_at_scri;
    SpinWeighted<ComplexDataVector, 2> u0bar_eth_dy_j =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&u0bar_dy_j), l_max) -
        dy_j_at_scri *
            conj(Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                make_not_null(&get(*u_0)), l_max)) +
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&r_buffer), l_max) /
            r_buffer * u0bar_dy_j;
    SpinWeighted<ComplexDataVector, 1> ethbar_dy_j =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&dy_j_at_scri), l_max) +
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&r_buffer), l_max) /
            r_buffer * dy_j_at_scri;
    // TEST
    SpinWeighted<ComplexDataVector, 2> u_term =
        0.5 * u0bar_eth_dy_j + 0.5 * get(*u_0) * ethbar_dy_j +
        .75 * dy_j_at_scri * conj(ethbar_u0) - 0.25 * ethbar_u0 * dy_j_at_scri;

    // additional phase factor delta set to zero.
    // Note: -2 * r extra factor due to derivative l to y
    // Note also: extra factor of 2.0 for conversion to strain.
    /// TODO currently using SpecH for this computation
    get(*news).data() =
        2.0 *
        ((-get(r).data() * exp(-2.0 * beta_at_scri.data()) *
          (dy_h_at_scri + get(du_r_divided_by_r).data() * dy_j_at_scri.data()
           /*+0.0 * u_term.data()*/)) +
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

template <>
struct CalculateScriPlusValue<Tags::CauchyGaugeScriPlus<Tags::Beta>> {
  using argument_tags =
      tmpl::list<Tags::GaugeOmega, Tags::InertialAngularCoords,
                 Tags::CauchyAngularCoords, Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::CauchyGaugeScriPlus<Tags::Beta>, Tags::Beta>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          cauchy_gauge_beta,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const tnsr::i<DataVector, 2>& x_tilde_of_x,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    SpinWeighted<ComplexDataVector, 0> beta_buffer;
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points =
        get(*beta).size() / number_of_angular_points;
    beta_buffer.data() = ComplexDataVector{
        get(*beta).data().data() +
            (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points};
    beta_buffer.data() -= 0.5 * log(get(omega).data());
    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*cauchy_gauge_beta)), make_not_null(&beta_buffer),
        get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), l_max);

    SpinWeighted<ComplexDataVector, 0> identity_test =
        Spectral::Swsh::swsh_interpolate_from_pfaffian(
            make_not_null(&get(*cauchy_gauge_beta)), get<0>(x_of_x_tilde),
            get<1>(x_of_x_tilde), l_max);
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
