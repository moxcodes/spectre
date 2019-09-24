// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/Systems/Cce/ComputePreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"

namespace Cce {

template <typename Tag>
struct CalculateScriPlusValue;

template <>
struct CalculateScriPlusValue<Tags::News> {
  using argument_tags = tmpl::list<
      Tags::SpecH, Tags::BondiJ, Tags::Dy<Tags::BondiJ>, Tags::BondiBeta,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
      Tags::EvolutionGaugeBoundaryValue<Tags::DuRDividedByR>,
      Spectral::Swsh::Tags::LMax, Spectral::Swsh::Tags::NumberOfRadialPoints>;
  using return_tags = tmpl::list<Tags::News, Tags::U0>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> news,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u_0,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& h,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*du_r_divided_by_r*/,
      const size_t l_max, const size_t number_of_radial_points) noexcept {
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    Scalar<SpinWeighted<ComplexDataVector, 2>> dy_h{get(h).size()};

    ComputePreSwshDerivatives<Tags::Dy<Tags::BondiH>>::apply(
        make_not_null(&dy_h), h, l_max);
    const auto dy_h_at_scri = ComplexDataVector{
        get(dy_h).data().data() +
            (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points};

    ComplexDataVector beta_buffer = get(beta).data();
    const auto beta_at_scri = make_const_view(
        get(beta), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    ComplexModalVector goldberg_modes_beta =
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(l_max, 1, beta_at_scri), l_max)
            .data();
    Parallel::printf("Beta check:\n");
    Parallel::printf("%e, %e ; %e %e\n", real(goldberg_modes_beta[6]),
                     imag(goldberg_modes_beta[6]), real(goldberg_modes_beta[8]),
                     imag(goldberg_modes_beta[8]));

    const auto j_at_scri = make_const_view(
        get(j), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    ComplexModalVector goldberg_modes_j =
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(l_max, 1, j_at_scri), l_max)
            .data();
    Parallel::printf("J check:\n");
    Parallel::printf("%e, %e ; %e %e\n", real(goldberg_modes_j[6]),
                     imag(goldberg_modes_j[6]), real(goldberg_modes_j[8]),
                     imag(goldberg_modes_j[8]));

    const auto j_at_boundary =
        make_const_view(get(j), 0, number_of_angular_points);

    goldberg_modes_j =
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(l_max, 1, j_at_boundary), l_max)
            .data();
    Parallel::printf("J boundary check:\n");
    Parallel::printf("%e, %e ; %e %e\n", real(goldberg_modes_j[6]),
                     imag(goldberg_modes_j[6]), real(goldberg_modes_j[8]),
                     imag(goldberg_modes_j[8]));

    const auto h_at_scri = make_const_view(
        get(h), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    ComplexModalVector goldberg_modes_h =
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(l_max, 1, h_at_scri), l_max)
            .data();
    Parallel::printf("H check:\n");
    Parallel::printf("%e, %e ; %e %e\n", real(goldberg_modes_h[6]),
                     imag(goldberg_modes_h[6]), real(goldberg_modes_h[8]),
                     imag(goldberg_modes_h[8]));

    const auto h_at_boundary =
        make_const_view(get(h), 0, number_of_angular_points);

    goldberg_modes_h =
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(l_max, 1, h_at_boundary), l_max)
            .data();
    Parallel::printf("H boundary check:\n");
    Parallel::printf("%e, %e ; %e %e\n", real(goldberg_modes_h[6]),
                     imag(goldberg_modes_h[6]), real(goldberg_modes_h[8]),
                     imag(goldberg_modes_h[8]));

    // in other contexts, it is worth worrying about whether these are at fixed
    // numerical radius or fixed Bondi radius, but those are equivalent at
    // scri+, so don't worry about it.
    const auto eth_beta_at_scri =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            l_max, 1, beta_at_scri);
    const auto eth_eth_beta_at_scri =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::EthEth>(
            l_max, 1, beta_at_scri);

    // additional phase factor delta set to zero.
    // Note: -2 * r extra factor due to derivative l to y
    // Note also: extra factor of 2.0 for conversion to strain.
    /// TODO currently using SpecH for this computation
    get(*news).data() =
        2.0 *
        (-get(r).data() * exp(-2.0 * beta_at_scri.data()) * dy_h_at_scri +
         eth_eth_beta_at_scri.data() + 2.0 * square(eth_beta_at_scri.data()));
  }
};

template <>
struct CalculateScriPlusValue<Tags::TimeIntegral<Tags::ScriPlus<Tags::Psi4>>> {
  using argument_tags =
      tmpl::list<Tags::Exp2Beta, Tags::Dy<Tags::BondiU>, Tags::Dy<Tags::BondiJ>,
                 Tags::BondiH, Tags::BondiR, Tags::EthRDividedByR,
                 Tags::DuRDividedByR, Spectral::Swsh::Tags::LMax,
                 Spectral::Swsh::Tags::NumberOfRadialPoints>;
  using return_tags =
      tmpl::list<Tags::TimeIntegral<Tags::ScriPlus<Tags::Psi4>>>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*>
          integral_of_psi_4,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_u,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& h,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r_divided_by_r,
      const size_t l_max, const size_t number_of_radial_points) noexcept {
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);

    const auto exp_2_beta_at_scri = make_const_view(
        get(exp_2_beta),
        (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    const auto dy_u_at_scri = make_const_view(
        get(dy_u), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto eth_dy_u_at_scri =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            l_max, 1, dy_u_at_scri);

    const auto dy_j_at_scri = make_const_view(
        get(dy_j), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    const auto r_view = make_const_view(
        get(r), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto eth_r_divided_by_r_view = make_const_view(
        get(eth_r_divided_by_r),
        (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto du_r_divided_by_r_view = make_const_view(
        get(du_r_divided_by_r),
        (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    Scalar<SpinWeighted<ComplexDataVector, 2>> dy_h{get(h).size()};
    ComputePreSwshDerivatives<Tags::Dy<Tags::BondiH>>::apply(
        make_not_null(&dy_h), h, l_max);
    const auto dy_h_at_scri = make_const_view(
        get(dy_h), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    get(*integral_of_psi_4) =
        -2.0 * r_view *
        ((conj(eth_dy_u_at_scri) +
          conj(eth_r_divided_by_r_view) * conj(dy_u_at_scri)) -
         (conj(dy_h_at_scri) + du_r_divided_by_r_view * conj(dy_j_at_scri))) /
        exp_2_beta_at_scri;
  }
};

template <>
struct CalculateScriPlusValue<Tags::ScriPlusFactor<Tags::Psi4>> {
  using argument_tags = tmpl::list<Tags::Exp2Beta, Spectral::Swsh::Tags::LMax,
                                   Spectral::Swsh::Tags::NumberOfRadialPoints>;
  using return_tags = tmpl::list<Tags::ScriPlusFactor<Tags::Psi4>>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          scri_plus_factor,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
      const size_t l_max, const size_t number_of_radial_points) noexcept {
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    const auto exp_2_beta_at_scri = make_const_view(
        get(exp_2_beta),
        (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    get(*scri_plus_factor) = -0.5 / exp_2_beta_at_scri;
  }
};

template <>
struct CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi3>> {
  using argument_tags =
      tmpl::list<Tags::Exp2Beta,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::Dy<Tags::BondiU>, Tags::Dy<Tags::BondiJ>, Tags::BondiH,
                 Tags::Dy<Tags::BondiW>, Tags::BondiR, Tags::EthRDividedByR,
                 Tags::DuRDividedByR, Spectral::Swsh::Tags::LMax,
                 Spectral::Swsh::Tags::NumberOfRadialPoints>;
  using return_tags = tmpl::list<Tags::ScriPlus<Tags::Psi3>>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*> psi_3,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_u,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& h,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_w,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r_divided_by_r,
      const size_t l_max, const size_t number_of_radial_points) noexcept {
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);

    Scalar<SpinWeighted<ComplexDataVector, 2>> dy_h{get(h).size()};
    ComputePreSwshDerivatives<Tags::Dy<Tags::BondiH>>::apply(
        make_not_null(&dy_h), h, l_max);
    const auto dy_h_at_scri = make_const_view(
        get(dy_h), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    const auto exp_2_beta_at_scri = make_const_view(
        get(exp_2_beta),
        (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    const auto eth_beta_at_scri = make_const_view(
        get(eth_beta), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    const auto dy_u_at_scri = make_const_view(
        get(dy_u), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto eth_dy_u_at_scri =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            l_max, 1, dy_u_at_scri);

    const auto dy_w_at_scri = make_const_view(
        get(dy_w), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto ethbar_dy_w_at_scri =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            l_max, 1, dy_w_at_scri);

    const auto dy_j_at_scri = make_const_view(
        get(dy_j), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    const auto r_view = make_const_view(
        get(r), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto eth_r_divided_by_r_view = make_const_view(
        get(eth_r_divided_by_r),
        (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto du_r_divided_by_r_view = make_const_view(
        get(du_r_divided_by_r),
        (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    SpinWeighted<ComplexDataVector, -2> linear_du_j_bar_at_scri =
        -2.0 * r_view *
        (conj(dy_h_at_scri) + du_r_divided_by_r_view * conj(dy_j_at_scri));
    const auto eth_linear_du_j_bar_at_scri =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            l_max, 1, linear_du_j_bar_at_scri);

    get(*psi_3) = (2.0 * exp_2_beta_at_scri * conj(eth_beta_at_scri) -
                   2.0 * r_view * eth_beta_at_scri *
                       (conj(eth_dy_u_at_scri) +
                        conj(eth_r_divided_by_r_view) * conj(dy_u_at_scri)) +
                   r_view * (ethbar_dy_w_at_scri +
                             conj(eth_r_divided_by_r_view) * dy_w_at_scri) +
                   eth_beta_at_scri * linear_du_j_bar_at_scri -
                   eth_linear_du_j_bar_at_scri -
                   eth_r_divided_by_r_view * linear_du_j_bar_at_scri) /
                  (sqrt(2.0) * exp_2_beta_at_scri);
  }
};

template <>
struct CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi2>> {
  using argument_tags =
      tmpl::list<Tags::Exp2Beta, Tags::Dy<Tags::BondiQ>, Tags::Dy<Tags::BondiU>,
                 Tags::Dy<Tags::BondiJ>, Tags::BondiH, Tags::BondiR,
                 Tags::EthRDividedByR, Tags::DuRDividedByR,
                 Spectral::Swsh::Tags::LMax,
                 Spectral::Swsh::Tags::NumberOfRadialPoints>;
  using return_tags = tmpl::list<Tags::ScriPlus<Tags::Psi2>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> psi_2,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_q,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_u,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& h,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r_divided_by_r,
      const size_t l_max, const size_t number_of_radial_points) noexcept {
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);

    Scalar<SpinWeighted<ComplexDataVector, 2>> dy_h{get(h).size()};
    ComputePreSwshDerivatives<Tags::Dy<Tags::BondiH>>::apply(
        make_not_null(&dy_h), h, l_max);
    const auto dy_h_at_scri = make_const_view(
        get(dy_h), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    const auto dy_j_at_scri = make_const_view(
        get(dy_j), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    const auto dy_q_at_scri = make_const_view(
        get(dy_q), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto ethbar_dy_q_at_scri =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            l_max, 1, dy_q_at_scri);

    const auto dy_u_at_scri = make_const_view(
        get(dy_u), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto eth_dy_u_at_scri =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            l_max, 1, dy_u_at_scri);

    const auto exp_2_beta_at_scri = make_const_view(
        get(exp_2_beta),
        (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    const auto r_view = make_const_view(
        get(r), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto eth_r_divided_by_r_view = make_const_view(
        get(eth_r_divided_by_r),
        (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto du_r_divided_by_r_view = make_const_view(
        get(du_r_divided_by_r),
        (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    get(*psi_2) =
        -0.5 * r_view *
        (-exp_2_beta_at_scri * (conj(ethbar_dy_q_at_scri) +
                                eth_r_divided_by_r_view * conj(dy_q_at_scri)) +
         2.0 * r_view * dy_j_at_scri *
             (conj(eth_dy_u_at_scri) +
              conj(eth_r_divided_by_r_view) * conj(dy_u_at_scri)) +
         2.0 * r_view * dy_j_at_scri *
             (conj(dy_h_at_scri) +
              du_r_divided_by_r_view * conj(dy_j_at_scri))) /
        exp_2_beta_at_scri;
  }
};

template <>
struct CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi1>> {
  using argument_tags =
      tmpl::list<Tags::Dy<Tags::Dy<Tags::BondiBeta>>, Tags::Dy<Tags::BondiJ>,
                 Tags::Dy<Tags::BondiQ>, Tags::Dy<Tags::Dy<Tags::BondiQ>>,
                 Tags::BondiR, Tags::EthRDividedByR, Spectral::Swsh::Tags::LMax,
                 Spectral::Swsh::Tags::NumberOfRadialPoints>;
  using return_tags = tmpl::list<Tags::ScriPlus<Tags::Psi1>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> psi_1,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_dy_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_q,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_dy_q,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      const size_t l_max, const size_t number_of_radial_points) noexcept {
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);

    const auto dy_dy_beta_at_scri = make_const_view(
        get(dy_dy_beta),
        (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto eth_dy_dy_beta_at_scri =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            l_max, 1, dy_dy_beta_at_scri);

    const auto dy_j_at_scri = make_const_view(
        get(dy_j), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto dy_q_at_scri = make_const_view(
        get(dy_q), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto dy_dy_q_at_scri = make_const_view(
        get(dy_dy_q), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    const auto r_view = make_const_view(
        get(r), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);
    const auto eth_r_divided_by_r_view = make_const_view(
        get(eth_r_divided_by_r),
        (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    get(*psi_1) = 0.5 * sqrt(2.0) * square(r_view) *
                  (6.0 * eth_dy_dy_beta_at_scri +
                   12.0 * eth_r_divided_by_r_view * dy_dy_beta_at_scri -
                   dy_j_at_scri * conj(dy_q_at_scri) - dy_dy_q_at_scri);
  }
};

template <>
struct CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi0>> {
  using argument_tags =
      tmpl::list<Tags::Dy<Tags::Dy<Tags::BondiBeta>>, Tags::Dy<Tags::BondiJ>,
                 Tags::Dy<Tags::Dy<Tags::BondiJ>>, Tags::BondiR,
                 Spectral::Swsh::Tags::LMax,
                 Spectral::Swsh::Tags::NumberOfRadialPoints>;
  using return_tags = tmpl::list<Tags::ScriPlus<Tags::Psi0>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> psi_0,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_dy_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, const size_t l_max,
      const size_t number_of_radial_points) noexcept {
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);

    Scalar<SpinWeighted<ComplexDataVector, 2>> dy_dy_dy_j{get(dy_dy_j).size()};

    ComputePreSwshDerivatives<Tags::Dy<Tags::Dy<Tags::Dy<Tags::BondiJ>>>>::
        apply(make_not_null(&dy_dy_dy_j), dy_dy_j, l_max);
    const auto dy_dy_dy_j_at_scri = make_const_view(
        get(dy_dy_dy_j),
        (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    const auto dy_j_at_scri = make_const_view(
        get(dy_j), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    const auto dy_dy_beta_at_scri = make_const_view(
        get(dy_dy_beta),
        (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    const auto r_view = make_const_view(
        get(r), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    get(*psi_0) =
        -4.0 * pow<3>(r_view) *
        (2.0 * dy_dy_beta_at_scri * dy_j_at_scri +
         conj(dy_j_at_scri) * square(dy_j_at_scri) - dy_dy_dy_j_at_scri);
  }
};

template <>
struct CalculateScriPlusValue<Tags::ScriPlus<Tags::Strain>> {
  using argument_tags =
      tmpl::list<Tags::Dy<Tags::BondiJ>, Tags::InertialRetardedTime,
                 Tags::BondiR, Spectral::Swsh::Tags::LMax,
                 Spectral::Swsh::Tags::NumberOfRadialPoints>;
  using return_tags = tmpl::list<Tags::ScriPlus<Tags::Strain>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> strain,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
      const Scalar<DataVector>& inertial_retarded_time,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, const size_t l_max,
      const size_t number_of_radial_points) noexcept {
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);

    SpinWeighted<ComplexDataVector, 0> complex_inertial_retarded_time;
    complex_inertial_retarded_time.data() =
        std::complex<double>(1.0, 0.0) * get(inertial_retarded_time);

    const auto eth_eth_retarded_time =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::EthEth>(
            l_max, 1, complex_inertial_retarded_time);

    const auto dy_j_at_scri = make_const_view(
        get(dy_j), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    const auto r_view = make_const_view(
        get(r), (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points);

    get(*strain) = -2.0 * r_view * dy_j_at_scri + eth_eth_retarded_time;
  }
};

template <>
struct CalculateScriPlusValue<::Tags::dt<Tags::InertialRetardedTime>> {
  using argument_tags = tmpl::list<Tags::Exp2Beta>;
  using return_tags = tmpl::list<::Tags::dt<Tags::InertialRetardedTime>>;

  static void apply(
      const gsl::not_null<Scalar<DataVector>*> dt_inertial_time,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp2beta) noexcept {
    ComplexDataVector buffer = get(exp2beta).data();
    get(*dt_inertial_time) = real(ComplexDataVector{
        buffer.data() + buffer.size() - get(*dt_inertial_time).size(),
        get(*dt_inertial_time).size()});
  }
};

template <>
struct CalculateScriPlusValue<Tags::CauchyGaugeScriPlus<Tags::BondiBeta>> {
  using argument_tags =
      tmpl::list<Tags::GaugeOmega, Tags::InertialAngularCoords,
                 Tags::CauchyAngularCoords, Spectral::Swsh::Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::CauchyGaugeScriPlus<Tags::BondiBeta>, Tags::BondiBeta>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          cauchy_gauge_beta,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const tnsr::i<DataVector, 2>& x_tilde_of_x,
      const tnsr::i<DataVector, 2>& /*x_of_x_tilde*/,
      const size_t l_max) noexcept {
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points =
        get(*beta).size() / number_of_angular_points;
    // FIXME this is no longer interpolating correctly
    SpinWeighted<ComplexDataVector, 0> beta_buffer{number_of_angular_points};
    ComplexDataVector beta_scri_view{
        get(*beta).data().data() +
            (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points};
    beta_buffer.data() = beta_scri_view - 0.5 * log(get(omega).data());

    Spectral::Swsh::SwshInterpolator interpolator{
        get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), 0, l_max};
    interpolator.interpolate(
        make_not_null(&get(*cauchy_gauge_beta).data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(l_max, 1, beta_buffer), l_max)
            .data());
  }
};

// template <>
// struct CalculateScriPlusValue<Tags::CauchyGaugeScriPlus<Tags::BondiQ>> {
//   using argument_tags =
//       tmpl::list<Tags::GaugeOmega, Tags::GaugeA, Tags::GaugeB>;
//   using return_tags =
//       tmpl::list<Tags::CauchyGaugeScriPlus<Tags::BondiQ>,
//       Tags::Dy<Tags::BondiU>>;
//   static void apply(
//       const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
//       q_scri,
//       const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> dy_u,
//       const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
//       const tnsr::i<DataVector, 2>& x_tilde_of_x,
//       const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max)
//       noexcept {
//     Spectral::Swsh::swsh_interpolate(
//         make_not_null(&get(*cauchy_gauge_u0)), make_not_null(&get(*u0)),
//         get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), l_max);
//   }
// };

template <>
struct CalculateScriPlusValue<Tags::CauchyGaugeScriPlus<Tags::U0>> {
  using argument_tags =
      tmpl::list<Tags::GaugeC, Tags::GaugeD, Tags::GaugeA, Tags::GaugeB,
                 Tags::GaugeOmega, Tags::GaugeOmegaCD,
                 Tags::InertialAngularCoords, Tags::CauchyAngularCoords,
                 Spectral::Swsh::Tags::LMax>;
  using return_tags = tmpl::list<Tags::CauchyGaugeScriPlus<Tags::U0>, Tags::U0>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          cauchy_gauge_u0,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u0,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& /*c*/,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*d*/,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*omega_cd*/,
      const tnsr::i<DataVector, 2>& x_tilde_of_x,
      const tnsr::i<DataVector, 2>& /*x_of_x_tilde*/,
      const size_t l_max) noexcept {
    Spectral::Swsh::SwshInterpolator interpolator{
        get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), 1, l_max};
    interpolator.interpolate(
        make_not_null(&get(*cauchy_gauge_u0).data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(l_max, 1, get(*u0)), l_max)
            .data());

    get(*cauchy_gauge_u0) = 0.5 / square(get(omega)) *
                            (conj(get(b)) * get(*cauchy_gauge_u0) -
                             get(a) * conj(get(*cauchy_gauge_u0)));
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&get(*cauchy_gauge_u0)), l_max, l_max - 2);
  }
};

template <typename Tag>
struct InitializeScriPlusValue;

template <>
struct InitializeScriPlusValue<Tags::InertialRetardedTime> {
  using argument_tags = tmpl::list<>;
  using return_tags = tmpl::list<Tags::InertialRetardedTime>;

  static void apply(const gsl::not_null<Scalar<DataVector>*> inertial_time,
                    const double& initial_time) noexcept {
    // this is arbitrary, has to do with choosing a BMS frame. We choose one
    // that has a particular specified initial time value.
    get(*inertial_time) = initial_time;
  }
};

// This needs extensive notes about which gauge each of these things needs to be
// in.
void calculate_non_inertial_news(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> news,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> du_j,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> boundary_r,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega_cd,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
    const tnsr::i<DataVector, 2> x_of_x_tilde,
    const tnsr::i<DataVector, 2> x_tilde_of_x, const size_t l_max,
    const bool interpolate_back) noexcept;

template <typename Tag>
struct CalculateCauchyGauge;

// Note these are for debugging output, and are still in the inertial angular
// grid. After any radial interpolation, the angular grid needs to be
// interpolated to the desired Cauchy grid.

template <>
struct CalculateCauchyGauge<Tags::CauchyGauge<Tags::BondiBeta>> {
  using argument_tags =
      tmpl::list<Tags::GaugeOmega, Tags::GaugeA, Tags::GaugeB, Tags::GaugeC,
                 Tags::GaugeD, Tags::InertialAngularCoords,
                 Tags::CauchyAngularCoords, Spectral::Swsh::Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::CauchyGauge<Tags::BondiBeta>, Tags::BondiBeta>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          cauchy_beta,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& /*a*/,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*b*/,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& /*c*/,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*d*/,
      const tnsr::i<DataVector, 2>& x_tilde_of_x,
      const tnsr::i<DataVector, 2>& /*x_of_x_tilde*/,
      const size_t l_max) noexcept {
    // angular slices
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points =
        get(*beta).size() / number_of_angular_points;

    // TEST double-checking identity properties
    // Note: the slices y = constant and y_tilde = constant should be the same
    // slices, so we test the consistency of the numerical derivatives.

    Spectral::Swsh::filter_swsh_volume_quantity(make_not_null(&get(*beta)),
                                                l_max, l_max - 2, 0.0, 8);

    // actually perform the transformation with the omega factor

    SpinWeighted<ComplexDataVector, 0> interpolated_cauchy_beta_slice{
        number_of_angular_points};
    SpinWeighted<ComplexDataVector, 0> beta_slice;
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      beta_slice.data() = ComplexDataVector{
          get(*beta).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      ComplexDataVector beta_view{
          get(*beta).data().data() + i * number_of_angular_points,
          number_of_angular_points};

      Spectral::Swsh::SwshInterpolator interpolator{
          get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), 0, l_max};
      interpolator.interpolate(
          make_not_null(&interpolated_cauchy_beta_slice.data()),
          Spectral::Swsh::libsharp_to_goldberg_modes(
              Spectral::Swsh::swsh_transform(l_max, 1, beta_slice), l_max)
              .data());

      ComplexDataVector cauchy_beta_slice{
          get(*cauchy_beta).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      cauchy_beta_slice =
          interpolated_cauchy_beta_slice.data() - 0.5 * log(get(omega).data());
    }

    // SpinWeighted<ComplexDataVector, 0> test_beta_cauchy{get(d).size()};
    // const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
    // Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
    // for (const auto& collocation_point : collocation) {
    //   test_beta_cauchy.data()[collocation_point.offset] =
    //       sin(collocation_point.theta) * sin(collocation_point.phi);
    // }

    // auto interpolated_cauchy_beta = Spectral::Swsh::swsh_interpolate(
    //     make_not_null(&test_beta_cauchy), get<0>(x_tilde_of_x),
    //     get<1>(x_tilde_of_x), l_max);

    // Spectral::Swsh::filter_swsh_volume_quantity(
    //     make_not_null(&interpolated_cauchy_beta), l_max, l_max - 2, 0.0, 8);

    // SpinWeighted<ComplexDataVector, 1> eth_tilde_beta =
    //     Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
    //         make_not_null(&test_beta_cauchy), l_max);
    // SpinWeighted<ComplexDataVector, -1> ethbar_tilde_beta =
    //     Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
    //         make_not_null(&test_beta_cauchy), l_max);
    // SpinWeighted<ComplexDataVector, 1> eth_beta =
    //     Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
    //         make_not_null(&interpolated_cauchy_beta), l_max);

    // auto eth_tilde_beta_interpolated = Spectral::Swsh::swsh_interpolate(
    //     make_not_null(&eth_tilde_beta), get<0>(x_tilde_of_x),
    //     get<1>(x_tilde_of_x), l_max);
    // auto ethbar_tilde_beta_interpolated = Spectral::Swsh::swsh_interpolate(
    //     make_not_null(&ethbar_tilde_beta), get<0>(x_tilde_of_x),
    //     get<1>(x_tilde_of_x), l_max);
    // SpinWeighted<ComplexDataVector, 1> identity_test_eth_beta =
    //     0.5 * (conj(get(b)) * eth_tilde_beta_interpolated +
    //            get(a) * ethbar_tilde_beta_interpolated);

    // Spectral::Swsh::filter_swsh_volume_quantity(
    //     make_not_null(&identity_test_eth_beta), l_max, l_max - 2, 0.0, 8);

    // Spectral::Swsh::filter_swsh_volume_quantity(
    //     make_not_null(&identity_test_eth_beta), l_max, l_max - 2, 0.0, 8);
    // printf("Identity test: Jacobian vs x_tilde_of_x\n");
    // for (size_t i = 0; i < identity_test_eth_beta.size(); ++i) {
    //   printf("(%e, %e) from (%e, %e)\n",
    //          real(identity_test_eth_beta.data()[i] - eth_beta.data()[i]),
    //          imag(identity_test_eth_beta.data()[i] - eth_beta.data()[i]),
    //          real(identity_test_eth_beta.data()[i]),
    //          imag(identity_test_eth_beta.data()[i]));
    // }
    // printf("done\n");

    // SpinWeighted<ComplexDataVector, 0> test_beta{get(d).size()};
    // for (const auto& collocation_point : collocation) {
    // test_beta.data()[collocation_point.offset] =
    // sin(collocation_point.theta) * sin(collocation_point.phi);
    // }
    // // Note: using the beta 'nice' values in both coordinate systems,
    // // subverting
    // // their true meaning in order to make a more desirable test
    // SpinWeighted<ComplexDataVector, -1> ethbar_beta =
    //     Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
    //         make_not_null(&test_beta), l_max);
    // SpinWeighted<ComplexDataVector, 1> eth_beta_cauchy =
    //     Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
    //         make_not_null(&test_beta), l_max);

    // auto eth_beta_interpolated_to_inertial =
    // Spectral::Swsh::swsh_interpolate(
    //     make_not_null(&eth_beta_cauchy), get<0>(x_of_x_tilde),
    //     get<1>(x_of_x_tilde), l_max);
    // auto ethbar_beta_interpolated_to_inertial =
    //     Spectral::Swsh::swsh_interpolate(make_not_null(&ethbar_beta),
    //                                      get<0>(x_of_x_tilde),
    //                                      get<1>(x_of_x_tilde), l_max);
    // SpinWeighted<ComplexDataVector, 1> identity_test_eth_tilde_beta =
    //     0.5 * (conj(get(d)) * eth_beta_interpolated_to_inertial +
    //            get(c) * ethbar_beta_interpolated_to_inertial);

    // auto beta_interpolated_to_inertial = Spectral::Swsh::swsh_interpolate(
    // make_not_null(&test_beta), get<0>(x_of_x_tilde), get<1>(x_of_x_tilde),
    // l_max);
    // SpinWeighted<ComplexDataVector, 1> eth_tilde_beta_inertial =
    // Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
    // make_not_null(&beta_interpolated_to_inertial), l_max);

    // Spectral::Swsh::filter_swsh_volume_quantity(
    //     make_not_null(&identity_test_eth_tilde_beta), l_max, l_max - 2, 0.0,
    //     8);
    // printf("Identity test: Jacobian vs x_of_x_tilde\n");
    // for (size_t i = 0; i < identity_test_eth_tilde_beta.size(); ++i) {
    // printf("(%e, %e) from (%e, %e)\n",
    // real(identity_test_eth_tilde_beta.data()[i] -
    // eth_tilde_beta_inertial.data()[i]),
    // imag(identity_test_eth_tilde_beta.data()[i] -
    // eth_tilde_beta_inertial.data()[i]),
    // real(identity_test_eth_tilde_beta.data()[i]),
    // imag(identity_test_eth_tilde_beta.data()[i]));
    // }
    // printf("done\n");

    // SpinWeighted<ComplexDataVector, 0> beta_interpolated_back_to_cauchy =
    // Spectral::Swsh::swsh_interpolate(
    // make_not_null(&beta_interpolated_to_inertial),
    // get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), l_max);
    // printf("Identity test: Inverse transformation\n");
    // for (size_t i = 0; i < test_beta.size(); ++i) {
    // printf("(%e, %e) from (%e, %e)\n",
    // real(test_beta.data()[i] -
    // beta_interpolated_back_to_cauchy.data()[i]),
    // imag(test_beta.data()[i] -
    // beta_interpolated_back_to_cauchy.data()[i]),
    // real(test_beta.data()[i]), imag(test_beta.data()[i]));
    // }
    // printf("done\n");
  }
};

template <>
struct CalculateCauchyGauge<Tags::CauchyGauge<Tags::SpecH>> {
  using argument_tags =
      tmpl::list<Tags::GaugeOmega, Tags::GaugeOmegaCD,
                 Tags::Du<Tags::GaugeOmegaCD>,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::GaugeA, Tags::GaugeB, Tags::BoundaryValue<Tags::SpecH>,
                 Tags::InertialAngularCoords, Spectral::Swsh::Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::CauchyGauge<Tags::SpecH>, Tags::SpecH, Tags::BondiJ,
                 Tags::CauchyGauge<Tags::BondiJ>, Tags::U0,
                 Tags::Dy<Tags::BondiJ>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> cauchy_h,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> h,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const gsl::not_null<
          Scalar<SpinWeighted<ComplexDataVector, 2>>*> /*cauchy_j*/,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u_0,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dy_j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          r_tilde_boundary,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& /*boundary_h*/,
      const tnsr::i<DataVector, 2>& x_tilde_of_x, const size_t l_max) noexcept {
    // TODO add interpolation first
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points = get(*j).size() / number_of_angular_points;

    const auto& one_minus_y_collocation =
        1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto>(
                  number_of_radial_points);

    SpinWeighted<ComplexDataVector, 2> interpolated_cauchy_h_slice{
        number_of_angular_points};
    SpinWeighted<ComplexDataVector, 2> interpolated_cauchy_j_slice{
        number_of_angular_points};
    SpinWeighted<ComplexDataVector, 2> h_slice{number_of_angular_points};
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector h_view{
          get(*h).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      // printf("debug: h view\n");
      // for(auto val : h_view){
      // printf("(%e, %e)\n", real(val), imag(val));
      // }
      // printf("done\n");
      SpinWeighted<ComplexDataVector, 2> j_view;
      j_view.data() = ComplexDataVector{
          get(*j).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      SpinWeighted<ComplexDataVector, 2> dy_j_view;
      dy_j_view.data() = ComplexDataVector{
          get(*dy_j).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      ComplexDataVector cauchy_h_view{
          get(*cauchy_h).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      // auto one_over_r_tilde =
      // one_minus_y_collocation[i] / (2.0 * get(*r_tilde_boundary).data());

      auto eth_r_tilde =
          Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
              l_max, 1, get(*r_tilde_boundary));

      SpinWeighted<ComplexDataVector, 1> u_0_bar_j_tilde =
          conj(get(*u_0)) * j_view;

      SpinWeighted<ComplexDataVector, 2> u_0_bar_eth_j_tilde =
          Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
              l_max, 1, u_0_bar_j_tilde) -
          j_view *
              conj(
                  Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                      l_max, 1, get(*u_0))) -
          conj(get(*u_0)) * eth_r_tilde * square(one_minus_y_collocation[i]) /
              (2.0 * get(*r_tilde_boundary)) * dy_j_view;

      SpinWeighted<ComplexDataVector, 2> angular_derivative_term =
          0.5 *
          (get(*u_0) *
               Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                   l_max, 1, j_view) -
           get(*u_0) * conj(eth_r_tilde) * square(one_minus_y_collocation[i]) /
               (2.0 * get(*r_tilde_boundary)) * dy_j_view +
           u_0_bar_eth_j_tilde);

      SpinWeighted<ComplexDataVector, 0> cauchy_du_omega_cd =
          get(du_omega_cd) - 0.5 * (get(*u_0) * conj(get(eth_omega_cd)) +
                                    conj(get(*u_0)) * get(eth_omega_cd));

      auto eth_u_0 = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
          l_max, 1, get(*u_0));
      auto ethbar_u_0 =
          Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
              l_max, 1, get(*u_0));

      ComplexDataVector k_tilde =
          sqrt(1.0 + j_view.data() * conj(j_view.data()));

      // the tensor transform of the h we are looking for
      h_slice.data() = h_view - angular_derivative_term.data() +
                       one_minus_y_collocation[i] * cauchy_du_omega_cd.data() /
                           get(omega_cd).data() * dy_j_view.data() -
                       2.0 * cauchy_du_omega_cd.data() / get(omega_cd).data() *
                           j_view.data() +
                       ethbar_u_0.data() * j_view.data() -
                       eth_u_0.data() * k_tilde;
      // TEST
      // h_slice.data() = h_view;
      Spectral::Swsh::SwshInterpolator interpolator{
          get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), 2, l_max};
      interpolator.interpolate(
          make_not_null(&interpolated_cauchy_h_slice.data()),
          Spectral::Swsh::libsharp_to_goldberg_modes(
              Spectral::Swsh::swsh_transform(l_max, 1, h_slice), l_max)
              .data());
      interpolator.interpolate(
          make_not_null(&interpolated_cauchy_j_slice.data()),
          Spectral::Swsh::libsharp_to_goldberg_modes(
              Spectral::Swsh::swsh_transform(l_max, 1, j_view), l_max)
              .data());

      ComplexDataVector interpolated_k =
          sqrt(1.0 + interpolated_cauchy_j_slice.data() *
                         conj(interpolated_cauchy_j_slice.data()));
      cauchy_h_view =
          0.25 / square(get(omega).data()) *
          (square(conj(get(b).data())) * interpolated_cauchy_h_slice.data() +
           square(get(a).data()) * conj(interpolated_cauchy_h_slice.data()) +
           get(a).data() * conj(get(b).data()) *
               (interpolated_cauchy_h_slice.data() *
                    conj(interpolated_cauchy_j_slice.data()) +
                interpolated_cauchy_j_slice.data() *
                    conj(interpolated_cauchy_h_slice.data())) /
               interpolated_k);
    }
    // Spectral::Swsh::filter_swsh_volume_quantity
    //(make_not_null(&get(*cauchy_w)),
    // l_max, l_max - 4, 0.0, 8);
    // printf("verifying boundary h\n");
    // for (size_t i = 0; i < number_of_angular_points; ++i) {
    // printf("(%e, %e) from (%e, %e)\n",
    // real(get(*cauchy_h).data()[i] - get(boundary_h).data()[i]),
    // imag(get(*cauchy_h).data()[i] - get(boundary_h).data()[i]),
    // real(get(*cauchy_h).data()[i]), imag(get(*cauchy_h).data()[i]));
    // }
    // printf("done\n");
  }
};

template <>
struct CalculateCauchyGauge<Tags::CauchyGauge<Tags::BondiW>> {
  using argument_tags =
      tmpl::list<Tags::GaugeOmegaCD, Tags::Du<Tags::GaugeOmegaCD>,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
                 Tags::BoundaryValue<Tags::BondiW>, Tags::InertialAngularCoords,
                 Spectral::Swsh::Tags::LMax>;
  using return_tags = tmpl::list<Tags::CauchyGauge<Tags::BondiW>, Tags::BondiW,
                                 Tags::BondiU, Tags::BondiBeta, Tags::BondiJ>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> cauchy_w,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> w,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r_tilde_boundary,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*boundary_w*/,
      const tnsr::i<DataVector, 2>& x_tilde_of_x, const size_t l_max) noexcept {
    // TODO add interpolation first
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points = get(*j).size() / number_of_angular_points;

    const auto& one_minus_y_collocation =
        1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto>(
                  number_of_radial_points);

    SpinWeighted<ComplexDataVector, 0> interpolated_cauchy_w_slice{
        number_of_angular_points};
    SpinWeighted<ComplexDataVector, 0> w_slice{number_of_angular_points};
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector w_view{
          get(*w).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      ComplexDataVector u_view{
          get(*u).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      ComplexDataVector beta_view{
          get(*beta).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      ComplexDataVector j_view{
          get(*j).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      ComplexDataVector cauchy_w_view{
          get(*cauchy_w).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      auto one_over_r_tilde =
          one_minus_y_collocation[i] / (2.0 * get(r_tilde_boundary).data());

      // TEST
      w_slice.data() =
          w_view - 1.0 * one_over_r_tilde * (get(omega_cd).data() - 1.0) +
          2.0 * get(du_omega_cd).data() / get(omega_cd).data() +
          (conj(get(eth_omega_cd).data()) * u_view +
           get(eth_omega_cd).data() * conj(u_view)) /
              get(omega_cd).data() -
          exp(2.0 * beta_view) * one_over_r_tilde /
              (2.0 * square(get(omega_cd).data())) *
              (square(conj(get(eth_omega_cd).data())) * j_view +
               square(get(eth_omega_cd).data()) * conj(j_view) -
               2.0 * get(eth_omega_cd).data() * conj(get(eth_omega_cd).data()) *
                   sqrt(1.0 + j_view * conj(j_view)));
      // w_slice.data() = w_view;
      // TEST

      Spectral::Swsh::SwshInterpolator interpolator{
          get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), 0, l_max};
      interpolator.interpolate(
          make_not_null(&interpolated_cauchy_w_slice.data()),
          Spectral::Swsh::libsharp_to_goldberg_modes(
              Spectral::Swsh::swsh_transform(l_max, 1, w_slice), l_max)
              .data());

      cauchy_w_view = interpolated_cauchy_w_slice.data();
    }
    // Spectral::Swsh::filter_swsh_volume_quantity
    // (make_not_null(&get(*cauchy_w)),
    // l_max, l_max - 4, 0.0, 8);
    // printf("verifying boundary w\n");
    // for (size_t i = 0; i < number_of_angular_points; ++i) {
    // printf("(%e, %e) from (%e, %e)\n",
    // real(get(*cauchy_w).data()[i] - get(boundary_w).data()[i]),
    // imag(get(*cauchy_w).data()[i] - get(boundary_w).data()[i]),
    // real(get(*cauchy_w).data()[i]), imag(get(*cauchy_w).data()[i]));
    // }
    // printf("done\n");
  }
};

template <>
struct CalculateCauchyGauge<Tags::CauchyGauge<Tags::BondiJ>> {
  using argument_tags =
      tmpl::list<Tags::GaugeA, Tags::GaugeB, Tags::GaugeOmega,
                 Tags::InertialAngularCoords, Spectral::Swsh::Tags::LMax>;
  using return_tags = tmpl::list<Tags::CauchyGauge<Tags::BondiJ>, Tags::BondiJ>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> cauchy_j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const tnsr::i<DataVector, 2>& x_tilde_of_x, const size_t l_max) noexcept {
    // TODO add interpolation first
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points = get(*j).size() / number_of_angular_points;
    Spectral::Swsh::filter_swsh_volume_quantity(make_not_null(&get(*j)), l_max,
                                                l_max - 4, 0.0, 8);

    SpinWeighted<ComplexDataVector, 2> interpolated_cauchy_j_slice{
        number_of_angular_points};
    SpinWeighted<ComplexDataVector, 2> j_slice{number_of_angular_points};
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector j_view{
          get(*j).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      j_slice.data() = j_view;

      Spectral::Swsh::SwshInterpolator interpolator{
          get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), 2, l_max};
      interpolator.interpolate(
          make_not_null(&interpolated_cauchy_j_slice.data()),
          Spectral::Swsh::libsharp_to_goldberg_modes(
              Spectral::Swsh::swsh_transform(l_max, 1, j_slice), l_max)
              .data());

      ComplexDataVector cauchy_j_view{
          get(*cauchy_j).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      interpolated_cauchy_j_slice =
          0.25 *
          (square(conj(get(b).data())) * interpolated_cauchy_j_slice.data() +
           square(get(a).data()) * conj(interpolated_cauchy_j_slice.data()) +
           2.0 * get(a).data() * conj(get(b).data()) *
               sqrt(1.0 + interpolated_cauchy_j_slice.data() *
                              conj(interpolated_cauchy_j_slice.data())));
      Spectral::Swsh::filter_swsh_boundary_quantity(
          make_not_null(&interpolated_cauchy_j_slice), l_max, l_max - 4);
      cauchy_j_view =
          interpolated_cauchy_j_slice.data() / square(get(omega).data());
    }
    Spectral::Swsh::filter_swsh_volume_quantity(make_not_null(&get(*cauchy_j)),
                                                l_max, l_max - 4, 0.0, 8);
  }
};

template <>
struct CalculateCauchyGauge<Tags::CauchyGauge<Tags::BondiU>> {
  using argument_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>, Tags::U0,
                 Tags::GaugeOmega, Tags::GaugeOmegaCD,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::GaugeA, Tags::GaugeB, Tags::BoundaryValue<Tags::BondiU>,
                 Tags::InertialAngularCoords, Spectral::Swsh::Tags::LMax>;
  using return_tags = tmpl::list<Tags::CauchyGauge<Tags::BondiU>, Tags::BondiU,
                                 Tags::BondiJ, Tags::BondiBeta>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> cauchy_u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r_tilde_boundary,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& u_0,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& /*boundary_u*/,
      const tnsr::i<DataVector, 2> x_tilde_of_x, const size_t l_max) noexcept {
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points = get(*u).size() / number_of_angular_points;

    const auto& one_minus_y_collocation =
        1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto>(
                  number_of_radial_points);

    SpinWeighted<ComplexDataVector, 1> interpolated_cauchy_u_slice{
        number_of_angular_points};
    SpinWeighted<ComplexDataVector, 1> u_slice{number_of_angular_points};
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector j_view{
          get(*j).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      ComplexDataVector u_view{
          get(*u).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      ComplexDataVector cauchy_u_view{
          get(*cauchy_u).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      ComplexDataVector beta_view{
          get(*beta).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      // u_undertilde
      u_slice.data() =
          (u_view + get(u_0).data()) +
          one_minus_y_collocation[i] * exp(2.0 * beta_view) /
              (2.0 * get(r_tilde_boundary).data() * get(omega_cd).data()) *
              (get(eth_omega_cd).data() * sqrt(1.0 + j_view * conj(j_view)) -
               conj(get(eth_omega_cd).data()) * j_view);

      Spectral::Swsh::SwshInterpolator interpolator{
          get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), 1, l_max};
      interpolator.interpolate(
          make_not_null(&interpolated_cauchy_u_slice.data()),
          Spectral::Swsh::libsharp_to_goldberg_modes(
              Spectral::Swsh::swsh_transform(l_max, 1, u_slice), l_max)
              .data());

      cauchy_u_view =
          0.5 *
          (interpolated_cauchy_u_slice.data() * conj(get(b).data()) -
           conj(interpolated_cauchy_u_slice.data()) * get(a).data()) /
          square(get(omega).data());
    }
    // double-check that the result is consistent with the originally
    // constructed boundary value
    // printf("verifying boundary u\n");
    // for (size_t i = 0; i < number_of_angular_points; ++i) {
    // printf("(%e, %e) from (%e, %e)\n",
    // real(get(*cauchy_u).data()[i] - get(boundary_u).data()[i]),
    // imag(get(*cauchy_u).data()[i] - get(boundary_u).data()[i]),
    // real(get(*cauchy_u).data()[i]), imag(get(*cauchy_u).data()[i]));
    // }
    // printf("done\n");
  }
};

template <>
struct CalculateCauchyGauge<Tags::CauchyGauge<Tags::BondiQ>> {
  using argument_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>, Tags::U0,
                 Tags::GaugeOmega,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::GaugeA, Tags::GaugeB, Spectral::Swsh::Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::CauchyGauge<Tags::BondiQ>, Tags::Dy<Tags::BondiU>,
                 Tags::CauchyGauge<Tags::BondiJ>,
                 Tags::CauchyGauge<Tags::BondiBeta>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> cauchy_q,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> dy_u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> cauchy_j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          cauchy_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& /*u_0*/,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& /*eth_omega*/,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const size_t l_max) noexcept {
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points =
        get(*dy_u).size() / number_of_angular_points;

    // FIXME this is currently specialized to omega = 1
    // const auto& one_minus_y_collocation =
    // 1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
    // Spectral::Quadrature::GaussLobatto>(
    // number_of_radial_points);

    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector dy_u_slice{
          get(*dy_u).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      ComplexDataVector cauchy_q_slice{
          get(*cauchy_q).data().data() + i * number_of_angular_points,
          number_of_angular_points};

      ComplexDataVector cauchy_j_slice{
          get(*cauchy_j).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      ComplexDataVector cauchy_beta_slice{
          get(*cauchy_beta).data().data() + i * number_of_angular_points,
          number_of_angular_points};

      // note a 1-y^2 / 1-y^2 has been canceled to avoid divergences.
      ComplexDataVector cauchy_dr_u =
          0.25 / (pow<3>(get(omega).data()) * get(r).data()) *
          (conj(get(b).data()) * dy_u_slice - get(a).data() * conj(dy_u_slice));

      cauchy_q_slice =
          4.0 * square(get(r).data() * get(omega).data()) *
          exp(-2.0 * cauchy_beta_slice) *
          (cauchy_j_slice * conj(cauchy_dr_u) +
           sqrt(1.0 + cauchy_j_slice * conj(cauchy_j_slice)) * cauchy_dr_u);
    }
  }
};

template <typename Tag>
struct CalculateInertialModes;

template <>
struct CalculateInertialModes<Tags::BondiJ> {
  template <typename DataBoxType>
  static ComplexModalVector compute(
      const gsl::not_null<DataBoxType*> box,
      const ComplexModalVector& goldberg_cauchy_modes,
      const ComplexDataVector& view_y) noexcept {
    const size_t l_max = db::get<Spectral::Swsh::Tags::LMax>(*box);
    SpinWeighted<ComplexModalVector, 2> libsharp_modes{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};

    const auto& coefficients =
        Spectral::Swsh::cached_coefficients_metadata(l_max);

    for (auto coefficient_iter = coefficients.begin();
         coefficient_iter != coefficients.end(); ++coefficient_iter) {
      // if(plus_m_index < goldberg_cauchy_modes.size()) {
      Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
          *coefficient_iter, make_not_null(&libsharp_modes), 0,
          goldberg_cauchy_modes[square((*coefficient_iter).l) +
                                (*coefficient_iter).l + (*coefficient_iter).m],
          goldberg_cauchy_modes[square((*coefficient_iter).l) +
                                (*coefficient_iter).l - (*coefficient_iter).m]);
      // } else {
      // Spectral::Swsh::set_libsharp_modes_from_goldberg_modes(
      // coefficient_iter, make_not_null(&libsharp_modes), 0, 0.0, 0.0);
      // }
    }
    Scalar<SpinWeighted<ComplexDataVector, 2>> cauchy_j;
    get(cauchy_j) =
        Spectral::Swsh::inverse_swsh_transform(l_max, 1, libsharp_modes);
    return CalculateInertialModes<Tags::BondiJ>::compute(
        box, make_not_null(&cauchy_j), view_y);
  }
  template <typename DataBoxType>
  static ComplexModalVector compute(
      const gsl::not_null<DataBoxType*> box,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> cauchy_j,
      const ComplexDataVector& /*view_y*/) noexcept {
    const size_t l_max = db::get<Spectral::Swsh::Tags::LMax>(*box);

    Scalar<SpinWeighted<ComplexDataVector, 2>> inertial_j;
    ComputeGaugeAdjustedBoundaryValue<Tags::BondiJ>::apply(
        make_not_null(&inertial_j), cauchy_j, db::get<Tags::GaugeC>(*box),
        db::get<Tags::GaugeD>(*box), db::get<Tags::GaugeOmegaCD>(*box),
        db::get<Tags::CauchyAngularCoords>(*box), l_max);
    auto goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
        Spectral::Swsh::swsh_transform(l_max, 1, get(inertial_j)), l_max);
    // TEST
    // auto goldberg_modes = Spectral::Swsh::libsharp_to_goldberg_modes(
    // Spectral::Swsh::swsh_transform(make_not_null(&get(*cauchy_j)), l_max),
    // l_max);
    return goldberg_modes.data();
  }
};

}  // namespace Cce