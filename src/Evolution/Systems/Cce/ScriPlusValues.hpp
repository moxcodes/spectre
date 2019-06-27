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
    Spectral::Swsh::swsh_interpolate(
        make_not_null(&get(*cauchy_gauge_beta)), make_not_null(&beta_buffer),
        get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), l_max);

    SpinWeighted<ComplexDataVector, 0> identity_test =
        Spectral::Swsh::swsh_interpolate(
            make_not_null(&get(*cauchy_gauge_beta)), get<0>(x_of_x_tilde),
            get<1>(x_of_x_tilde), l_max);
  }
};

// template <>
// struct CalculateScriPlusValue<Tags::CauchyGaugeScriPlus<Tags::Q>> {
//   using argument_tags =
//       tmpl::list<Tags::GaugeOmega, Tags::GaugeA, Tags::GaugeB>;
//   using return_tags =
//       tmpl::list<Tags::CauchyGaugeScriPlus<Tags::Q>, Tags::Dy<Tags::U>>;
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
                 Tags::LMax>;
  using return_tags = tmpl::list<Tags::CauchyGaugeScriPlus<Tags::U0>, Tags::U0>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          cauchy_gauge_u0,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u0,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const tnsr::i<DataVector, 2>& x_tilde_of_x,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::swsh_interpolate(
        make_not_null(&get(*cauchy_gauge_u0)), make_not_null(&get(*u0)),
        get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), l_max);
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

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          inertial_time,
      const double& initial_time) noexcept {
    // this is arbitrary, has to do with choosing a BMS frame. We choose one
    // that has a particular specified initial time value.
    get(*inertial_time).data() = initial_time;
  }
};

template <typename Tag>
struct CalculateCauchyGauge;

// Note these are for debugging output, and are still in the inertial angular
// grid. After any radial interpolation, the angular grid needs to be
// interpolated to the desired Cauchy grid.

template <>
struct CalculateCauchyGauge<Tags::CauchyGauge<Tags::Beta>> {
  using argument_tags =
      tmpl::list<Tags::GaugeOmega, Tags::GaugeA, Tags::GaugeB, Tags::GaugeC,
                 Tags::GaugeD, Tags::InertialAngularCoords,
                 Tags::CauchyAngularCoords, Tags::LMax>;
  using return_tags = tmpl::list<Tags::CauchyGauge<Tags::Beta>, Tags::Beta>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          cauchy_beta,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
      const tnsr::i<DataVector, 2>& x_tilde_of_x,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
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
      Spectral::Swsh::swsh_interpolate(
          make_not_null(&interpolated_cauchy_beta_slice),
          make_not_null(&beta_slice), get<0>(x_tilde_of_x),
          get<1>(x_tilde_of_x), l_max);
      ComplexDataVector cauchy_beta_slice{
          get(*cauchy_beta).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      cauchy_beta_slice =
          interpolated_cauchy_beta_slice.data() - 0.5 * log(get(omega).data());
    }

    SpinWeighted<ComplexDataVector, 0> test_beta_cauchy{get(d).size()};
    const auto& collocation = Spectral::Swsh::precomputed_collocation<
        Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
    for (const auto& collocation_point : collocation) {
      test_beta_cauchy.data()[collocation_point.offset] =
          sin(collocation_point.theta) * sin(collocation_point.phi);
    }

    auto interpolated_cauchy_beta = Spectral::Swsh::swsh_interpolate(
        make_not_null(&test_beta_cauchy), get<0>(x_tilde_of_x),
        get<1>(x_tilde_of_x), l_max);

    Spectral::Swsh::filter_swsh_volume_quantity(
        make_not_null(&interpolated_cauchy_beta), l_max, l_max - 2, 0.0, 8);

    SpinWeighted<ComplexDataVector, 1> eth_tilde_beta =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&test_beta_cauchy), l_max);
    SpinWeighted<ComplexDataVector, -1> ethbar_tilde_beta =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&test_beta_cauchy), l_max);
    SpinWeighted<ComplexDataVector, 1> eth_beta =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&interpolated_cauchy_beta), l_max);

    auto eth_tilde_beta_interpolated = Spectral::Swsh::swsh_interpolate(
        make_not_null(&eth_tilde_beta), get<0>(x_tilde_of_x),
        get<1>(x_tilde_of_x), l_max);
    auto ethbar_tilde_beta_interpolated = Spectral::Swsh::swsh_interpolate(
        make_not_null(&ethbar_tilde_beta), get<0>(x_tilde_of_x),
        get<1>(x_tilde_of_x), l_max);
    SpinWeighted<ComplexDataVector, 1> identity_test_eth_beta =
        0.5 * (conj(get(b)) * eth_tilde_beta_interpolated +
               get(a) * ethbar_tilde_beta_interpolated);

    Spectral::Swsh::filter_swsh_volume_quantity(
        make_not_null(&identity_test_eth_beta), l_max, l_max - 2, 0.0, 8);

    Spectral::Swsh::filter_swsh_volume_quantity(
        make_not_null(&identity_test_eth_beta), l_max, l_max - 2, 0.0, 8);
    printf("Identity test: Jacobian vs x_tilde_of_x\n");
    for (size_t i = 0; i < identity_test_eth_beta.size(); ++i) {
      printf("(%e, %e) from (%e, %e)\n",
             real(identity_test_eth_beta.data()[i] - eth_beta.data()[i]),
             imag(identity_test_eth_beta.data()[i] - eth_beta.data()[i]),
             real(identity_test_eth_beta.data()[i]),
             imag(identity_test_eth_beta.data()[i]));
    }
    printf("done\n");

    SpinWeighted<ComplexDataVector, 0> test_beta{get(d).size()};
    for (const auto& collocation_point : collocation) {
      test_beta.data()[collocation_point.offset] =
          sin(collocation_point.theta) * sin(collocation_point.phi);
    }
    // Note: using the beta 'nice' values in both coordinate systems, subverting
    // their true meaning in order to make a more desirable test
    SpinWeighted<ComplexDataVector, -1> ethbar_beta =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&test_beta), l_max);
    SpinWeighted<ComplexDataVector, 1> eth_beta_cauchy =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&test_beta), l_max);

    auto eth_beta_interpolated_to_inertial = Spectral::Swsh::swsh_interpolate(
        make_not_null(&eth_beta_cauchy), get<0>(x_of_x_tilde),
        get<1>(x_of_x_tilde), l_max);
    auto ethbar_beta_interpolated_to_inertial =
        Spectral::Swsh::swsh_interpolate(make_not_null(&ethbar_beta),
                                         get<0>(x_of_x_tilde),
                                         get<1>(x_of_x_tilde), l_max);
    SpinWeighted<ComplexDataVector, 1> identity_test_eth_tilde_beta =
        0.5 * (conj(get(d)) * eth_beta_interpolated_to_inertial +
               get(c) * ethbar_beta_interpolated_to_inertial);

    auto beta_interpolated_to_inertial = Spectral::Swsh::swsh_interpolate(
        make_not_null(&test_beta), get<0>(x_of_x_tilde), get<1>(x_of_x_tilde),
        l_max);
    SpinWeighted<ComplexDataVector, 1> eth_tilde_beta_inertial =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&beta_interpolated_to_inertial), l_max);

    Spectral::Swsh::filter_swsh_volume_quantity(
        make_not_null(&identity_test_eth_tilde_beta), l_max, l_max - 2, 0.0, 8);
    printf("Identity test: Jacobian vs x_of_x_tilde\n");
    for (size_t i = 0; i < identity_test_eth_tilde_beta.size(); ++i) {
      printf("(%e, %e) from (%e, %e)\n",
             real(identity_test_eth_tilde_beta.data()[i] -
                  eth_tilde_beta_inertial.data()[i]),
             imag(identity_test_eth_tilde_beta.data()[i] -
                  eth_tilde_beta_inertial.data()[i]),
             real(identity_test_eth_tilde_beta.data()[i]),
             imag(identity_test_eth_tilde_beta.data()[i]));
    }
    printf("done\n");

    SpinWeighted<ComplexDataVector, 0> beta_interpolated_back_to_cauchy =
        Spectral::Swsh::swsh_interpolate(
            make_not_null(&beta_interpolated_to_inertial), get<0>(x_tilde_of_x),
            get<1>(x_tilde_of_x), l_max);
    printf("Identity test: Inverse transformation\n");
    for (size_t i = 0; i < identity_test_eth_tilde_beta.size(); ++i) {
      printf("(%e, %e) from (%e, %e)\n",
             real(test_beta.data()[i] -
                  beta_interpolated_back_to_cauchy.data()[i]),
             imag(test_beta.data()[i] -
                  beta_interpolated_back_to_cauchy.data()[i]),
             real(test_beta.data()[i]), imag(test_beta.data()[i]));
    }
    printf("done\n");
  }
};

template <>
struct CalculateCauchyGauge<Tags::CauchyGauge<Tags::J>> {
  using argument_tags = tmpl::list<Tags::GaugeA, Tags::GaugeB, Tags::GaugeOmega,
                                   Tags::InertialAngularCoords, Tags::LMax>;
  using return_tags = tmpl::list<Tags::CauchyGauge<Tags::J>, Tags::J>;

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
      Spectral::Swsh::swsh_interpolate(
          make_not_null(&interpolated_cauchy_j_slice), make_not_null(&j_slice),
          get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), l_max);
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
struct CalculateCauchyGauge<Tags::CauchyGauge<Tags::U>> {
  using argument_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::R>, Tags::U0,
                 Tags::GaugeOmega,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::GaugeA, Tags::GaugeB, Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::CauchyGauge<Tags::U>, Tags::U, Tags::J, Tags::Beta>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> cauchy_u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& u_0,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const size_t l_max) noexcept {
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points = get(*u).size() / number_of_angular_points;

    const auto& one_minus_y_collocation =
        1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto>(
                  number_of_radial_points);

    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector u_slice{
          get(*u).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      ComplexDataVector cauchy_u_slice{
          get(*cauchy_u).data().data() + i * number_of_angular_points,
          number_of_angular_points};

      ComplexDataVector beta_slice{
          get(*beta).data().data() + i * number_of_angular_points,
          number_of_angular_points};
      ComplexDataVector j_tilde_slice{
          get(*j).data().data() + i * number_of_angular_points,
          number_of_angular_points};

      ComplexDataVector u_undertilde =
          u_slice + exp(2.0 * beta_slice) * one_minus_y_collocation[i] /
                        (2.0 * get(r).data() * get(omega).data()) *
                        (conj(get(eth_omega).data()) * j_tilde_slice -
                         get(eth_omega).data() *
                             sqrt(1.0 + j_tilde_slice * conj(j_tilde_slice)));
      cauchy_u_slice = 0.5 / square(get(omega).data()) *
                           (conj(get(b).data()) * u_undertilde -
                            get(a).data() * conj(u_undertilde)) +
                       get(u_0).data();
    }
  }
};

template <>
struct CalculateCauchyGauge<Tags::CauchyGauge<Tags::Q>> {
  using argument_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::R>, Tags::U0,
                 Tags::GaugeOmega,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::GaugeA, Tags::GaugeB, Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::CauchyGauge<Tags::Q>, Tags::Dy<Tags::U>,
                 Tags::CauchyGauge<Tags::J>, Tags::CauchyGauge<Tags::Beta>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> cauchy_q,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> dy_u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> cauchy_j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          cauchy_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& u_0,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const size_t l_max) noexcept {
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points =
        get(*dy_u).size() / number_of_angular_points;

    // FIXME this is currently specialized to omega = 1
    const auto& one_minus_y_collocation =
        1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto>(
                  number_of_radial_points);

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

}  // namespace Cce
