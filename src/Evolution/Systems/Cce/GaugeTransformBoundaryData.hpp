// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/math/interpolators/barycentric_rational.hpp>
#include <boost/math/tools/roots.hpp>

#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Utilities/MakeArray.hpp"

namespace Cce {
// TOOD most of these computations require intermediate quantities that could be
// cached or grouped together like in the main evolution computations. These
// calculations are smaller, so there is significantly less savings, but this
// might be a spot to keep an eye on for optimization.

using compute_gauge_adjustments_setup_tags =
    tmpl::list<Tags::R, Tags::J, Tags::Dr<Tags::J>>;

template <typename Tag>
struct ComputeGaugeAdjustedBoundaryValue;

// this just needs to be interpolated to the new grid
// Consider rolling this into the beta or R one
template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::R> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::R>,
                                 Tags::BoundaryValue<Tags::R>>;
  using argument_tags =
      tmpl::list<Tags::CauchyAngularCoords, Tags::GaugeOmega, Tags::LMax>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_r,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> r,
      const tnsr::i<DataVector, 2>& x_of_x_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const size_t l_max) noexcept {
    Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(*r)),
                                                  l_max, l_max - 2);
    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*evolution_gauge_r)), make_not_null(&get(*r)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);

    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&get(*evolution_gauge_r)), l_max, l_max - 2);
    get(*evolution_gauge_r) = get(*evolution_gauge_r) / get(omega);
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::DuRDividedByR> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::DuRDividedByR>,
                 Tags::BoundaryValue<Tags::DuRDividedByR>>;
  using argument_tags =
      tmpl::list<Tags::U0, Tags::EvolutionGaugeBoundaryValue<Tags::R>,
                 Tags::GaugeA, Tags::GaugeB, Tags::GaugeOmega,
                 Tags::Du<Tags::GaugeOmega>, Tags::CauchyAngularCoords,
                 Tags::LMax>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_du_r_divided_by_r,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          du_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& u_0,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& evolution_gauge_r,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&get(*du_r_divided_by_r)), l_max, l_max - 2);

    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*evolution_gauge_du_r_divided_by_r)),
        make_not_null(&get(*du_r_divided_by_r)), get<0>(x_of_x_tilde),
        get<1>(x_of_x_tilde), l_max);

    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&get(*evolution_gauge_du_r_divided_by_r)), l_max,
        l_max - 2);

    // taking this as argument saves an interpolation, which is significantly
    // more expensive than the extra multiplication.
    SpinWeighted<ComplexDataVector, 0> r_buffer =
        get(evolution_gauge_r) * get(omega);
    auto eth_r = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        make_not_null(&r_buffer), l_max);
    get(*evolution_gauge_du_r_divided_by_r) +=
        0.25 *
            (get(u_0) * get(b) * conj(eth_r) +
             conj(get(u_0)) * get(a) * conj(eth_r) +
             conj(get(u_0)) * conj(get(b)) * eth_r +
             get(u_0) * conj(get(a)) * eth_r) /
            r_buffer -
        get(du_omega) / get(omega);
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::J> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::J>,
                                 Tags::BoundaryValue<Tags::J>>;
  using argument_tags = tmpl::list<Tags::GaugeA, Tags::GaugeB, Tags::GaugeOmega,
                                   Tags::CauchyAngularCoords, Tags::LMax>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          evolution_gauge_j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(*j)),
                                                  l_max, l_max - 2);

    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*evolution_gauge_j)), make_not_null(&get(*j)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);

    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&get(*evolution_gauge_j)), l_max, l_max - 2);

    get(*evolution_gauge_j).data() =
        ((0.25 * square(get(b).data())) * get(*evolution_gauge_j).data() +
         0.25 * square(get(a).data()) * conj(get(*evolution_gauge_j).data()) -
         0.5 * get(a).data() * get(b).data() *
             sqrt(1.0 + get(*evolution_gauge_j).data() *
                            conj(get(*evolution_gauge_j).data()))) /
        square(get(omega).data());
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::Dr<Tags::J>> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::Dr<Tags::J>>,
                 Tags::BoundaryValue<Tags::Dr<Tags::J>>>;
  using argument_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::J>,
                                   Tags::GaugeA, Tags::GaugeB, Tags::GaugeOmega,
                                   Tags::CauchyAngularCoords, Tags::LMax>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          evolution_gauge_dr_j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& evolution_gauge_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(*dr_j)),
                                                  l_max, l_max - 2);

    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*evolution_gauge_dr_j)), make_not_null(&get(*dr_j)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);

    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&get(*evolution_gauge_dr_j)), l_max, l_max - 2);

    get(*evolution_gauge_dr_j).data() =
        ((0.25 * square(get(b).data())) * get(*evolution_gauge_dr_j).data() +
         0.25 * square(get(a).data()) *
             conj(get(*evolution_gauge_dr_j).data()) -
         0.25 * get(a).data() * get(b).data() *
             (get(*evolution_gauge_dr_j).data() *
                  conj(get(evolution_gauge_j).data()) +
              conj(get(*evolution_gauge_dr_j).data()) *
                  get(evolution_gauge_j).data()) /
             sqrt(1.0 + get(evolution_gauge_j).data() *
                            conj(get(evolution_gauge_j).data()))) /
        get(omega).data();
  }
};

// beta is the same in both gauges. This still should have separate tags for the
// time being for compatibility with original evolution code.
template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::Beta> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::Beta>,
                                 Tags::BoundaryValue<Tags::Beta>>;
  using argument_tags = tmpl::list<Tags::GaugeOmega, Tags::CauchyAngularCoords,
                                   Tags::InertialAngularCoords, Tags::LMax>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_beta,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const tnsr::i<DataVector, 2>& x_of_x_tilde,
      const tnsr::i<DataVector, 2>& x_tilde_of_x, const size_t l_max) noexcept {
    // try filtering to try to get the interpolations to match up.
    Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(*beta)),
                                                  l_max, l_max - 2);

    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*evolution_gauge_beta)), make_not_null(&get(*beta)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);

    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&get(*evolution_gauge_beta)), l_max, l_max - 2);

    get(*evolution_gauge_beta) += 0.5 * log(get(omega).data());
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::Q> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::Q>,
                                 Tags::BoundaryValue<Tags::Dr<Tags::U>>,
                                 Tags::BoundaryValue<Tags::Q>>;
  using argument_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::J>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::Dr<Tags::J>>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::R>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::Beta>, Tags::GaugeA,
                 Tags::GaugeB, Tags::GaugeOmega,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::CauchyAngularCoords, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          evolution_gauge_bondi_q,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> dr_u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          boundary_q,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_j_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    apply_impl(make_not_null(&get(*evolution_gauge_bondi_q)),
               make_not_null(&get(*dr_u)), make_not_null(&get(*boundary_q)),
               get(j_tilde), get(dr_j_tilde), get(r_tilde), get(beta_tilde),
               get(a), get(b), get(omega), get(eth_omega), x_of_x_tilde, l_max);
  }
  static void apply_impl(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*>
          evolution_gauge_bondi_q,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> dr_u,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> boundary_q,
      const SpinWeighted<ComplexDataVector, 2>& j_tilde,
      const SpinWeighted<ComplexDataVector, 2>& dr_j_tilde,
      const SpinWeighted<ComplexDataVector, 0>& r_tilde,
      const SpinWeighted<ComplexDataVector, 0>& beta_tilde,
      const SpinWeighted<ComplexDataVector, 2>& a,
      const SpinWeighted<ComplexDataVector, 0>& b,
      const SpinWeighted<ComplexDataVector, 0>& omega,
      const SpinWeighted<ComplexDataVector, 1>& eth_omega,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::filter_swsh_boundary_quantity(dr_u, l_max, l_max - 6);

    SpinWeighted<ComplexDataVector, 1> evolution_coords_dr_u =
        Spectral::Swsh::swsh_interpolate_from_pfaffian(
            dr_u, get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&evolution_coords_dr_u), l_max, l_max - 6);

    auto k = SpinWeighted<ComplexDataVector, 0>{
        sqrt(1.0 + j_tilde.data() * conj(j_tilde.data()))};

    auto exp_minus_2_beta =
        SpinWeighted<ComplexDataVector, 0>{exp(-2.0 * beta_tilde.data())};

    evolution_coords_dr_u =
        0.5 * omega *
            (b * evolution_coords_dr_u + a * conj(evolution_coords_dr_u)) +
        (eth_omega * k - conj(eth_omega) * j_tilde) *
            (1.0 / (r_tilde * exp_minus_2_beta)) *
            (-1.0 / r_tilde + 0.25 * r_tilde *
                                  (dr_j_tilde * conj(dr_j_tilde) -
                                   0.25 *
                                       square(j_tilde * conj(dr_j_tilde) +
                                              dr_j_tilde * conj(j_tilde)) /
                                       (1.0 + j_tilde * conj(j_tilde)))) /
            omega +
        1.0 / (r_tilde * exp_minus_2_beta) *
            (-conj(eth_omega) * dr_j_tilde +
             0.5 * eth_omega / k *
                 (j_tilde * conj(dr_j_tilde) + conj(j_tilde) * dr_j_tilde)) /
            omega;
    // TEST
    // evolution_coords_dr_u =
    // 0.5 * omega *
    // (b * evolution_coords_dr_u + a * conj(evolution_coords_dr_u));

    *evolution_gauge_bondi_q =
        square(r_tilde) * exp_minus_2_beta *
        (j_tilde * conj(evolution_coords_dr_u) + k * evolution_coords_dr_u);
  }
};

// NOTE! this gets the boundary value for \f$\hat{U}\f$
template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::U> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::U>,
                                 Tags::BoundaryValue<Tags::U>>;
  using argument_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::J>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::R>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::Beta>, Tags::GaugeA,
                 Tags::GaugeB, Tags::GaugeOmega,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::CauchyAngularCoords, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          evolution_gauge_u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(*u)),
                                                  l_max, l_max - 6);
    SpinWeighted<ComplexDataVector, 1> evolution_coords_u =
        Spectral::Swsh::swsh_interpolate_from_pfaffian(
            make_not_null(&get(*u)), get<0>(x_of_x_tilde), get<1>(x_of_x_tilde),
            l_max);
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&evolution_coords_u), l_max, l_max - 6);

    SpinWeighted<ComplexDataVector, 0> k_tilde;
    k_tilde.data() =
        sqrt(1.0 + get(j_tilde).data() * conj(get(j_tilde).data()));

    SpinWeighted<ComplexDataVector, 0> exp_2_beta_tilde;
    exp_2_beta_tilde.data() = exp(2.0 * get(beta_tilde).data());

    // TEST
    // get(*evolution_gauge_u) =
    // 0.5 * (get(a) * conj(evolution_coords_u) + get(b) * evolution_coords_u);

    get(*evolution_gauge_u) =
        0.5 *
            (get(a) * conj(evolution_coords_u) + get(b) * evolution_coords_u) -
        exp_2_beta_tilde / (get(r_tilde) * get(omega)) *
            (conj(get(eth_omega)) * get(j_tilde) - get(eth_omega) * k_tilde);
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::W> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::W>,
                 Tags::BoundaryValue<Tags::W>, Tags::BoundaryValue<Tags::U>,
                 Tags::BoundaryValue<Tags::Beta>>;
  using argument_tags =
      tmpl::list<Tags::U0, Tags::EvolutionGaugeBoundaryValue<Tags::R>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::J>, Tags::GaugeOmega,
                 Tags::Du<Tags::GaugeOmega>,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::GaugeB, Tags::GaugeA, Tags::CauchyAngularCoords,
                 Tags::LMax>;

  // TODO arguments apply impl etc
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_w,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> w,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          boundary_u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          boundary_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& u_0,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    apply_impl(make_not_null(&get(*evolution_gauge_w)), make_not_null(&get(*w)),
               make_not_null(&get(*boundary_u)),
               make_not_null(&get(*boundary_beta)), get(u_0), get(r_tilde),
               get(j_tilde), get(omega), get(du_omega), get(eth_omega), get(b),
               get(a), x_of_x_tilde, l_max);
  }

 private:
  static void apply_impl(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>
          evolution_gauge_w,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> w,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> boundary_u,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> boundary_beta,
      const SpinWeighted<ComplexDataVector, 1>& u_0,
      const SpinWeighted<ComplexDataVector, 0>& r_tilde,
      const SpinWeighted<ComplexDataVector, 2>& j_tilde,
      const SpinWeighted<ComplexDataVector, 0>& omega,
      const SpinWeighted<ComplexDataVector, 0>& du_omega,
      const SpinWeighted<ComplexDataVector, 1>& eth_omega,
      const SpinWeighted<ComplexDataVector, 0>& b,
      const SpinWeighted<ComplexDataVector, 2>& a,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::filter_swsh_boundary_quantity(w, l_max, l_max - 2);
    Spectral::Swsh::swsh_interpolate_from_pfaffian(evolution_gauge_w, w,
                                                   get<0>(x_of_x_tilde),
                                                   get<1>(x_of_x_tilde), l_max);
    Spectral::Swsh::filter_swsh_boundary_quantity(evolution_gauge_w, l_max,
                                                  l_max - 2);

    SpinWeighted<ComplexDataVector, 1> boundary_u_of_x_tilde{};

    Spectral::Swsh::filter_swsh_boundary_quantity(boundary_u, l_max, l_max - 2);

    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&boundary_u_of_x_tilde), boundary_u, get<0>(x_of_x_tilde),
        get<1>(x_of_x_tilde), l_max);
    Spectral::Swsh::filter_swsh_boundary_quantity(boundary_u, l_max, l_max - 2);

    // TODO use beta_tilde instead for efficiency (fewer interpolations)
    SpinWeighted<ComplexDataVector, 0> boundary_beta_of_x_tilde{};
    Spectral::Swsh::filter_swsh_boundary_quantity(boundary_beta, l_max,
                                                  l_max - 2);
    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&boundary_beta_of_x_tilde), boundary_beta,
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&boundary_beta_of_x_tilde), l_max, l_max - 2);

    SpinWeighted<ComplexDataVector, 0> k_tilde;
    k_tilde.data() = sqrt(1.0 + j_tilde.data() * conj(j_tilde.data()));
    SpinWeighted<ComplexDataVector, 0> exp_2_beta_of_x_tilde;
    exp_2_beta_of_x_tilde.data() = exp(2.0 * boundary_beta_of_x_tilde.data());

    // TEST
    *evolution_gauge_w =
        *evolution_gauge_w + 1.0 / r_tilde * (1.0 / omega - 1.0) +
        2.0 * du_omega / omega +
        (1.0 / (2.0 * square(omega))) *
            (conj(eth_omega) * (b * (boundary_u_of_x_tilde - u_0) +
                                a * conj(boundary_u_of_x_tilde - u_0)) +
             eth_omega * (conj(a) * (boundary_u_of_x_tilde - u_0) +
                          conj(b) * conj(boundary_u_of_x_tilde - u_0))) -
        exp_2_beta_of_x_tilde / (2.0 * square(omega) * r_tilde) *
            (square(conj(eth_omega)) * j_tilde +
             square(eth_omega) * conj(j_tilde) -
             2.0 * eth_omega * conj(eth_omega) * k_tilde);

    // TEST
    // *evolution_gauge_w = *evolution_gauge_w + 2.0 * du_omega;
    // *evolution_gauge_w = *w;
    // TEST
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::H> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::H>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::J>,
                 Tags::BoundaryValue<Tags::SpecH>, Tags::BoundaryValue<Tags::J>,
                 Tags::BoundaryValue<Tags::Dr<Tags::J>>, Tags::U0,
                 Tags::EvolutionGaugeBoundaryValue<Tags::R>, Tags::GaugeA>;
  using argument_tags =
      tmpl::list<Tags::GaugeB, Tags::GaugeOmega, Tags::Du<Tags::GaugeOmega>,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::Du<Tags::GaugeA>, Tags::Du<Tags::GaugeB>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::DuRDividedByR>,
                 Tags::CauchyAngularCoords, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> h_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> spec_h,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dr_j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u_0,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> r_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& du_a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_b,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r_divided_by_r_tilde,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    apply_impl(make_not_null(&get(*h_tilde)), make_not_null(&get(*j_tilde)),
               make_not_null(&get(*spec_h)), make_not_null(&get(*j)),
               make_not_null(&get(*dr_j)), make_not_null(&get(*u_0)),
               make_not_null(&get(*r_tilde)), make_not_null(&get(*a)), get(b),
               get(omega), get(du_omega), get(eth_omega), get(du_a), get(du_b),
               get(du_r_divided_by_r_tilde), x_of_x_tilde, l_max);
  }

  static void apply_impl(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> h_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> j_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> spec_h,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> j,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> dr_j,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> u_0,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> r_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> a,
      const SpinWeighted<ComplexDataVector, 0>& b,
      const SpinWeighted<ComplexDataVector, 0>& omega,
      const SpinWeighted<ComplexDataVector, 0>& du_tilde_omega,
      const SpinWeighted<ComplexDataVector, 1>& eth_omega,
      const SpinWeighted<ComplexDataVector, 2>& du_a,
      const SpinWeighted<ComplexDataVector, 0>& du_b,
      const SpinWeighted<ComplexDataVector, 0>& du_r_divided_by_r_tilde,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    // optimization note: this has many allocations. They can be made fewer.
    // optimization note: this has several spin-weighted derivatives, they can
    // be aggregated
    Spectral::Swsh::filter_swsh_boundary_quantity(j, l_max, l_max - 2);
    Spectral::Swsh::filter_swsh_boundary_quantity(dr_j, l_max, l_max - 2);
    Spectral::Swsh::filter_swsh_boundary_quantity(spec_h, l_max, l_max - 2);
    SpinWeighted<ComplexDataVector, 2> j_of_x_tilde =
        Spectral::Swsh::swsh_interpolate_from_pfaffian(
            j, get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);
    SpinWeighted<ComplexDataVector, 2> dr_j_of_x_tilde =
        Spectral::Swsh::swsh_interpolate_from_pfaffian(
            dr_j, get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);
    SpinWeighted<ComplexDataVector, 2> h_of_x_tilde =
        Spectral::Swsh::swsh_interpolate_from_pfaffian(
            spec_h, get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);
    Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&j_of_x_tilde),
                                                  l_max, l_max - 2);
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&dr_j_of_x_tilde), l_max, l_max - 2);
    Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&h_of_x_tilde),
                                                  l_max, l_max - 2);

    // this manipulation is needed because spin weight 3 or higher is not
    // supported
    SpinWeighted<ComplexDataVector, 0> abar_j_of_x_tilde =
        conj(*a) * j_of_x_tilde;
    SpinWeighted<ComplexDataVector, 1> u0bar_j_of_x_tilde =
        conj(*u_0) * j_of_x_tilde;

    auto eth_r_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(r_tilde,
                                                                   l_max);
    SpinWeighted<ComplexDataVector, 1> abar_eth_j_of_x_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&abar_j_of_x_tilde), l_max) -
        conj(*a) * eth_r_tilde * omega * dr_j_of_x_tilde -
        j_of_x_tilde *
            conj(Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                a, l_max));
    SpinWeighted<ComplexDataVector, 2> u0bar_eth_j_of_x_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&u0bar_j_of_x_tilde), l_max) -
        conj(*u_0) * eth_r_tilde * omega * dr_j_of_x_tilde -
        j_of_x_tilde *
            conj(Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                u_0, l_max));
    SpinWeighted<ComplexDataVector, 1> ethbar_j_of_x_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&j_of_x_tilde), l_max) -
        conj(eth_r_tilde) * omega * dr_j_of_x_tilde;

    // TODO check factors of omega
    SpinWeighted<ComplexDataVector, 0> k;
    k.data() = sqrt(1.0 + j_of_x_tilde.data() * conj(j_of_x_tilde.data()));
    SpinWeighted<ComplexDataVector, 2> angular_derivative_part =
        0.25 * (*u_0) * b * ethbar_j_of_x_tilde +
        0.25 * (*u_0) * abar_eth_j_of_x_tilde +
        0.25 * conj(*u_0) * (*a) * ethbar_j_of_x_tilde +
        0.25 * conj(b) * u0bar_eth_j_of_x_tilde;
    SpinWeighted<ComplexDataVector, 0> du_omega =
        (du_tilde_omega -
         0.25 * (b * (*u_0) + (*a) * conj(*u_0)) * conj(eth_omega) -
         0.25 * (conj(*a) * (*u_0) + conj(b * (*u_0))) * eth_omega);
    SpinWeighted<ComplexDataVector, 2> du_tilde_j =
        h_of_x_tilde + angular_derivative_part +
        (du_omega) * (*r_tilde) * dr_j_of_x_tilde;

    *h_tilde =
        (0.5 * b * du_b * j_of_x_tilde +
         0.5 * (*a) * du_a * conj(j_of_x_tilde) -
         0.5 * (*a * du_b + b * du_a) * k + 0.25 * square(b) * du_tilde_j +
         0.25 * square(*a) * conj(du_tilde_j) -
         0.25 * (*a) * b *
             (du_tilde_j * conj(j_of_x_tilde) +
              j_of_x_tilde * conj(du_tilde_j)) /
             k) /
            square(omega) -
        0.5 * du_tilde_omega / pow<3>(omega) *
            (square(b) * j_of_x_tilde + square(*a) * conj(j_of_x_tilde) -
             2.0 * (*a) * b * k) +
        du_r_divided_by_r_tilde * (*r_tilde) * 0.25 *
            (square(b) * dr_j_of_x_tilde + square(*a) * conj(dr_j_of_x_tilde) -
             (*a) * b *
                 (dr_j_of_x_tilde * conj(j_of_x_tilde) +
                  j_of_x_tilde * conj(dr_j_of_x_tilde)) /
                 k) /
            omega;

    // REMOVE BELOW
    // auto time_derivative_part =
    // 0.25 * (square(b) * h_of_x_tilde + square(*a) * conj(h_of_x_tilde) -
    // (*a) * b *
    // (j_of_x_tilde * conj(h_of_x_tilde) +
    // h_of_x_tilde * conj(j_of_x_tilde)) /
    // (1.0 + j_of_x_tilde * conj(j_of_x_tilde))) +
    // 0.5 * b * du_b * j_of_x_tilde + 0.5 * (*a) * du_a * conj(j_of_x_tilde) -
    // 0.5 * ((*a) * du_b + b * du_a) * k;

    // note below we change the du_omega which is actually a derivative at fixed
    // tilded angles to the version at fixed worldtube angle

    auto eth_u0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(u_0, l_max);
    auto ethbar_u0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(u_0,
                                                                      l_max);
    // *h_tilde =
    // h_of_x_tilde + du_b * j_of_x_tilde +
    // 0.5 * (*u_0) * ethbar_j_of_x_tilde + 0.5 * u0bar_eth_j_of_x_tilde +
    // -du_a * k + 0.5 * (ethbar_u0 + conj(ethbar_u0)) * j_of_x_tilde -
    // 0.25 * (ethbar_u0 + conj(ethbar_u0)) * (*r_tilde) * dr_j_of_x_tilde +
    // du_r_divided_by_r_tilde * (*r_tilde) * (dr_j_of_x_tilde);

    // *h_tilde =
    // h_of_x_tilde + du_b * j_of_x_tilde + angular_derivative_part -
    // du_a * k -
    // 0.5 * du_tilde_omega / pow<3>(omega) *
    // (square(b) * j_of_x_tilde +
    // square(*a) * conj(j_of_x_tilde - 2.0 * (*a) * b * k)) -
    // 0.25 * (ethbar_u0 + conj(ethbar_u0)) * (*r_tilde) * dr_j_of_x_tilde +
    // du_r_divided_by_r_tilde * (*r_tilde) * (dr_j_of_x_tilde);
  }
};

struct GaugeUpdateU {
  using argument_tags = tmpl::list<Tags::CauchyAngularCoords, Tags::GaugeOmega,
                                   Tags::Exp2Beta, Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::DuCauchyAngularCoords, Tags::Du<Tags::GaugeA>,
                 Tags::Du<Tags::GaugeB>, Tags::U0, Tags::U,
                 Tags::Du<Tags::GaugeOmega>, Tags::GaugeA, Tags::GaugeB>;

  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 2>*> du_x,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> du_a,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> du_b,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u_0,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> du_omega,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> a,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> b,
      const tnsr::i<DataVector, 2> x_of_x_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp2beta,
      const size_t l_max) noexcept {
    size_t number_of_radial_points =
        get(*u).size() /
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    // u_hat_0
    ComplexDataVector u_scri_slice{
        get(*u).data().data() +
            (number_of_radial_points - 1) *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    get(*u_0).data() = u_scri_slice;
    // subtract u_hat_0 from u
    // TEST
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector angular_view{
          get(*u).data().data() +
              i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
      angular_view -= get(*u_0).data();
    }
    // TEST
    // u_under_tilde_0 (note: other corrections from radial coordinate change
    // don't appear as we are evaluating at scri+).
    get(*u_0) = get(*u_0);
    // u_0
    get(*u_0) = (2.0 / (get(*b) * conj(get(*b)) - get(*a) * conj(get(*a)))) *
                (conj(get(*b)) * get(*u_0) - get(*a) * conj(get(*u_0)));

    // Note we store x^\phi * sin(theta) rather than x^\phi to reduce the number
    // of places sin(theta)'s appear. Unfortunately, without reformulating the
    // spin-weighted spherical harmonics themselves, we won't be able to
    // eliminate sin(theta)s completely

    get<0>(*du_x) = -real(get(*u_0).data());
    // // note this is actually sin theta * du_phi, so we need to include a
    // // correction.
    get<1>(*du_x) = -imag(get(*u_0).data()) + get<1>(x_of_x_tilde) *
                                                  cos(get<0>(x_of_x_tilde)) *
                                                  get<0>(*du_x);
    // TEST
    // get<0>(*du_x) = 0.0;
    // // note this is actually sin theta * du_phi, so we need to include a
    // // correction.
    // get<1>(*du_x) = 0.0;

    // at some point we might want to split these out into their own
    // computational function
    SpinWeighted<ComplexDataVector, 1> b_u0 = get(*b) * get(*u_0);
    SpinWeighted<ComplexDataVector, 1> a_u0bar = get(*a) * conj(get(*u_0));

    auto eth_a_u0bar =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&a_u0bar), l_max);
    auto ethbar_a_u0bar =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&a_u0bar), l_max);

    auto eth_b = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        make_not_null(&get(*b)), l_max);
    auto ethbar_b =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&get(*b)), l_max);

    SpinWeighted<ComplexDataVector, 0> a_abar = conj(get(*a)) * get(*a);

    auto ethbar_a =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&get(*a)), l_max);

    SpinWeighted<ComplexDataVector, 1> abar_eth_a =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&a_abar), l_max) -
        get(*a) * conj(ethbar_a);

    ComplexDataVector exp2beta_buffer = get(exp2beta).data();
    ComplexDataVector exp2beta_slice = ComplexDataVector{
        exp2beta_buffer.data() +
            (number_of_radial_points - 1) *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    // note these are the angular derivatives in the asymptotically inertial
    // frame
    auto ethbar_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&get(*u_0)), l_max);
    auto eth_u_0 = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        make_not_null(&get(*u_0)), l_max);

    get(*du_a) = -0.25 * (get(*a) * conj(get(*b)) * conj(ethbar_u_0) +
                          square(get(*a)) * conj(eth_u_0) +
                          get(*b) * conj(get(*b)) * eth_u_0 +
                          get(*b) * get(*a) * ethbar_u_0);

    get(*du_b) = -0.25 * (get(*a) * get(*b) * conj(eth_u_0) +
                          get(*a) * conj(get(*a)) * conj(ethbar_u_0) +
                          square(get(*b)) * ethbar_u_0 +
                          get(*b) * conj(get(*a)) * eth_u_0);

    // get(*du_a) = -eth_u_0;
    // get(*du_b) = -ethbar_u_0;

    // TEST TEST
    // get(*du_a).data() = 0.0;
    // get(*du_b).data() = 0.0;

    get(*du_omega) =
        -0.125 *
        (get(*b) * ethbar_u_0 + conj(get(*a)) * eth_u_0 +
         get(*a) * conj(eth_u_0) + conj(get(*b)) * conj(ethbar_u_0)) *
        get(omega);
    // TEST
    // get(*du_omega) = -0.25 * (ethbar_u_0 + conj(ethbar_u_0)) * get(omega);
  }
};

struct GaugeUpdateJacobianFromCoords {
  using argument_tags = tmpl::list<Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::GaugeA, Tags::GaugeB, Tags::InertialAngularCoords,
                 Tags::CauchyAngularCoords>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> a,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> b,
      const gsl::not_null<tnsr::i<DataVector, 2>*> x_tilde_of_x,
      const gsl::not_null<tnsr::i<DataVector, 2>*> x_of_x_tilde,
      const size_t l_max) noexcept {
    // first, interpolate to a new grid. For this, we assume that the theta
    // (more slowly varying) is already on a gauss-legendre grid.
    // Therefore, for each constant-in-theta view, we construct a similar-order
    // Gauss-Legendre grid using Barycentric rational interpolation.
    const size_t number_of_theta_points =
        Spectral::Swsh::number_of_swsh_theta_collocation_points(l_max);
    const size_t number_of_phi_points =
        Spectral::Swsh::number_of_swsh_phi_collocation_points(l_max);
    tnsr::i<DataVector, 2> x_tilde_of_x_gl{get(*a).size()};
    DataVector phi_angles{number_of_phi_points};
    for (size_t i = 0; i < number_of_phi_points; ++i) {
      phi_angles[i] = static_cast<double>(i) * 2.0 * M_PI /
                      static_cast<double>(number_of_phi_points);
    }
    // TODO turn this into gauss-lobatto and add the additional duplicate
    // collocation point at the end to make the interpolation work.
    const auto& gl_collocation =
        Spectral::collocation_points<Spectral::Basis::Legendre,
                                     Spectral::Quadrature::Gauss>(
            number_of_phi_points);
    DataVector phi_gl_collocation = gl_collocation * M_PI + M_PI;
    const auto& theta_size_gl_collocation =
        Spectral::collocation_points<Spectral::Basis::Legendre,
                                     Spectral::Quadrature::Gauss>(
            number_of_theta_points);
    DataVector theta_gl_collocation = gl_collocation * M_PI / 2.0 + M_PI / 2.0;

    // TODO this needs to be changed so that it doesn't demand so large a
    // collocation set, this is unreasonably expensive at ~50 points
    printf("collocation %zu, %zu\n", phi_gl_collocation.size(),
           number_of_phi_points);
    for (auto val : phi_gl_collocation) {
      printf("%e\n", val);
    }
    printf("interpolating to gl grid\n");
    for (size_t i = 0; i < number_of_theta_points; ++i) {
      DataVector theta_gl_view{
          get<0>(x_tilde_of_x_gl).data() + i * number_of_phi_points,
          number_of_phi_points};
      DataVector phi_gl_view{
          get<1>(x_tilde_of_x_gl).data() + i * number_of_phi_points,
          number_of_phi_points};
      DataVector theta_view{
          get<0>(*x_tilde_of_x).data() + i * number_of_phi_points,
          number_of_phi_points};
      DataVector phi_view{
          get<1>(*x_tilde_of_x).data() + i * number_of_phi_points,
          number_of_phi_points};
      boost::math::barycentric_rational<double> theta_interpolator(
          phi_angles.data(), theta_view.data(), number_of_phi_points,
          number_of_phi_points / 2);
      boost::math::barycentric_rational<double> phi_interpolator(
          phi_angles.data(), phi_view.data(), number_of_phi_points,
          number_of_phi_points / 2);
      for (size_t j = 0; j < number_of_phi_points; ++j) {
        printf("original coordinates, %e, %e\n", theta_view[j], phi_view[j]);
        printf("Anticipated theta collocation value %e\n",
               theta_gl_collocation[i]);
      }
      for (size_t j = 0; j < number_of_phi_points; ++j) {
        theta_gl_view[j] = theta_interpolator(phi_gl_collocation[j]);
        printf("theta : %e, %e\n", theta_gl_view[j], phi_gl_collocation[j]);
        phi_gl_view[j] = phi_interpolator(phi_gl_collocation[j]);
        printf("phi : %e, %e\n", phi_gl_view[j], phi_gl_collocation[j]);
      }
    }

    // With the new grid, we can now use the standard differentiation matrices
    // applied in the two respective directions to obtain the four Jacobian
    // components. By convention, the first index is the derivative and the
    // second is the coordinate being differentiated. Note that this is
    // \partial_x \tilde{x}.
    tnsr::iJ<DataVector, 2> dx_of_x_tilde_gl{get(*a).size()};
    printf("applying differentiation matrix\n");
    const Matrix& phi_differentiation_matrix =
        Spectral::differentiation_matrix<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::Gauss>(
            number_of_phi_points);
    const Matrix& theta_differentiation_matrix =
        Spectral::differentiation_matrix<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::Gauss>(
            number_of_theta_points);
    const std::array<Matrix, 2> theta_derivative =
        make_array(Matrix{}, theta_differentiation_matrix);
    const std::array<Matrix, 2> phi_derivative =
        make_array(phi_differentiation_matrix, Matrix{});
    Index<2> extents{number_of_phi_points, number_of_theta_points};
    apply_matrices(make_not_null(&get<0, 0>(dx_of_x_tilde_gl)),
                   theta_derivative, get<0>(x_tilde_of_x_gl), extents);
    apply_matrices(make_not_null(&get<0, 1>(dx_of_x_tilde_gl)),
                   theta_derivative, get<1>(x_tilde_of_x_gl), extents);
    apply_matrices(make_not_null(&get<1, 0>(dx_of_x_tilde_gl)), phi_derivative,
                   get<0>(x_tilde_of_x_gl), extents);
    apply_matrices(make_not_null(&get<1, 1>(dx_of_x_tilde_gl)), phi_derivative,
                   get<1>(x_tilde_of_x_gl), extents);

    // Apply extra factors of Pi to the derivatives.
    get<0, 0>(dx_of_x_tilde_gl) /= M_PI / 2.0;
    printf("dtheta theta tilde\n");
    get<0, 1>(dx_of_x_tilde_gl) /= M_PI / 2.0;
    get<1, 0>(dx_of_x_tilde_gl) /= M_PI;
    get<1, 1>(dx_of_x_tilde_gl) /= M_PI;

    for (size_t A = 0; A < 2; ++A) {
      for (size_t B = 0; B < 2; ++B) {
        printf("dx of x tilde %zu, %zu\n", A, B);
        for (auto val : dx_of_x_tilde_gl.get(A, B)) {
          printf("%e\n", val);
        }
      }
    }
    printf("interpolating back to angular grid\n");
    tnsr::iJ<DataVector, 2> dx_of_x_tilde{get(*a).size()};
    // interpolate the derivatives back to the equiangular grid
    for (size_t A = 0; A < 2; ++A) {
      for (size_t B = 0; B < 2; ++B) {
        for (size_t i = 0; i < number_of_theta_points; ++i) {
          DataVector gl_view{
              dx_of_x_tilde_gl.get(A, B).data() + i * number_of_phi_points,
              number_of_phi_points};
          DataVector target_view{
              dx_of_x_tilde.get(A, B).data() + i * number_of_phi_points,
              number_of_phi_points};
          boost::math::barycentric_rational<double> interpolator(
              phi_gl_collocation.data(), gl_view.data(), number_of_phi_points,
              number_of_phi_points / 2);
          for (size_t j = 0; j < number_of_phi_points; ++j) {
            target_view[j] = interpolator(phi_angles[j]);
          }
        }
      }
    }

    // We also need to populate a DataVector of the theta collocation points for
    // use in the spin-weighted jacobian factors below
    DataVector theta_collocation{get(*a).size()};
    const auto& collocation = Spectral::Swsh::precomputed_collocation<
        Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
    for (const auto& collocation_point : collocation) {
      theta_collocation[collocation_point.offset] = collocation_point.theta;
    }
    printf("creating a and b spin-weighted scalars\n");

    SpinWeighted<ComplexDataVector, 2> a_of_x;
    a_of_x.data() =
        std::complex<double>(1.0, 0.0) *
            (get<0, 0>(dx_of_x_tilde) - sin(get<0>(*x_tilde_of_x)) /
                                            sin(theta_collocation) *
                                            get<1, 1>(dx_of_x_tilde)) +
        std::complex<double>(0.0, 1.0) *
            (1.0 / sin(theta_collocation) * get<1, 0>(dx_of_x_tilde) +
             sin(get<0>(*x_tilde_of_x)) * get<0, 1>(dx_of_x_tilde));
    SpinWeighted<ComplexDataVector, 0> b_of_x;
    b_of_x.data() =
        std::complex<double>(1.0, 0.0) *
            (get<0, 0>(dx_of_x_tilde) + sin(get<0>(*x_tilde_of_x)) /
                                            sin(theta_collocation) *
                                            get<1, 1>(dx_of_x_tilde)) +
        std::complex<double>(0.0, 1.0) *
            (-1.0 / sin(theta_collocation) * get<1, 0>(dx_of_x_tilde) +
             sin(get<0>(*x_tilde_of_x)) * get<0, 1>(dx_of_x_tilde));

    // extra allocation for debugging purposes - need to copy in order to check
    // the current values of a and b.
    auto a_interpolated = Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&a_of_x), get<0>(*x_of_x_tilde), get<1>(*x_of_x_tilde),
        l_max);
    auto b_interpolated = Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&b_of_x), get<0>(*x_of_x_tilde), get<1>(*x_of_x_tilde),
        l_max);
    printf("Identity check : Jacobian vs explicit derivatives : a");
    for (size_t i = 0; i < a_interpolated.size(); ++i) {
      printf("(%e, %e) from (%e, %e)\n",
             real(get(*a).data()[i] - a_interpolated.data()[i]),
             imag(get(*a).data()[i] - a_interpolated.data()[i]),
             real(get(*a).data()[i]), imag(get(*a).data()[i]));
    }

    printf("Identity check : Jacobian vs explicit derivatives : b");
    for (size_t i = 0; i < b_interpolated.size(); ++i) {
      printf("(%e, %e) from (%e, %e)\n",
             real(get(*b).data()[i] - b_interpolated.data()[i]),
             imag(get(*b).data()[i] - b_interpolated.data()[i]),
             real(get(*b).data()[i]), imag(get(*b).data()[i]));
    }
  }
};

struct GaugeUpdateDuXtildeOfX {
  using argument_tags = tmpl::list<Tags::InertialAngularCoords, Tags::LMax>;
  using return_tags = tmpl::list<Tags::DuInertialAngularCoords, Tags::GaugeA,
                                 Tags::GaugeB, Tags::U0>;

  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 2>*> du_x_tilde_of_x,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          a_of_x_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          b_of_x_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          u_0_of_x_tilde,
      const tnsr::i<DataVector, 2>& x_tilde_of_x, const size_t l_max) noexcept {
    // interpolate a and b back to the non-inertial coordinates.
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&get(*a_of_x_tilde)), l_max, l_max - 2);
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&get(*b_of_x_tilde)), l_max, l_max - 2);
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&get(*u_0_of_x_tilde)), l_max, l_max - 2);

    SpinWeighted<ComplexDataVector, 2> a_of_x =
        Spectral::Swsh::swsh_interpolate_from_pfaffian(
            make_not_null(&get(*a_of_x_tilde)), get<0>(x_tilde_of_x),
            get<1>(x_tilde_of_x), l_max);

    SpinWeighted<ComplexDataVector, 0> b_of_x =
        Spectral::Swsh::swsh_interpolate_from_pfaffian(
            make_not_null(&get(*b_of_x_tilde)), get<0>(x_tilde_of_x),
            get<1>(x_tilde_of_x), l_max);

    SpinWeighted<ComplexDataVector, 1> u_0_of_x =
        Spectral::Swsh::swsh_interpolate_from_pfaffian(
            make_not_null(&get(*u_0_of_x_tilde)), get<0>(x_tilde_of_x),
            get<1>(x_tilde_of_x), l_max);

    Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&a_of_x), l_max,
                                                  l_max - 2);
    Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&b_of_x), l_max,
                                                  l_max - 2);
    Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&u_0_of_x),
                                                  l_max, l_max - 2);

    // \partial_u \tilde{x}^{\tilde{A}} = -U^A \partial_A \tilde{x}^{\tilde{A}}
    // these are awkward for spin-weight reasoning because the coordinates
    // themselves are not well represented by any particular spin weighted
    // decomposition.
    /// FIXME Something is wrong with the coordinate evolutions.

    // get<0>(*du_x_tilde_of_x) = real(
    // 0.25 * conj(u_0_of_x.data()) * (a_of_x.data() + conj(b_of_x.data())) +
    // 0.25 * u_0_of_x.data() * (conj(a_of_x.data()) + b_of_x.data()));
    // get<1>(*du_x_tilde_of_x) =
    // real(0.25 * std::complex<double>(0.0, -1.0) * conj(u_0_of_x.data()) *
    // (a_of_x.data() - conj(b_of_x.data())) +
    // 0.25 * std::complex<double>(0.0, -1.0) * u_0_of_x.data() *
    // (-conj(a_of_x.data()) + b_of_x.data())) +
    // get<1>(x_tilde_of_x) * cos(get<0>(x_tilde_of_x)) *
    // get<0>(*du_x_tilde_of_x);

    get<0>(*du_x_tilde_of_x) =
        0.25 *
        real(u_0_of_x.data() * (b_of_x.data() + conj(a_of_x.data())) +
             conj(u_0_of_x.data()) * (a_of_x.data() + conj(b_of_x.data())));
    get<1>(*du_x_tilde_of_x) =
        0.25 * real(std::complex<double>(0.0, -1.0) *
                    (u_0_of_x.data() * (b_of_x.data() - conj(a_of_x.data())) +
                     conj(u_0_of_x.data()) *
                         (a_of_x.data() - conj(b_of_x.data())))) +
        get<1>(x_tilde_of_x) * cos(get<0>(x_tilde_of_x)) *
            get<0>(*du_x_tilde_of_x);

    // TEST linearized approximation
    // get<0>(*du_x_tilde_of_x) = real(u_0_of_x.data());
    // get<1>(*du_x_tilde_of_x) =
    // imag(u_0_of_x.data()) + get<1>(x_tilde_of_x) *
    // cos(get<0>(x_tilde_of_x)) *
    // get<0>(*du_x_tilde_of_x);

    // get<0>(*du_x_tilde_of_x) = 0.0;
    // get<1>(*du_x_tilde_of_x) = 0.0;
  }
};

struct InitializeXtildeOfX {
  using return_tags = tmpl::list<Tags::InertialAngularCoords>;
  using argument_tags = tmpl::list<Tags::LMax>;

  static void apply(const gsl::not_null<tnsr::i<DataVector, 2>*> x_tilde_of_x,
                    const size_t l_max) noexcept {
    const auto& collocation = Spectral::Swsh::precomputed_collocation<
        Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
    for (const auto& collocation_point : collocation) {
      get<0>(*x_tilde_of_x)[collocation_point.offset] = collocation_point.theta;
      get<1>(*x_tilde_of_x)[collocation_point.offset] =
          collocation_point.phi * sin(collocation_point.theta);
      // TEST: attempting a deliberately screwball set of coordinates to
      // highlight mistakes in first timestep

      // full test: has nontrivial a, b, and omega
      // get<0>(*x_tilde_of_x)[collocation_point.offset] =
      // collocation_point.theta; auto rootfind = boost::math::tools::bisect(
      //     [&collocation_point](double x) {
      //       return collocation_point.phi - (x - 1.0e-3 * sin(x));
      //     },
      //     collocation_point.phi - 2.0e-3, collocation_point.phi + 2.0e-3,
      //     [](double x, double y) { return abs(x - y) < 1.0e-14; });
      // // printf("rootfind test %e, %e, %e\n", collocation_point.phi,
      // // rootfind.first, rootfind.second);
      // get<1>(*x_tilde_of_x)[collocation_point.offset] =
      //     0.5 * (rootfind.first + rootfind.second) *
      //     sin(collocation_point.theta);

      // partial test: omega=1
      // get<0>(*x_tilde_of_x)[collocation_point.offset] =
      // collocation_point.theta;
      // get<1>(*x_tilde_of_x)[collocation_point.offset] =
      // ((collocation_point.phi + 1.0e-3 * cos(collocation_point.theta))) *
      // sin(collocation_point.theta);
    }
  }
};

struct GaugeUpdateOmega {
  using argument_tags = tmpl::list<Tags::GaugeA, Tags::GaugeB, Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::GaugeOmega,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> omega,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          eth_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const size_t l_max) noexcept {
    get(*omega) = 0.5 * sqrt(get(b).data() * conj(get(b).data()) -
                             get(a).data() * conj(get(a).data()));
    // TEST
    // get(*omega).data() = 1.0;

    Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        make_not_null(&get(*eth_omega)), make_not_null(&get(*omega)), l_max);
  }
};

struct InitializeGauge {
  using argument_tags = tmpl::list<Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::CauchyAngularCoords, Tags::GaugeA, Tags::GaugeB>;

  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 2>*> x_of_x_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> a,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> b,
      const size_t l_max) noexcept {
    const auto& collocation = Spectral::Swsh::precomputed_collocation<
        Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
    for (const auto& collocation_point : collocation) {
      get<0>(*x_of_x_tilde)[collocation_point.offset] = collocation_point.theta;
      get<1>(*x_of_x_tilde)[collocation_point.offset] =
          collocation_point.phi * sin(collocation_point.theta);
      // TEST: attempting a deliberately screwball set of coordinates to
      // highlight mistakes in first timestep

      // full test, has nontrivial a, b, omega, and angular derivatives of omega
      // NOT REPRESENTABLE AS SPIN-2, EXPECT BAD TRANSFORMS
      // get<0>(*x_of_x_tilde)[collocation_point.offset] =
      // collocation_point.theta;
      // get<1>(*x_of_x_tilde)[collocation_point.offset] =
      // (collocation_point.phi - 1.0e-3 * sin(collocation_point.phi)) *
      // sin(collocation_point.theta);
      // get(*a).data()[collocation_point.offset] =
      // 1.0e-3 * cos(collocation_point.phi);
      // get(*b).data()[collocation_point.offset] =
      // 2.0 - 1.0e-3 * cos(collocation_point.phi);

      // partial test, has nontrivial a, and b, but omega is 1
      // get<0>(*x_of_x_tilde)[collocation_point.offset] =
      // collocation_point.theta;
      // get<1>(*x_of_x_tilde)[collocation_point.offset] =
      // (collocation_point.phi - 1.0e-3 * cos(collocation_point.theta)) *
      // sin(collocation_point.theta);
      // get(*a).data()[collocation_point.offset] =
      // -std::complex<double>(0.0, 1.0) * 1.0e-3 *
      // square(sin(collocation_point.theta));
      // get(*b).data()[collocation_point.offset] =
      // 2.0 - std::complex<double>(0.0, 1.0) * 1.0e-3 *
      // square(sin(collocation_point.theta));

      // TEST
    }
    get(*a).data() = 0.0;
    get(*b).data() = 2.0;
  }
};
}  // namespace Cce
