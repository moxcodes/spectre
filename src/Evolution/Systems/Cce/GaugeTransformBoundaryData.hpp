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
    // Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(*r)),
    // l_max, l_max - 2);
    SpinWeighted<ComplexDataVector, 0> r_over_omega = get(*r) / get(omega);

    Spectral::Swsh::swsh_interpolate(
        make_not_null(&get(*evolution_gauge_r)), make_not_null(&r_over_omega),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);

    // Spectral::Swsh::filter_swsh_boundary_quantity(
    // make_not_null(&get(*evolution_gauge_r)), l_max, l_max - 2);
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::DuRDividedByR> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::DuRDividedByR>,
                 Tags::BoundaryValue<Tags::DuRDividedByR>>;
  using argument_tags =
      tmpl::list<Tags::U0, Tags::EvolutionGaugeBoundaryValue<Tags::R>,
                 Tags::GaugeA, Tags::GaugeB, Tags::GaugeOmegaCD,
                 Tags::Du<Tags::GaugeOmegaCD>, Tags::CauchyAngularCoords,
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
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega_cd,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    // Spectral::Swsh::filter_swsh_boundary_quantity(
    // make_not_null(&get(*du_r_divided_by_r)), l_max, l_max - 2);

    Spectral::Swsh::swsh_interpolate(
        make_not_null(&get(*evolution_gauge_du_r_divided_by_r)),
        make_not_null(&get(*du_r_divided_by_r)), get<0>(x_of_x_tilde),
        get<1>(x_of_x_tilde), l_max);

    // taking this as argument saves an interpolation, which is significantly
    // more expensive than the extra multiplication.
    SpinWeighted<ComplexDataVector, 0> r_buffer =
        get(evolution_gauge_r) / get(omega_cd);
    auto eth_r = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        make_not_null(&r_buffer), l_max);
    get(*evolution_gauge_du_r_divided_by_r) +=
        -0.5 * (get(u_0) * conj(eth_r) + conj(get(u_0)) * eth_r) / r_buffer +
        get(du_omega_cd) / get(omega_cd);
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::J> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::J>,
                                 Tags::BoundaryValue<Tags::J>>;
  using argument_tags =
      tmpl::list<Tags::GaugeC, Tags::GaugeD, Tags::GaugeOmegaCD,
                 Tags::CauchyAngularCoords, Tags::LMax>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          evolution_gauge_j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    // Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(*j)),
    // l_max, l_max - 2);

    Spectral::Swsh::swsh_interpolate(
        make_not_null(&get(*evolution_gauge_j)), make_not_null(&get(*j)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);

    // NOTE this might require a filter between the jacobian factor and the
    // conformal factor to maximize precision.
    get(*evolution_gauge_j).data() =
        0.25 *
        (square(conj(get(d).data())) * get(*evolution_gauge_j).data() +
         square(get(c).data()) * conj(get(*evolution_gauge_j).data()) +
         2.0 * get(c).data() * conj(get(d).data()) *
             sqrt(1.0 + get(*evolution_gauge_j).data() *
                            conj(get(*evolution_gauge_j).data()))) /
        square(get(omega_cd).data());

    // Spectral::Swsh::filter_swsh_boundary_quantity(
    // make_not_null(&get(*evolution_gauge_j)), l_max, l_max - 2);
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::Dr<Tags::J>> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::Dr<Tags::J>>,
                 Tags::BoundaryValue<Tags::Dr<Tags::J>>>;
  using argument_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::J>, Tags::GaugeC,
                 Tags::GaugeD, Tags::GaugeOmegaCD, Tags::CauchyAngularCoords,
                 Tags::LMax>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          evolution_gauge_dr_j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& evolution_gauge_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    // Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(*dr_j)),
    // l_max, l_max - 2);

    Spectral::Swsh::swsh_interpolate(
        make_not_null(&get(*evolution_gauge_dr_j)), make_not_null(&get(*dr_j)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);

    // NOTE this might require a filter between the jacobian factor and the
    // conformal factor to maximize precision.
    get(*evolution_gauge_dr_j).data() =
        ((0.25 * square(conj(get(d).data()))) *
             get(*evolution_gauge_dr_j).data() +
         0.25 * square(get(c).data()) *
             conj(get(*evolution_gauge_dr_j).data()) +
         0.25 * get(c).data() * conj(get(d).data()) *
             (get(*evolution_gauge_dr_j).data() *
                  conj(get(evolution_gauge_j).data()) +
              conj(get(*evolution_gauge_dr_j).data()) *
                  get(evolution_gauge_j).data()) /
             sqrt(1.0 + get(evolution_gauge_j).data() *
                            conj(get(evolution_gauge_j).data()))) /
        pow<3>(get(omega_cd).data());

    // Spectral::Swsh::filter_swsh_boundary_quantity(
    // make_not_null(&get(*evolution_gauge_dr_j)), l_max, l_max - 2);
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
    SpinWeighted<ComplexDataVector, 0> beta_buffer;
    // Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(*beta)),
    // l_max, l_max - 2);
    beta_buffer.data() = get(*beta).data() + 0.5 * log(get(omega).data());

    Spectral::Swsh::swsh_interpolate(
        make_not_null(&get(*evolution_gauge_beta)), make_not_null(&beta_buffer),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);
    // Spectral::Swsh::filter_swsh_boundary_quantity(
    // make_not_null(&get(*evolution_gauge_beta)), l_max, l_max - 2);
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
                 Tags::EvolutionGaugeBoundaryValue<Tags::Beta>, Tags::GaugeC,
                 Tags::GaugeD, Tags::GaugeOmegaCD,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
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
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    apply_impl(make_not_null(&get(*evolution_gauge_bondi_q)),
               make_not_null(&get(*dr_u)), make_not_null(&get(*boundary_q)),
               get(j_tilde), get(dr_j_tilde), get(r_tilde), get(beta_tilde),
               get(c), get(d), get(omega_cd), get(eth_omega_cd), x_of_x_tilde,
               l_max);
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
      const SpinWeighted<ComplexDataVector, 2>& c,
      const SpinWeighted<ComplexDataVector, 0>& d,
      const SpinWeighted<ComplexDataVector, 0>& omega_cd,
      const SpinWeighted<ComplexDataVector, 1>& eth_omega_cd,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    // TODO fix transforms for representability concerns

    SpinWeighted<ComplexDataVector, 1> evolution_coords_dr_u =
        Spectral::Swsh::swsh_interpolate(dr_u, get<0>(x_of_x_tilde),
                                         get<1>(x_of_x_tilde), l_max);

    SpinWeighted<ComplexDataVector, 0> k;
    k.data() = sqrt(1.0 + j_tilde.data() * conj(j_tilde.data()));

    SpinWeighted<ComplexDataVector, 0> exp_minus_2_beta;
    exp_minus_2_beta.data() = exp(-2.0 * beta_tilde.data());
    // NOTE extra sign change in later terms due to omega_cd = 1/omega
    // TEST
    evolution_coords_dr_u =
        0.5 / pow<3>(omega_cd) *
            (conj(d) * evolution_coords_dr_u -
             c * conj(evolution_coords_dr_u)) -
        (eth_omega_cd * k - conj(eth_omega_cd) * j_tilde) *
            (1.0 / (r_tilde * exp_minus_2_beta)) *
            (-1.0 / r_tilde + 0.25 * r_tilde *
                                  (dr_j_tilde * conj(dr_j_tilde) -
                                   0.25 *
                                       square(j_tilde * conj(dr_j_tilde) +
                                              dr_j_tilde * conj(j_tilde)) /
                                       (1.0 + j_tilde * conj(j_tilde)))) /
            omega_cd -
        1.0 / (r_tilde * exp_minus_2_beta) *
            (-conj(eth_omega_cd) * dr_j_tilde +
             0.5 * eth_omega_cd / k *
                 (j_tilde * conj(dr_j_tilde) + conj(j_tilde) * dr_j_tilde)) /
            omega_cd;

    // TEST
    // evolution_coords_dr_u =
    // 0.5 * omega *
    // (b * evolution_coords_dr_u + a * conj(evolution_coords_dr_u));

    // evolution_coords_dr_u =
    // 0.5 / pow<3>(omega_cd) *
    // (conj(d) * evolution_coords_dr_u - c * conj(evolution_coords_dr_u));

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
                 Tags::EvolutionGaugeBoundaryValue<Tags::Beta>, Tags::GaugeC,
                 Tags::GaugeD, Tags::GaugeOmegaCD,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::CauchyAngularCoords, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          evolution_gauge_u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    // TODO
    SpinWeighted<ComplexDataVector, 1> evolution_coords_u =
        Spectral::Swsh::swsh_interpolate(make_not_null(&get(*u)),
                                         get<0>(x_of_x_tilde),
                                         get<1>(x_of_x_tilde), l_max);

    SpinWeighted<ComplexDataVector, 0> k_tilde;
    k_tilde.data() =
        sqrt(1.0 + get(j_tilde).data() * conj(get(j_tilde).data()));

    SpinWeighted<ComplexDataVector, 0> exp_2_beta_tilde;
    exp_2_beta_tilde.data() = exp(2.0 * get(beta_tilde).data());

    // TEST
    // get(*evolution_gauge_u) =
    // 0.5 * (get(a) * conj(evolution_coords_u) + get(b) * evolution_coords_u);

    // Note extra sign change on final term due to omega_cd = 1/omega
    get(*evolution_gauge_u) = 0.5 / square(get(omega_cd)) *
                                  (-get(c) * conj(evolution_coords_u) +
                                   conj(get(d)) * evolution_coords_u) +
                              exp_2_beta_tilde /
                                  (get(r_tilde) * get(omega_cd)) *
                                  (conj(get(eth_omega_cd)) * get(j_tilde) -
                                   get(eth_omega_cd) * k_tilde);
    // Spectral::Swsh::filter_swsh_boundary_quantity(
    // make_not_null(&get(*evolution_gauge_u)), l_max, l_max - 2);
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::W> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::W>,
                 Tags::BoundaryValue<Tags::W>, Tags::BoundaryValue<Tags::U>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::U>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::Beta>>;
  using argument_tags =
      tmpl::list<Tags::U0, Tags::EvolutionGaugeBoundaryValue<Tags::R>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::J>, Tags::GaugeOmegaCD,
                 Tags::Du<Tags::GaugeOmegaCD>,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::GaugeD, Tags::GaugeC, Tags::CauchyAngularCoords,
                 Tags::LMax>;

  // TODO arguments apply impl etc
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_w,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> w,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          boundary_u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          boundary_u_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          beta_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& u_0,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& c,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    apply_impl(make_not_null(&get(*evolution_gauge_w)), make_not_null(&get(*w)),
               make_not_null(&get(*boundary_u)),
               make_not_null(&get(*boundary_u_tilde)),

               make_not_null(&get(*beta_tilde)), get(u_0), get(r_tilde),
               get(j_tilde), get(omega_cd), get(du_omega_cd), get(eth_omega_cd),
               get(d), get(c), x_of_x_tilde, l_max);
  }

  // TODO change variable names and implementation
 private:
  static void apply_impl(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>
          evolution_gauge_w,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> w,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> boundary_u,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> boundary_u_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> beta_tilde,
      const SpinWeighted<ComplexDataVector, 1>& u_0,
      const SpinWeighted<ComplexDataVector, 0>& r_tilde,
      const SpinWeighted<ComplexDataVector, 2>& j_tilde,
      const SpinWeighted<ComplexDataVector, 0>& omega_cd,
      const SpinWeighted<ComplexDataVector, 0>& du_omega_cd,
      const SpinWeighted<ComplexDataVector, 1>& eth_omega_cd,
      const SpinWeighted<ComplexDataVector, 0>& d,
      const SpinWeighted<ComplexDataVector, 2>& c,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    // Spectral::Swsh::filter_swsh_boundary_quantity(w, l_max, l_max - 2);
    Spectral::Swsh::swsh_interpolate(evolution_gauge_w, w, get<0>(x_of_x_tilde),
                                     get<1>(x_of_x_tilde), l_max);

    SpinWeighted<ComplexDataVector, 1> boundary_u_of_x_tilde{};

    // Spectral::Swsh::filter_swsh_boundary_quantity(boundary_u, l_max, l_max -
    // 2);

    Spectral::Swsh::swsh_interpolate(make_not_null(&boundary_u_of_x_tilde),
                                     boundary_u, get<0>(x_of_x_tilde),
                                     get<1>(x_of_x_tilde), l_max);

    SpinWeighted<ComplexDataVector, 0> k_tilde;
    k_tilde.data() = sqrt(1.0 + j_tilde.data() * conj(j_tilde.data()));
    SpinWeighted<ComplexDataVector, 0> exp_2_beta_of_x_tilde;
    exp_2_beta_of_x_tilde.data() = exp(2.0 * beta_tilde->data());

    // TEST
    *evolution_gauge_w =
        *evolution_gauge_w + 1.0 / r_tilde * (omega_cd - 1.0) -
        2.0 * du_omega_cd / omega_cd -
        (conj(eth_omega_cd) * (*boundary_u_tilde - u_0) +
         eth_omega_cd * (conj(*boundary_u_tilde) - conj(u_0))) /
            omega_cd +
        exp_2_beta_of_x_tilde / (2.0 * square(omega_cd) * r_tilde) *
            (square(conj(eth_omega_cd)) * j_tilde +
             square(eth_omega_cd) * conj(j_tilde) -
             2.0 * eth_omega_cd * conj(eth_omega_cd) * k_tilde);
    // TEST

    // *evolution_gauge_w =
    // *evolution_gauge_w + 1.0 / r_tilde * (omega_cd - 1.0) -
    // 2.0 * du_omega_cd / omega_cd -
    // 0.5 * (conj(eth_omega_cd) * ((conj(d) * boundary_u_of_x_tilde -
    // c * conj(boundary_u_of_x_tilde)) /
    // square(omega_cd) -
    // u_0) +
    // eth_omega_cd * ((-conj(c) * boundary_u_of_x_tilde +
    // d * conj(boundary_u_of_x_tilde)) /
    // square(omega_cd) -
    // conj(u_0))) -
    // exp_2_beta_of_x_tilde / (2.0 * square(omega_cd) * r_tilde) *
    // (square(conj(eth_omega_cd)) * j_tilde +
    // square(eth_omega_cd) * conj(j_tilde) -
    // 2.0 * eth_omega_cd * conj(eth_omega_cd) * k_tilde);

    // TEST
    // *evolution_gauge_w = *evolution_gauge_w + 2.0 * du_omega_cd;
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
                 Tags::BoundaryValue<Tags::Dr<Tags::J>>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::Dr<Tags::J>>, Tags::U0,
                 Tags::EvolutionGaugeBoundaryValue<Tags::R>,
                 Tags::BoundaryValue<Tags::R>, Tags::GaugeC>;
  using argument_tags =
      tmpl::list<Tags::GaugeD, Tags::GaugeA, Tags::GaugeB, Tags::GaugeOmegaCD,
                 Tags::GaugeOmega, Tags::Du<Tags::GaugeOmegaCD>,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::Du<Tags::GaugeC>, Tags::Du<Tags::GaugeD>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::DuRDividedByR>,
                 Tags::CauchyAngularCoords, Tags::InertialAngularCoords,
                 Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> h_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> spec_h,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dr_j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          dr_j_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u_0,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> r_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> r_of_x,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& du_c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r_divided_by_r_tilde,
      const tnsr::i<DataVector, 2>& x_of_x_tilde,
      const tnsr::i<DataVector, 2>& x_tilde_of_x, const size_t l_max) noexcept {
    apply_impl(make_not_null(&get(*h_tilde)), make_not_null(&get(*j_tilde)),
               make_not_null(&get(*spec_h)), make_not_null(&get(*j)),
               make_not_null(&get(*dr_j)), make_not_null(&get(*dr_j_tilde)),
               make_not_null(&get(*u_0)), make_not_null(&get(*r_tilde)),
               make_not_null(&get(*r_of_x)), make_not_null(&get(*c)), get(d),
               get(a), get(b), get(omega_cd), get(omega), get(du_omega_cd),
               get(eth_omega_cd), get(du_c), get(du_d),
               get(du_r_divided_by_r_tilde), x_of_x_tilde, x_tilde_of_x, l_max);
  }

  // TODO change variable names and implementation
  static void apply_impl(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> h_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> j_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> spec_h,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> j,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> dr_j,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> dr_j_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> u_0,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> r_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> r_of_x,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> c,
      const SpinWeighted<ComplexDataVector, 0>& d,
      const SpinWeighted<ComplexDataVector, 2>& a,
      const SpinWeighted<ComplexDataVector, 0>& b,
      const SpinWeighted<ComplexDataVector, 0>& omega_cd,
      const SpinWeighted<ComplexDataVector, 0>& omega,
      const SpinWeighted<ComplexDataVector, 0>& du_tilde_omega_cd,
      const SpinWeighted<ComplexDataVector, 1>& eth_omega_cd,
      const SpinWeighted<ComplexDataVector, 2>& du_c,
      const SpinWeighted<ComplexDataVector, 0>& du_d,
      const SpinWeighted<ComplexDataVector, 0>& du_r_divided_by_r_tilde,
      const tnsr::i<DataVector, 2>& x_of_x_tilde,
      const tnsr::i<DataVector, 2>& x_tilde_of_x, const size_t l_max) noexcept {
    // optimization note: this has many allocations. They can be made fewer.
    // optimization note: this has several spin-weighted derivatives, they can
    // be aggregated
    // Spectral::Swsh::filter_swsh_boundary_quantity(u_0, l_max, l_max - 2);
    SpinWeighted<ComplexDataVector, 1> u_0_of_x =
        Spectral::Swsh::swsh_interpolate(u_0, get<0>(x_tilde_of_x),
                                         get<1>(x_tilde_of_x), l_max);
    // Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&u_0_of_x),
    // l_max, l_max - 2);

    // Spectral::Swsh::filter_swsh_boundary_quantity(j, l_max, l_max - 2);
    // Spectral::Swsh::filter_swsh_boundary_quantity(dr_j, l_max, l_max - 2);
    SpinWeighted<ComplexDataVector, 2> j_of_x_tilde =
        Spectral::Swsh::swsh_interpolate(j, get<0>(x_of_x_tilde),
                                         get<1>(x_of_x_tilde), l_max);
    SpinWeighted<ComplexDataVector, 2> dr_j_of_x_tilde =
        Spectral::Swsh::swsh_interpolate(dr_j, get<0>(x_of_x_tilde),
                                         get<1>(x_of_x_tilde), l_max);

    u_0_of_x = 0.5 / square(omega) * (conj(b) * u_0_of_x - a * conj(u_0_of_x));
    auto eth_r = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        r_of_x, l_max);
    SpinWeighted<ComplexDataVector, 1> u_0_bar_j_cauchy = conj(u_0_of_x) * (*j);

    // TODO combine some expressions to minimize allocations where possible
    SpinWeighted<ComplexDataVector, 2> u_0_bar_eth_j_cauchy =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&u_0_bar_j_cauchy), l_max) -
        *j * conj(Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                 make_not_null(&u_0_of_x), l_max)) -
        conj(u_0_of_x) * eth_r * (*dr_j);

    SpinWeighted<ComplexDataVector, 2> du_tilde_j_cauchy_first_term =
        *spec_h +
        0.5 *
            (u_0_of_x *
                 Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                     j, l_max) -
             u_0_of_x * conj(eth_r) * (*dr_j) + u_0_bar_eth_j_cauchy);
    // Spectral::Swsh::filter_swsh_boundary_quantity(
    // make_not_null(&du_tilde_j_cauchy_first_term), l_max, l_max - 2);
    SpinWeighted<ComplexDataVector, 2> du_tilde_j_first_term =
        Spectral::Swsh::swsh_interpolate(
            make_not_null(&du_tilde_j_cauchy_first_term), get<0>(x_of_x_tilde),
            get<1>(x_of_x_tilde), l_max);

    SpinWeighted<ComplexDataVector, 2> du_tilde_j =
        du_tilde_j_first_term -
        (*r_tilde) * du_tilde_omega_cd / square(omega_cd) * dr_j_of_x_tilde;

    SpinWeighted<ComplexDataVector, 0> k_of_x_tilde;
    k_of_x_tilde.data() =
        sqrt(1.0 + j_of_x_tilde.data() * conj(j_of_x_tilde.data()));

    SpinWeighted<ComplexDataVector, 0> k_tilde;
    k_tilde.data() = sqrt(1.0 + j_tilde->data() * conj(j_tilde->data()));

    auto eth_r_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(r_tilde,
                                                                   l_max);

    SpinWeighted<ComplexDataVector, 1> u_0_bar_j_tilde =
        conj(*u_0) * (*j_tilde);

    // TODO combine some expressions to minimize allocations where possible
    SpinWeighted<ComplexDataVector, 2> u_0_bar_eth_j_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&u_0_bar_j_tilde), l_max) -
        *j_tilde *
            conj(Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                u_0, l_max)) -
        conj(*u_0) * eth_r_tilde * (*dr_j_tilde);

    SpinWeighted<ComplexDataVector, 2> angular_derivative_term =
        0.5 *
        (*u_0 * Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                    j_tilde, l_max) -
         *u_0 * conj(eth_r_tilde) * (*dr_j_tilde) + u_0_bar_eth_j_tilde);

    SpinWeighted<ComplexDataVector, 0> cauchy_du_omega_cd =
        du_tilde_omega_cd -
        0.5 * (*u_0 * conj(eth_omega_cd) + conj(*u_0) * eth_omega_cd);

    auto eth_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(u_0, l_max);
    auto ethbar_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(u_0,
                                                                      l_max);

    // Spectral::Swsh::filter_swsh_boundary_quantity(spec_h, l_max, l_max - 2);
    SpinWeighted<ComplexDataVector, 2> h_of_x_tilde =
        Spectral::Swsh::swsh_interpolate(spec_h, get<0>(x_of_x_tilde),
                                         get<1>(x_of_x_tilde), l_max);

    *h_tilde =
        angular_derivative_term -
        *r_tilde * cauchy_du_omega_cd / omega_cd * (*dr_j_tilde) +
        2.0 * cauchy_du_omega_cd / omega_cd * (*j_tilde) -
        ethbar_u_0 * (*j_tilde) + eth_u_0 * k_tilde +
        0.25 / square(omega_cd) *
            (square(conj(d)) * h_of_x_tilde + square(*c) * conj(h_of_x_tilde) +
             *c * conj(d) *
                 (h_of_x_tilde * conj(j_of_x_tilde) +
                  j_of_x_tilde * conj(h_of_x_tilde)) /
                 k_of_x_tilde) +
        du_r_divided_by_r_tilde * (*r_tilde) * (*dr_j_tilde);
    // TEST
    // *h_tilde =
    // 0.25 / square(omega_cd) *
    // (square(conj(d)) * h_of_x_tilde + square(*c) * conj(h_of_x_tilde) +
    // *c * conj(d) *
    // (h_of_x_tilde * conj(j_of_x_tilde) +
    // j_of_x_tilde * conj(h_of_x_tilde)) /
    // k_of_x_tilde) +
    // du_r_divided_by_r_tilde * (*r_tilde) * (*dr_j_tilde);

    // Spectral::Swsh::filter_swsh_boundary_quantity(h_tilde, l_max, l_max - 2);
  }
};

struct GaugeUpdateU {
  using argument_tags =
      tmpl::list<Tags::GaugeA, Tags::GaugeB, Tags::CauchyAngularCoords,
                 Tags::InertialAngularCoords, Tags::GaugeOmegaCD,
                 Tags::GaugeOmega,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::Exp2Beta, Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::DuCauchyCartesianCoords, Tags::Du<Tags::GaugeC>,
                 Tags::Du<Tags::GaugeD>, Tags::U0, Tags::U,
                 Tags::Du<Tags::GaugeOmegaCD>, Tags::GaugeC, Tags::GaugeD>;

  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*> du_x,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> du_c,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> du_d,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u_0,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          du_omega_cd,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> c,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> d,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const tnsr::i<DataVector, 2> x_of_x_tilde,
      const tnsr::i<DataVector, 2> x_tilde_of_x,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
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
    // TEST
    // get(*u_0).data() = 0.0;
    // get(*u_0).data() = 0.0 * u_scri_slice;

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
    // Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(*u_0)),
    // l_max, l_max - 4);

    // NOTE currently just storing u_0_hat -- most of these factors just have to
    // be reversed for the rest of the boundary conditions to use them anyway.

    SpinWeighted<ComplexDataVector, 1> u_0_cauchy =
        0.5 * (get(*d) * get(*u_0) + get(*c) * conj(get(*u_0)));

    get<0>(*du_x) =
        real(((-cos(get<0>(x_of_x_tilde)) * cos(get<1>(x_of_x_tilde))) -
              std::complex<double>(0.0, 1.0) * sin(get<1>(x_of_x_tilde))) *
             u_0_cauchy.data());
    get<1>(*du_x) =
        real(((-cos(get<0>(x_of_x_tilde)) * sin(get<1>(x_of_x_tilde))) +
              std::complex<double>(0.0, 1.0) * cos(get<1>(x_of_x_tilde))) *
             u_0_cauchy.data());
    get<2>(*du_x) = real(sin(get<0>(x_of_x_tilde)) * u_0_cauchy.data());

    // Unfortunately, this is the best way I can think of to guarantee that
    // these derivatives are correctly evaluated, and the derivatives are needed
    // for the evolution equations below.
    auto cauchy_coords_u_0 = Spectral::Swsh::swsh_interpolate(
        make_not_null(&get(*u_0)), get<0>(x_tilde_of_x), get<1>(x_tilde_of_x),
        l_max);
    cauchy_coords_u_0 =
        0.5 / square(get(omega)) *
        (conj(get(b)) * cauchy_coords_u_0 - get(a) * conj(cauchy_coords_u_0));

    auto ethbar_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&cauchy_coords_u_0), l_max);
    auto eth_u_0 = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        make_not_null(&cauchy_coords_u_0), l_max);

    // Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&ethbar_u_0),
    // l_max, l_max - 4);
    // Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&eth_u_0),
    // l_max, l_max - 4);

    auto inertial_coords_ethbar_u_0 = Spectral::Swsh::swsh_interpolate(
        make_not_null(&ethbar_u_0), get<0>(x_of_x_tilde), get<1>(x_of_x_tilde),
        l_max);
    auto inertial_coords_eth_u_0 = Spectral::Swsh::swsh_interpolate(
        make_not_null(&eth_u_0), get<0>(x_of_x_tilde), get<1>(x_of_x_tilde),
        l_max);

    auto inertial_ethbar_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&get(*u_0)), l_max);
    auto inertial_eth_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&get(*u_0)), l_max);

    // auto inertial_coords_ethbar_u_0 =
    // Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
    // make_not_null(&get(*u_0)), l_max);
    // auto inertial_coords_eth_u_0 =
    // Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
    // make_not_null(&get(*u_0)), l_max);

    // the du_omega should be representable, the du_c and du_d are likely not,
    // similar to the jacobian factors c and d themselves.
    // get(*du_omega_cd) =
    // 0.125 *
    // (conj(get(*d)) * conj(inertial_ethbar_u_0) -
    // get(*c) * conj(inertial_eth_u_0) + get(*d) * inertial_ethbar_u_0 -
    // conj(get(*c)) * inertial_eth_u_0) /
    // get(omega_cd);

    // get(*du_omega_cd) =
    // 0.25 * (inertial_coords_ethbar_u_0 + conj(inertial_coords_ethbar_u_0)) *
    // get(omega_cd);

    get(*du_omega_cd) = 0.25 *
                            (inertial_ethbar_u_0 + conj(inertial_ethbar_u_0)) *
                            get(omega_cd) +
                        0.5 * (get(*u_0) * conj(get(eth_omega_cd)) +
                               conj(get(*u_0)) * get(eth_omega_cd));

    // TEST
    // get(*du_omega_cd).data() = 0.0;

    get(*du_c) = 0.5 * (conj(get(*d)) * inertial_coords_eth_u_0 -
                        get(*c) * conj(inertial_coords_ethbar_u_0)) +
                 2.0 * get(*c) / get(omega_cd) * get(*du_omega_cd);

    get(*du_d) = -0.5 * (get(*d) * conj(inertial_coords_ethbar_u_0) -
                         conj(get(*c)) * inertial_coords_eth_u_0) +
                 2.0 * get(*d) / get(omega_cd) * get(*du_omega_cd);
  }
};

template <typename AngularTag, typename CartesianTag>
struct GaugeUpdateAngularFromCartesian {
  using argument_tags = tmpl::list<Tags::LMax>;
  using return_tags = tmpl::list<AngularTag, CartesianTag>;
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 2>*> angular_coords,
      const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_coords,
      const size_t l_max) noexcept {
    // normalize the cartesian coordinates
    DataVector cartesian_r = sqrt(square(get<0>(*cartesian_coords)) +
                                  square(get<1>(*cartesian_coords)) +
                                  square(get<2>(*cartesian_coords)));

    // FIXME this brutal renormalization will probably do bad things for the
    // time stepper.
    // TRY: no renormalization, because we use atan2 for both angular
    // coordinates, which shouldn't care whether we are actually on the sphere.
    // In principle, the drift should be minor anyways.

    // printf("cartesian r check\n");
    // for(auto val : cartesian_r) {
    // printf("%e\n", 1.0 - val);
    // }
    // printf("done\n");

    get<0>(*cartesian_coords) /= cartesian_r;
    get<1>(*cartesian_coords) /= cartesian_r;
    get<2>(*cartesian_coords) /= cartesian_r;

    auto x = get<0>(*cartesian_coords);
    auto y = get<1>(*cartesian_coords);
    auto z = get<2>(*cartesian_coords);

    get<0>(*angular_coords) = atan2(sqrt(square(x) + square(y)), z);
    get<1>(*angular_coords) = atan2(y, x);
  }
};

template <typename GaugeFactorSpin2, typename GaugeFactorSpin0,
          typename CartesianCoordinateTag, typename TargetCoordinateTag,
          typename DuCartesianCoordinateTag>
struct GaugeUpdateJacobianFromCoords {
  using argument_tags = tmpl::list<Tags::LMax>;
  using return_tags =
      tmpl::list<GaugeFactorSpin2, GaugeFactorSpin0, Tags::Du<GaugeFactorSpin2>,
                 Tags::Du<GaugeFactorSpin0>, TargetCoordinateTag,
                 DuCartesianCoordinateTag, CartesianCoordinateTag>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> a,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> b,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> du_a,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> du_b,
      const gsl::not_null<tnsr::i<DataVector, 2>*> x_tilde_of_x,
      const gsl::not_null<tnsr::i<DataVector, 3>*> du_x_tilde_of_x_cartesian,
      const gsl::not_null<tnsr::i<DataVector, 3>*> x_tilde_of_x_cartesian,
      const size_t l_max) noexcept {
    // first, interpolate to a new grid. For this, we assume that the theta
    // (more slowly varying) is already on a gauss-legendre grid.
    // Therefore, for each constant-in-theta view, we construct a similar-order
    // Gauss-Legendre grid using Barycentric rational interpolation.
    const size_t number_of_theta_points =
        Spectral::Swsh::number_of_swsh_theta_collocation_points(l_max);
    const size_t number_of_phi_points =
        Spectral::Swsh::number_of_swsh_phi_collocation_points(l_max);
    tnsr::I<ComplexDataVector, 3> x_tilde_cartesian{get(*a).size()};

    get<0>(x_tilde_cartesian) =
        std::complex<double>(1.0, 0.0) * get<0>(*x_tilde_of_x_cartesian);
    get<1>(x_tilde_cartesian) =
        std::complex<double>(1.0, 0.0) * get<1>(*x_tilde_of_x_cartesian);
    get<2>(x_tilde_cartesian) =
        std::complex<double>(1.0, 0.0) * get<2>(*x_tilde_of_x_cartesian);

    tnsr::iJ<DataVector, 3> dx_x_tilde_cartesian{get(*a).size()};
    SpinWeighted<ComplexDataVector, 0> buffer{get(*a).size()};
    SpinWeighted<ComplexDataVector, 1> derivative_buffer{get(*a).size()};
    for (size_t i = 0; i < 3; ++i) {
      buffer = x_tilde_cartesian.get(i);
      Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
          make_not_null(&derivative_buffer), make_not_null(&buffer), l_max);
      dx_x_tilde_cartesian.get(0, i) = -real(derivative_buffer.data());
      dx_x_tilde_cartesian.get(1, i) = -imag(derivative_buffer.data());
    }
    tnsr::iJ<DataVector, 2> dx_x_tilde_of_x{get(*a).size()};

    for (size_t i = 0; i < 2; ++i) {
      dx_x_tilde_of_x.get(i, 0) =
          cos(get<1>(*x_tilde_of_x)) * cos(get<0>(*x_tilde_of_x)) *
              dx_x_tilde_cartesian.get(i, 0) +
          cos(get<0>(*x_tilde_of_x)) * sin(get<1>(*x_tilde_of_x)) *
              dx_x_tilde_cartesian.get(i, 1) -
          sin(get<0>(*x_tilde_of_x)) * dx_x_tilde_cartesian.get(i, 2);
      dx_x_tilde_of_x.get(i, 1) =
          -sin(get<1>(*x_tilde_of_x)) * dx_x_tilde_cartesian.get(i, 0) +
          cos(get<1>(*x_tilde_of_x)) * dx_x_tilde_cartesian.get(i, 1);
    }

    SpinWeighted<ComplexDataVector, 2> a_of_x;
    a_of_x.data() =
        std::complex<double>(1.0, 0.0) *
            (get<0, 0>(dx_x_tilde_of_x) - get<1, 1>(dx_x_tilde_of_x)) +
        std::complex<double>(0.0, 1.0) *
            (get<1, 0>(dx_x_tilde_of_x) + get<0, 1>(dx_x_tilde_of_x));
    SpinWeighted<ComplexDataVector, 0> b_of_x;
    b_of_x.data() =
        std::complex<double>(1.0, 0.0) *
            (get<0, 0>(dx_x_tilde_of_x) + get<1, 1>(dx_x_tilde_of_x)) +
        std::complex<double>(0.0, 1.0) *
            (-get<1, 0>(dx_x_tilde_of_x) + get<0, 1>(dx_x_tilde_of_x));

    get(*a) = a_of_x;
    get(*b) = b_of_x;
  }
};

struct GaugeUpdateDuXtildeOfX {
  using argument_tags =
      tmpl::list<Tags::GaugeOmega, Tags::InertialAngularCoords, Tags::LMax>;
  using return_tags = tmpl::list<Tags::DuInertialCartesianCoords, Tags::GaugeA,
                                 Tags::GaugeB, Tags::U, Tags::U0>;
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*> du_x_tilde_of_x,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> a,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> b,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u_0_hat,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const tnsr::i<DataVector, 2>& x_tilde_of_x, const size_t l_max) noexcept {
    // interpolate a and b back to the non-inertial coordinates.
    // Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null
    //(&get(*u_0_hat)),
    // l_max, l_max - 2);

    SpinWeighted<ComplexDataVector, 1> u_0_hat_of_x =
        Spectral::Swsh::swsh_interpolate(make_not_null(&get(*u_0_hat)),
                                         get<0>(x_tilde_of_x),
                                         get<1>(x_tilde_of_x), l_max);

    get<0>(*du_x_tilde_of_x) =
        real(((-cos(get<0>(x_tilde_of_x)) * cos(get<1>(x_tilde_of_x))) -
              std::complex<double>(0.0, 1.0) * sin(get<1>(x_tilde_of_x))) *
             (-u_0_hat_of_x.data()));

    get<1>(*du_x_tilde_of_x) =
        real(((-cos(get<0>(x_tilde_of_x)) * sin(get<1>(x_tilde_of_x))) +
              std::complex<double>(0.0, 1.0) * cos(get<1>(x_tilde_of_x))) *
             (-u_0_hat_of_x.data()));

    get<2>(*du_x_tilde_of_x) =
        real(sin(get<0>(x_tilde_of_x)) * (-u_0_hat_of_x.data()));
  }
};

struct InitializeXtildeOfX {
  using return_tags =
      tmpl::list<Tags::InertialAngularCoords, Tags::InertialCartesianCoords>;
  using argument_tags = tmpl::list<Tags::LMax>;

  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 2>*> x_tilde_of_x,
      const gsl::not_null<tnsr::i<DataVector, 3>*> x_tilde_of_x_cartesian,
      const size_t l_max) noexcept {
    const auto& collocation = Spectral::Swsh::precomputed_collocation<
        Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
    for (const auto& collocation_point : collocation) {
      get<0>(*x_tilde_of_x)[collocation_point.offset] = collocation_point.theta;
      get<1>(*x_tilde_of_x)[collocation_point.offset] = collocation_point.phi;

      // TEST nonunity conformal factor
      // get<1>(*x_tilde_of_x)[collocation_point.offset] =
      // collocation_point.phi +
      // 1.0e-3 * cos(collocation_point.phi) * sin(collocation_point.theta);
      // TEST omega = 1
      // get<1>(*x_tilde_of_x)[collocation_point.offset] =
      // collocation_point.phi + 1.0e-3 * sin(collocation_point.theta);
    }
    get<0>(*x_tilde_of_x_cartesian) =
        sin(get<0>(*x_tilde_of_x)) * cos(get<1>(*x_tilde_of_x));
    get<1>(*x_tilde_of_x_cartesian) =
        sin(get<0>(*x_tilde_of_x)) * sin(get<1>(*x_tilde_of_x));
    get<2>(*x_tilde_of_x_cartesian) = cos(get<0>(*x_tilde_of_x));
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
    // Spectral::Swsh::filter_swsh_boundary_quantity
    //(make_not_null(&get(*omega)),
    // l_max, l_max - 4);

    Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        make_not_null(&get(*eth_omega)), make_not_null(&get(*omega)), l_max);
  }
};

// maybe merge this with previous
struct GaugeUpdateOmegaCD {
  using argument_tags = tmpl::list<Tags::GaugeC, Tags::GaugeD, Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::GaugeOmegaCD,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                  Spectral::Swsh::Tags::Eth>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> omega_cd,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          eth_omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
      const size_t l_max) noexcept {
    get(*omega_cd) = 0.5 * sqrt(get(d).data() * conj(get(d).data()) -
                                get(c).data() * conj(get(c).data()));
    // Spectral::Swsh::filter_swsh_boundary_quantity(
    // make_not_null(&get(*omega_cd)), l_max, l_max - 4);

    Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        make_not_null(&get(*eth_omega_cd)), make_not_null(&get(*omega_cd)),
        l_max);
  }
};

struct InitializeGauge {
  using argument_tags = tmpl::list<Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords,
                 Tags::GaugeC, Tags::GaugeD>;

  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 2>*> x_of_x_tilde,
      const gsl::not_null<tnsr::i<DataVector, 3>*> x_of_x_tilde_cartesian,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> c,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> d,
      const size_t l_max) noexcept {
    const auto& collocation = Spectral::Swsh::precomputed_collocation<
        Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
    for (const auto& collocation_point : collocation) {
      get<0>(*x_of_x_tilde)[collocation_point.offset] = collocation_point.theta;
      get<1>(*x_of_x_tilde)[collocation_point.offset] = collocation_point.phi;
      // TEST nonunity conformal factor

      // auto rootfind = boost::math::tools::bisect(
      // [&collocation_point](double x) {
      // return collocation_point.phi -
      // (x + 1.0e-3 * cos(x) * sin(collocation_point.theta));
      // },
      // collocation_point.phi - 2.0e-3, collocation_point.phi + 2.0e-3,
      // [](double x, double y) { return abs(x - y) < 1.0e-14; });

      // printf("rootfind test %e, %e, %e\n", collocation_point.phi,
      // rootfind.first, rootfind.second);

      // get<1>(*x_of_x_tilde)[collocation_point.offset] =
      // 0.5 * (rootfind.first + rootfind.second);
      // get<1>(*x_of_x_tilde)[collocation_point.offset] =
      // collocation_point.phi - 1.0e-3 * sin(collocation_point.theta);
    }
    get<0>(*x_of_x_tilde_cartesian) =
        sin(get<0>(*x_of_x_tilde)) * cos(get<1>(*x_of_x_tilde));
    get<1>(*x_of_x_tilde_cartesian) =
        sin(get<0>(*x_of_x_tilde)) * sin(get<1>(*x_of_x_tilde));
    get<2>(*x_of_x_tilde_cartesian) = cos(get<0>(*x_of_x_tilde));

    get(*c).data() = 0.0;
    get(*d).data() = 2.0;
  }
};
}  // namespace Cce
