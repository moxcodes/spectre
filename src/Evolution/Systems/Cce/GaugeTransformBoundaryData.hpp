// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"

namespace Cce {
// TOOD most of these computations require intermediate quantities that could be
// cached or grouped together like in the main evolution computations. These
// calculations are smaller, so there is significantly less savings, but this
// might be a spot to keep an eye on for optimization.

using compute_gauge_adjustments_setup_tags =
    tmpl::list<Tags::R, Tags::DuRDividedByR, Tags::J, Tags::Dr<Tags::J>>;

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
    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*evolution_gauge_r)), make_not_null(&get(*r)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);
    get(*evolution_gauge_r) /= omega;
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::DuRDividedByR> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::DuRDividedByR>,
                 Tags::BoundaryValue<Tags::DuRDividedByR>>;
  using argument_tags =
      tmpl::list<Tags::CauchyAngularCoords, Tags::U0,
                 Tags::EvolutionGaugeBoundaryValue<Tags::R>, Tags::GaugeOmega,
                 Tags::Du<Tags::GaugeOmega>, Tags::LMax>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_du_r_divided_by_r,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          du_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& u_0,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& evolution_gauge_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*evolution_gauge_du_r_divided_by_r)),
        make_not_null(&get(*du_r_divided_by_r)), get<0>(x_of_x_tilde),
        get<1>(x_of_x_tilde), l_max);
    // taking this as argument saves an interpolation, which is significantly
    // more expensive than the extra multiplication.
    SpinWeighted<ComplexDataVector, 0> r_buffer =
        get(evolution_gauge_r) * omega;
    auto eth_r = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        r_buffer, l_max);
    get(*evolution_gauge_du_r_divided_by_r) +=
        0.5 * (get(u_0) * conj(eth_r) + conj(get(u_0)) eth_r) / r_buffer -
        du_omega / omega;
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
      const Scalar<SpinWeighted<ComplexDataVvector, 0>>& omega,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*evolution_gauge_j)), make_not_null(&get(*j)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);

    get(*evolution_gauge_j).data() =
        ((0.25 * square(get(b).data())) * get(*evolution_gauge_j).data() +
         0.25 * square(get(a).data()) * conj(get(*evolution_gauge_j).data()) -
         0.5 * get(a).data() * get(b).data() *
             sqrt(1.0 + get(*evolution_gauge_j).data() *
                            conj(get(*evolution_gauge_j).data()))) /
        square(omega);
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
    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*evolution_gauge_dr_j)), make_not_null(&get(*dr_j)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);
    // two powers of omega from converting the angular part of the metric, one
    // from the derivative
    get(*evolution_gauge_dr_j).data() =
        ((0.25 * square(get(b).data())) * get(*evolution_gauge_dr_j).data() +
         0.25 * square(get(a).data()) *
             conj(get(*evolution_gauge_dr_j).data()) -
         0.25 * get(a).data() * get(b).data() *
             (get(*evolution_gauge_dr_j).data() *
                  conj(get(evolution_gauge_j).data()) +
              conj(get(*evolution_gauge_dr_j).data()) *
                  get(evolution_gauge_j).data()) /
             (1.0 + get(evolution_gauge_j).data() *
                        conj(get(evolution_gauge_j).data()))) /
        pow<3>(omega);
  }
};

// beta is the same in both gauges. This still should have separate tags for the
// time being for compatibility with original evolution code.
template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::Beta> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::Beta>,
                                 Tags::BoundaryValue<Tags::Beta>>;
  using argument_tags =
      tmpl::list<Tags::CauchyAngularCoords, Tags::GaugeOmega, Tags::LMax>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_beta,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*evolution_gauge_beta)), make_not_null(&get(*beta)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);
    get(*evolution_gauge_beta) -= 0.5 * log(omega);
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::Q> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::Q>,
                                 Tags::BoundaryValue<Tags::Dr<Tags::U>>>;
  using argument_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::J>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::Dr<Tags::J>>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::R>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::Beta>, Tags::GaugeA,
                 Tags::GaugeB, Tags::GaugeOmega,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::CauchyAngularCoords, Tags::LMax>;

  template <typename... Args>
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          evolution_gauge_bondi_q,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> dr_u,
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
               make_not_null(&get(*dr_u)), get(j_tilde), get(dr_j_tilde),
               get(r_tilde), get(beta_tilde), get(a), get(b), get(omega),
               get(eth_omega), x_of_x_tilde, l_max);
  }
  static void apply_impl(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*>
          evolution_gauge_bondi_q,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> dr_u,
      const SpinWeighted<ComplexDataVector, 2>& j_tilde,
      const SpinWeighted<ComplexDataVector, 2>& dr_j_tilde,
      const SpinWeighted<ComplexDataVector, 0>& r_tilde,
      const SpinWeighted<ComplexDataVector, 0>& beta_tilde,
      const SpinWeighted<ComplexDataVector, 2>& a,
      const SpinWeighted<ComplexDataVector, 0>& b,
      const SpinWeighted<ComplexDataVector, 0>& omega,
      const SpinWeighted<ComplexDataVector, 1>& eth_omega,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    SpinWeighted<ComplexDataVector, 1> evolution_coords_dr_u =
        Spectral::Swsh::swsh_interpolate_from_pfaffian(
            dr_u, get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);

    auto k = SpinWeighted<ComplexDataVector, 0>{
        sqrt(1.0 + j.data() * conj(j.data()))};
    evolution_coords_dr_u =
        0.5 / square(omega) *
            (b * evolution_coords_dr_u + a * conj(evolution_coords_dr_u)) -
        1.0 / (square(r_tilde) * square(omega)) *
            (conj(eth_omega) * j_tilde - eth_omega * k) +
        1.0 / (square(r_tilde) * square(omega)) *
            (conj(eth_omega) * dr_j_tilde -
             0.5 * eth_omega / (1.0 + j_tilde * conj(j_tilde)) *
                 (j_tilde * conj(dr_j_tilde) + conj(j_tilde) * dr_j_tilde));
    auto exp_2_beta =
        SpinWeighted<ComplexDataVector, 0>{exp(-2.0 * beta.data())};
    *evolution_gauge_bondi_q =
        square(bondi_r) * exp_2_beta * 0.5 *
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
                 Tags::EvolutionGaugeBoundaryValue<Tags::R>, Tags::GaugeA,
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
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    SpinWeighted<ComplexDataVector, 1> evolution_coords_u =
        Spectral::Swsh::swsh_interpolate_from_pfaffian(
            make_not_null(&get(*u)), get<0>(x_of_x_tilde), get<1>(x_of_x_tilde),
            l_max);

    SpinWeighted<ComplexDataVector, 0> k_tilde;
    k_tilde.data() = sqrt(1.0 + j_tilde * conj(j_tilde));
    get(*evolution_gauge_u) =
        0.5 *
            (get(a) * conj(evolution_coords_u) + get(b) * evolution_coords_u) /
            square(omega) +
        1.0 / (r_tilde * omega) *
            (conj(eth_omega) * j_tilde - eth_omega * k_tilde);
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::W> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::W>,
                                 Tags::BoundaryValue<Tags::W>>;
  using argument_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::R>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::J>,
                 Tags::BoundaryValue<Tags::U>, Tags::BoundaryValue<Tags::Beta>,
                 Tags::GaugeOmega, Tags::Du<Tags::GaugeOmega>,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::GaugeB, Tags::GaugeA, Tags::CauchyAngularCoords,
                 Tags::LMax>;

  // TODO arguments apply impl etc
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_w,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> w,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& boundary_u,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    apply_impl(make_not_null(&get(evolution_gauge_w)), make_not_null(&get(w)),
               get(r_tilde), get(j_tilde), get(boundary_u), get(boundary_beta),
               get(omega), get(du_omega), get(eth_omega), get(b), get(a),
               x_of_x_tilde, l_max);
  }

 private:
  static void apply_impl(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>
          evolution_gauge_w,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> w,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> boundary_u,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> boundary_beta,
      const SpinWeighted<ComplexDataVector, 0>& r_tilde,
      const SpinWeighted<ComplexDataVector, 2>& j_tilde,
      const SpinWeighted<ComplexDataVector, 0>& omega,
      const SpinWeighted<ComplexDataVector, 0>& du_omega,
      const SpinWeighted<ComplexDataVector, 1>& eth_omega,
      const SpinWeighted<ComplexDataVector, 0>& b,
      const SpinWeighted<ComplexDataVector, 2>& a,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*evolution_gauge_w)), make_not_null(&get(*w)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);
    SpinWeighted<ComplexDataVector, 1> boundary_u_of_x_tilde{};
    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&boundary_u_of_x_tilde), make_not_null(&get(*boundary_u)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);

    SpinWeighted<ComplexDataVector, 0> boundary_beta_of_x_tilde{};

    SpinWeighted<ComplexDataVector, 0> k_tilde;
    k_tilde.data() = sqrt(1.0 + j_tilde * conj(j_tilde));
    evolution_gauge_w =
        evolution_gauge_w / omega +
        1.0 / r_tilde * (1.0 / square(omega) - 1.0) -
        2.0 * du_omega / square(omega) -
        (1.0 / (2.0 * pow<4>(omega) * r_tilde)) *
            (conj(eth_omega) *
                 (b * (boundary_u - u_0) + a * conj(boundary_u - u_0)) +
             eth_omega * (b * (conj(a) * (boundary_u - u_0) +
                               conj(b) * conj(boundary_u - u_0)))) *
            exp(2.0 * boundary_beta) -
        exp(2.0 * boundary_beta) / (2.0 * pow<4>(omega) * r_tilde) *
            (square(conj(eth_omega)) * j_tilde +
             square(eth_omega) * conj(j_tilde) -
             2.0 * eth_omega * conj(eth_omega) * k_tilde);
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::H> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::H>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::J>,
                 Tags::BoundaryValue<Tags::H>, Tags::BoundaryValue<Tags::J>,
                 Tags::U0, Tags::EvolutionGaugeBoundaryValue<Tags::R>,
                 Tags::GaugeA>;
  using argument_tags =
      tmpl::list<Tags::GaugeB, Tags::Du<Tags::GaugeA>, Tags::Du<Tags::GaugeB>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::Dr<Tags::J>>,
                 Tags::CauchyAngularCoords, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> h_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> h,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u0,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> r_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& du_a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_b,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_j_tilde,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    apply_impl(make_not_null(&get(*h_tilde)), make_not_null(&get(*j_tilde)),
               make_not_null(&get(*h)), make_not_null(&get(*j)),
               make_not_null(&get(*u0)), make_not_null(&get(*r_tilde)),
               make_not_null(&get(*a)), get(b), get(omega), get(du_omega),
               get(eth_omega), get(du_a), get(du_b), get(dr_j_tilde),
               x_of_x_tilde, l_max);
  }

  static void apply_impl(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> h_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> j_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> h,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> j,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> u0,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> r_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> a,
      const SpinWeighted<ComplexDataVector, 0>& b,
      const SpinWeighted<ComplexDataVector, 0>& omega,
      const SpinWeighted<ComplexDataVector, 0>& du_omega,
      const SpinWeighted<ComplexDataVector, 1>& eth_omega,
      const SpinWeighted<ComplexDataVector, 2>& du_a,
      const SpinWeighted<ComplexDataVector, 0>& du_b,
      const SpinWeighted<ComplexDataVector, 2>& dr_j_tilde,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    // optimization note: this has many allocations. They can be made fewer.
    // optimization note: this has several spin-weighted derivatives, they can
    // be aggregated
    SpinWeighted<ComplexDataVector, 2> j_of_x_tilde =
        Spectral::Swsh::swsh_interpolate_from_pfaffian(
            j, get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);
    SpinWeighted<ComplexDataVector, 2> h_of_x_tilde =
        Spectral::Swsh::swsh_interpolate_from_pfaffian(
            h, get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);

    // this manipulation is needed because spin weight 3 or higher is not
    // supported
    SpinWeighted<ComplexDataVector, 0> abar_j_tilde = conj(*a) * (*j_tilde);
    SpinWeighted<ComplexDataVector, 1> u0bar_j_tilde = conj(*u0) * (*j_tilde);
    // TODO all of these should be evaluated at fixed Bondi radius, not fixed
    // compactified radius, so we need radial derivatives and more inputs
    auto eth_r_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(r_tilde,
                                                                   l_max);
    auto abar_eth_j_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&abar_j_tilde), l_max) -
        conj(*a) * eth_r_tilde * dr_j_tilde -
        (*j_tilde) *
            conj(Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                a, l_max));
    auto u0bar_eth_j_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&u0bar_j_tilde), l_max) -
        conj(*u0) * eth_r_tilde * dr_j_tilde -
        (*j_tilde) *
            conj(Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                u0, l_max));
    auto ethbar_j_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(j_tilde,
                                                                      l_max) -
        conj(eth_r_tilde) * dr_j_tilde;

    SpinWeighted<ComplexDataVector, 0> k;
    k.data() = sqrt(1.0 + j_of_x_tilde.data() * conj(j_of_x_tilde.data()));
    auto angular_derivative_part = 0.25 * (*u0) * b * ethbar_j_tilde +
                                   0.25 * (*u0) * abar_eth_j_tilde +
                                   0.25 * conj(*u0) * (*a) * ethbar_j_tilde +
                                   0.25 * conj(b) * u0bar_eth_j_tilde;
    auto time_derivative_part =
        0.25 * (square(b) * h_of_x_tilde + square(*a) * conj(h_of_x_tilde) -
                (*a) * b *
                    (j_of_x_tilde * conj(h_of_x_tilde) +
                     h_of_x_tilde * conj(j_of_x_tilde)) /
                    (1.0 + j_of_x_tilde * conj(j_of_x_tilde))) +
        0.5 * b * du_b * j_of_x_tilde + 0.5 * (*a) * du_a * conj(j_of_x_tilde) -
        0.5 * ((*a) * du_b + b * du_a) * k;
    *h_tilde =
        angular_derivative_part + time_derivative_part / square(omega) -
        2.0 *
            (du_omega - 0.25 * (b * u_0 + (*a) * conj(u_0)) * conj(eth_omega) -
             0.25 * (conj(*a) * u_0 + conj(b * u_0)) * eth_omega) /
            pow<3>(omega) *
            (0.25 * (square(b) * j_of_x_tilde + square(a) * conj(j_of_x_tilde) -
                     2.0 * a * b * k));
  }
};

struct GaugeUpdateU {
  using argument_tags =
      tmpl::list<Tags::GaugeA, Tags::GaugeB, Tags::Exp2Beta, Tags::LMax>;
  using return_tags = tmpl::list<Tags::DuCauchyAngularCoords,
                                 Tags::Du<Tags::GaugeA>, Tags::Du<Tags::GaugeB>,
                                 Tags::U0, Tags::U, Tags::Du<Tags::GaugeOmega>>;

  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 2>*> du_x,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> du_a,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> du_b,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u_0,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> du_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp2beta,
      const size_t l_max) noexcept {
    size_t number_of_radial_points =
        get(*u).size() /
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    // u_hat_0
    get(*u_0).data() = ComplexDataVector{
        get(*u).data().data() +
            (number_of_radial_points - 1) *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    // subtract u_hat_0 from u
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector angular_view{
          get(*u).data().data() +
              i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
      angular_view -= get(*u_0).data();
    }
    // u_under_tilde_0 (note: other corrections from radial coordinate change
    // don't appear as we are evaluating at scri+).
    get(*u_0) = square(get(omega)) * u_0;
    // u_0
    get(*u_0) = (2.0 / (get(b) * conj(get(b)) - get(a) * conj(get(a)))) *
                (conj(get(b)) * get(*u_0) - get(a) * conj(get(*u_0)));

    // Note we store x^\phi * sin(theta) rather than x^\phi to reduce the number
    // of places sin(theta)'s appear. Unfortunately, without reformulating the
    // spin-weighted spherical harmonics themselves, we won't be able to
    // eliminate sin(theta)s completely

    get<0>(*du_x) = -real(get(*u_0).data());
    get<1>(*du_x) = -imag(get(*u_0).data());
    // at some point we might want to split these out into their own
    // computational function
    SpinWeighted<ComplexDataVector, 1> a_u0bar_plus_b_u0 =
        get(a) * conj(get(*u_0)) + get(b) * get(*u_0);

    auto eth_a_u0bar_plus_b_u0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&a_u0bar_plus_b_u0), l_max);
    auto ethbar_a_u0bar_plus_b_u0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&a_u0bar_plus_b_u0), l_max);

    ComplexDataVector exp2beta_buffer = get(exp2beta).data();
    ComplexDataVector exp2beta_slice = ComplexDataVector{
        exp2beta_buffer.data() +
            (number_of_radial_points - 1) *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    get(*du_a) = -0.25 * get(a) * ethbar_a_u0bar_plus_b_u0 -
                 0.25 * get(b) * eth_a_u0bar_plus_b_u0;
    get(*du_b) = -0.25 * conj(get(b)) * ethbar_a_u0bar_plus_b_u0 -
                 0.25 * conj(get(a)) * eth_a_u0bar_plus_b_u0;
    auto ethbar_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&get(*u_0)), l_max);
    get(*du_gauge_omega) = 0.25 * (ethbar_u_0 + conj(ethbar_u_0));
  }
};

struct GaugeUpdateOmega {
  using argument_tags = tmpl::list<Tags::GaugeA, Tags::GaugeB, Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::GaugeOmega,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> a,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> b,
      const size_t l_max) noexcept {
    const auto& collocation = Spectral::Swsh::precomputed_collocation<
        Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
    for (const auto& collocation_point : collocation) {
      get<0>(*x_of_x_tilde)[collocation_point.offset] = collocation_point.theta;
      get<1>(*x_of_x_tilde)[collocation_point.offset] =
          collocation_point.phi * sin(collocation_point.theta);
      get(*a).data() = 0.0;
      get(*b).data() = 2.0;
    }
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
      get(*a).data() = 0.0;
      get(*b).data() = 2.0;
    }
  }
};
}  // namespace Cce
