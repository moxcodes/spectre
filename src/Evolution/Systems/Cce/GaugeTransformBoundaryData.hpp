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
    tmpl::list<Tags::BondiR, Tags::BondiJ, Tags::Dr<Tags::BondiJ>>;

template <typename Tag>
struct ComputeGaugeAdjustedBoundaryValue;

// this just needs to be interpolated to the new grid
// Consider rolling this into the beta or R one
template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::BondiR> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
                 Tags::BoundaryValue<Tags::BondiR>>;
  using argument_tags =
      tmpl::list<Tags::CauchyAngularCoords, Tags::GaugeOmegaCD, Tags::LMax>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_r,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> r,
      const tnsr::i<DataVector, 2>& x_of_x_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const size_t l_max) noexcept {
    // Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(*r)),
    // l_max, l_max - 2);
    // SpinWeighted<ComplexDataVector, 0> r_over_omega = get(*r) / get(omega);
    Spectral::Swsh::SwshInterpolator interpolator{
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), 0, l_max};
    interpolator.interpolate(
        make_not_null(&get(*evolution_gauge_r).data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(make_not_null(&get(*r)), l_max),
            l_max)
            .data());

    get(*evolution_gauge_r) = get(*evolution_gauge_r) * get(omega_cd);

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
      tmpl::list<Tags::U0, Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
                 Tags::GaugeOmegaCD, Tags::Du<Tags::GaugeOmegaCD>,
                 Tags::CauchyAngularCoords, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_du_r_divided_by_r,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          du_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& u_0,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& evolution_gauge_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega_cd,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    // Spectral::Swsh::filter_swsh_boundary_quantity(
    // make_not_null(&get(*du_r_divided_by_r)), l_max, l_max - 2);

    Spectral::Swsh::SwshInterpolator interpolator{
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), 0, l_max};
    interpolator.interpolate(
        make_not_null(&get(*evolution_gauge_du_r_divided_by_r).data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(
                make_not_null(&get(*du_r_divided_by_r)), l_max),
            l_max)
            .data());

    // taking this as argument saves an interpolation, which is significantly
    // more expensive than the extra multiplication.
    SpinWeighted<ComplexDataVector, 0> r_buffer =
        get(evolution_gauge_r) / get(omega_cd);
    auto eth_r = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        make_not_null(&r_buffer), l_max);
    // TEST
    get(*evolution_gauge_du_r_divided_by_r) +=
        0.5 * (get(u_0) * conj(eth_r) + conj(get(u_0)) * eth_r) / r_buffer +
        get(du_omega_cd) / get(omega_cd);
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::BondiJ> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>,
                 Tags::BoundaryValue<Tags::BondiJ>>;
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

    Spectral::Swsh::SwshInterpolator interpolator{
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), 2, l_max};
    interpolator.interpolate(
        make_not_null(&get(*evolution_gauge_j).data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(make_not_null(&get(*j)), l_max),
            l_max)
            .data());

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
struct ComputeGaugeAdjustedBoundaryValue<Tags::Dr<Tags::BondiJ>> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::Dr<Tags::BondiJ>>,
                 Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>;
  using argument_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>, Tags::GaugeC,
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

    Spectral::Swsh::SwshInterpolator interpolator{
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), 2, l_max};
    interpolator.interpolate(
        make_not_null(&get(*evolution_gauge_dr_j).data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(make_not_null(&get(*dr_j)), l_max),
            l_max)
            .data());

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
struct ComputeGaugeAdjustedBoundaryValue<Tags::BondiBeta> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>,
                 Tags::BoundaryValue<Tags::BondiBeta>>;
  using argument_tags =
      tmpl::list<Tags::GaugeOmegaCD, Tags::CauchyAngularCoords, Tags::LMax>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_beta,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::SwshInterpolator interpolator{
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), 0, l_max};
    interpolator.interpolate(
        make_not_null(&get(*evolution_gauge_beta).data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(make_not_null(&get(*beta)), l_max),
            l_max)
            .data());

    get(*evolution_gauge_beta).data() -= 0.5 * log(get(omega_cd).data());
    // Spectral::Swsh::filter_swsh_boundary_quantity(
    // make_not_null(&get(*evolution_gauge_beta)), l_max, l_max - 2);
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::BondiQ> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiQ>,
                 Tags::BoundaryValue<Tags::Dr<Tags::BondiU>>, Tags::BondiJ,
                 Tags::Dy<Tags::BondiJ>>;

    using argument_tags =
        tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>,
                   Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
                   Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>,
                   Tags::GaugeC, Tags::GaugeD, Tags::GaugeOmegaCD,
                   Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                    Spectral::Swsh::Tags::Eth>,
                   Tags::CauchyAngularCoords, Tags::LMax>;

    static void apply(
        const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
            evolution_gauge_bondi_q,
        const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> dr_u,
        const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
            j_tilde,
        const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
            dy_j_tilde,
        const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
        const Scalar<SpinWeighted<ComplexDataVector, 0>>& r_tilde,
        const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta_tilde,
        const Scalar<SpinWeighted<ComplexDataVector, 2>>& c,
        const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
        const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
        const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
        const tnsr::i<DataVector, 2>& x_of_x_tilde,
        const size_t l_max) noexcept {
      apply_impl(make_not_null(&get(*evolution_gauge_bondi_q)),
                 make_not_null(&get(*dr_u)), make_not_null(&get(*j_tilde)),
                 make_not_null(&get(*dy_j_tilde)), get(j), get(r_tilde),
                 get(beta_tilde), get(c), get(d), get(omega_cd),
                 get(eth_omega_cd), x_of_x_tilde, l_max);
    }
    static void apply_impl(
        const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*>
            evolution_gauge_bondi_q,
        const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> dr_u,
        const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> j_tilde,
        const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> dy_j_tilde,
        const SpinWeighted<ComplexDataVector, 2>& j,
        const SpinWeighted<ComplexDataVector, 0>& r_tilde,
        const SpinWeighted<ComplexDataVector, 0>& beta_tilde,
        const SpinWeighted<ComplexDataVector, 2>& c,
        const SpinWeighted<ComplexDataVector, 0>& d,
        const SpinWeighted<ComplexDataVector, 0>& omega_cd,
        const SpinWeighted<ComplexDataVector, 1>& eth_omega_cd,
        const tnsr::i<DataVector, 2>& x_of_x_tilde,
        const size_t l_max) noexcept {
      // TODO fix transforms for representability concerns

      SpinWeighted<ComplexDataVector, 1> evolution_coords_dr_u{dr_u->size()};
      Spectral::Swsh::SwshInterpolator interpolator{
          get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), 1, l_max};
      interpolator.interpolate(
          make_not_null(&evolution_coords_dr_u.data()),
          Spectral::Swsh::libsharp_to_goldberg_modes(
              Spectral::Swsh::swsh_transform(dr_u, l_max), l_max)
              .data());

      SpinWeighted<ComplexDataVector, 2> boundary_j_tilde;
      boundary_j_tilde.data() = ComplexDataVector{
          j_tilde->data().data(),
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
      // TEST
      boundary_j_tilde.data() = j.data();

      SpinWeighted<ComplexDataVector, 0> k;
      k.data() =
          sqrt(1.0 + boundary_j_tilde.data() * conj(boundary_j_tilde.data()));

      SpinWeighted<ComplexDataVector, 2> dr_j_tilde;
      dr_j_tilde.data() =
          2.0 / r_tilde.data() *
          ComplexDataVector{
              dy_j_tilde->data().data(),
              Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

      SpinWeighted<ComplexDataVector, 0> exp_minus_2_beta;
      exp_minus_2_beta.data() = exp(-2.0 * beta_tilde.data());
      // NOTE extra sign change in later terms due to omega_cd = 1/omega
      // TEST
      evolution_coords_dr_u =
          0.5 / pow<3>(omega_cd) *
              (conj(d) * evolution_coords_dr_u -
               c * conj(evolution_coords_dr_u)) -
          (eth_omega_cd * k - conj(eth_omega_cd) * boundary_j_tilde) *
              (1.0 / (r_tilde * exp_minus_2_beta)) *
              (-1.0 / r_tilde +
               0.25 * r_tilde *
                   (dr_j_tilde * conj(dr_j_tilde) -
                    0.25 *
                        square(boundary_j_tilde * conj(dr_j_tilde) +
                               dr_j_tilde * conj(boundary_j_tilde)) /
                        (1.0 + boundary_j_tilde * conj(boundary_j_tilde)))) /
              omega_cd -
          1.0 / (r_tilde * exp_minus_2_beta) *
              (-conj(eth_omega_cd) * dr_j_tilde +
               0.5 * eth_omega_cd / k *
                   (boundary_j_tilde * conj(dr_j_tilde) +
                    conj(boundary_j_tilde) * dr_j_tilde)) /
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
          (boundary_j_tilde * conj(evolution_coords_dr_u) +
           k * evolution_coords_dr_u);
    }
};

// NOTE! this gets the boundary value for \f$\hat{U}\f$
template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::BondiU> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiU>,
                 Tags::BoundaryValue<Tags::BondiU>, Tags::BondiJ>;
  using argument_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>,
                 Tags::GaugeC, Tags::GaugeD, Tags::GaugeOmegaCD,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::CauchyAngularCoords, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          evolution_gauge_u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::SwshInterpolator interpolator{
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), 1, l_max};
    SpinWeighted<ComplexDataVector, 1> evolution_coords_u{get(*u).size()};
    interpolator.interpolate(
        make_not_null(&evolution_coords_u.data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(make_not_null(&get(*u)), l_max),
            l_max)
            .data());

    SpinWeighted<ComplexDataVector, 2> boundary_j_tilde;
    boundary_j_tilde = ComplexDataVector{
        get(*j_tilde).data().data(),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    boundary_j_tilde.data() = get(j).data();

    SpinWeighted<ComplexDataVector, 0> k_tilde;
    k_tilde.data() =
        sqrt(1.0 + boundary_j_tilde.data() * conj(boundary_j_tilde.data()));

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
                                  (conj(get(eth_omega_cd)) * boundary_j_tilde -
                                   get(eth_omega_cd) * k_tilde);
    // Spectral::Swsh::filter_swsh_boundary_quantity(
    // make_not_null(&get(*evolution_gauge_u)), l_max, l_max - 2);
  }
};

template <>
struct ComputeGaugeAdjustedBoundaryValue<Tags::BondiW> {
  using return_tags = tmpl::list<
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiW>,
      Tags::BoundaryValue<Tags::BondiW>, Tags::BoundaryValue<Tags::BondiU>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiU>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>, Tags::BondiJ>;
  using argument_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>, Tags::U0,
                 Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
                 Tags::GaugeOmegaCD, Tags::Du<Tags::GaugeOmegaCD>,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::CauchyAngularCoords, Tags::LMax>;

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
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& u_0,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    apply_impl(make_not_null(&get(*evolution_gauge_w)), make_not_null(&get(*w)),
               make_not_null(&get(*boundary_u)),
               make_not_null(&get(*boundary_u_tilde)),
               make_not_null(&get(*beta_tilde)), make_not_null(&get(*j_tilde)),
               get(j), get(u_0), get(r_tilde), get(omega_cd), get(du_omega_cd),
               get(eth_omega_cd), x_of_x_tilde, l_max);
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
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> j_tilde,
      const SpinWeighted<ComplexDataVector, 2>& j,
      const SpinWeighted<ComplexDataVector, 1>& u_0,
      const SpinWeighted<ComplexDataVector, 0>& r_tilde,
      const SpinWeighted<ComplexDataVector, 0>& omega_cd,
      const SpinWeighted<ComplexDataVector, 0>& du_omega_cd,
      const SpinWeighted<ComplexDataVector, 1>& eth_omega_cd,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::SwshInterpolator w_interpolator{
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), 0, l_max};
    w_interpolator.interpolate(
        make_not_null(&evolution_gauge_w->data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(w, l_max), l_max)
            .data());

    SpinWeighted<ComplexDataVector, 1> boundary_u_of_x_tilde{
        boundary_u->size()};
    Spectral::Swsh::SwshInterpolator u_interpolator{
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), 1, l_max};
    u_interpolator.interpolate(
        make_not_null(&boundary_u_of_x_tilde.data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(boundary_u, l_max), l_max)
            .data());

    SpinWeighted<ComplexDataVector, 2> boundary_j_tilde;
    boundary_j_tilde = ComplexDataVector{
        j_tilde->data().data(),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    // TEST
    boundary_j_tilde.data() = j.data();

    SpinWeighted<ComplexDataVector, 0> k_tilde;
    k_tilde.data() =
        sqrt(1.0 + boundary_j_tilde.data() * conj(boundary_j_tilde.data()));
    SpinWeighted<ComplexDataVector, 0> exp_2_beta_of_x_tilde;
    exp_2_beta_of_x_tilde.data() = exp(2.0 * beta_tilde->data());

    SpinWeighted<ComplexDataVector, 0> w_copy = *evolution_gauge_w;
    SpinWeighted<ComplexDataVector, 0> cauchy_du_omega_cd =
        du_omega_cd -
        0.5 * (u_0 * conj(eth_omega_cd) + conj(u_0) * eth_omega_cd);

    *evolution_gauge_w =
        *evolution_gauge_w + 1.0 / r_tilde * (omega_cd - 1.0) -
        2.0 * cauchy_du_omega_cd / omega_cd -
        (conj(eth_omega_cd) * (*boundary_u_tilde - u_0) +
         eth_omega_cd * (conj(*boundary_u_tilde) - conj(u_0))) /
            omega_cd +
        exp_2_beta_of_x_tilde / (2.0 * square(omega_cd) * r_tilde) *
            (square(conj(eth_omega_cd)) * boundary_j_tilde +
             square(eth_omega_cd) * conj(boundary_j_tilde) -
             2.0 * eth_omega_cd * conj(eth_omega_cd) * k_tilde);

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
struct ComputeGaugeAdjustedBoundaryValue<Tags::BondiH> {
  using return_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiH>, Tags::BondiJ,
                 Tags::BoundaryValue<Tags::SpecH>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>,
                 Tags::Dy<Tags::BondiJ>, Tags::U0,
                 Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>, Tags::GaugeC>;
  using argument_tags =
      tmpl::list<Tags::GaugeD, Tags::GaugeOmegaCD, Tags::Du<Tags::GaugeOmegaCD>,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::DuRDividedByR>,
                 Tags::CauchyAngularCoords, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> h_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> spec_h,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dy_j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u_0,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> r_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r_divided_by_r_tilde,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    apply_impl(make_not_null(&get(*h_tilde)), make_not_null(&get(*j_tilde)),
               make_not_null(&get(*spec_h)), make_not_null(&get(*j)),
               make_not_null(&get(*dy_j)), make_not_null(&get(*u_0)),
               make_not_null(&get(*r_tilde)), make_not_null(&get(*c)), get(d),
               get(omega_cd), get(du_omega_cd), get(eth_omega_cd),
               get(du_r_divided_by_r_tilde), x_of_x_tilde,  l_max);
  }

  // TODO change variable names and implementation
  static void apply_impl(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> h_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> j_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> spec_h,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> j,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> dy_j,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> u_0,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> r_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> c,
      const SpinWeighted<ComplexDataVector, 0>& d,
      const SpinWeighted<ComplexDataVector, 0>& omega_cd,
      const SpinWeighted<ComplexDataVector, 0>& du_tilde_omega_cd,
      const SpinWeighted<ComplexDataVector, 1>& eth_omega_cd,
      const SpinWeighted<ComplexDataVector, 0>& du_r_divided_by_r_tilde,
      const tnsr::i<DataVector, 2>& x_of_x_tilde,const size_t l_max) noexcept {
    // optimization note: this has many allocations. They can be made fewer.
    // optimization note: this has several spin-weighted derivatives, they can
    // be aggregated
    // Spectral::Swsh::filter_swsh_boundary_quantity(u_0, l_max, l_max - 2);

    SpinWeighted<ComplexDataVector, 2> boundary_j_tilde;
    boundary_j_tilde.data() = ComplexDataVector{
        j_tilde->data().data(),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    // TEST
    boundary_j_tilde.data() = j->data();

    SpinWeighted<ComplexDataVector, 2> j_of_x_tilde;
    j_of_x_tilde.data() =
        0.25 * (square(d.data()) * boundary_j_tilde.data() +
                square(c->data()) * conj(boundary_j_tilde.data()) -
                2.0 * c->data() * d.data() *
                    sqrt(1.0 + boundary_j_tilde.data() *
                                   conj(boundary_j_tilde.data())));

    SpinWeighted<ComplexDataVector, 2> dr_j_tilde;
    dr_j_tilde.data() =
        2.0 / r_tilde->data() *
        ComplexDataVector{
            dy_j->data().data(),
            Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    SpinWeighted<ComplexDataVector, 0> k_of_x_tilde;
    k_of_x_tilde.data() =
        sqrt(1.0 + j_of_x_tilde.data() * conj(j_of_x_tilde.data()));

    SpinWeighted<ComplexDataVector, 0> k_tilde;
    k_tilde.data() =
        sqrt(1.0 + boundary_j_tilde.data() * conj(boundary_j_tilde.data()));

    auto eth_r_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(r_tilde,
                                                                   l_max);

    SpinWeighted<ComplexDataVector, 1> u_0_bar_j_tilde =
        conj(*u_0) * boundary_j_tilde;

    // TODO combine some expressions to minimize allocations where possible
    SpinWeighted<ComplexDataVector, 2> u_0_bar_eth_j_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&u_0_bar_j_tilde), l_max) -
        boundary_j_tilde *
            conj(Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                u_0, l_max)) -
        conj(*u_0) * eth_r_tilde * (dr_j_tilde);
    SpinWeighted<ComplexDataVector, 2> angular_derivative_term =
        0.5 *
        (*u_0 * Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                    make_not_null(&boundary_j_tilde), l_max) -
         *u_0 * conj(eth_r_tilde) * (dr_j_tilde) + u_0_bar_eth_j_tilde);

    SpinWeighted<ComplexDataVector, 0> cauchy_du_omega_cd =
        du_tilde_omega_cd -
        0.5 * (*u_0 * conj(eth_omega_cd) + conj(*u_0) * eth_omega_cd);

    auto eth_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(u_0, l_max);
    auto ethbar_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(u_0,
                                                                      l_max);

    // Spectral::Swsh::filter_swsh_boundary_quantity(spec_h, l_max, l_max - 2);
    SpinWeighted<ComplexDataVector, 2> h_of_x_tilde{spec_h->size()};
    Spectral::Swsh::SwshInterpolator h_interpolator{
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), 2, l_max};
    h_interpolator.interpolate(
        make_not_null(&h_of_x_tilde.data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(spec_h, l_max), l_max)
            .data());

    *h_tilde =
        angular_derivative_term -
        *r_tilde * cauchy_du_omega_cd / omega_cd * (dr_j_tilde) +
        2.0 * cauchy_du_omega_cd / omega_cd * boundary_j_tilde -
        ethbar_u_0 * (boundary_j_tilde) + eth_u_0 * k_tilde +
        0.25 / square(omega_cd) *
            (square(conj(d)) * h_of_x_tilde + square(*c) * conj(h_of_x_tilde) +
             *c * conj(d) *
                 (h_of_x_tilde * conj(j_of_x_tilde) +
                  j_of_x_tilde * conj(h_of_x_tilde)) /
                 k_of_x_tilde) +
        du_r_divided_by_r_tilde * (*r_tilde) * (dr_j_tilde);
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
      tmpl::list<Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords,
                 Tags::GaugeOmegaCD,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::DuCauchyCartesianCoords, Tags::U0, Tags::BondiU,
                 Tags::Du<Tags::GaugeOmegaCD>, Tags::GaugeC, Tags::GaugeD>;

  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*> du_x,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u_0,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          du_omega_cd,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> c,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> d,
      const tnsr::i<DataVector, 2> x_of_x_tilde,
      const tnsr::i<DataVector, 3> cartesian_x_of_x_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
      const size_t l_max, const bool gauge_transform_u = true) noexcept {
    size_t number_of_radial_points =
        get(*u).size() /
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    // u_hat_0
    ComplexDataVector u_scri_slice{
        get(*u).data().data() +
            (number_of_radial_points - 1) *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    // RT test
    // get(*u_0).data() = 0.0;
    get(*u_0).data() = u_scri_slice;

    // printf("debug: u at scri\n");
    // for(auto val : u_scri_slice) {
    // printf("%e, %e\n", real(val), imag(val));
    // }
    // printf("done\n");

    // TEST
    // get(*u_0).data() = 0.0;
    // get(*u_0).data() = 0.0 * u_scri_slice;

    // TEST
    DataVector theta{Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    DataVector phi{Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    const auto& collocation = Spectral::Swsh::precomputed_collocation<
        Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
    for (const auto& collocation_point : collocation) {
      theta[collocation_point.offset] = collocation_point.theta;
      phi[collocation_point.offset] = collocation_point.phi;
    }
    // get(*u_0).data() = 1.0e-3 * std::complex<double>(1.0, 0.0) * cos(theta) *
    // sin(theta);

    // subtract u_hat_0 from u
    // TEST
    if (gauge_transform_u) {
      for (size_t i = 0; i < number_of_radial_points; ++i) {
        ComplexDataVector angular_view{
            get(*u).data().data() +
                i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
            Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
        angular_view -= get(*u_0).data();
      }
    }
    // TEST
    // Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(*u_0)),
    // l_max, l_max - 4);

    // NOTE currently just storing u_0_hat -- most of these factors just have to
    // be reversed for the rest of the boundary conditions to use them anyway.

    // TODO need to interpolate if not transforming u because of coordinate
    // choice
    Spectral::Swsh::SwshInterpolator u_interpolator{
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), 1, l_max};
    SpinWeighted<ComplexDataVector, 1> u_0_cauchy{
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    if (gauge_transform_u) {
      u_0_cauchy = 0.5 * (get(*d) * get(*u_0) + get(*c) * conj(get(*u_0)));
    } else {
      u_interpolator.interpolate(
          make_not_null(&u_0_cauchy.data()),
          Spectral::Swsh::libsharp_to_goldberg_modes(
              Spectral::Swsh::swsh_transform(make_not_null(&get(*u_0)), l_max),
              l_max)
              .data());
    }

    if (gauge_transform_u) {
      SpinWeighted<ComplexDataVector, 0> x;
      x.data() =
          std::complex<double>(1.0, 0.0) * get<0>(cartesian_x_of_x_tilde);
      SpinWeighted<ComplexDataVector, 0> y;
      y.data() =
          std::complex<double>(1.0, 0.0) * get<1>(cartesian_x_of_x_tilde);
      SpinWeighted<ComplexDataVector, 0> z;
      z.data() =
          std::complex<double>(1.0, 0.0) * get<2>(cartesian_x_of_x_tilde);

      auto eth_x = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
          make_not_null(&x), l_max);
      auto eth_y = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
          make_not_null(&y), l_max);
      auto eth_z = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
          make_not_null(&z), l_max);
      get<0>(*du_x) = real(conj(get(*u_0).data()) * eth_x.data());
      get<1>(*du_x) = real(conj(get(*u_0).data()) * eth_y.data());
      get<2>(*du_x) = real(conj(get(*u_0).data()) * eth_z.data());
    } else {
      get<0>(*du_x) = -cos(get<0>(x_of_x_tilde)) * cos(get<1>(x_of_x_tilde)) *
                          real(u_0_cauchy.data()) +
                      sin(get<1>(x_of_x_tilde)) * imag(u_0_cauchy.data());
      get<1>(*du_x) = -cos(get<0>(x_of_x_tilde)) * sin(get<1>(x_of_x_tilde)) *
                          real(u_0_cauchy.data()) -
                      cos(get<1>(x_of_x_tilde)) * imag(u_0_cauchy.data());
      get<2>(*du_x) = sin(get<0>(x_of_x_tilde)) * real(u_0_cauchy.data());
    }

    // Unfortunately, this is the best way I can think of to guarantee that
    // these derivatives are correctly evaluated, and the derivatives are needed
    // for the evolution equations below.

    auto inertial_ethbar_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&get(*u_0)), l_max);
    auto inertial_eth_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&get(*u_0)), l_max);

    // NOTE if we are not transforming u, then the u_0 we have is actually not
    // the 'hatted' version, and is expressed in the noninertial coordinates.
    // So, we need to adjust the derivation.
    if (gauge_transform_u) {
      get(*du_omega_cd) =
          0.25 * (inertial_ethbar_u_0 + conj(inertial_ethbar_u_0)) *
              get(omega_cd) +
          0.5 * (get(*u_0) * conj(get(eth_omega_cd)) +
                 conj(get(*u_0)) * get(eth_omega_cd));
    } else {
      SpinWeighted<ComplexDataVector, 0> du_omega_cd_source_coords =
          0.25 * (inertial_ethbar_u_0 + conj(inertial_ethbar_u_0)) *
          get(omega_cd);
      Spectral::Swsh::SwshInterpolator omega_interpolator{
          get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), 0, l_max};
      omega_interpolator.interpolate(
          make_not_null(&get(*du_omega_cd).data()),
          Spectral::Swsh::libsharp_to_goldberg_modes(
              Spectral::Swsh::swsh_transform(
                  make_not_null(&du_omega_cd_source_coords), l_max),
              l_max)
              .data());
    }
  }
};

// This takes the u_0 from the databox, assuming it has already been set
// separately.
struct GaugeUpdateUManualTransform {
  using argument_tags =
      tmpl::list<Tags::CauchyCartesianCoords, Tags::GaugeOmegaCD,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmegaCD,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::LMax>;
  using return_tags = tmpl::list<Tags::DuCauchyCartesianCoords, Tags::U0,
                                 Tags::BondiU, Tags::Du<Tags::GaugeOmegaCD>>;

  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*> du_x,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u_0,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          du_omega_cd,
      const tnsr::i<DataVector, 3> cartesian_x_of_x_tilde,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_omega_cd,
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

    // subtract u_hat_0 from u
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector angular_view{
          get(*u).data().data() +
              i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
      angular_view -= get(*u_0).data();
    }

    SpinWeighted<ComplexDataVector, 0> x;
    x.data() = std::complex<double>(1.0, 0.0) * get<0>(cartesian_x_of_x_tilde);
    SpinWeighted<ComplexDataVector, 0> y;
    y.data() = std::complex<double>(1.0, 0.0) * get<1>(cartesian_x_of_x_tilde);
    SpinWeighted<ComplexDataVector, 0> z;
    z.data() = std::complex<double>(1.0, 0.0) * get<2>(cartesian_x_of_x_tilde);

    auto eth_x = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        make_not_null(&x), l_max);
    auto eth_y = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        make_not_null(&y), l_max);
    auto eth_z = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        make_not_null(&z), l_max);
    get<0>(*du_x) = real(conj(get(*u_0).data()) * eth_x.data());
    get<1>(*du_x) = real(conj(get(*u_0).data()) * eth_y.data());
    get<2>(*du_x) = real(conj(get(*u_0).data()) * eth_z.data());

    // Unfortunately, this is the best way I can think of to guarantee that
    // these derivatives are correctly evaluated, and the derivatives are needed
    // for the evolution equations below.

    auto inertial_ethbar_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&get(*u_0)), l_max);
    auto inertial_eth_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&get(*u_0)), l_max);

    get(*du_omega_cd) = 0.25 *
                            (inertial_ethbar_u_0 + conj(inertial_ethbar_u_0)) *
                            get(omega_cd) +
                        0.5 * (get(*u_0) * conj(get(eth_omega_cd)) +
                               conj(get(*u_0)) * get(eth_omega_cd));
  }
};

template <typename AngularTag, typename CartesianTag>
struct GaugeUpdateAngularFromCartesian {
  using argument_tags = tmpl::list<>;
  using return_tags = tmpl::list<AngularTag, CartesianTag>;
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 2>*> angular_coords,
      const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_coords) noexcept {
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
    // for (auto val : cartesian_r) {
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
          typename CartesianCoordinateTag, typename TargetCoordinateTag>
struct GaugeUpdateJacobianFromCoords {
  using argument_tags = tmpl::list<Tags::LMax>;
  using return_tags = tmpl::list<GaugeFactorSpin2, GaugeFactorSpin0,
                                 TargetCoordinateTag, CartesianCoordinateTag>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> a,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> b,
      const gsl::not_null<tnsr::i<DataVector, 2>*> x_tilde_of_x,
      const gsl::not_null<tnsr::i<DataVector, 3>*> x_tilde_of_x_cartesian,
      const size_t l_max) noexcept {
    // first, interpolate to a new grid. For this, we assume that the theta
    // (more slowly varying) is already on a gauss-legendre grid.
    // Therefore, for each constant-in-theta view, we construct a similar-order
    // Gauss-Legendre grid using Barycentric rational interpolation.
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
  using return_tags = tmpl::list<Tags::DuInertialCartesianCoords, Tags::U0,
                                 Tags::Du<Tags::GaugeOmega>>;

  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*> du_x_tilde_of_x,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u_0_hat,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> du_omega,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega,
      const tnsr::i<DataVector, 2>& x_tilde_of_x, const size_t l_max) noexcept {
    // interpolate a and b back to the non-inertial coordinates.
    // Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null
    //(&get(*u_0_hat)),
    // l_max, l_max - 2);

    SpinWeighted<ComplexDataVector, 1> u_0_hat_of_x{get(*u_0_hat).size()};
    Spectral::Swsh::SwshInterpolator u_interpolator{
        get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), 1, l_max};
    Spectral::Swsh::SwshInterpolator omega_interpolator{
        get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), 0, l_max};

    auto inertial_ethbar_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
            make_not_null(&get(*u_0_hat)), l_max);
    auto inertial_eth_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&get(*u_0_hat)), l_max);

    SpinWeighted<ComplexDataVector, 0> du_omega_of_x_tilde =
        -0.25 * (inertial_ethbar_u_0 + conj(inertial_ethbar_u_0)) * get(omega);

    omega_interpolator.interpolate(
        make_not_null(&get(*du_omega).data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(make_not_null(&du_omega_of_x_tilde),
                                           l_max),
            l_max)
            .data());

    u_interpolator.interpolate(make_not_null(&u_0_hat_of_x.data()),
                               Spectral::Swsh::libsharp_to_goldberg_modes(
                                   Spectral::Swsh::swsh_transform(
                                       make_not_null(&get(*u_0_hat)), l_max),
                                   l_max)
                                   .data());

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
    // SpinWeighted<ComplexDataVector, 0> new_omega;
    // new_omega.data() = 0.5 * sqrt(get(d).data() * conj(get(d).data()) -
    // get(c).data() * conj(get(c).data()));

    // printf("testing omega vs evolved\n");
    // for(size_t i = 0; i < new_omega.size(); ++i) {
    // printf("%e, %e\n", real(new_omega.data()[i] - get(*omega_cd).data()[i]),
    // imag(new_omega.data()[i] - get(*omega_cd).data()[i]));
    // }
    // printf("done\n");

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
                 Tags::GaugeC, Tags::GaugeD, Tags::GaugeOmegaCD>;

  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 2>*> x_of_x_tilde,
      const gsl::not_null<tnsr::i<DataVector, 3>*> x_of_x_tilde_cartesian,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> c,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> d,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> omega_cd,
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
    get(*omega_cd).data() = 1.0;
    get(*c).data() = 0.0;
    get(*d).data() = 2.0;
  }
};
}  // namespace Cce
