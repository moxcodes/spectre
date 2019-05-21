// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"

// TOOD most of these computations require intermediate quantities that could be
// cached or grouped together like in the main evolution computations. These
// calculations are smaller, so there is significantly less savings, but this
// might be a spot to keep an eye on for optimization.

template <typename Tag>
ComputeGaugeAdjustedBoundaryValue;

// this just needs to be interpolated to the new grid
// Consider rolling this into the beta or R one
template <>
ComputeGaugeAdjustedBoundaryValue<Tags::R> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::R>,
                                 Tags::BoundaryValue<Tags::R>>;
  using argument_tags = tmpl::list<Tags::CauchyAngularCoords, Tags::LMax>;
  void apply(const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
                 evolution_gauge_r,
             const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> r,
             const tnr::i<DataVector, 2>& x_of_x_tilde,
             const size_t l_max) noexcept {
    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*evolution_gauge_r)), make_not_null(&get(*r)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);
  }
};

template <>
ComputeGaugeAdjustedBoundaryValue<Tags::J> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::J>,
                                 Tags::BoundaryValue<Tags::J>>;
  using argument_tags = tmpl::list<Tags::GaugeA, Tags::GaugeB,
                                   Tags::CauchyAngularCoords, Tags::LMax>;
  void apply(const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
                 evolution_gauge_j,
             const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
             const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
             const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
             const tnr::i<DataVector, 2>& x_of_x_tilde,
             const size_t l_max) noexcept {
    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*evolution_gauge_j)), make_not_null(&get(*j)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);

    get(*evolution_gauge_j).data() =
        (0.25 * square(get(b).data())) * get(*evolution_gauge_j).data() +
        0.25 * square(get(a).data()) * conj(get(*evolution_gauge_j).data()) +
        0.5 * get(a).data() * get(b).data() *
            sqrt(1.0 + get(*evolution_gauge_j).data() *
                           conj(get(*evolution_gauge_j).data()));
  }
};

// beta is the same in both gauges. This still should have separate tags for the
// time being for compatibility with original evolution code.
template <>
ComputeGaugeAdjustedBoundaryValue<Tags::Beta> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::Beta>,
                                 Tags::BoundaryValue<Tags::Beta>>;
  using argument_tags = tmpl::list<Tags::CauchyAngularCoords, Tags::LMax>;
  void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_gauge_beta,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
      const tnr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    Spectral::Swsh::swsh_interpolate_from_pfaffian(
        make_not_null(&get(*evolution_gauge_beta)), make_not_null(&get(*beta)),
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max);
  }
};

template <>
ComputeGaugeAdjustedBoundaryValue<Tags::Q> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::Q>>;
  using argument_tags =
      tmpl::list<Tags::BoundaryValue<Tags::Dr<Tags::U>>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::J>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::R>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::Beta>, Tags::GaugeA,
                 Tags::GaugeB, Tags::CauchyAngularCoords, Tags::LMax>;

  template <typename... Args>
  void apply(const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
                 evolution_gauge_bondi_q,
             const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>,
             dr_u const Args&... args) noexcept {
    apply_impl(make_not_null(&get(*evolution_gauge_bondi_q)),
               make_not_null(&get(*dr_u)), ger(args)...);
  }
  void apply_impl(const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*>
                      evolution_gauge_bondi_q,
                  const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> dr_u,
                  const SpinWeighted<ComplexDataVector, 2>& j,
                  const SpinWeighted<ComplexDataVector, 0>& bondi_r,
                  const SpinWeighted<ComplexDataVector, 0>& beta,
                  const SpinWeighted<ComplexDataVector, 2>& a,
                  const SpinWeighted<ComplexDataVector, 0>& b,
                  const tnsr::i<DataVector, 2>& x_of_x_tilde,
                  const size_t l_max) noexcept {
    SpinWeighted<ComplexDataVector, 1> evolution_coords_dr_u =
        swsh_interpolate_from_pfaffian(dr_u, get<0>(x_of_x_tilde),
                                       get<1>(x_of_x_tilde), l_max);
    auto k = SpinWeighted<ComplexDataVector, 0>{
        sqrt(1.0 + j.data() * conj(j.data()))};
    *evolution_gauge_bondi_q = square(bondi_r) * exp(-2.0 * beta) * 0.5 *
                               (j * (conj(a) * evolution_coords_dr_u +
                                     conj(b) * conj(evolution_coords_dr_u)) +
                                    k * (a * conj(evolution_coords_dr_u) +
                                         b * evolution_coords_dr_u););
  }
};

// NOTE! this gets the boundary value for \f$\hat{U}\f$
template <>
ComputeGaugeAdjustedBoundaryValue<Tags::U> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::U>,
                                 Tags::BoundaryValue<Tags::U>>;
  using argument_tags =
      tmpl::list<Tags::GaugeA, Tags::GaugeB, Tags::CauchyAngularCoords>;

  void apply(const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
                 evolution_gauge_u,
             const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
             const tnsr::i<DataVector, 2>& x_of_x_tilde,
             const size_t l_max) noexcept {
    SpinWeighted<ComplexDataVector, 0> evolution_coords_u =
        swsh_interpolate_from_pfaffian(make_not_null(&get(*u)),
                                       get<0>(x_of_x_tilde),
                                       get<1>(x_of_x_tilde), l_max);
    get(*evolution_gauge) =
        0.5 * (get(a) * conj(evolution_coords_u) + get(b) * evolution_coords_u);
  }
};

template <>
ComputeGaugeAdjustedBoundaryValue<Tags::W> {
  using return_tags = tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::W>,
                                 Tags::BoundaryValue<Tags::W>>;
  using argument_tags = tmpl::list<Tags::CauchyAngularCoords, Tags::LMax>;

  void apply(const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
                 evolution_gauge_w,
             const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> w,
             const tnsr::i<DataVector, 2>& x_of_x_tilde,
             const size_t l_max) noexcept {
    Spectral::Swsh::swsh_interpolate_from_pfaffian(get(*evolution_gauge_w),
                                                   get(w), get<0>(x_of_x_tilde),
                                                   get<1>(x_of_x_tilde), l_max);
  }
};

template <>
ComputeGaugeAdjustedBoundaryValue<Tags::H> {
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;

  void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> h_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j_tilde,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> h,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u0,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    apply_impl(make_not_null(&get(*h_tilde)), make_not_null(&get(*j_tilde)),
               make_not_null(&get(*h)), make_not_null(&get(*j)),
               make_not_null(&get(*u0)), make_not_null(&get(*a)), get(b),
               x_of_x_tilde, l_max);
  }

  void apply_impl(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> h_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> j_tilde,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> h,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> j,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> u0,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> a,
      const SpinWeighted<ComplexDataVector, 0>& b,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    // optimization note: this has many allocations. They can be made fewer.
    // optimization note: this has several spin-weighted derivatives, they can
    // be aggregated
    SpinWeighted<ComplexDataVector, 1> a_u0bar_plus_b_u0 =
        (*a) * conj(u0) + (*b) * u0;
    auto eth_a_u0bar_plus_b_u0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&a_ubar_plus_b_u), l_max);
    auto ethbar_a_u0bar_plus_b_u0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&a_ubar_plus_b_u), l_max);

    auto du_a = -0.25 * (*a) * ethbar_a_u0bar_plus_b_u0 -
                0.25 * (*b) * eth_a_u0bar_plus_b_u0;
    auto du_b = -0.25 * (*b) * ethbar_a_u0bar_plus_b_u0 -
                0.25 * (*a) * eth_a_u0bar_plus_b_u0;

    SpinWeighted<ComplexDataVector, 2> j_of_x_tilde =
        swsh_interpolate_from_pfaffian(make_not_null(&get(*j)),
                                       get<0>(x_of_x_tilde),
                                       get<1>(x_of_x_tilde), l_max);
    SpinWeighted<ComplexDataVector, 2> h_of_x_tilde =
        swsh_interpolate_from_pfaffian(make_not_null(&get(*h)),
                                       get<0>(x_of_x_tilde),
                                       get<1>(x_of_x_tilde), l_max);

    // this manipulation is needed because spin weight 3 or higher is not
    // supported
    SpinWeighted<ComplexDataVector, 1> abar_j_tilde = conj(*a) * (*j_tilde);
    SpinWeighted<ComplexDataVector, 1> u0bar_j_tilde = conj(*u0) * (*j_tilde);
    auto abar_eth_j_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&abar_j_tilde), l_max) -
        (*j_tilde) *
            conj(Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                a, l_max));
    auto u0bar_eth_j_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&u0bar_j_tilde), l_max) -
        (*j_tilde) *
            conj(Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                u0, l_max));
    auto ethbar_j_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(j_tilde,
                                                                      l_max);

    SpinWeighted<ComplexDataVector, 0> k =
        sqrt(1.0 + j_of_x_tilde * conj(j_of_x_tilde));
    auto angular_derivative_part = -0.25 * (*u0) * b * ethbar_j_tilde -
                                   0.25 * (*u0) * abar_eth_j_tilde -
                                   0.25 * conj(*u0) * a * ethbar_j_tilde -
                                   0.25 * conj(b) * u0bar_eth_j_tilde;
    auto time_derivative_part =
        0.25 * (square(b) * h_of_x_tilde + square(*a) * conj(h_of_x_tilde) -
                (*a) * b *
                    (j_of_x_tilde * conj(h_of_x_tilde) +
                     h_of_x_tilde * conj(j_of_x_tilde)) /
                    (1.0 + j_of_x_tilde * conj(j_of_x_tilde))) +
        0.5 * b * du_b * j_of_x_tilde + 0.5 * (*a) * du_a * conj(j_of_x_tilde) -
        0.5 * ((*a) * du_b + b * du_a) * k;
    *h_tilde = angular_derivative_part + time_derivative_part;
  }
};

struct GaugeUpdateU {
  using argument_tags = tmpl::list<Tags::GaugeA, Tags::GaugeB, Tags::LMax>;
  using return_tags =
      tmpl::list<Tags::Du<Tags::CauchyAngularCoords>, Tags::U0, Tags::U>;

  void apply(
      const gsl::not_null<tnsr::i<DataVector, 2>*> du_x,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u_0,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> u,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& a,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& b,
      const size_t l_max) noexcept {
    size_t number_of_radial_points =
        get(*u).size() /
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    // u_hat_0
    get(*u_0).data() = ComplexDataVector{
        get(*u).data().data(),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    // subtract u_hat_0 from u
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector angular_view{
          get(*u).data() +
              i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
      angular_view -= get(*u_0).data();
    }
    // u_0
    get(*u_0) = (2.0 / (get(b) * conj(get(b)) - get(a) * conj(get(a)))) *
                (conj(get(b)) * get(*u_0) - get(a) * conj(get(*u_0)));

    // Note we store x^\phi * sin(theta) rather than x^\phi to reduce the number
    // of places sin(theta)'s appear. Unfortunately, without reformulating the
    // spin-weighted spherical harmonics themselves, we won't be able to
    // eliminate sin(theta)s completely
    get<0>(*du_x) = -real(u_0);
    get<1>(*du_x) = -imag(u_0);
  }
};

// inversion from the coordinate values \f$x(\tilde{x})\f to the \f$a\f$ and
// \f$b\f$ which give easy expressions for the spin-weighted quantities.
struct GaugeUpdateAB {
  using argument_tags<Tags::CauchyAngularCoords, Tags::LMax>;
  using return_tags<Tags::GaugeA, Tags::GaugeB>;

  void apply(const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> a,
             const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> b,
             const tnsr::i<DataVector, 2>& x_of_x_tilde,
             const size_t l_max) noexcept {
    size_t size = get<0>(x_of_x_tilde).size();
    tnsr::ii<DataVector, 2> d_tilde_x{size};
    SpinWeighted<ComplexDataVector, 0> derivative_buffer{size};
    SpinWeighted<ComplexDataVector, 1> eth_buffer{size};

    derivative_buffer = std::complex<double>(1.0, 0.0) * get<0>(x_of_x_tilde);
    swsh_derivative(make_not_null(&eth_buffer),
                    make_not_null(&derivative_buffer), l_max);
    // \partial_{\tilde{\theta}} x^\theta
    get<0, 0>(d_tilde_x) = -real(eth_buffer);
    // (1/sin(\tilde{\theta})) \partial_{\tilde{phi}} x^theta
    get<1, 0>(d_tilde_x) = -imag(eth_buffer);

    derivative_buffer = std::complex<double>(1.0, 0.0) * get<1>(x_of_x_tilde);
    swsh_derivative(make_not_null(&eth_buffer),
                    make_not_null(&derivative_buffer), l_max);

    // \sin(\theta) \partial_{\tilde{\theta}} x^\phi
    get<0, 1>(d_tilde_x) = -real(eth_buffer) - get<1>(x_of_x_tilde) *
                                                   cos(get<0>(x_of_x_tilde)) *
                                                   get<0, 0>(d_tilde_x);
    // (sin(\theta)/sin(\tilde{\theta})) \partial_{\tilde{phi}} x^\phi
    d_tilde_x.get(1, 1) = -imag(eth_buffer) - get<1>(x_of_x_tilde) *
                                                  cos(get<0>(x_of_x_tilde)) *
                                                  get<1, 0>(d_tilde_x);

    // note no sin factors from the q's, they cancel the implicit factors from
    // above
    // note no interpolation is required in this process, as each of the above
    // tensor quantities is on the \tilde{x} grid
    auto d_x_tilde = determinant_and_inverse(d_tilde_x).second;
    get(*a).data() = std::complex<double>(1.0, 0.0) *
                         (get<0, 0>(d_tilde_x) - get<1, 1>(d_tilde_x)) +
                     std::complex<double>(0.0, 1.0) *
                         (get<1, 0>(d_tilde_x) + get<0, 1>(d_tilde_x));
    get(*b).data() = std::complex<double>(1.0, 0.0) *
                         (get<0, 0>(d_tilde_x) + get<1, 1>(d_tilde_x)) +
                     std::complex<double>(0.0, 1.0) *
                         (-get<1, 0>(d_tilde_x) + get<0, 1>(d_tilde_x));
  }
};
