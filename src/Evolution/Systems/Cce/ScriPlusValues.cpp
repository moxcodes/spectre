// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/ScriPlusValues.hpp"

#include "Evolution/Systems/Cce/ComputePreSwshDerivatives.hpp"

namespace Cce {

void calculate_inertial_h(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> h_tilde,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> h,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> j,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> j_tilde,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> u_hat_0,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> dy_j_tilde,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> r_tilde_boundary,
    const SpinWeighted<ComplexDataVector, 0>& omega_cd,
    const SpinWeighted<ComplexDataVector, 0>& du_omega_cd,
    const SpinWeighted<ComplexDataVector, 1>& eth_omega_cd,
    const SpinWeighted<ComplexDataVector, 2>& c,
    const SpinWeighted<ComplexDataVector, 0>& d,
    const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points = j->size() / number_of_angular_points;

  const auto& one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);

  SpinWeighted<ComplexDataVector, 2> interpolated_h_slice{
      number_of_angular_points};
  SpinWeighted<ComplexDataVector, 2> interpolated_j_slice{
      number_of_angular_points};
  SpinWeighted<ComplexDataVector, 2> h_slice{number_of_angular_points};
  for (size_t i = 0; i < number_of_radial_points; ++i) {
    h_slice.data() =
        ComplexDataVector{h->data().data() + i * number_of_angular_points,
                          number_of_angular_points};
    SpinWeighted<ComplexDataVector, 2> j_view;
    j_view.data() =
        ComplexDataVector{j->data().data() + i * number_of_angular_points,
                          number_of_angular_points};
    SpinWeighted<ComplexDataVector, 2> j_tilde_view;
    j_tilde_view.data() =
        ComplexDataVector{j_tilde->data().data() + i * number_of_angular_points,
                          number_of_angular_points};
    SpinWeighted<ComplexDataVector, 2> dy_j_tilde_view;
    dy_j_tilde_view.data() = ComplexDataVector{
        dy_j_tilde->data().data() + i * number_of_angular_points,
        number_of_angular_points};
    ComplexDataVector h_tilde_view{
        h_tilde->data().data() + i * number_of_angular_points,
        number_of_angular_points};

    auto eth_r_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            r_tilde_boundary, l_max);

    SpinWeighted<ComplexDataVector, 1> u_0_bar_j_tilde =
        conj(*u_hat_0) * j_tilde_view;

    SpinWeighted<ComplexDataVector, 2> u_0_bar_eth_j_tilde =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&u_0_bar_j_tilde), l_max) -
        j_tilde_view *
            conj(Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                u_hat_0, l_max)) -
        conj(*u_hat_0) * eth_r_tilde * square(one_minus_y_collocation[i]) /
            (2.0 * (*r_tilde_boundary)) * dy_j_tilde_view;

    SpinWeighted<ComplexDataVector, 2> angular_derivative_term =
        0.5 *
        ((*u_hat_0) *
             Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(
                 make_not_null(&j_tilde_view), l_max) -
         (*u_hat_0) * conj(eth_r_tilde) * square(one_minus_y_collocation[i]) /
             (2.0 * (*r_tilde_boundary)) * dy_j_tilde_view +
         u_0_bar_eth_j_tilde);

    SpinWeighted<ComplexDataVector, 0> cauchy_du_omega_cd =
        du_omega_cd -
        0.5 * ((*u_hat_0) * conj(eth_omega_cd) + conj(*u_hat_0) * eth_omega_cd);

    Spectral::Swsh::SwshInterpolator interpolator{
        get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), 2, l_max};
    interpolator.interpolate(
        make_not_null(&interpolated_h_slice.data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(make_not_null(&h_slice), l_max),
            l_max)
            .data());
    interpolator.interpolate(
        make_not_null(&interpolated_j_slice.data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(make_not_null(&j_view), l_max),
            l_max)
            .data());

    SpinWeighted<ComplexDataVector, 0> k_tilde;
    k_tilde.data() =
        sqrt(1.0 + j_tilde_view.data() * conj(j_tilde_view.data()));
    SpinWeighted<ComplexDataVector, 0> interpolated_k;
    interpolated_k.data() = sqrt(1.0 + interpolated_j_slice.data() *
                                           conj(interpolated_j_slice.data()));

    auto eth_u_0 = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        u_hat_0, l_max);
    auto ethbar_u_0 =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Ethbar>(u_hat_0,
                                                                      l_max);

    SpinWeighted<ComplexDataVector, 2> computed_h_tilde =
        angular_derivative_term -
        one_minus_y_collocation[i] * cauchy_du_omega_cd / omega_cd *
            dy_j_tilde_view +
        2.0 * cauchy_du_omega_cd / omega_cd * j_tilde_view -
        ethbar_u_0 * j_tilde_view + eth_u_0 * k_tilde +
        0.25 / square(omega_cd) *
            (square(conj(d)) * interpolated_h_slice +
             square(c) * conj(interpolated_h_slice) +
             c * conj(d) *
                 (interpolated_h_slice * conj(interpolated_j_slice) +
                  conj(interpolated_h_slice) * interpolated_j_slice) /
                 interpolated_k);
    h_tilde_view = computed_h_tilde.data();
  }
}

// As a diagnostic computation, this is over-thorough and rather expensive
// We construct the coordinate transforms such that:
// - j is in the source gauge (untilded)
// - beta is in the source gauge (untilded)
// - scri u is in the source gauge (untilded)
// - boundary_r is in the source gauge (untilded)
// - gauge_c is in the target gauge (tilded)
// - gauge_d is in the target gauge (tilded)
// - omega_cd is in the target gauge (tilded)
// - du_omega_cd is in the target gauge (tilded)
// - eth_omega_cd is in the target gauge (tilded)
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
    const bool interpolate_back) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points =
      get(*j).size() / number_of_angular_points;

  // j_tilde
  Scalar<SpinWeighted<ComplexDataVector, 2>> j_tilde{number_of_angular_points *
                                                     number_of_radial_points};
  // This function works just as well in reverse
  CalculateCauchyGauge<Tags::CauchyGauge<Tags::BondiJ>>::apply(
      make_not_null(&j_tilde), j, gauge_c, gauge_d, omega_cd, x_of_x_tilde,
      l_max);
  SpinWeighted<ComplexDataVector, 2> scri_j_tilde{number_of_angular_points};
  scri_j_tilde.data() = ComplexDataVector{
      get(j_tilde).data().data() +
          (number_of_radial_points - 1) * number_of_angular_points,
      number_of_angular_points};

  // dy_j_tilde
  Scalar<SpinWeighted<ComplexDataVector, 2>> dy_j_tilde{
      number_of_angular_points * number_of_radial_points};
  ComputePreSwshDerivatives<Tags::Dy<Tags::BondiJ>>::apply(
      make_not_null(&dy_j_tilde), j_tilde, l_max);

  // scri_u_hat
  // scri U version is simpler than the volume form.
  SpinWeighted<ComplexDataVector, 1> scri_u;
  scri_u.data() =
      ComplexDataVector{get(*u).data().data() + (number_of_radial_points - 1) *
                                                    number_of_angular_points,
                        number_of_angular_points};
  SpinWeighted<ComplexDataVector, 1> scri_u_hat{number_of_angular_points};
  Spectral::Swsh::SwshInterpolator u_interpolator{
      get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), 1, l_max};
  u_interpolator.interpolate(
      make_not_null(&scri_u_hat.data()),
      Spectral::Swsh::libsharp_to_goldberg_modes(
          Spectral::Swsh::swsh_transform(make_not_null(&scri_u), l_max), l_max)
          .data());
  scri_u_hat =
      0.5 * (scri_u_hat * conj(get(gauge_d)) - conj(scri_u_hat) * get(gauge_c));
  // boundary_r_tilde
  SpinWeighted<ComplexDataVector, 0> boundary_r_tilde{number_of_angular_points};
  Spectral::Swsh::SwshInterpolator interpolator{get<0>(x_of_x_tilde),
                                                get<1>(x_of_x_tilde), 0, l_max};
  interpolator.interpolate(make_not_null(&boundary_r_tilde.data()),
                           Spectral::Swsh::libsharp_to_goldberg_modes(
                               Spectral::Swsh::swsh_transform(
                                   make_not_null(&get(*boundary_r)), l_max),
                               l_max)
                               .data());
  boundary_r_tilde = boundary_r_tilde * get(omega_cd);

  // h_tilde
  Scalar<SpinWeighted<ComplexDataVector, 2>> h_tilde{number_of_angular_points *
                                                     number_of_radial_points};
  calculate_inertial_h(
      make_not_null(&get(h_tilde)), make_not_null(&get(*du_j)),
      make_not_null(&get(*j)), make_not_null(&get(j_tilde)),
      make_not_null(&scri_u_hat), make_not_null(&get(dy_j_tilde)),
      make_not_null(&boundary_r_tilde), get(omega_cd), get(du_omega_cd),
      get(eth_omega_cd), get(gauge_c), get(gauge_d), x_of_x_tilde, l_max);

  SpinWeighted<ComplexDataVector, 0> scri_beta;
  scri_beta.data() = ComplexDataVector{
      get(*beta).data().data() +
          (number_of_radial_points - 1) * number_of_angular_points,
      number_of_angular_points};
  SpinWeighted<ComplexDataVector, 0> scri_beta_tilde;
  interpolator.interpolate(
      make_not_null(&scri_beta_tilde.data()),
      Spectral::Swsh::libsharp_to_goldberg_modes(
          Spectral::Swsh::swsh_transform(make_not_null(&scri_beta), l_max),
          l_max)
          .data());
  scri_beta_tilde -= 0.5 * log(get(omega_cd).data());

  auto eth_beta_tilde_at_scri =
      Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
          make_not_null(&scri_beta_tilde), l_max);
  auto eth_eth_beta_tilde_at_scri =
      Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::EthEth>(
          make_not_null(&scri_beta_tilde), l_max);

  Scalar<SpinWeighted<ComplexDataVector, 2>> dy_h_tilde{get(h_tilde).size()};

  ComputePreSwshDerivatives<Tags::Dy<Tags::BondiH>>::apply(
      make_not_null(&dy_h_tilde), h_tilde, l_max);
  auto dy_h_at_scri = ComplexDataVector{
      get(dy_h_tilde).data().data() +
          (number_of_radial_points - 1) * number_of_angular_points,
      number_of_angular_points};

  SpinWeighted<ComplexDataVector, 2> news_in_inertial_frame;
  news_in_inertial_frame.data() =
      2.0 * (-boundary_r_tilde.data() * exp(-2.0 * scri_beta_tilde.data()) *
                 dy_h_at_scri +
             eth_eth_beta_tilde_at_scri.data() +
             2.0 * square(eth_beta_tilde_at_scri.data()));

  if (interpolate_back) {
    Spectral::Swsh::SwshInterpolator news_interpolator{
        get<0>(x_tilde_of_x), get<1>(x_tilde_of_x), 2, l_max};
    news_interpolator.interpolate(
        make_not_null(&get(*news).data()),
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(
                make_not_null(&news_in_inertial_frame), l_max),
            l_max)
            .data());
  } else {
    get(*news) = news_in_inertial_frame;
  }
}
}  // namespace Cce
