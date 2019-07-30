// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>

#include "DataStructures/Matrix.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {

/*!
 * \brief Initialize \f$J\f$ on the first hypersurface from provided boundary
 * values of \f$J\f$, \f$R\f$, and \f$\partial_r J\f$.
 *
 * \details This initial data is chosen to take the function:
 *
 *\f[ J = \frac{A}{r} + \frac{B}{r^3},
 *\f]
 *
 * where
 *
 * \f{align*}{
 * A = R \left( \frac{3}{2} J|_{r = R} + \frac{1}{2} R \partial_r J|_{r =
 * R}\right) \notag\\
 * B = - \frac{1}{2} R^3 (J|_{r = R} + R \partial_r J|_{r = R})
 * \f}
 */
template <template <typename> class BoundaryPrefix>
struct InitializeJ {
  using boundary_tags = tmpl::list<Tags::BoundaryValue<Tags::BondiJ>,
                                   Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
                                   Tags::BoundaryValue<Tags::BondiR>>;

  using return_tags = tmpl::list<Tags::BondiJ>;
  using argument_tags = tmpl::append<boundary_tags>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r) noexcept {
    const size_t number_of_radial_points =
        get(*j).size() / get(boundary_j).size();

    const auto& one_minus_y_collocation =
        1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto>(
                  number_of_radial_points);

    for (size_t i = 0; i < number_of_radial_points; i++) {
      ComplexDataVector angular_view_j{
          get(*j).data().data() + get(boundary_j).size() * i,
          get(boundary_j).size()};
      const auto one_minus_y_coefficient =
          0.25 * (3.0 * get(boundary_j).data() +
                  get(r).data() * get(boundary_dr_j).data());
      const auto one_minus_y_cubed_coefficient =
          -0.0625 *
          (get(boundary_j).data() + get(r).data() * get(boundary_dr_j).data());
      angular_view_j =
          one_minus_y_collocation[i] * one_minus_y_coefficient +
          pow<3>(one_minus_y_collocation[i]) * one_minus_y_cubed_coefficient;
    }
  }
};

struct GaugeAdjustInitialJ {
  using boundary_tags =
      tmpl::list<Tags::GaugeC, Tags::GaugeD, Tags::GaugeOmegaCD,
                 Tags::CauchyAngularCoords, Tags::LMax>;
  using return_tags = tmpl::list<Tags::BondiJ>;
  using argument_tags = tmpl::append<boundary_tags>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
      const tnsr::i<DataVector, 2>& x_of_x_tilde, const size_t l_max) noexcept {
    const size_t number_of_radial_points = get(*j).size() / get(c).size();

    for (size_t i = 0; i < number_of_radial_points; i++) {
      // TODO rewrite
      ComplexDataVector angular_view_j{
          get(*j).data().data() + get(c).size() * i, get(c).size()};
      // TODO it's probably better to use the volume filtering to combine
      // evaluations of the swsh transform
      SpinWeighted<ComplexDataVector, 2> j_view;
      j_view.data() = angular_view_j;
      Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&j_view),
                                                    l_max, l_max - 4);
      // TODO review all filtering to double-check their necessity

      SpinWeighted<ComplexDataVector, 2> evolution_coords_j_view =
          Spectral::Swsh::swsh_interpolate(make_not_null(&j_view),
                                           get<0>(x_of_x_tilde),
                                           get<1>(x_of_x_tilde), l_max);
      // finish adjusting the gauge in place
      // ComplexDataVector cd_omega =
          // 0.5 * sqrt(get(d).data() * conj(get(d).data()) +
                     // get(c).data() * conj(get(c).data()));

      evolution_coords_j_view =
          0.25 * (square(conj(get(d).data())) * evolution_coords_j_view.data() +
                  square(get(c).data()) * conj(evolution_coords_j_view.data()) +
                  2.0 * conj(get(d).data()) * get(c).data() *
                      sqrt(1.0 + evolution_coords_j_view.data() *
                                     conj(evolution_coords_j_view.data())));
      Spectral::Swsh::filter_swsh_boundary_quantity(
          make_not_null(&evolution_coords_j_view), l_max, l_max - 4);
      angular_view_j =
          evolution_coords_j_view.data() / square(get(omega_cd).data());

      // TEST
      // SpinWeighted<ComplexDataVector, 2> identity_test_inertial_j;
      // identity_test_inertial_j.data() = angular_view_j;

      // SpinWeighted<ComplexDataVector, 2> cauchy_coords_j_view =
          // Spectral::Swsh::swsh_interpolate(
              // make_not_null(&identity_test_inertial_j), get<0>(x_tilde_of_x),
              // get<1>(x_tilde_of_x), l_max);

      // SpinWeighted<ComplexDataVector, 2> identity_test_cauchy_j;
      // identity_test_cauchy_j.data() =
          // 0.25 *
          // (square(conj(get(b).data())) * cauchy_coords_j_view.data() +
           // square(get(a).data()) * conj(cauchy_coords_j_view.data()) +
           // 2.0 * get(a).data() * conj(get(b).data()) *
               // sqrt(1.0 + cauchy_coords_j_view.data() *
                              // conj(cauchy_coords_j_view.data())));
      // Spectral::Swsh::filter_swsh_boundary_quantity(
          // make_not_null(&identity_test_cauchy_j), l_max, l_max - 4);
      // identity_test_cauchy_j.data() /= square(get(omega).data());

      // printf("Identity test: J transformation\n");
      // for (size_t i = 0; i < identity_test_cauchy_j.size(); ++i) {
        // printf("(%e, %e) from (%e, %e)\n",
               // real(identity_test_cauchy_j.data()[i] - j_view.data()[i]),
               // imag(identity_test_cauchy_j.data()[i] - j_view.data()[i]),
               // real(identity_test_cauchy_j.data()[i]),
               // imag(identity_test_cauchy_j.data()[i]));
      // }
      // printf("done\n");
      // TEST

      // for(size_t i = 0; i < angular_view_j.size(); ++i) {
      // printf(
      // "(%e, %e)\n",
      // real(evolution_coords_j_view.data()[i] - angular_slice_j.data()[i]),
      // imag(evolution_coords_j_view.data()[i] -
      // angular_slice_j.data()[i]));
      // }

      // TEST

      // TEST: just interpolate
      // angular_view_j = evolution_coords_j_view.data();
      // TEST
      // angular_view_j = 0.25 *
      // (square(get(b).data()) * angular_slice_j.data() +
      // square(get(a).data()) * conj(angular_slice_j.data()) -
      // 2.0 * get(a).data() * get(b).data() *
      // sqrt(1.0 + angular_slice_j.data() *
      // conj(angular_slice_j.data()))) /
      // square(get(omega).data());
    }
    Spectral::Swsh::filter_swsh_volume_quantity(make_not_null(&get(*j)), l_max,
                                                l_max - 4, 0.0, 8);
  }
};

}  // namespace Cce
