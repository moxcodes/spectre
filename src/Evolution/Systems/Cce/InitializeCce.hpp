// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>

#include "DataStructures/Matrix.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
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
}  // namespace Cce
