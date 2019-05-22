// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/LinearOperators.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/VectorAlgebra.hpp"

namespace Cce {

namespace detail {
// A convenience function for computing the spin-weighted derivatives of \f$R\f$
// divided by \f$R\f$, which appears often in Jacobians to transform between
// Bondi coordinates and the numerical coordinates used in CCE.
template <typename DerivKind>
void angular_derivative_of_r_divided_by_r_impl(
    const gsl::not_null<
        SpinWeighted<ComplexDataVector,
                     Spectral::Swsh::Tags::derivative_spin_weight<DerivKind>>*>
        d_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& boundary_r,
    const size_t l_max) noexcept;

}  // namespace detail

/*!
 * \brief A set of procedures for computing the set of inputs to the CCE
 * integrand computations that can be computed before any of the intermediate
 * integrands are evaluated. The template specializations of this template are
 * compatible with acting as a the mutator in a `DataBox` mutation.
 *
 * \details For the storage model in which a set of `Variables` are stored in a
 * `DataBox`:
 * - output/input : `integration_independent` with tags `pre_computation_tags`
 * - input : `boundary_values` with at least tags
 *   `pre_computation_boundary_tags`
 * - input : `pre_swsh_derivatives` with at least `Tags::J`.
 *
 * The `BoundaryPrefix` tag allows easy switching between the
 * regularity-preserving version and standard CCE
 *
 */
template <template <typename> class BoundaryPrefix, typename Tag>
struct PrecomputeCceDependencies;


/// Computes \f$1 - y\f$ for the Cce system.
template <template <typename> class BoundaryPrefix>
struct PrecomputeCceDependencies<BoundaryPrefix, Tags::OneMinusY> {
  using boundary_tags = tmpl::list<>;
  using pre_swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::OneMinusY>;
  using argument_tags = tmpl::list<Spectral::Swsh::Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          one_minus_y,
      const size_t l_max) noexcept {
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points =
        get(one_minus_y)->size() / number_of_angular_points;
    const auto& one_minus_y_collocation =
        1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto>(
                  number_of_radial_points);
    // iterate through the angular 'chunks' and set them to their 1-y value
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector angular_view{
          get(*one_minus_y).data().data() + number_of_angular_points * i,
          number_of_angular_points};
      angular_view = one_minus_y_collocation[i];
    }
  }
};

/// Computes \f$R\f$ from its boundary value (by repeating it over the radial
/// dimension)
template <template <typename> class BoundaryPrefix>
struct PrecomputeCceDependencies<BoundaryPrefix, Tags::BondiR> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::BondiR>>;
  using pre_swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::BondiR>;
  using argument_tags =
      tmpl::append<tmpl::list<Spectral::Swsh::Tags::LMax>, boundary_tags>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> r,
      const size_t l_max,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r) noexcept {
    size_t number_of_radial_points =
        get(*r).size() /
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    repeat(make_not_null(&get(*r).data()), get(boundary_r).data(),
           number_of_radial_points);
  }
};

/// Computes \f$\partial_u R / R\f$ from its boundary value (by repeating it
/// over the radial dimension).
template <template <typename> class BoundaryPrefix>
struct PrecomputeCceDependencies<BoundaryPrefix, Tags::DuRDividedByR> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::DuRDividedByR>>;
  using pre_swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::DuRDividedByR>;
  using argument_tags =
      tmpl::append<tmpl::list<Spectral::Swsh::Tags::LMax>, boundary_tags>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          du_r_divided_by_r,
      const size_t l_max,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&
          boundary_du_r_divided_by_r) noexcept {
    size_t number_of_radial_points =
        get(*du_r_divided_by_r).size() /
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    repeat(make_not_null(&get(*du_r_divided_by_r).data()),
           get(boundary_du_r_divided_by_r).data(), number_of_radial_points);
  }
};

/// Computes \f$\eth R / R\f$ by differentiating and repeating the boundary
/// value of \f$R\f$.
template <template <typename> class BoundaryPrefix>
struct PrecomputeCceDependencies<BoundaryPrefix, Tags::EthRDividedByR> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::BondiR>>;
  using pre_swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::EthRDividedByR>;
  using argument_tags =
      tmpl::append<tmpl::list<Spectral::Swsh::Tags::LMax>, boundary_tags>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          eth_r_divided_by_r,
      const size_t l_max,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r) noexcept {
    detail::angular_derivative_of_r_divided_by_r_impl<
        Spectral::Swsh::Tags::Eth>(make_not_null(&get(*eth_r_divided_by_r)),
                                   get(boundary_r), l_max);
  }
};

/// Computes \f$\eth \eth R / R\f$ by differentiating and repeating the boundary
/// value of \f$R\f$.
template <template <typename> class BoundaryPrefix>
struct PrecomputeCceDependencies<BoundaryPrefix, Tags::EthEthRDividedByR> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::BondiR>>;
  using pre_swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::EthEthRDividedByR>;
  using argument_tags =
      tmpl::append<tmpl::list<Spectral::Swsh::Tags::LMax>, boundary_tags>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          eth_eth_r_divided_by_r,
      const size_t l_max,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r) noexcept {
    detail::angular_derivative_of_r_divided_by_r_impl<
        Spectral::Swsh::Tags::EthEth>(
        make_not_null(&get(*eth_eth_r_divided_by_r)), get(boundary_r), l_max);
  }
};

/// Computes \f$\eth \bar{\eth} R / R\f$ by differentiating and repeating the
/// boundary value of \f$R\f$.
template <template <typename> class BoundaryPrefix>
struct PrecomputeCceDependencies<BoundaryPrefix, Tags::EthEthbarRDividedByR> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::BondiR>>;
  using pre_swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::EthEthbarRDividedByR>;
  using argument_tags =
      tmpl::append<tmpl::list<Spectral::Swsh::Tags::LMax>, boundary_tags>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          eth_ethbar_r_divided_by_r,
      const size_t l_max,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r) noexcept {
    detail::angular_derivative_of_r_divided_by_r_impl<
        Spectral::Swsh::Tags::EthEthbar>(
        make_not_null(&get(*eth_ethbar_r_divided_by_r)), get(boundary_r),
        l_max);
  }
};

/// Computes \f$K = \sqrt{1 + J \bar{J}}\f$.
template <template <typename> class BoundaryPrefix>
struct PrecomputeCceDependencies<BoundaryPrefix, Tags::BondiK> {
  using boundary_tags = tmpl::list<>;
  using pre_swsh_derivative_tags = tmpl::list<Tags::BondiJ>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::BondiK>;
  using argument_tags = tmpl::append<tmpl::list<Spectral::Swsh::Tags::LMax>,
                                     pre_swsh_derivative_tags>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> k,
      const size_t /*l_max*/,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j) noexcept {
    get(*k).data() = sqrt(1.0 + get(j).data() * conj(get(j)).data());
  }
};

/*!
 * \brief Convenience routine for computing all of the CCE inputs to integrand
 * computation which do not depend on intermediate integrand results, to be
 * executed before moving through the hierarchy of integrands.
 *
 * \details Provided a `DataBox` with the appropriate tags (including
 * `pre_computation_boundary_tags`, `pre_computation_tags`, `Tags::J` and
 * `Spectral::Swsh::Tags::LMax`), this function will apply all of the necessary
 * mutations to update the `pre_computation_tags` to their correct values for
 * the current values for the remaining (input) tags.
 *
 * The `BoundaryPrefix` tag allows easy switching between the
 * regularity-preserving version and standard CCE
 */
template <template <typename> class BoundaryPrefix, typename DataBoxType>
void mutate_all_precompute_cce_dependencies(
    const gsl::not_null<DataBoxType*> box) noexcept {
  tmpl::for_each<pre_computation_tags>([&box](auto x) {
    using integration_independent_tag = typename decltype(x)::type;
    using mutation =
        PrecomputeCceDependencies<BoundaryPrefix, integration_independent_tag>;
    db::mutate_apply<mutation>(box);
  });
}
}  // namespace Cce
