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
 */
template <typename Tag>
struct PrecomputeCceDependencies;

/// Computes \f$1 - y\f$ for the Cce system.
template <>
struct PrecomputeCceDependencies<Tags::OneMinusY> {
  using boundary_tags = tmpl::list<>;
  using pre_swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::OneMinusY>;
  using argument_tags = tmpl::list<Spectral::Swsh::Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          one_minus_y,
      const size_t l_max) noexcept;
};

/// Computes \f$R\f$ from its boundary value (by repeating it over the radial
/// dimension)
template <>
struct PrecomputeCceDependencies<Tags::BondiR> {
  using boundary_tags = tmpl::list<Tags::BoundaryValue<Tags::BondiR>>;
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
template <>
struct PrecomputeCceDependencies<Tags::DuRDividedByR> {
  using boundary_tags = tmpl::list<Tags::BoundaryValue<Tags::DuRDividedByR>>;
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
template <>
struct PrecomputeCceDependencies<Tags::EthRDividedByR> {
  using boundary_tags = tmpl::list<Tags::BoundaryValue<Tags::BondiR>>;
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
template <>
struct PrecomputeCceDependencies<Tags::EthEthRDividedByR> {
  using boundary_tags = tmpl::list<Tags::BoundaryValue<Tags::BondiR>>;
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
template <>
struct PrecomputeCceDependencies<Tags::EthEthbarRDividedByR> {
  using boundary_tags = tmpl::list<Tags::BoundaryValue<Tags::BondiR>>;
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
template <>
struct PrecomputeCceDependencies<Tags::BondiK> {
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
 */
template <typename DataBoxType>
void mutate_all_precompute_cce_dependencies(
    const gsl::not_null<DataBoxType*> box) noexcept {
  tmpl::for_each<pre_computation_tags>([&box](auto x) {
    using integration_independent_tag = typename decltype(x)::type;
    using mutation = PrecomputeCceDependencies<integration_independent_tag>;
    db::mutate_apply<mutation>(box);
  });
}
}  // namespace Cce
