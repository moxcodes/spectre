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

namespace Cce {
// Precomputation routines for the quantities stored in the `BondiAndSWCache`
// Variables. A set of these will need to be evaluated for each of the Bondi
// variables which will be evolved,and they evaluate both the set of
// quantities which are used in the hypersurface integrands and the quantities
// needed to be cached before angular derivatives can be applied

/*!
 * \brief A set of procedures for computing the set of inputs to the CCE
 * integrand computations that are to be performed prior to the spin-weighted
 * spherical harmonic differentiation (and for the first step in the series of
 * integrations, after the `PrecomputeCceDependencies`)
 *
 * \details For the storage model in which a set of `Variables` are stored in a
 * `DataBox`:
 * - output/input : `pre_swsh_derivatives` with tags
 * `all_pre_swsh_derivative_tags`
 * - input : `swsh_derivative_tags` with at least tags
 *   `all_swsh_derivative_tags`
 */
template <typename Tag>
struct ComputePreSwshDerivatives;

/// Compute \f$\bar{J}\f$.
/// \note This should be unnecessary in most execution procedures, as all
/// quantities should be derived from \f$J\f$ and its derivatives followed by
/// explicit conjugation operations, which are expected to be sufficiently cheap
/// to avoid the storage cost of recording the conjugates
template <>
struct ComputePreSwshDerivatives<Tags::BondiJbar> {
  using pre_swsh_derivative_tags = tmpl::list<Tags::BondiJ>;
  using swsh_derivative_tags = tmpl::list<>;
  using integrand_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::BondiJbar>;
  using argument_tags = pre_swsh_derivative_tags;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> jbar,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j) noexcept {
    get(*jbar) = conj(get(j));
  }
};

/// Compute \f$\bar{U}\f$.
/// \note This should be unnecessary in most execution procedures, as all
/// quantities should be derived from \f$U\f$ and its derivatives followed by
/// explicit conjugation operations, which are expected to be sufficiently cheap
/// to avoid the storage cost of recording the conjugates
template <>
struct ComputePreSwshDerivatives<Tags::BondiUbar> {
  using pre_swsh_derivative_tags = tmpl::list<Tags::BondiU>;
  using swsh_derivative_tags = tmpl::list<>;
  using integrand_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::BondiUbar>;
  using argument_tags = pre_swsh_derivative_tags;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*> ubar,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& u) noexcept {
    get(*ubar) = conj(get(u));
  }
};

/// Compute \f$\bar{Q}\f$.
/// \note This should be unnecessary in most execution procedures, as all
/// quantities should be derived from \f$Q\f$ and its derivatives followed by
/// explicit conjugation operations, which are expected to be sufficiently cheap
/// to avoid the storage cost of recording the conjugates
template <>
struct ComputePreSwshDerivatives<Tags::BondiQbar> {
  using pre_swsh_derivative_tags = tmpl::list<Tags::BondiQ>;
  using swsh_derivative_tags = tmpl::list<>;
  using integrand_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::BondiQbar>;
  using argument_tags = pre_swsh_derivative_tags;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*> qbar,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& q) noexcept {
    get(*qbar) = conj(get(q));
  }
};

/// Compute the product of `Lhs` and `Rhs`.
template <typename Lhs, typename Rhs>
struct ComputePreSwshDerivatives<::Tags::Multiplies<Lhs, Rhs>> {
  using pre_swsh_derivative_tags = tmpl::list<Lhs, Rhs>;
  using swsh_derivative_tags = tmpl::list<>;
  using integrand_tags = tmpl::list<>;

  using return_tags = tmpl::list<::Tags::Multiplies<Lhs, Rhs>>;
  using argument_tags = pre_swsh_derivative_tags;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<
          ComplexDataVector, ::Tags::Multiplies<Lhs, Rhs>::type::type::spin>>*>
          result,
      const Scalar<SpinWeighted<ComplexDataVector, Lhs::type::type::spin>>& lhs,
      const Scalar<SpinWeighted<ComplexDataVector, Rhs::type::type::spin>>&
          rhs) noexcept {
    get(*result) = get(lhs) * get(rhs);
  }
};

/// Compute the product of \f$\bar{J}\f$ and the quantity represented by `Rhs`.
/// In this function, \f$\bar{J}\f$ is obtained via conjugation of
/// `Tags::BondiJ` inline, rather than accessing `Tags::BondiJbar` in storage.
template <typename Rhs>
struct ComputePreSwshDerivatives<::Tags::Multiplies<Tags::BondiJbar, Rhs>> {
  using pre_swsh_derivative_tags = tmpl::list<Tags::BondiJ, Rhs>;
  using swsh_derivative_tags = tmpl::list<>;
  using integrand_tags = tmpl::list<>;

  using return_tags = tmpl::list<::Tags::Multiplies<Tags::BondiJbar, Rhs>>;
  using argument_tags = pre_swsh_derivative_tags;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<
          ComplexDataVector,
          ::Tags::Multiplies<Tags::BondiJbar, Rhs>::type::type::spin>>*>
          result,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
      const Scalar<SpinWeighted<ComplexDataVector, Rhs::type::type::spin>>&
          rhs) noexcept {
    get(*result) = conj(get(j)) * get(rhs);
  }
};

/// Compute the product of the quantity represented by `Lhs` and \f$\bar{J}\f$.
/// In this function, \f$\bar{J}\f$ is obtained via conjugation of
/// `Tags::BondiJ` inline, rather than accessing `Tags::BondiJbar` in storage.
template <typename Lhs>
struct ComputePreSwshDerivatives<::Tags::Multiplies<Lhs, Tags::BondiJbar>> {
  using pre_swsh_derivative_tags = tmpl::list<Tags::BondiJ, Lhs>;
  using swsh_derivative_tags = tmpl::list<>;
  using integrand_tags = tmpl::list<>;

  using return_tags = tmpl::list<::Tags::Multiplies<Lhs, Tags::BondiJbar>>;
  using argument_tags = pre_swsh_derivative_tags;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<
          ComplexDataVector,
          ::Tags::Multiplies<Lhs, Tags::BondiJbar>::type::type::spin>>*>
          result,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
      const Scalar<SpinWeighted<ComplexDataVector, Lhs::type::type::spin>>&
          lhs) noexcept {
    get(*result) = get(lhs) * conj(get(j));
  }
};

/// Compute the product of \f$\bar{U}\f$ and the quantity represented by `Rhs`.
/// In this function, \f$\bar{U}\f$ is obtained via conjugation of
/// `Tags::BondiU` inline, rather than accessing `Tags::BondiUbar` in storage.
template <typename Rhs>
struct ComputePreSwshDerivatives<::Tags::Multiplies<Tags::BondiUbar, Rhs>> {
  using pre_swsh_derivative_tags = tmpl::list<Tags::BondiU, Rhs>;
  using swsh_derivative_tags = tmpl::list<>;
  using integrand_tags = tmpl::list<>;

  using return_tags = tmpl::list<::Tags::Multiplies<Tags::BondiUbar, Rhs>>;
  using argument_tags = pre_swsh_derivative_tags;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<
          ComplexDataVector,
          ::Tags::Multiplies<Tags::BondiUbar, Rhs>::type::type::spin>>*>
          result,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& u,
      const Scalar<SpinWeighted<ComplexDataVector, Rhs::type::type::spin>>&
          rhs) noexcept {
    get(*result) = conj(get(u)) * get(rhs);
  }
};

/// Compute \f$\bar{J} * (Q - 2 \eth \beta)\f$.
/// \note the conjugates for this are accessed by their non-conjugate
/// counterparts (`Tags::BondiJ` and `Tags::BondiQ`) then conjugated inline
template <>
struct ComputePreSwshDerivatives<Tags::JbarQMinus2EthBeta> {
  using pre_swsh_derivative_tags = tmpl::list<Tags::BondiJ, Tags::BondiQ>;
  using swsh_derivative_tags =
      tmpl::list<Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                  Spectral::Swsh::Tags::Eth>>;
  using integrand_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::JbarQMinus2EthBeta>;
  using argument_tags =
      tmpl::append<pre_swsh_derivative_tags, swsh_derivative_tags>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*>
          jbar_q_minus_2_eth_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& q,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta) noexcept {
    get(*jbar_q_minus_2_eth_beta) =
        conj(get(j)) * (get(q) - 2.0 * get(eth_beta));
  }
};

/// Compute \f$\exp(2 \beta)\f$
template <>
struct ComputePreSwshDerivatives<Tags::Exp2Beta> {
  using pre_swsh_derivative_tags = tmpl::list<Tags::BondiBeta>;
  using swsh_derivative_tags = tmpl::list<>;
  using integrand_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::Exp2Beta>;
  using argument_tags = tmpl::append<pre_swsh_derivative_tags>;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          exp_2_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta) noexcept {
    get(*exp_2_beta).data() = exp(2.0 * get(beta).data());
  }
};

/// Compute the derivative of the quantity represented by `Tag` with respect to
/// the numerical radial coordinate \f$y\f$.
template <typename Tag>
struct ComputePreSwshDerivatives<Tags::Dy<Tag>> {
  using pre_swsh_derivative_tags = tmpl::list<Tag>;
  using swsh_derivative_tags = tmpl::list<>;
  using integrand_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::Dy<Tag>>;
  using argument_tags = tmpl::append<tmpl::list<Spectral::Swsh::Tags::LMax>,
                                     pre_swsh_derivative_tags>;

  static void apply(
      const gsl::not_null<
          Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>*>
          dy_val,
      const size_t l_max,
      const Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>&
          val) noexcept {
    logical_partial_directional_derivative_of_complex(
        make_not_null(&get(*dy_val).data()), get(val).data(),
        Mesh<3>{
            {{Spectral::Swsh::number_of_swsh_theta_collocation_points(l_max),
              Spectral::Swsh::number_of_swsh_phi_collocation_points(l_max),
              get(val).size() /
                  Spectral::Swsh::number_of_swsh_collocation_points(l_max)}},
            Spectral::Basis::Legendre,
            Spectral::Quadrature::GaussLobatto},
        2);
  }
};

// @{
/// In an actual CCE evolution, the values of the first derivatives of
/// `Tags::BondiBeta` and `Tags::BondiU` can just be copied from the integrands.
template <>
struct ComputePreSwshDerivatives<Tags::Dy<Tags::BondiBeta>> {
  using pre_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
  using integrand_tags = tmpl::list<Tags::Integrand<Tags::BondiBeta>>;

  using return_tags = tmpl::list<Tags::Dy<Tags::BondiBeta>>;
  using argument_tags = integrand_tags;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> dy_val,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&
          integrand_val) noexcept {
    *dy_val = integrand_val;
  }
};

template <>
struct ComputePreSwshDerivatives<Tags::Dy<Tags::BondiU>> {
  using pre_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
  using integrand_tags = tmpl::list<Tags::Integrand<Tags::BondiU>>;

  using return_tags = tmpl::list<Tags::Dy<Tags::BondiU>>;
  using argument_tags = integrand_tags;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> dy_val,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>&
          integrand_val) noexcept {
    *dy_val = integrand_val;
  }
};
// @}

/// Compute the derivative with respect to the numerical radial coordinate
/// \f$y\f$ of a quantity which is a spin-weighted spherical harmonic
/// derivative. This is separate from the generic case of a derivative with
/// respect to \f$y\f$ because the tags associated with this computation are
/// suggested to be in different `Variables` in the `DataBox`, so the typelists
/// indicating this function's needs from those variables are differently
/// arranged.
template <typename Tag, typename DerivKind>
struct ComputePreSwshDerivatives<
    Tags::Dy<Spectral::Swsh::Tags::Derivative<Tag, DerivKind>>> {
  using pre_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags =
      tmpl::list<Spectral::Swsh::Tags::Derivative<Tag, DerivKind>>;
  using integrand_tags = tmpl::list<>;

  using return_tags =
      tmpl::list<Tags::Dy<Spectral::Swsh::Tags::Derivative<Tag, DerivKind>>>;
  using argument_tags = tmpl::append<tmpl::list<Spectral::Swsh::Tags::LMax>,
                                     swsh_derivative_tags>;

  static constexpr int spin =
      Spectral::Swsh::Tags::Derivative<Tag, DerivKind>::spin;
  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, spin>>*>
          dy_val,
      const size_t l_max,
      const Scalar<SpinWeighted<ComplexDataVector, spin>>& val) noexcept {
    logical_partial_directional_derivative_of_complex(
        make_not_null(&get(*dy_val).data()), get(val).data(),
        Mesh<3>{
            {{Spectral::Swsh::number_of_swsh_theta_collocation_points(l_max),
              Spectral::Swsh::number_of_swsh_phi_collocation_points(l_max),
              get(val).size() /
                  Spectral::Swsh::number_of_swsh_collocation_points(l_max)}},
            Spectral::Basis::Legendre,
            Spectral::Quadrature::GaussLobatto},
        2);
  }
};

/*!
 * \brief This routine evaluates the set of inputs to the CCE integrand for
 * `BondiValueTag` which do not involve spin-weighted angular differentiation.
 * This function is to be called on the `DataBox` holding the relevant CCE data
 * on each hypersurface integration step, prior to evaluating the spin-weighted
 * derivatives needed for the same CCE integrand.
 *
 * \details Provided a `DataBox` with the appropriate tags (including
 * `all_pre_swsh_derivative_tags`, `all_swsh_derivative_tags` and
 * `Spectral::Swsh::Tags::LMax`), this function will apply all of the necessary
 * mutations to update `all_pre_swsh_derivatives_for_tag<BondiValueTag>` to
 * their correct values for the current values for the remaining (input) tags.
 */
template <typename BondiValueTag, typename DataBoxType>
void mutate_all_pre_swsh_derivatives_for_tag(
    const gsl::not_null<DataBoxType*> box) noexcept {
  tmpl::for_each<pre_swsh_derivative_tags_to_compute_for<BondiValueTag>>(
      [&box](auto x) {
        using pre_swsh_derivative_tag = typename decltype(x)::type;
        using mutation = ComputePreSwshDerivatives<pre_swsh_derivative_tag>;
        db::mutate_apply<mutation>(box);
      });
}
}  // namespace Cce
