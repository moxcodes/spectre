// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/NPTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"

namespace Cce {

template <typename Tag>
struct CalculateNPSpinCoefficient;

template <>
struct CalculateNPSpinCoefficient<Tags::NpAlpha> {
  using argument_tags =
      tmpl::list<Tags::BondiJ,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                                  Spectral::Swsh::Tags::Ethbar>,
                 Spectral::Swsh::Tags::Derivative<
                     ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
                     Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::BondiQ, Tags::BondiR, Tags::BondiK, Tags::OneMinusY,
                 Spectral::Swsh::Tags::LMax,
                 Spectral::Swsh::Tags::NumberOfRadialPoints>;
  using return_tags = tmpl::list<Tags::NpAlpha>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*>
          np_alpha,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& ethbar_j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_j_jbar,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& q,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& k,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const size_t l_max, const size_t number_of_radial_points) noexcept {
    apply_impl(make_not_null(&get(*np_alpha)), get(j), get(ethbar_j),
               get(eth_j_jbar), get(eth_beta), get(q), get(r), get(k),
               get(one_minus_y), l_max, number_of_radial_points);
  }

  static void apply_impl(
      const gsl::not_null<SpinWeighted<ComplexDataVector, -1>*> np_alpha,
      const SpinWeighted<ComplexDataVector, 2>& j,
      const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
      const SpinWeighted<ComplexDataVector, 1>& eth_j_jbar,
      const SpinWeighted<ComplexDataVector, 1>& eth_beta,
      const SpinWeighted<ComplexDataVector, 1>& q,
      const SpinWeighted<ComplexDataVector, 0>& r,
      const SpinWeighted<ComplexDataVector, 0>& k,
      const SpinWeighted<ComplexDataVector, 0>& one_minus_y, const size_t l_max,
      const size_t number_of_radial_points) noexcept {
    SpinWeighted<ComplexDataVector, 0> sqrt_one_plus_k;
    sqrt_one_plus_k.data() = sqrt(k.data() + 1.0);
    *no_alpha = 0.125 * one_minus_y / r *
                (sqrt_one_plus_k * (conj(eth_beta) + 0.5 * conj(q)) +
                 1.0 / sqrt_one_plus_k *
                     (-0.5 * conj(j) * q - 0.25 * conj(eth_j_jbar) / k +
                      0.25 * conj(ethbar_j) / k - conj(j) * eth_beta +
                      0.75 * conj(ethbar_j)) -
                 0.25 / ((1.0 + k) * sqrt_one_plus_k) * conj(j) *
                     (eth_j_jbar - j * conj(ethbar_j)) / k)
  }
};

template <>
struct CalculateNPSpinCoefficient<Tags::NpBeta> {
  using argument_tags =
      tmpl::list<Tags::BondiJ,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                                  Spectral::Swsh::Tags::Ethbar>,
                 Spectral::Swsh::Tags::Derivative<
                     ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
                     Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::BondiQ, Tags::BondiR, Tags::BondiK, Tags::OneMinusY,
                 Spectral::Swsh::Tags::LMax,
                 Spectral::Swsh::Tags::NumberOfRadialPoints>;
  using return_tags = tmpl::list<Tags::NpBeta>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> np_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& ethbar_j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_j_jbar,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& q,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& k,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const size_t l_max, const size_t number_of_radial_points) noexcept {
    apply_impl(make_not_null(&get(*np_beta)), get(j), get(ethbar_j),
               get(eth_j_jbar), get(eth_beta), get(q), get(r), get(k),
               get(one_minus_y), l_max, number_of_radial_points);
  }

  static void apply_impl(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> np_beta,
      const SpinWeighted<ComplexDataVector, 2>& j,
      const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
      const SpinWeighted<ComplexDataVector, 1>& eth_j_jbar,
      const SpinWeighted<ComplexDataVector, 1>& eth_beta,
      const SpinWeighted<ComplexDataVector, 1>& q,
      const SpinWeighted<ComplexDataVector, 0>& r,
      const SpinWeighted<ComplexDataVector, 0>& k,
      const SpinWeighted<ComplexDataVector, 0>& one_minus_y, const size_t l_max,
      const size_t number_of_radial_points) noexcept {
    SpinWeighted<ComplexDataVector, 0> sqrt_one_plus_k;
    sqrt_one_plus_k.data() = sqrt(k.data() + 1.0);
    *np_beta =
        0.125 * one_minus_y / r *
        (sqrt_one_plus_k * (eth_beta + 0.5 * q) +
         1.0 / sqrt_one_plus_k *
             (-0.5 * j * conj(q) + 0.25 * eth_j_jbar / k - 0.25 * ethbar_j / k -
              j * conj(eth_beta) - 0.75 * ethbar_j) +
         0.25 / ((1.0 + k) * sqrt_one_plus_k) * j *
             (conj(eth_j_jbar) - conj(j) * ethbar_j) / k)
  }
};

template <>
struct CalculateNPSpinCoefficient<Tags::NpGamma> {
  using argument_tags = tmpl::list<
      Tags::SpecH, Tags::BondiJ,
      Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
          Spectral::Swsh::Tags::Eth>,
      Tags::Dy<Tags::BondiJ>, Tags::Exp2Beta, Tags::BondiU,
      Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiU, Spectral::Swsh::Tags::Eth>,
      Tags::BondiW, Tags::Dy<Tags::BondiW>, Tags::BondiR, Tags::BondiK,
      Tags::OneMinusY, Spectral::Swsh::Tags::LMax,
      Spectral::Swsh::Tags::NumberOfRadialPoints>;
  using return_tags = tmpl::list<Tags::NpGamma>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> np_gamma,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& spec_h,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& ethbar_j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_j_jbar,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& exp_2_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& u,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& ethbar_u,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_u,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& w,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_w,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& k,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const size_t l_max, const size_t number_of_radial_points) noexcept {
    apply_impl(make_not_null(&get(*np_gamma)), get(spec_h), get(j),
               get(ethbar_j), get(eth_j_jbar), get(dy_j), get(exp_2_beta),
               get(u), get(ethbar_u), get(eth_u), get(w), get(dy_w), get(r),
               get(k), get(one_minus_y), l_max, number_of_radial_points);
  }

  static void apply_impl(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> np_gamma,
      const SpinWeighted<ComplexDataVector, 2>& spec_h,
      const SpinWeighted<ComplexDataVector, 2>& j,
      const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
      const SpinWeighted<ComplexDataVector, 1>& eth_j_jbar,
      const SpinWeighted<ComplexDataVector, 2>& dy_j,
      const SpinWeighted<ComplexDataVector, 1>& exp_2_beta,
      const SpinWeighted<ComplexDataVector, 1>& u,
      const SpinWeighted<ComplexDataVector, 0>& ethbar_u,
      const SpinWeighted<ComplexDataVector, 2>& eth_u,
      const SpinWeighted<ComplexDataVector, 0>& w,
      const SpinWeighted<ComplexDataVector, 0>& dy_w,
      const SpinWeighted<ComplexDataVector, 0>& r,
      const SpinWeighted<ComplexDataVector, 0>& k,
      const SpinWeighted<ComplexDataVector, 0>& one_minus_y, const size_t l_max,
      const size_t number_of_radial_points) noexcept {
    *np_gamma =
        0.25 / exp_2_beta *
        (0.125 * square(one_minus_y) / (r * (1.0 + k)) *
             (conj(j) * dy_j - j * conj(dy_j)) +
         one_minus_y *
             (dy_w + 0.25 * w / (1.0 + k) * (conj(j) * dy_j - j * conj(dy_j))) +
         (0.5 * conj(eth_u) * j - 0.5 * eth_u * conj(j) -
          0.5 * conj(ethbar_u) * k + 0.5 * ethbar_u * k + w +
          0.5 / (1.0 + k) *
              (j * conj(spec_h) - spec_h * conj(j) -
               0.5 * conj(u) * (eth_j_jbar - 2.0 * j * conj(ethbar_j)) +
               0.5 * u * (conj(eth_j_jbar) - 2.0 * ethbar_j * conj(j)))));
  }
};

}  // namespace Cce
