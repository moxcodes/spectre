// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/math/interpolators/barycentric_rational.hpp>
#include <boost/math/tools/roots.hpp>

#include "DataStructures/VectorAlgebra.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Utilities/MakeArray.hpp"

namespace Cce {

struct InitializeRobinsonTrautman {
  using return_tags = tmpl::list<Tags::RobinsonTrautmanW>;
  using argument_tags = tmpl::list<Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> rt_w,
      const size_t l_max) noexcept {
    // 1, 1 trial starting data
    Spectral::Swsh::SpinWeightedSphericalHarmonic swsh{0, 1, 1};
    for (const auto& collocation_point :
         Spectral::Swsh::precomputed_collocation<
             Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max)) {
      get(*rt_w).data()[collocation_point.offset] =
          0.01 * swsh.evaluate(collocation_point.theta, collocation_point.phi);
    }
  }
};

template <typename Tag>
struct CalculateRobinsonTrautman;

template <>
struct CalculateRobinsonTrautman<Tags::BoundaryValue<Tags::R>> {
  using return_tags = tmpl::list<Tags::BoundaryValue<Tags::R>>;
  using argument_tags = tmpl::list<>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          boundary_cauchy_r) noexcept {
    get(*boundary_cauchy_r).data() = 100.0;
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::BoundaryValue<Tags::DuRDividedByR>> {
  using return_tags = tmpl::list<Tags::BoundaryValue<Tags::DuRDividedByR>>;
  using argument_tags = tmpl::list<>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          boundary_cauchy_du_r_divided_by_r) noexcept {
    get(*boundary_cauchy_du_r_divided_by_r).data() = 0.0;
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::BoundaryValue<Tags::J>> {
  using return_tags = tmpl::list<Tags::BoundaryValue<Tags::J>>;
  using argument_tags = tmpl::list<>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          boundary_cauchy_j) noexcept {
    // For the RT solution, J is just 0.
    get(*boundary_cauchy_j).data() = 0.0;
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::CauchyGauge<Tags::J>> {
  using return_tags = tmpl::list<Tags::CauchyGauge<Tags::J>>;
  using argument_tags = tmpl::list<>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          cauchy_j) noexcept {
    // For the RT solution, J is just 0.
    get(*cauchy_j).data() = 0.0;
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::BoundaryValue<Tags::Dr<Tags::J>>> {
  using return_tags = tmpl::list<Tags::BoundaryValue<Tags::Dr<Tags::J>>>;
  using argument_tags = tmpl::list<>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          boundary_dr_cauchy_j) noexcept {
    // For the RT solution, J is just 0.
    get(*boundary_dr_cauchy_j).data() = 0.0;
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::BoundaryValue<Tags::H>> {
  using return_tags = tmpl::list<Tags::BoundaryValue<Tags::H>>;
  using argument_tags = tmpl::list<>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          boundary_cauchy_h) noexcept {
    // For the RT solution, J is just 0.
    get(*boundary_cauchy_h).data() = 0.0;
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::CauchyGauge<Tags::H>> {
  using return_tags = tmpl::list<Tags::CauchyGauge<Tags::H>>;
  using argument_tags = tmpl::list<>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          cauchy_h) noexcept {
    // For the RT solution, J is just 0.
    get(*cauchy_h).data() = 0.0;
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::BoundaryValue<Tags::Beta>> {
  using return_tags = tmpl::list<Tags::BoundaryValue<Tags::Beta>>;
  using argument_tags = tmpl::list<Tags::RobinsonTrautmanW, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          boundary_cauchy_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& rt_w,
      const size_t l_max) noexcept {
    get(*boundary_cauchy_beta).data() = 0.5 * log(get(rt_w).data());
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::CauchyGauge<Tags::Beta>> {
  using return_tags = tmpl::list<Tags::CauchyGauge<Tags::Beta>>;
  using argument_tags = tmpl::list<Tags::RobinsonTrautmanW, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          cauchy_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& rt_w,
      const size_t l_max) noexcept {
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points =
        get(*cauchy_beta).size() / number_of_angular_points;
    get(*cauchy_beta).data() = outer(
        get(rt_w).data(), ComplexDataVector{number_of_radial_points, 1.0});
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::BoundaryValue<Tags::Q>> {
  using return_tags =
      tmpl::list<Tags::BoundaryValue<Tags::Q>, Tags::RobinsonTrautmanW>;
  using argument_tags = tmpl::list<>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          boundary_cauchy_q,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> rt_w,
      const size_t l_max) noexcept {
    get(*boundary_cauchy_q) =
        -Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&get(*rt_w)), l_max) /
        get(*rt_w);
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::CauchyGauge<Tags::Q>> {
  using return_tags =
      tmpl::list<Tags::CauchyGauge<Tags::Q>, Tags::RobinsonTrautmanW>;
  using argument_tags = tmpl::list<Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> cauchy_q,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> rt_w,
      const size_t l_max) noexcept {
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points =
        get(*cauchy_q).size() / number_of_angular_points;

    get(*cauchy_q).data() =
        outer(-Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
                   make_not_null(&get(*rt_w)), l_max)
                      .data() /
                  get(*rt_w).data(),
              ComplexDataVector{number_of_radial_points, 1.0});
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::BoundaryValue<Tags::U>> {
  using return_tags =
      tmpl::list<Tags::BoundaryValue<Tags::U>, Tags::RobinsonTrautmanW>;
  using argument_tags = tmpl::list<Tags::BoundaryValue<Tags::R>, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          boundary_cauchy_u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> rt_w,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      const size_t l_max) noexcept {
    get(*boundary_cauchy_u) =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&get(*rt_w)), l_max) /
        get(boundary_r);
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::CauchyGauge<Tags::U>> {
  using return_tags =
      tmpl::list<Tags::CauchyGauge<Tags::U>, Tags::RobinsonTrautmanW>;
  using argument_tags =
      tmpl::list<Tags::BoundaryValue<Tags::R>, Tags::OneMinusY, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> cauchy_u,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> rt_w,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const size_t l_max) noexcept {
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points =
        get(*cauchy_u).size() / number_of_angular_points;
    ComplexDataVector boundary_data =
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
            make_not_null(&get(*rt_w)), l_max)
            .data() /
        get(boundary_r).data();
    get(*cauchy_u).data() =
        outer(boundary_data, ComplexDataVector{number_of_radial_points, 1.0}) *
        get(one_minus_y).data() / 2.0;
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::BoundaryValue<Tags::W>> {
  using return_tags =
      tmpl::list<Tags::BoundaryValue<Tags::W>, Tags::RobinsonTrautmanW>;
  using argument_tags = tmpl::list<Tags::BoundaryValue<Tags::R>, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          boundary_cauchy_w,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> rt_w,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      const size_t l_max) noexcept {
    get(*boundary_cauchy_w) =
        (get(*rt_w).data() +
         Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::EthEthbar>(
             make_not_null(&get(*rt_w)), l_max)
             .data() -
         1.0) /
            get(boundary_r).data() -
        2.0 / (square(get(*rt_w).data()) * get(boundary_r).data());
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::CauchyGauge<Tags::W>> {
  using return_tags =
      tmpl::list<Tags::CauchyGauge<Tags::W>, Tags::RobinsonTrautmanW>;
  using argument_tags =
      tmpl::list<Tags::BoundaryValue<Tags::R>, Tags::OneMinusY, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> cauchy_w,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> rt_w,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      const size_t l_max) noexcept {
    size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    size_t number_of_radial_points =
        get(*cauchy_w).size() / number_of_angular_points;
    // For the RT solution, J is just 0.
    ComplexDataVector boundary_data_inverse_r =
        (get(*rt_w).data() +
         Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::EthEthbar>(
             make_not_null(&get(*rt_w)), l_max)
             .data() -
         1.0) /
        get(boundary_r).data();
    ComplexDataVector boundary_data_inverse_r_squared =
        -2.0 / (square(get(*rt_w).data()) * get(boundary_r).data());
    get(*cauchy_w) = outer(boundary_data_inverse_r,
                           ComplexDataVector{number_of_radial_points, 1.0}) *
                         one_minus_y / 2.0 +
                     outer(boundary_data_inverse_r,
                           ComplexDataVector{number_of_radial_points, 1.0}) *
                         square(one_minus_y) / 4.0;
  }
};

template <>
struct CalculateRobinsonTrautman<Tags::Du<Tags::RobinsonTrautmanW>> {
  using return_tags =
      tmpl::list<Tags::Du<Tags::RobinsonTrautmanW>, Tags::RobinsonTrautmanW>;
  using argument_tags = tmpl::list<Tags::LMax>;

  static void apply(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> du_rt_w,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> rt_w,
      const size_t l_max) noexcept {
    auto eth_rt_w = Spectral::Swsh::swsh_derivative<Spectral::Swsh::Eth>(
        make_not_null(&get(*rt_w)), l_max);
    SpinWeighted<ComplexDataVector, 0> rt_k =
        square(get(*rt_w)) - eth_rt_w * conj(eth_rt_w) +
        get(*rt_w) *
            Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::EthEthbar>(
                make_not_null(&get(*rt_w)), l_max);
    get(*du_rt_w) =
        pow<3>(get(*rt_w).data()) *
        Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::EthEthbar>(
            make_not_null(&rt_k), l_max) /
        12.0;
  }
};

}  // namespace Cce
