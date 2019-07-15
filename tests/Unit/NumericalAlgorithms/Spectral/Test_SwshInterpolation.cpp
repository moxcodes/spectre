// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <complex>

#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransformJob.hpp"
#include "tests/Unit/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Spectral {
namespace Swsh {
namespace {

template <typename Generator>
void test_basis_function(const gsl::not_null<Generator*> generator) noexcept {
  UniformCustomDistribution<double> phi_dist{0.0, 2.0 * M_PI};
  const double phi = phi_dist(*generator);
  UniformCustomDistribution<double> theta_dist{0.01, M_PI - 0.01};
  const double theta = theta_dist(*generator);
  UniformCustomDistribution<int> spin_dist{-2, 2};
  const int spin = spin_dist(*generator);
  UniformCustomDistribution<size_t> l_dist{static_cast<size_t>(abs(spin)), 16};
  const size_t l = l_dist(*generator);
  UniformCustomDistribution<int> m_dist{-static_cast<int>(l),
                                        static_cast<int>(l)};
  const int m = m_dist(*generator);

  std::complex<double> expected = TestHelpers::spin_weighted_spherical_harmonic(
      spin, static_cast<int>(l), m, theta, phi);
  auto test_harmonic = SpinWeightedSphericalHarmonic{spin, l, m};
  std::complex<double> test = test_harmonic.evaluate(theta, phi);
  std::complex<double> test_pfaffian =
      test_harmonic.evaluate_from_pfaffian(theta, phi * sin(theta));
  CAPTURE(spin);
  CAPTURE(l);
  CAPTURE(m);
  CAPTURE(theta);
  CAPTURE(phi);
  // need a slightly looser approx to accommodate the explicit factorials in the
  // simpler TestHelper form
  Approx factorial_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  CHECK_ITERABLE_CUSTOM_APPROX(test, expected, factorial_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(test_pfaffian, expected, factorial_approx);
}

template <int spin, typename Generator>
void test_interpolation(const gsl::not_null<Generator*> generator) noexcept {
  INFO("Testing interpolation for spin " << spin);
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};
  const size_t l_max = 16;
  UniformCustomDistribution<double> phi_dist{0.0, 2.0 * M_PI};
  UniformCustomDistribution<double> theta_dist{0.01, M_PI - 0.01};
  const size_t number_of_target_points = 10;

  const DataVector target_phi = make_with_random_values<DataVector>(
      generator, make_not_null(&phi_dist), number_of_target_points);
  const DataVector target_theta = make_with_random_values<DataVector>(
      generator, make_not_null(&theta_dist), number_of_target_points);
  const DataVector target_phi_pfaffian = target_phi * sin(target_theta);

  SpinWeighted<ComplexModalVector, spin> generated_modes{
      2 * number_of_swsh_coefficients(l_max)};
  TestHelpers::generate_swsh_modes<spin>(
      make_not_null(&generated_modes.data()), generator,
      make_not_null(&coefficient_distribution), 1, l_max);
  auto generated_collocation =
      inverse_swsh_transform(make_not_null(&generated_modes), l_max);

  auto goldberg_modes = libsharp_to_goldberg_modes(generated_modes, l_max);

  ComplexDataVector expected{number_of_target_points, 0.0};
  ComplexDataVector another_expected{number_of_target_points, 0.0};
  auto interpolator = SwshInterpolator{target_theta, target_phi, spin, l_max};

  for(int l = 0; l <= l_max; ++l) {
    for(int m = -l; m <= l; ++m) {
      auto sYlm =
          SpinWeightedSphericalHarmonic{spin, static_cast<size_t>(l), m};
      if (l == std::max(abs(m), abs(spin))) {
        ComplexDataVector harmonic_test;
        interpolator.direct_evaluation_swsh_at_l_min(
            make_not_null(&harmonic_test), l, m);
        for(size_t i = 0; i < number_of_target_points; ++i) {
          CHECK_ITERABLE_APPROX(sYlm.evaluate(target_theta[i], target_phi[i]),
                                harmonic_test[i]);
        }
      }
      if (l == std::max(abs(m), abs(spin)) + 1) {
        ComplexDataVector harmonic_test_l_min;
        interpolator.direct_evaluation_swsh_at_l_min(
            make_not_null(&harmonic_test_l_min), l - 1, m);

        ComplexDataVector harmonic_test_l_min_plus_one;
        interpolator.evaluate_swsh_at_l_min_plus_one(
            make_not_null(&harmonic_test_l_min_plus_one), harmonic_test_l_min,
            l - 1, m);

        for(size_t i = 0; i < number_of_target_points; ++i) {
          CHECK_ITERABLE_APPROX(sYlm.evaluate(target_theta[i], target_phi[i]),
                                harmonic_test_l_min_plus_one[i]);
        }
      }
      if (l == std::max(abs(m), abs(spin)) and abs(m) > abs(spin)) {
        if(m > 0) {
          ComplexDataVector harmonic_test;
          interpolator.direct_evaluation_swsh_at_l_min(
              make_not_null(&harmonic_test), l - 1, m - 1);
          interpolator.evaluate_swsh_m_recurrence_at_l_min(
              make_not_null(&harmonic_test), l, m);
          INFO("checking l=" << l <<" m=" << m);
          for(size_t i = 0; i < number_of_target_points; ++i) {
            CHECK_ITERABLE_APPROX(sYlm.evaluate(target_theta[i], target_phi[i]),
                                  harmonic_test[i]);
          }
        } else {
          ComplexDataVector harmonic_test;
          interpolator.direct_evaluation_swsh_at_l_min(
              make_not_null(&harmonic_test), l - 1, m + 1);
          interpolator.evaluate_swsh_m_recurrence_at_l_min(
              make_not_null(&harmonic_test), l, m);
          INFO("checking l=" << l <<" m=" << m);
          for(size_t i = 0; i < number_of_target_points; ++i) {
            CHECK_ITERABLE_APPROX(sYlm.evaluate(target_theta[i], target_phi[i]),
                                  harmonic_test[i]);
          }
        }
      }
      for(size_t i = 0; i < number_of_target_points; ++i) {
        expected[i] +=
            goldberg_modes.data()[static_cast<size_t>(square(l) + l + m)] *
            TestHelpers::spin_weighted_spherical_harmonic(
                spin, l, m, target_theta[i], target_phi[i]);
        another_expected[i] +=
            goldberg_modes.data()[static_cast<size_t>(square(l) + l + m)] *
            sYlm.evaluate(target_theta[i], target_phi[i]);
      }
    }
  }

  Approx factorial_approx =
      Approx::custom()
      .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
      .scale(1.0);

  // direct test Clenshaw sums
  for(int m = -static_cast<int>(l_max); m <= static_cast<int>(l_max); ++m) {
    ComplexDataVector expected_clenshaw_sum{number_of_target_points, 0.0};
    for (int l = std::max(abs(m), abs(spin)); l <= static_cast<int>(l_max);
         ++l) {
      auto sYlm =
          SpinWeightedSphericalHarmonic{spin, static_cast<size_t>(l), m};
      for(size_t i = 0; i < number_of_target_points; ++i) {
        expected_clenshaw_sum[i] +=
            goldberg_modes.data()[static_cast<size_t>(square(l) + l + m)] *
            sYlm.evaluate(target_theta[i], target_phi[i]);
      }
    }
    ComplexDataVector clenshaw{number_of_target_points, 0.0};

    ComplexDataVector harmonic_test_l_min;
    interpolator.direct_evaluation_swsh_at_l_min(
        make_not_null(&harmonic_test_l_min), std::max(abs(spin), abs(m)), m);

    ComplexDataVector harmonic_test_l_min_plus_one;
    interpolator.evaluate_swsh_at_l_min_plus_one(
        make_not_null(&harmonic_test_l_min_plus_one), harmonic_test_l_min,
        std::max(abs(spin), abs(m)), m);

    interpolator.clenshaw_sum(
        make_not_null(&clenshaw), harmonic_test_l_min,
        harmonic_test_l_min_plus_one,
        libsharp_to_goldberg_modes(
            swsh_transform(make_not_null(&generated_collocation), l_max), l_max)
            .data(),
        m);
    INFO("checking clenshaw sum for m=" << m);
    for(size_t i = 0; i < number_of_target_points; ++i) {
      CHECK_ITERABLE_CUSTOM_APPROX(clenshaw[i], expected_clenshaw_sum[i],
                                   factorial_approx);
    }
  }
  auto pfaffian_interp =
      swsh_interpolate_from_pfaffian(make_not_null(&generated_collocation),
                                     target_theta, target_phi_pfaffian, l_max);
  auto standard_interp = swsh_interpolate(make_not_null(&generated_collocation),
                                          target_theta, target_phi, l_max);

  ComplexDataVector clenshaw_interpolation;
  interpolator.interpolate(
      make_not_null(&clenshaw_interpolation),
      libsharp_to_goldberg_modes(
          swsh_transform(make_not_null(&generated_collocation), l_max), l_max)
          .data());

  CHECK_ITERABLE_CUSTOM_APPROX(expected, another_expected, factorial_approx);

  CHECK_ITERABLE_CUSTOM_APPROX(pfaffian_interp.data(), expected,
                               factorial_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(standard_interp.data(), expected,
                               factorial_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(clenshaw_interpolation, expected,
                               factorial_approx);
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.SwshInterpolation",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  for(size_t i = 0; i < 10; ++i) {
    test_basis_function(make_not_null(&generator));
  }

  test_interpolation<-1>(make_not_null(&generator));
  test_interpolation<0>(make_not_null(&generator));
  test_interpolation<2>(make_not_null(&generator));
}
}  // namespace
}  // namespace Swsh
}  // namespace Spectral
