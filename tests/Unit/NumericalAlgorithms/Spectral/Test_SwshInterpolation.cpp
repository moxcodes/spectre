// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <complex>

#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "tests/Unit/NumericalAlgorithms/Spectra/SwshTestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

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

  std::complex<double> expected =
      spin_weighted_spherical_harmonic(s, static_cast<int>(l), m, theta, phi);
  auto test_harmonic = SpinWeightedSphericalHarmonic(spin, l, m);
  std::complex<double> test = test_harmonic.evaluate(theta, phi);
  std::complex<double> test_pfaffian =
      test_harmonic.evaluate_pfaffian(theta, phi * sin(theta));

  CHECK_ITERABLE_APPROX(test, expected);
  CHECK_ITERABLE_APPROX(test_pfaffian, expected);
}

template <int spin, typename Generator>
void test_interpolation(const gsl::not_null<Generator*> generator) noexcept {
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};
  const size_t l_max = 16;
  UniformCustomDistribution<double> phi_dist{0.0, 2.0 * M_PI};
  UniformCustomDistribution<double> theta_dist{0.01, M_PI - 0.01};
  const size_t number_of_target_points = 10;

  const DataVector target_phi = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&phi_dist), 10);
  const DataVector target_theta = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&theta_dist), 10);
  const DataVector target_phi_pfaffian = target_phi * sin(target_theta);

  SpinWeighted<ComplexModalVector, spin> generated_modes{
      2 * Spectral::Swsh::number_of_swsh_coefficients(l_max)};
  TestHelpers::generate_swsh_modes<spin>(make_not_null(&generated_modes.data()),
                                         make_not_null(&generator), 1,
                                         l_max);
  auto generated_coefficients = Spectral::Swsh::inverse_swsh_transform(
      make_not_null(&generated_modes), l_max);
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
