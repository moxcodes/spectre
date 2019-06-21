// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/math/special_functions/binomial.hpp>
#include <cmath>
#include <complex>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransformJob.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Spectral {
namespace Swsh {

class SpinWeightedSphericalHarmonic {
 public:
  // TODO try to limit the casts
  SpinWeightedSphericalHarmonic(const int spin, const size_t l,
                                const int m) noexcept
      : spin_{spin}, l_{l}, m_{m} {
    overall_prefactor_ = 1.0;
    if (abs(m) > abs(spin)) {
      for (size_t i = 0; i < static_cast<size_t>(abs(m) - abs(spin)); ++i) {
        overall_prefactor_ *=
            (static_cast<double>(l + static_cast<size_t>(abs(m))) -
             static_cast<double>(i)) /
            (static_cast<double>(l) -
             static_cast<double>(static_cast<size_t>(abs(spin)) + i));
      }
    } else if (abs(spin) > abs(m)) {
      for (size_t i = 0; i < static_cast<size_t>(abs(spin) - abs(m)); ++i) {
        overall_prefactor_ *=
            (static_cast<double>(l) -
             static_cast<double>(static_cast<size_t>(abs(m)) + i)) /
            (static_cast<double>(l + static_cast<size_t>(abs(spin))) -
             static_cast<double>(i));
      }
    }
    // if neither is greater (they are equal), then the prefactor is 1.0
    overall_prefactor_ *= (2.0 * l + 1.0) / (4.0 * M_PI);
    overall_prefactor_ = sqrt(overall_prefactor_);
    overall_prefactor_ *= (m % 2) == 0 ? 1.0 : -1.0;

    if (l < static_cast<size_t>(abs(spin))) {
      if (spin < 0) {
        r_prefactors_ =
            std::vector<double>(l + static_cast<size_t>(abs(spin)) + 1, 0.0);
      }
    } else {
      for (int r = 0; r <= (static_cast<int>(l) - spin); ++r) {
        if (static_cast<int>(r) + spin - m >= 0 and
            static_cast<int>(l) - static_cast<int>(r) + m >= 0) {
          r_prefactors_.push_back(
              boost::math::binomial_coefficient<double>(
                  static_cast<size_t>(static_cast<int>(l) - spin),
                  static_cast<size_t>(r)) *
              boost::math::binomial_coefficient<double>(
                  static_cast<size_t>(static_cast<int>(l) + spin),
                  static_cast<size_t>(spin - m + static_cast<int>(r))) *
              (((static_cast<int>(l) - r - spin) % 2) == 0 ? 1.0 : -1.0));
        } else {
          r_prefactors_.push_back(0.0);
        }
      }
    }
  }

  // TEST to see if blaze causes power problems
  DataVector power(const DataVector& arg, const int exponent) {
    DataVector result{arg.size(), 1.0};
    if (exponent > 0) {
      for (size_t i = 0; i < static_cast<int>(exponent); ++i) {
        result *= arg;
      }
    } else {
      for (size_t i = 0; i < static_cast<int>(-exponent); ++i) {
        result /= arg;
      }
    }
  }

  void evaluate(const gsl::not_null<ComplexDataVector*> result,
                const DataVector& theta, const DataVector& phi,
                const DataVector& sin_theta_over_2,
                const DataVector& cos_theta_over_2) noexcept {
    check_and_resize(result, theta.size());
    *result = 0.0;
    // TODO there should be a way to do away with this allocation. My first pass
    // involved a nasty ternary that made Blaze sad :(
    DataVector theta_factor{theta.size(), 1.0};
    for (int r = 0; r <= (static_cast<int>(l_) - spin_); ++r) {
      theta_factor = 1.0;
      if (2 * static_cast<int>(l_) > 2 * r + spin_ - m_) {
        theta_factor = pow(cos_theta_over_2, 2 * r + spin_ - m_) *
                       pow(sin_theta_over_2,
                           2 * static_cast<int>(l_) - (2 * r + spin_ - m_));
      } else if (2 * static_cast<int>(l_) < 2 * r + spin_ - m_) {
        theta_factor = pow(cos_theta_over_2 / sin_theta_over_2,
                           2 * r + spin_ - m_ - 2 * static_cast<int>(l_)) *
                       pow(cos_theta_over_2, 2 * l_);
      } else {
        theta_factor = pow(cos_theta_over_2, 2 * l_);
      }
      *result +=
          gsl::at(r_prefactors_, r) * theta_factor;
    }
    *result *=
        overall_prefactor_ *
        (std::complex<double>(1.0, 0.0) * cos(static_cast<double>(m_) * phi) +
         std::complex<double>(0.0, 1.0) * sin(static_cast<double>(m_) * phi));
  }

  ComplexDataVector evaluate(const DataVector& theta, const DataVector& phi,
                             const DataVector& sin_theta_over_2,
                             const DataVector& cos_theta_over_2) noexcept {

    ComplexDataVector result{theta.size(), 0.0};
    evaluate(make_not_null(&result), theta, phi, sin_theta_over_2,
             cos_theta_over_2);
    return result;
  }


  std::complex<double> evaluate(const double theta, const double phi) noexcept {
    std::complex<double> accumulator = 0.0;
    for (int r = 0; r <= (static_cast<int>(l_) - spin_); ++r) {
      std::complex<double> theta_factor = 1.0;
      if (2 * static_cast<int>(l_) > 2 * r + spin_ - m_) {
        theta_factor = pow(cos(theta / 2.0), 2 * r + spin_ - m_) *
                       pow(sin(theta / 2.0),
                           2 * static_cast<int>(l_) - (2 * r + spin_ - m_));
      } else if (2 * static_cast<int>(l_) < 2 * r + spin_ - m_) {
        theta_factor = pow(1.0 / tan(theta / 2.0),
                           2 * r + spin_ - m_ - 2 * static_cast<int>(l_)) *
                       pow(cos(theta / 2.0), 2 * l_);
      } else {
        theta_factor = pow(cos(theta / 2.0), 2 * l_);
      }
      accumulator += gsl::at(r_prefactors_, r) *
                     std::complex<double>(cos(static_cast<double>(m_) * phi),
                                          sin(static_cast<double>(m_) * phi)) *
                     theta_factor;
    }
    accumulator *= overall_prefactor_;
    return accumulator;
  }

  std::complex<double> evaluate_from_pfaffian(const double theta,
                                              const double phi) noexcept {
    // note: trouble near poles. Ideally, there would be an alternative formula
    // for this representation, but I do not know it.
    return evaluate(theta, phi / sin(theta));
  }

 private:
  int spin_;
  size_t l_;
  int m_;
  double overall_prefactor_;
  std::vector<double> r_prefactors_;
};

// TODO these interpolation methods can be improved. Some ideas for making them
// better:
// - matrix multiplies for small l
// - find some sort of Clenshaw-esque recurrence relation for spin-weighted
// quantities
// - at least cache trig functions of theta and phi.

// @{
template <int Spin>
void swsh_interpolate_from_pfaffian(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*>
        target_collocation,
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*>
        source_collocation,
    const DataVector& target_theta,
    const DataVector& target_phi_times_sin_theta, const size_t l_max) noexcept {
  check_and_resize(target_collocation, target_theta.size());
  SpinWeighted<ComplexModalVector, Spin> goldberg_modes =
      libsharp_to_goldberg_modes(swsh_transform(source_collocation, l_max),
                                 l_max);
  target_collocation->data() = 0.0;
  DataVector sin_theta_over_2 = sin(target_theta / 2.0);
  DataVector cos_theta_over_2 = cos(target_theta / 2.0);
  DataVector phi = target_phi_times_sin_theta / sin(target_theta);
  for (size_t l = abs(Spin); l <= l_max; ++l) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); ++m) {
      auto sYlm = SpinWeightedSphericalHarmonic{Spin, l, m};
      target_collocation->data() +=
          goldberg_modes.data()[static_cast<size_t>(
              static_cast<int>(square(l) + l) + m)] *
          sYlm.evaluate(target_theta, phi, sin_theta_over_2, cos_theta_over_2);
    }
  }
}

template <int Spin>
SpinWeighted<ComplexDataVector, Spin> swsh_interpolate_from_pfaffian(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*>
        source_collocation,
    const DataVector& target_theta, const DataVector& target_phi,
    const size_t l_max) noexcept {
  SpinWeighted<ComplexDataVector, Spin> result{target_theta.size()};
  swsh_interpolate_from_pfaffian(make_not_null(&result), source_collocation,
                                 target_theta, target_phi, l_max);
  return result;
}
// @}

// @{

template <int Spin>
void swsh_interpolate(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*>
        target_collocation,
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*>
        source_collocation,
    const DataVector& target_theta, const DataVector& target_phi,
    const size_t l_max) noexcept {
  check_and_resize(target_collocation, target_theta.size());
  SpinWeighted<ComplexModalVector, Spin> goldberg_modes =
      libsharp_to_goldberg_modes(swsh_transform(source_collocation, l_max),
                                 l_max);
  target_collocation->data() = 0.0;
  DataVector sin_theta_over_2 = sin(target_theta / 2);
  DataVector cos_theta_over_2 = cos(target_theta / 2);
  for (size_t l = abs(Spin); l <= l_max; ++l) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); ++m) {
      auto sYlm = SpinWeightedSphericalHarmonic{Spin, l, m};
      target_collocation->data() +=
          goldberg_modes.data()[static_cast<size_t>(
              static_cast<int>(square(l) + l) + m)] *
          sYlm.evaluate(target_theta, target_phi, sin_theta_over_2,
                        cos_theta_over_2);
    }
  }
}

template <int Spin>
SpinWeighted<ComplexDataVector, Spin> swsh_interpolate(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*>
        source_collocation,
    const DataVector& target_theta, const DataVector& target_phi,
    const size_t l_max) noexcept {
  SpinWeighted<ComplexDataVector, Spin> result{target_theta.size()};
  swsh_interpolate(make_not_null(&result), source_collocation, target_theta,
                   target_phi, l_max);
  return result;
}
// @}

}  // namespace Swsh
}  // namespace Spectral
