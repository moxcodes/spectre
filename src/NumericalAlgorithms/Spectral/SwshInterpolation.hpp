// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/math/special_functions/binomial.hpp>
#include <complex>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Spectral {
namespace Swsh {

class SpinWeightedSphericalHarmonic {
 public:
  SpinWeightedSphericalHarmonic(const int spin, const size_t l,
                                const int m) noexcept
      : spin_{spin}, l_{l}, m_{m} {
    overall_prefactor_ = 1.0;
    if (abs(spin) > abs(m)) {
      for (size_t i = 0; i < abs(m) - abs(spin)) {
        overall_prefactor *=
            (static_cast<double>(l + abs(m)) - static_cast<double>(i)) /
            (static_cast<double>(l) - static_cast<double>(abs(spin) + i));
      }
    } else if (abs(m) > abs(spin)) {
      for (size_t i = 0; i < abs(m) - abs(spin)) {
        overall_prefactor *=
            (static_cast<double>(l) - static_cast<double>(abs(m) + i)) /
            (static_cast<double>(l + abs(spin)) - static_cast<double>(i));
      }
    }
    // if neither is greater (they are equal), then the prefactor is 1.0
    overall_prefactor_ = sqrt(overall_prefactor_);
    overall_prefactor_ *= (m % 2) == 0 ? 1.0 : -1.0;

    if (l < abs(spin)) {
      if (spin < 0) {
        r_prefactors = std::vector<double>(l + abs(spin), 0.0);
      }
    } else {
      for (int r = 0; r < (static_cast<int>(l) - spin); ++r) {
        if (static_cast<int>(r) + spin - m >= 0 and
            static_cast<int>(l) - static_cast<int>(r) + m >= 0) {
          r_prefactors_.push_back(
              boost::math::binomial_coefficient<double>(
                  static_cast<size_t>(static_cast<int>(l) - spin), r) *
              boost::math::binomial_coefficient<double>(
                  static_cast<size_t>(static_cast<int>(l) + spin),
                  static_cast<size_t>(spin - m + static_cast<int>(r)))) *
              (((static_cast<int>(l) - r - spin) % 2) == 0 ? 1.0 : -1.0);
        } else {
          r_prefactors_.push_back(0.0);
        }
      }
    }
  }

  std::complex<double> evaluate(const double theta, const double phi) noexcept {
    std::complex<double> accumulator = 0.0;
    for (int r = 0; r < (static_cast<int>(l_) - spin_); ++r) {
      std::complex<double> theta_factor = 1.0;
      if (2 * static_cast<int>(l_) > 2 * r + spin_ - m_) {
        theta_factor = pow(cos(theta / 2.0), 2 * r + spin_ - m_) *
                       pow(sin(theta / 2.0),
                           2 * static_cast<int>(l_) - (2 * r + spin_ - m_));
      } else if (2 * static_cast<int>(l) < 2 * r + s - m) {
        theta_factor = pow(cot(theta / 2.0),
                           2 * r + spin_ - m_ - 2 * static_cast<int>(l_)) *
                       pow(cos(theta / 2.0), 2 * l_);
      } else {
        theta_factor = pow(cos(theta / 2.0), 2 * l_);
      }

      accumulator += gsl::at(r_prefactors_, r) std::complex<double>(
                         cos(static_cast<double>(m) * phi),
                         sin(static_cast<double>(m) * phi)) *
                     theta_factor;
    }
    accumulator *= ovarall_prefactor_;
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

// @{
template <int Spin>
void swsh_interpolate_from_pfaffian(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*>
        target_collocation,
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*>
        source_collocation,
    const DataVector& target_theta, const DataVector& target_phi,
    const l_max) noexcept {
  check_and_resize(target_collocation, target_theta.size());
  SpinWeighted<ComplexModalVector, Spin> goldberg_modes =
      libsharp_to_goldberg_modes(swsh_transform(source_collocation, l_max),
                                 l_max);
  for (size_t l = 0; l < l_max; ++l) {
    for (int m = -static_cast<int>(l); m < static_cast<int>(l); ++m) {
      auto sYlm = SpinWeightedSphericalHarmonic(Spin, l, m);
      for (size_t i = 0; i < target_theta.size(); ++i) {
        target_collocation->data()[i] +=
            goldberg_modes.data()[static_cast<int>(square(l) + l) + m] *
            SpinWeightedSphericalHarmonic.evaluate_from_pfaffian(
                target_theta[i], target_phi[i]);
      }
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
                                 target_points, l_max);
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
    const l_max) noexcept {
  check_and_resize(target_collocation, target_theta.size());
  SpinWeighted<ComplexModalVector, Spin> goldberg_modes =
      libsharp_to_goldberg_modes(swsh_transform(source_collocation, l_max),
                                 l_max);
  for (size_t l = 0; l < l_max; ++l) {
    for (int m = -static_cast<int>(l); m < static_cast<int>(l); ++m) {
      auto sYlm = SpinWeightedSphericalHarmonic(Spin, l, m);
      for (size_t i = 0; i < target_theta.size(); ++i) {
        target_collocation->data()[i] +=
            goldberg_modes.data()[static_cast<int>(square(l) + l) + m] *
            SpinWeightedSphericalHarmonic.evaluate(target_theta[i],
                                                   target_phi[i]);
      }
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
  swsh_interpolate(make_not_null(&result), source_collocation, target_points,
                   l_max);
  return result;
}
// @}

}  // namespace Swsh
}  // namespace Spectral
