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
      *result += gsl::at(r_prefactors_, r) * theta_factor;
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

class SwshInterpolator {
 public:
  SwshInterpolator(const DataVector& theta, const DataVector& phi,
                   const int spin, const size_t l_max) noexcept
      : spin_{spin}, l_max_{l_max} {
    cos_theta_ = cos(theta);
    cos_theta_over_two_ = cos(theta / 2.0);
    sin_theta_over_two = sin(theta / 2.0);
    exp_i_phi_ = std::complex<double>(1.0, 0.0) * cos(phi) +
                 std::complex<double>(0.0, 1.0) * sin(phi);
    double k_plus_l;
    double a;
    double b;
    DataVector alpha;
    double beta;

    for (int m = -static_cast<int>(l_max); m <= static_cast<int>(l_max); ++m) {
      alpha_table.push_back(std::vector<DataVector>{});
      beta_table.push_back(std::vector<double>{});
      lambda_table.push_back(std::vector<int>{});
      for (int l = min(abs(m), abs(spin)); l <= l_max; ++l) {
        a = static_cast<double>(abs(spin + m));
        b = static_cast<double>(abs(spin - m));
        k_plus_l = static_cast<double>(-(a + b) / 2 + l);

        alpha = (2.0 * l_plus_k + b + a - 1) *
                sqrt((2.0 * static_cast<double>(l) + 1.0) /
                     ((2.0 * static_cast<double>(l) - 1.0) * l_plus_k *
                      (l_plus_k + a + b) * (l_plus_k + a) * (l_plus_k + b))) *
                ((2.0 * l_plus_k + a + b) * cos_theta_ +
                 (square(a) - square(b)) / (2.0 * l_plus_k + a + b - 2.0));
        alpha_table[m + static_cast<int>(l_max)].push_back(alpha);

        beta = -sqrt((2.0 * static_cast<double>(l) + 1.0) *
                     (l_plus_k + a - 1.0) * (l_plus_k + b - 1.0) *
                     (l_plus_k - 1.0) * (l_plus_k + a + b - 1.0) /
                     ((2.0 * static_cast<double>(l) - 3.0) * l_plus_k *
                      (l_plus_k + a + b) * (l_plus_k + a) * (l_plus_k + b))) *
               (2.0 * l_plus_k + a + b) / (2.0 * l_plus_k + a + b - 2.0);
        beta_table[m + static_cast<int>(l_max)].push_back(beta);

        lambda_table[m + static_cast<int>(l_max)].push_back(s >= -m ? 0
                                                                    : s + m);
      }
    }
  }

  void interpolate(const gsl::not_null<ComplexDataVector*> interpolated,
                   const ComplexModalVector& goldberg_modes) noexcept {
    check_and_resize(interpolated, cos_theta_.size());
    *interpolated = 0.0;

    // used only if s=0;
    ComplexDataVector cached_base_harmonic;
    int cached_base_harmonic_m;
    int cached_base_harmonic_l;

    // used during both recurrence legs
    ComplexDataVector current_cached_harmonic;
    ComplexDataVector current_cached_harmonic_l_plus_one;

    // perform the Clenshaw sums over positive m >= 0.
    for (int m = 0; m <= static_cast<int>(l_max); ++m) {
      if (abs(s) >= abs(m)) {
        direct_evaluation_swsh_at_l_min(make_not_null(&current_cached_harmonic),
                                        abs(s), m);
        evaluate_swsh_at_l_min_plus_one(
            make_not_null(&current_cached_harmonic_l_plus_one), abs(s), m);
      } else {
        evaluate_swsh_m_recurrence_at_l_min(
            make_not_null(&current_cached_harmonic), abs(m), m);
        evaluate_swsh_at_l_min_plus_one(
            make_not_null(&current_cached_harmonic_l_plus_one), abs(m), m);
      }
      if (s == 0 and m == 0) {
        cached_base_harmonic = current_cached_harmonic;
      }
      clenshaw_sum(interpolated, current_cached_harmonic,
                   current_cahced_harmonic_l_plus_one, goldberg_modes);
    }
    // perform the Clenshaw sums over m < 0.
    for (int m = -1; m > -static_cast<int>(l_max); --m) {
      if (m == -1 and s == 0) {
        current_cached_harmonic = cached_base_harmonic
      }
      if (abs(s) >= abs(m)) {
        direct_evaluation_swsh_at_l_min(make_not_null(&current_cached_harmonic),
                                        abs(s), m);
        evaluate_swsh_at_l_min_plus_one(
            make_not_null(&current_cached_harmonic_l_plus_one),
            current_cached_harmonic, abs(s), m);
      } else {
        evaluate_swsh_m_recurrence_at_l_min(
            make_not_null(&current_cached_harmonic), abs(m), m);
        evaluate_swsh_at_l_min_plus_one(
            make_not_null(&current_cached_harmonic_l_plus_one), abs(m), m);
      }
      clenshaw_sum(interpolated, current_cached_harmonic,
                   current_cahced_harmonic_l_plus_one, goldberg_modes);
    }
  }

 private:
  void direct_evaluation_swsh_at_l_min(
      const gsl::not_null<ComplexDataVector*> harmonic, const int l,
      const int m) noexcept {
    double prefactor = 1.0;
    double a = static_cast<double>(abs(spin_ + m));
    int b = abs(spin - m);
    double l_plus_k = static_cast<double>(-(abs(spin_ + m) + b) / 2 + l);
    for (int i = 0; factor <= b; ++i) {
      prefactor *= (l_plus_k + a + static_cast<double>(i)) /
                   (l_plus_k + static_cast<double>(i));
    }
    prefactor = sqrt(prefactor) prefactor *=
        ((m + lambda_table[m + static_cast<int>(l_max_)]
                          [l - min(abs(spin_), abs(m))]) %
         2) == 0
            ? 1.0
            : -1.0;
    // note: Jacobi Polynomial contribution is just 1 for l_min.
    *harmonic = prefactor *
                pow(sin_theta_over_two, static_cast<size_t>(abs(spin_ + m))) *
                pow(cos_theta_over_two, static_cast<size_t>(abs(spin_ - m)));
  }

  void evaluate_swsh_at_l_min_plus_one(
      const gsl::not_null<ComplexDataVector*> harmonic,
      const ComplexDataVector& harmonic_at_l_min, const int l_min,
      const int m) noexcept {
    double a = static_cast<double>(abs(spin_ + m));
    double b = static_cast<double>(abs(spin - m));
    double l_min_plus_k =
        static_cast<double>(-(abs(spin_ + m) + b) / 2 + l_min);
    double prefactor =
        sqrt((2.0 * static_cast<double>(l_min) + 3.0) * (l_min_plus_k + 1.0) *
             (l_min_plus_k + a + b + 1.0) /
             ((2.0 * static_cast<double>(l_min) + 1.0) *
              (l_min_plus_k + a + 1.0) * (l_min_plus_k + b + 1.0)));
    harmonic = prefactor * harmonic_at_l_min *
               (a + 1.0 + 0.5 * (a + b + 2.0) * (cos_theta_ - 1.0));
  }

  void evaluate_swsh_m_recurrence_at_l_min(
      const gsl::not_null<ComplexDataVector*> harmonic, const int l_min,
      const int m) noexcept {
    double a = static_cast<double>(abs(spin_ + m));
    double b = static_cast<double>(abs(spin - m));
    double l_min_plus_k =
        static_cast<double>(-(abs(spin_ + m) + b) / 2 + l_min);

    double prefactor =
        sqrt((2.0 * static_cast<double>(abs(m)) + 1.0) *
             (l_min_plus_k + a + b - 1.0) * (l_min_plus_k + a + b) /
             ((2.0 * static_cast<double>(abs(m)) - 1.0) * (l_min_plus_k + a) *
              (l_min_plus_k + b)));
    int lambda_difference = lambda_table[m + static_cast<int>(l_max_)][0];
    if (m > 0) {
      lambda_difference -= lambda_table[m - 1 + static_cast<int>(l_max_)][0];
    } else {
      lambda_difference -= lambda_table[m + 1 + static_cast<int>(l_max_)][0];
    }
    prefactor *= lambda_difference % 2 == 0 ? 1.0 : -1.0;
    *harmonic = prefactor * sin_theta_over_two_ * cos_theta_over_two_ *
                exp_i_phi_ * *harmonic;
  }

  void clenshaw_sum(const gsl::not_null<ComplexDataVector*> interpolation,
                    const ComplexDataVector& l_min_harmonic,
                    const ComplexDataVector& l_min_plus_one_harmonic,
                    const ComplexModalVector& goldberg_modes,
                    const int m) noexcept {
    // for efficiency, write recurrence results to a cyclic three-element cache
    std::array<ComplexDataVector, 3> recurrence_cache;
    recurrence_cache[2] = ComplexDataVector{interpolation->size(), 0.0};
    recurrence_cache[1] = ComplexDataVector{interpolation->size(), 0.0};
    for (int l = static_cast<int>(l_max_); l > min(abs(spin_), abs(m)); l--) {
      int cache_offset = (l - static_cast<int>(l_max));
      recurrence_cache[(cache_offset) % 3] = goldberg_modes[square(l) + l + m];
      if (l < static_cast<int>(l_max_)) {
        recurrence_cache[(cache_offset) % 3] +=
            alpha_table[m + static_cast<int>(l_max_)]
                       [l - min(abs(spin_), abs(m)) + 1] *
            recurrence_cache[(cache_offset + 1) % 3];
      }
      if (l < static_cast<int>(l_max_) - 1) {
        recurrence_cache[(cache_offset) % 3] +=
            beta_table[m + static_cast<int>(l_max_)]
                      [l - min(abs(spin_), abs(m)) + 2] *
            recurrence_cache[(cache_offset + 2) % 3];
      }
    }
    int l_min = min(abs(spin_), abs(m));
    int cache_offset = (l_min - static_cast<int>(l_max));
    *interpolation +=
        l_min_harmonic * goldberg_modes[square(l_min) + l_min + m] +
        l_min_plus_one_harmonic * recurrence_cache[(cache_offset + 1) % 3] +
        l_min_harmonic * recurrence_cache[(cache_offset + 2) % 3] *
            beta_table[m + static_cast(l_max_)][2];
  }

  int spin_;
  size_t l_max_;
  DataVector cos_theta_;
  DataVector cos_theta_over_two_;
  DataVector sin_theta_over_two_;
  ComplexDataVector exp_i_phi_;
  // Tables are stored in a triangular m-varies-fastest manner. The first
  // element for each m is for l = min(|m|, |s|), which is the first needed in
  // the recurrence relations
  std::vector<std::vector<DataVector>> alpha_table;
  std::vector<std::vector<double>> beta_table;
  std::vector<std::vector<int>> lambda_table;
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
