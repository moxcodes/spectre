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
struct CalculateNPSpinCoefficient<> {
  using argument_tags = tmpl::list<>;
  using return_tags = tmpl::list<Spectral::Swsh::Tags::LMax,
                                 Spectral::Swsh::Tags::NumberOfRadialPoints>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*>
          np_alpha,
      const size_t l_max, const size_t number_of_radial_points) noexcept {}
};

}  // namespace Cce
