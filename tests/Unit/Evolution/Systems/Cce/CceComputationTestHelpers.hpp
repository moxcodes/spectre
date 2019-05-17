// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/VectorAlgebra.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {
namespace TestHelpers {

// explicit power method avoids some troublesome Blaze behavior with powers of
// complex
ComplexDataVector power(const ComplexDataVector& val,
                        const size_t exponent) noexcept;

// For representing a primitive series of powers in inverse r for diagnostic
// computations
template <typename T>
struct RadialPolyCoefficientsFor : db::SimpleTag {
  // for the test, we just store a quadratic in radius
  using type = Scalar<ComplexModalVector>;
  using tag = T;
  static std::string name() {
    return "RadialPolyCoefficientsFor(" + T::name() + ")";
  }
};

// For representing the angular function in a separable quantity.
template <typename T>
struct AngularCollocationsFor : db::SimpleTag {
  using type = typename T::type;
  using tag = T;
  static std::string name() {
    return "AngularCollocationsFor(" + T::name() + ")";
  }
};

void compute_one_minus_y(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        one_minus_y,
    const size_t l_max) noexcept {
  size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  size_t number_of_radial_points =
      get(*one_minus_y).size() / number_of_angular_points;
  const auto& one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                                             number_of_radial_points);
  // iterate through the angular 'chunks' and set them to their 1-y value
  for (size_t i = 0; i < number_of_radial_points; ++i) {
    ComplexDataVector angular_view{
      get(*one_minus_y).data().data() + number_of_angular_points * i,
      number_of_angular_points};
    angular_view = one_minus_y_collocation[i];
  }
}

// needed due to an internal trouble for blaze powers of complex values
ComplexDataVector power(const ComplexDataVector& value,
                        const size_t exponent) noexcept {
  ComplexDataVector return_value{value.size(), 1.0};
  for(size_t i = 0; i < exponent; ++i) {
    return_value *= value;
  }
  return return_value;
}

}  // namespace TestHelpers
}  // namespace Cce
