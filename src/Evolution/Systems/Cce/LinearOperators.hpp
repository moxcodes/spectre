// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>

#include "DataStructures/Matrix.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

/*!
 * \ingroup CceGroup
 * \brief Computes the partial derivative along a particular direction
 * determined by the `dimension_to_differentiate` field in `DerivativeTag`. The
 * pre-derivative input is taken from `input_with_u` and placed in
 * `output_with_directional_derivative_of_u`
 *
 * \note This is placed in Cce Utilities for its currently narrow use-case. If
 * more general uses desire a partial derivative of complex values, this should
 * be moved to `NumericalAlgorithms`. This utility currently assumes the
 * spatial dimensionality is 3, which would also need to be generalized, likely
 * by creating a wrapping struct with partial template specializations.
 */
void logical_partial_directional_derivative_of_complex(
    const gsl::not_null<ComplexDataVector*> d_u, const ComplexDataVector& u,
    const Mesh<3>& mesh, size_t dimension_to_differentiate) noexcept;
