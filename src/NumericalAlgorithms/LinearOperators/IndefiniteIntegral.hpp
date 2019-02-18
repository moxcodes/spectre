// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
template <size_t>
class Mesh;
/// \endcond

// @{
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Compute the indefinite integral of a grid-function over a manifold
 * using gaussian quadrature, from the lower end of the range.
 *
 * The integral is computed on the reference element by multiplying the
 * DataVector with the Spectral::quadrature_weights() in that
 * dimension.
 * \requires number of points in `integrand` and `mesh` are equal.
 * \param integrand the grid function to integrate.
 * \param mesh the Mesh of the manifold on which `integrand` is located.
 * \param boundary_values a set of boundary values, of a size equal to the
 * number of slices along the axis to be integrated (i.e. the number of
 * integrations to perform).
 * \param axis the number of the axis to be integrated along.
 */
template <size_t Dim>
void indefinite_integral(const gsl::not_null<DataVector*> integral,
                         const DataVector& integrand, const Mesh<Dim>& mesh,
                         const DataVector& boundary_values,
                         const size_t axis) noexcept;

// @}
