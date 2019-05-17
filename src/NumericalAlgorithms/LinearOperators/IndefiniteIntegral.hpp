// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
template <size_t>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

// @{
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Compute the indefinite integral of a function in the
 * `dim_to_integrate`. Applying a zero boundary condition on each stripe.
 *
 * \requires number of points in `integrand` and `mesh` are equal.
 */
template <size_t Dim, typename VectorType>
void indefinite_integral(gsl::not_null<VectorType*> integral,
                         const VectorType& integrand, const Mesh<Dim>& mesh,
                         size_t dim_to_integrate) noexcept;

template <size_t Dim, typename VectorType>
VectorType indefinite_integral(const VectorType& integrand,
                               const Mesh<Dim>& mesh,
                               size_t dim_to_integrate) noexcept;
// @}
