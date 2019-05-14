// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cmath>

#include "DataStructures/Matrix.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/LinearOperators.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

void logical_partial_directional_derivative_of_complex(
    const gsl::not_null<ComplexDataVector*> d_u, const ComplexDataVector& u,
    const Mesh<3>& mesh, size_t dimension_to_differentiate) noexcept {
  auto matrix_array = make_array(Matrix{}, Matrix{}, Matrix{});
  matrix_array[dimension_to_differentiate] = Spectral::differentiation_matrix(
      mesh.slice_through(dimension_to_differentiate));
  apply_matrices(d_u, matrix_array, u, mesh.extents());
}
