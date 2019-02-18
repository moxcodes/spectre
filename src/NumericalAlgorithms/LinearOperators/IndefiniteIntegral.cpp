// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/IndefiniteIntegral.hpp"
#include "NumericalAlgorithms/LinearOperators/Transpose.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"

// TODO: add a version which takes temporary buffers by pointer
template <>
void indefinite_integral(const gsl::not_null<DataVector*> integral,
                         const DataVector& integrand, const Mesh<3>& mesh,
                         const DataVector& boundary_values,
                         const size_t axis) noexcept {
  Matrix integral_operator =
      Spectral::integration_matrix(mesh.slice_through(axis));

  const size_t integration_size = integrand.size();
  size_t chunk_size = 1;
  for (size_t faster_axis = 0; faster_axis < axis; ++faster_axis) {
    chunk_size *= mesh.extents(faster_axis);
  }

  const size_t number_of_chunks = integration_size / chunk_size;
  size_t number_of_slices = integration_size / mesh.extents(axis);
  DataVector transpose_buffer{integration_size};
  // optimization note: there's effectively extra copies which can be avoided
  // if axis = 0
  transpose(make_not_null(&transpose_buffer), integrand, chunk_size,
            number_of_chunks);

  for (size_t i = 0; i < boundary_values.size(); ++i) {
    transpose_buffer[i * mesh.extents(axis)] = boundary_values[i];
  }

  DataVector integrated_buffer{integration_size};
  dgemm_<true>('N', 'N', mesh.extents(axis), number_of_slices,
               mesh.extents(axis), 1.0, integral_operator.data(),
               mesh.extents(axis), transpose_buffer.data(), mesh.extents(axis),
               0.0, integrated_buffer.data(), mesh.extents(axis));

  transpose(integral, integrated_buffer, number_of_chunks, chunk_size);
}
