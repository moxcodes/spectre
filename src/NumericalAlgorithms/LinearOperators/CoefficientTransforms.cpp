// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"

#include <array>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"  // IWYU pragma: keep
#include "DataStructures/ModalVector.hpp"
#include "Domain/Mesh.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Matrix

namespace {
template <size_t Dim, size_t... Is>
std::array<Matrix, Dim> make_transform_matrices(
    const Mesh<Dim>& mesh, const bool nodal_to_modal,
    std::index_sequence<Is...> /*meta*/) noexcept {
  return nodal_to_modal
             ? std::array<Matrix, Dim>{{Spectral::nodal_to_modal_matrix(
                   mesh.slice_through(Is))...}}
             : std::array<Matrix, Dim>{{Spectral::modal_to_nodal_matrix(
                   mesh.slice_through(Is))...}};
}
}  // namespace

template <typename ModalVectorType, typename NodalVectorType, size_t Dim>
void to_modal_coefficients(
    const gsl::not_null<ModalVectorType*> modal_coefficients,
    const NodalVectorType& nodal_coefficients, const Mesh<Dim>& mesh) noexcept {
  if (modal_coefficients->size() != nodal_coefficients.size()) {
    ASSERT(modal_coefficients->is_owning(),
           "Cannot resize a non-owning ModalVector");
    *modal_coefficients = ModalVectorType(nodal_coefficients.size());
  }
  apply_matrices<ModalVectorType>(
      modal_coefficients,
      make_transform_matrices<Dim>(mesh, true, std::make_index_sequence<Dim>{}),
      nodal_coefficients, mesh.extents());
}

template <size_t Dim>
ModalVector to_modal_coefficients(const DataVector& nodal_coefficients,
                                  const Mesh<Dim>& mesh) noexcept {
  ModalVector modal_coefficients(nodal_coefficients.size());
  to_modal_coefficients(make_not_null(&modal_coefficients), nodal_coefficients,
                        mesh);
  return modal_coefficients;
}

template <size_t Dim>
ComplexModalVector to_modal_coefficients(
    const ComplexDataVector& nodal_coefficients,
    const Mesh<Dim>& mesh) noexcept {
  ComplexModalVector modal_coefficients(nodal_coefficients.size());
  to_modal_coefficients(make_not_null(&modal_coefficients), nodal_coefficients,
                        mesh);
  return modal_coefficients;
}

template <typename ModalVectorType, typename NodalVectorType, size_t Dim>
void to_nodal_coefficients(
    const gsl::not_null<NodalVectorType*> nodal_coefficients,
    const ModalVectorType& modal_coefficients, const Mesh<Dim>& mesh) noexcept {
  if (nodal_coefficients->size() != modal_coefficients.size()) {
    ASSERT(nodal_coefficients->is_owning(),
           "Cannot resize a non-owning DataVector");
    *nodal_coefficients = NodalVectorType(modal_coefficients.size());
  }
  apply_matrices<NodalVectorType>(
      nodal_coefficients,
      make_transform_matrices<Dim>(mesh, false,
                                   std::make_index_sequence<Dim>{}),
      modal_coefficients, mesh.extents());
}

template <size_t Dim>
DataVector to_nodal_coefficients(const ModalVector& modal_coefficients,
                                 const Mesh<Dim>& mesh) noexcept {
  DataVector nodal_coefficients(modal_coefficients.size());
  to_nodal_coefficients(make_not_null(&nodal_coefficients), modal_coefficients,
                        mesh);
  return nodal_coefficients;
}

template <size_t Dim>
ComplexDataVector to_nodal_coefficients(
    const ComplexModalVector& modal_coefficients,
    const Mesh<Dim>& mesh) noexcept {
  ComplexDataVector nodal_coefficients(modal_coefficients.size());
  to_nodal_coefficients(make_not_null(&nodal_coefficients), modal_coefficients,
                        mesh);
  return nodal_coefficients;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(r, data)                                       \
  template void to_modal_coefficients(                             \
      const gsl::not_null<ModalVector*> modal_coefficients,        \
      const DataVector& nodal_coefficients,                        \
      const Mesh<GET_DIM(data)>& mesh) noexcept;                   \
  template void to_modal_coefficients(                             \
      const gsl::not_null<ComplexModalVector*> modal_coefficients, \
      const ComplexDataVector& nodal_coefficients,                 \
      const Mesh<GET_DIM(data)>& mesh) noexcept;                   \
  template ModalVector to_modal_coefficients(                      \
      const DataVector& nodal_coefficients,                        \
      const Mesh<GET_DIM(data)>& mesh) noexcept;                   \
  template ComplexModalVector to_modal_coefficients(               \
      const ComplexDataVector& nodal_coefficients,                 \
      const Mesh<GET_DIM(data)>& mesh) noexcept;                   \
  template void to_nodal_coefficients(                             \
      const gsl::not_null<DataVector*> nodal_coefficients,         \
      const ModalVector& modal_coefficients,                       \
      const Mesh<GET_DIM(data)>& mesh) noexcept;                   \
  template void to_nodal_coefficients(                             \
      const gsl::not_null<ComplexDataVector*> nodal_coefficients,  \
      const ComplexModalVector& modal_coefficients,                \
      const Mesh<GET_DIM(data)>& mesh) noexcept;                   \
  template DataVector to_nodal_coefficients(                       \
      const ModalVector& modal_coefficients,                       \
      const Mesh<GET_DIM(data)>& mesh) noexcept;                   \
  template ComplexDataVector to_nodal_coefficients(                \
      const ComplexModalVector& modal_coefficients,                \
      const Mesh<GET_DIM(data)>& mesh) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))
#undef GET_DIM
#undef INSTANTIATE
