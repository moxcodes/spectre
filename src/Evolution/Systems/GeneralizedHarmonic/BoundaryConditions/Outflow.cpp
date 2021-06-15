// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Outflow.hpp"

#include <memory>
#include <pup.h>

#include "Utilities/GenerateInstantiations.hpp"

namespace GeneralizedHarmonic::BoundaryConditions {
template <size_t Dim>
Outflow<Dim>::Outflow(CkMigrateMessage* const msg) noexcept
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Outflow<Dim>::get_clone() const noexcept {
  return std::make_unique<Outflow>(*this);
}

template <size_t Dim>
void Outflow<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
}

// NOLINTNEXTLINE
template <size_t Dim>
PUP::able::PUP_ID Outflow<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class Outflow<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace GeneralizedHarmonic::BoundaryConditions
