// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ApparentHorizons/SpherepackIterator.hpp"
#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Transpose.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"

namespace Cce {

/// \brief constructs the collocation values for \f$\cos(\phi)\f$,
/// \f$\cos(\theta)\f$, \f$\sin(\phi)\f$, and \f$\sin(\theta)\f$, returned by
/// `not_null` pointer in that order. These are needed for coordinate
/// transformations from the input cartesian-like coordinates.
void trigonometric_functions_on_swsh_collocation(
    const gsl::not_null<Scalar<DataVector>*> cos_phi,
    const gsl::not_null<Scalar<DataVector>*> cos_theta,
    const gsl::not_null<Scalar<DataVector>*> sin_phi,
    const gsl::not_null<Scalar<DataVector>*> sin_theta,
    const size_t l_max) noexcept;

void cartesian_to_angular_coordinates_and_derivatives(
    const gsl::not_null<tnsr::I<DataVector, 3>*> cartesian_coords,
    const gsl::not_null<tnsr::iJ<DataVector, 3>*> cartesian_to_angular_jacobian,
    const gsl::not_null<tnsr::iJ<DataVector, 3>*>
        inverse_cartesian_to_angular_jacobian,
    const Scalar<DataVector>& cos_phi, const Scalar<DataVector>& cos_theta,
    const Scalar<DataVector>& sin_phi, const Scalar<DataVector>& sin_theta,
    const double extraction_radius) noexcept;

void cartesian_spatial_metric_and_derivatives(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> cartesian_spatial_metric,
    const gsl::not_null<tnsr::II<DataVector, 3>*> inverse_spatial_metric,
    const gsl::not_null<tnsr::ijj<DataVector, 3>*> d_cartesian_spatial_metric,
    const gsl::not_null<tnsr::ii<DataVector, 3>*> dt_cartesian_spatial_metric,
    const tnsr::ii<DataVector, 3>& spatial_metric_coefficients,
    const tnsr::ii<DataVector, 3>& dr_spatial_metric_coefficients,
    const tnsr::ii<DataVector, 3>& dt_spatial_metric_coefficients,
    const tnsr::iJ<DataVector, 3>& inverse_cartesian_to_angular_jacobian,
    const tnsr::I<DataVector, 3>& cartesian_coords,
    const YlmSpherepack& spherical_harmonics, const bool radial_renormalize,
    const size_t l_max) noexcept;

void cartesian_shift_and_derivatives(
    gsl::not_null<tnsr::I<DataVector, 3>*> cartesian_shift,
    gsl::not_null<tnsr::iJ<DataVector, 3>*> d_cartesian_shift,
    gsl::not_null<tnsr::I<DataVector, 3>*> dt_cartesian_shift,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const tnsr::I<DataVector, 3>& shift_coefficients,
    const tnsr::I<DataVector, 3>& dr_shift_coefficients,
    const tnsr::I<DataVector, 3>& dt_shift_coefficients,
    const tnsr::iJ<DataVector, 3>& inverse_cartesian_to_angular_jacobian,
    const tnsr::I<DataVector, 3>& cartesian_coords,
    const YlmSpherepack& spherical_harmonics, const bool radial_renormalize,
    const size_t l_max) noexcept;

void cartesian_lapse_and_derivatives(
    const gsl::not_null<Scalar<DataVector>*> cartesian_lapse,
    const gsl::not_null<tnsr::i<DataVector, 3>*> d_cartesian_lapse,
    const gsl::not_null<Scalar<DataVector>*> dt_cartesian_lapse,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const Scalar<DataVector>& lapse_coefficients,
    const Scalar<DataVector>& dr_lapse_coefficients,
    const Scalar<DataVector>& dt_lapse_coefficients,
    const tnsr::iJ<DataVector, 3>& inverse_cartesian_to_angular_jacobian,
    const tnsr::I<DataVector, 3>& cartesian_coords,
    const YlmSpherepack& spherical_harmonics, const bool radial_renormalize,
    const size_t l_max) noexcept;

void generalized_harmonic_quantities(
    const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi,
    const gsl::not_null<tnsr::aa<DataVector, 3>*> dt_psi,
    const gsl::not_null<tnsr::aa<DataVector, 3>*> psi,
    const gsl::not_null<tnsr::II<DataVector, 3>*> inverse_spatial_metric,
    const tnsr::ii<DataVector, 3>& cartesian_spatial_metric,
    const tnsr::ijj<DataVector, 3>& d_cartesian_spatial_metric,
    const tnsr::ii<DataVector, 3>& dt_cartesian_spatial_metric,
    const tnsr::I<DataVector, 3>& cartesian_shift,
    const tnsr::iJ<DataVector, 3>& d_cartesian_shift,
    const tnsr::I<DataVector, 3>& dt_cartesian_shift,
    const Scalar<DataVector>& cartesian_lapse,
    const tnsr::i<DataVector, 3>& d_cartesian_lapse,
    const Scalar<DataVector>& dt_cartesian_lapse) noexcept;

void null_metric_and_derivative(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> du_null_metric,
    const gsl::not_null<tnsr::aa<DataVector, 3>*> null_metric,
    const tnsr::iJ<DataVector, 3>& cartesian_to_angular_jacobian,
    const tnsr::aa<DataVector, 3>& dt_psi,
    const tnsr::aa<DataVector, 3>& psi) noexcept;

void worldtube_normal_and_derivatives(
    const gsl::not_null<tnsr::iJ<DataVector, 3>*> angular_d_worldtube_normal,
    const gsl::not_null<tnsr::I<DataVector, 3>*> worldtube_normal,
    const gsl::not_null<tnsr::I<DataVector, 3>*> dt_worldtube_normal,
    const tnsr::iJ<DataVector, 3>& cartesian_to_angular_jacobian,
    const Scalar<DataVector>& cos_phi, const Scalar<DataVector>& cos_theta,
    const tnsr::iaa<DataVector, 3>& phi, const tnsr::aa<DataVector, 3>& psi,
    const tnsr::aa<DataVector, 3>& dt_psi, const Scalar<DataVector>& sin_phi,
    const Scalar<DataVector>& sin_theta,
    const tnsr::II<DataVector, 3> inverse_spatial_metric) noexcept;

void null_vector_l_and_derivatives(
    const gsl::not_null<tnsr::iA<DataVector, 3>*> angular_d_null_l,
    const gsl::not_null<tnsr::A<DataVector, 3>*> du_null_l,
    const gsl::not_null<tnsr::A<DataVector, 3>*> null_l,
    const tnsr::iJ<DataVector, 3>& angular_d_worldtube_normal,
    const tnsr::I<DataVector, 3>& dt_worldtube_normal,
    const tnsr::iJ<DataVector, 3>& cartesian_to_angular_jacobian,
    const tnsr::i<DataVector, 3>& d_lapse, const tnsr::iaa<DataVector, 3>& phi,
    const tnsr::iJ<DataVector, 3>& d_shift, const Scalar<DataVector>& dt_lapse,
    const tnsr::aa<DataVector, 3>& dt_psi,
    const tnsr::I<DataVector, 3>& dt_shift, const Scalar<DataVector>& lapse,
    const tnsr::aa<DataVector, 3>& psi, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& wordltube_normal) noexcept;

// this gets the down-index versions of various derivatives of the metric in
// tilded coordinates (lambda-null coordinates)
void dlambda_null_metric_and_inverse(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> dlambda_null_metric,
    const gsl::not_null<tnsr::AA<DataVector, 3>*> dlambda_inverse_null_metric,
    const tnsr::iA<DataVector, 3> angular_d_null_l,
    const tnsr::iJ<DataVector, 3>& cartesian_to_angular_jacobian,
    const tnsr::iaa<DataVector, 3>& phi, const tnsr::aa<DataVector, 3>& dt_psi,
    const tnsr::A<DataVector, 3>& du_null_l,
    const tnsr::AA<DataVector, 3>& inverse_null_metric,
    const tnsr::A<DataVector, 3>& null_l,
    const tnsr::aa<DataVector, 3>& psi) noexcept;

void bondi_r(const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> r,
             const tnsr::aa<DataVector, 3>& null_metric) noexcept;

void d_bondi_r(
    const gsl::not_null<tnsr::a<DataVector, 3>*> d_r,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> r,
    const tnsr::aa<DataVector, 3>& dlambda_null_metric,
    const tnsr::aa<DataVector, 3>& du_null_metric,
    const tnsr::AA<DataVector, 3>& inverse_null_metric, size_t l_max,
    const YlmSpherepack /*spherical_harmonic*/) noexcept;

void dyads(
    const gsl::not_null<tnsr::i<ComplexDataVector, 2>*> down_dyad,
    const gsl::not_null<tnsr::i<ComplexDataVector, 2>*> up_dyad) noexcept;

void beta_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
    const tnsr::a<DataVector, 3>& d_r) noexcept;

void bondi_u_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> bondi_u,
    const tnsr::i<ComplexDataVector, 2>& down_dyad,
    const tnsr::a<DataVector, 3>& d_r,
    const tnsr::AA<DataVector, 3>& inverse_null_metric) noexcept;

void bondi_w_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> bondi_w,
    const tnsr::a<DataVector, 3>& d_r,
    const tnsr::AA<DataVector, 3>& inverse_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r) noexcept;

void bondi_j_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> bondi_j,
    const tnsr::aa<DataVector, 3>& null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>> r,
    const tnsr::i<ComplexDataVector, 2>& up_dyad) noexcept;

void dr_bondi_j(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dr_j,
    const tnsr::aa<DataVector, 3>& dlambda_null_metric,
    const tnsr::a<DataVector, 3>& d_r,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
    const tnsr::i<ComplexDataVector, 2>& up_dyad) noexcept;

void d2lambda_bondi_r(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> d2lambda_r,
    const tnsr::a<DataVector, 3>& d_r,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r) noexcept;

void bondi_q_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> bondi_q,
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> dr_bondi_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& d2lambda_r,
    const tnsr::AA<DataVector, 3>& dlambda_inverse_null_metric,
    const tnsr::a<DataVector, 3>& d_r,
    const tnsr::i<ComplexDataVector, 2> down_dyad,
    const tnsr::i<DataVector, 2> angular_d_dlambda_r,
    const tnsr::AA<DataVector, 3>& inverse_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_u) noexcept;

void bondi_h_worldtube_data(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> bondi_h,
    const tnsr::a<DataVector, 3>& d_r,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const tnsr::aa<DataVector, 3>& du_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
    const tnsr::i<ComplexDataVector, 2>& up_dyad) noexcept;

/*!
 * \brief Process the worldtube data from metric and derivatives to desired
 * Bondi quantities
 *
 * \details
 * The mathematics are a bit complicated for all of the coordinate
 * transformations that are necessary to obtain the Bondi gauge quantities.
 * For full mathematical details, see ...
 *
 * This function takes as input the full set of ADM metric data and its radial
 * and time derivatives on a two-dimensional surface of constant r and t in
 * numerical coordinates. This data must be provided as spherical harmonic
 * coefficients in the SpherePack format (\see SpherePackIterator). This data
 * is provided in  nine `Tensor`s. For further details on the definition of
 * these quantities, see [genralized harmonic paper].
 *
 * Sufficient tags to provide full worldtube boundary data at a particular
 * time are set in `bondi_boundary_data`. In particular, the following tags
 * are set:
 * - Cce::Tags::Beta
 * - Cce::Tags::U
 * - Cce::Tags::Q
 * - Cce::Tags::W
 * - Cce::Tags::J
 * - Cce::Tags::H
 * - Cce::Tags::R
 * - Cce::Tags::Du<Cce::Tags::R>
 *
 * The mathematical transformations are implemented as a set of individual
 * cascaded functions below. The details of the manipulations that are
 * performed to the input data may be found in the individual functions
 * themselves, which are called in the following order:
 * - `trigonometric_functions_on_swsh_collocation()`
 */
template <typename BoundaryTags>
void create_bondi_boundary_data_from_cauchy(
    const gsl::not_null<Variables<BoundaryTags>*> bondi_boundary_data,
    const tnsr::ii<DataVector, 3>& spatial_metric_coefficients,
    const tnsr::ii<DataVector, 3>& dt_spatial_metric_coefficients,
    const tnsr::ii<DataVector, 3>& dr_spatial_metric_coefficients,
    const tnsr::I<DataVector, 3>& shift_coefficients,
    const tnsr::I<DataVector, 3>& dt_shift_coefficients,
    const tnsr::I<DataVector, 3>& dr_shift_coefficients,
    const Scalar<DataVector>& lapse_coefficients,
    const Scalar<DataVector>& dt_lapse_coefficients,
    const Scalar<DataVector>& dr_lapse_coefficients,
    const double extraction_radius, const size_t l_max,
    const YlmSpherepack spherical_harmonic,
    const bool radial_renormalize) noexcept {
  // optimization note: revisit to merge most allocations into this variables
  size_t size = Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // This needs to be restructured to first move everything to an angular
  // basis due to the way that things are provided from the input file.
  Scalar<DataVector> sin_theta{size};
  Scalar<DataVector> cos_theta{size};
  Scalar<DataVector> sin_phi{size};
  Scalar<DataVector> cos_phi{size};
  trigonometric_functions_on_swsh_collocation(
      make_not_null(&cos_phi), make_not_null(&cos_theta),
      make_not_null(&sin_phi), make_not_null(&sin_theta), l_max);

  // NOTE: to handle the singular values of polar coordinates, the phi
  // components of all tensors are scaled according to their sin(theta)
  // prefactors.
  // so, any down-index component get<2>(A) represents 1/sin(theta) A_\phi,
  // and any up-index component get<2>(A) represents sin(theta) A^\phi.
  // This holds for Jacobians, and so direct application of the Jacobians
  // brings the factors through.

  tnsr::I<DataVector, 3> cartesian_coords{size};
  tnsr::iJ<DataVector, 3> cartesian_to_angular_jacobian{size};
  tnsr::iJ<DataVector, 3> inverse_cartesian_to_angular_jacobian{size};
  cartesian_to_angular_coordinates_and_derivatives(
      make_not_null(&cartesian_coords),
      make_not_null(&cartesian_to_angular_jacobian),
      make_not_null(&inverse_cartesian_to_angular_jacobian), cos_phi, cos_theta,
      sin_phi, sin_theta, extraction_radius);

  tnsr::ii<DataVector, 3> cartesian_spatial_metric{size};
  tnsr::II<DataVector, 3> inverse_spatial_metric{size};
  tnsr::ijj<DataVector, 3> d_cartesian_spatial_metric{size};
  tnsr::ii<DataVector, 3> dt_cartesian_spatial_metric{size};
  cartesian_spatial_metric_and_derivatives(
      make_not_null(&cartesian_spatial_metric),
      make_not_null(&inverse_spatial_metric),
      make_not_null(&d_cartesian_spatial_metric),
      make_not_null(&dt_cartesian_spatial_metric), spatial_metric_coefficients,
      dr_spatial_metric_coefficients, dt_spatial_metric_coefficients,
      inverse_cartesian_to_angular_jacobian, cartesian_coords,
      spherical_harmonic, radial_renormalize, l_max);

  tnsr::I<DataVector, 3> cartesian_shift{size};
  tnsr::iJ<DataVector, 3> d_cartesian_shift{size};
  tnsr::I<DataVector, 3> dt_cartesian_shift{size};
  cartesian_shift_and_derivatives(
      make_not_null(&cartesian_shift), make_not_null(&d_cartesian_shift),
      make_not_null(&dt_cartesian_shift), inverse_spatial_metric,
      shift_coefficients, dr_shift_coefficients, dt_shift_coefficients,
      inverse_cartesian_to_angular_jacobian, cartesian_coords,
      spherical_harmonic, radial_renormalize, l_max);

  Scalar<DataVector> cartesian_lapse{size};
  tnsr::i<DataVector, 3> d_cartesian_lapse{size};
  Scalar<DataVector> dt_cartesian_lapse{size};
  cartesian_lapse_and_derivatives(
      make_not_null(&cartesian_lapse), make_not_null(&d_cartesian_lapse),
      make_not_null(&dt_cartesian_lapse), inverse_spatial_metric,
      lapse_coefficients, dr_lapse_coefficients, dt_lapse_coefficients,
      inverse_cartesian_to_angular_jacobian, cartesian_coords,
      spherical_harmonic, radial_renormalize, l_max);

  tnsr::iaa<DataVector, 3> phi{size};
  tnsr::aa<DataVector, 3> dt_psi{size};
  tnsr::aa<DataVector, 3> psi{size};
  generalized_harmonic_quantities(
      make_not_null(&phi), make_not_null(&dt_psi), make_not_null(&psi),
      make_not_null(&inverse_spatial_metric), cartesian_spatial_metric,
      d_cartesian_spatial_metric, dt_cartesian_spatial_metric, cartesian_shift,
      d_cartesian_shift, dt_cartesian_shift, cartesian_lapse, d_cartesian_lapse,
      dt_cartesian_lapse);

  tnsr::aa<DataVector, 3> null_metric{size};
  tnsr::aa<DataVector, 3> du_null_metric{size};
  null_metric_and_derivative(make_not_null(&du_null_metric),
                             make_not_null(&null_metric),
                             cartesian_to_angular_jacobian, dt_psi, psi);

  tnsr::AA<DataVector, 3> inverse_null_metric =
      determinant_and_inverse(null_metric).second;

  tnsr::I<DataVector, 3> dt_worldtube_normal{size};
  tnsr::I<DataVector, 3> worldtube_normal{size};
  tnsr::iJ<DataVector, 3> angular_d_worldtube_normal{size};
  worldtube_normal_and_derivatives(
      make_not_null(&angular_d_worldtube_normal),
      make_not_null(&worldtube_normal), make_not_null(&dt_worldtube_normal),
      cartesian_to_angular_jacobian, cos_phi, cos_theta, phi, psi, dt_psi,
      sin_phi, sin_theta, inverse_spatial_metric);

  tnsr::iA<DataVector, 3> angular_d_null_l{size};
  tnsr::A<DataVector, 3> du_null_l{size};
  tnsr::A<DataVector, 3> null_l{size};
  null_vector_l_and_derivatives(
      make_not_null(&angular_d_null_l), make_not_null(&du_null_l),
      make_not_null(&null_l), angular_d_worldtube_normal, dt_worldtube_normal,
      cartesian_to_angular_jacobian, d_cartesian_lapse, phi, d_cartesian_shift,
      dt_cartesian_lapse, dt_psi, dt_cartesian_shift, cartesian_lapse, psi,
      cartesian_shift, worldtube_normal);

  SpinWeighted<ComplexDataVector, 0> buffer_for_derivatives{
      get<0>(null_l).size()};
  SpinWeighted<ComplexDataVector, 1> eth_buffer{get<0>(null_l).size()};
  for (size_t a = 0; a < 4; ++a) {
    buffer_for_derivatives.data() =
        std::complex<double>(1.0, 0.0) * null_l.get(a);
    Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
        make_not_null(&eth_buffer), make_not_null(&buffer_for_derivatives),
        l_max);
    angular_d_null_l.get(1, a) = -real(eth_buffer.data());
    angular_d_null_l.get(2, a) = -imag(eth_buffer.data());
    angular_d_null_l.get(0, a) = 0.0;
  }

  tnsr::aa<DataVector, 3> dlambda_null_metric{size};
  tnsr::AA<DataVector, 3> dlambda_inverse_null_metric{size};
  dlambda_null_metric_and_inverse(make_not_null(&dlambda_null_metric),
                                  make_not_null(&dlambda_inverse_null_metric),
                                  angular_d_null_l,
                                  cartesian_to_angular_jacobian, phi, dt_psi,
                                  du_null_l, inverse_null_metric, null_l, psi);

  auto& r = get<Tags::BoundaryValue<Tags::R>>(*bondi_boundary_data);
  bondi_r(make_not_null(&r), null_metric);

  tnsr::a<DataVector, 3> d_r{size};
  d_bondi_r(make_not_null(&d_r), make_not_null(&r), dlambda_null_metric,
            du_null_metric, inverse_null_metric, l_max, spherical_harmonic);
  get(get<Tags::BoundaryValue<Tags::DuRDividedByR>>(*bondi_boundary_data))
      .data() = std::complex<double>{1.0, 0.0} * get<0>(d_r) / get(r).data();

  tnsr::i<ComplexDataVector, 2> down_dyad{size};
  tnsr::i<ComplexDataVector, 2> up_dyad{size};
  dyads(make_not_null(&down_dyad), make_not_null(&up_dyad));

  beta_worldtube_data(make_not_null(&get<Tags::BoundaryValue<Tags::Beta>>(
                          *bondi_boundary_data)),
                      d_r);

  auto& bondi_u = get<Tags::BoundaryValue<Tags::U>>(*bondi_boundary_data);
  bondi_u_worldtube_data(make_not_null(&bondi_u), down_dyad, d_r,
                         inverse_null_metric);

  bondi_w_worldtube_data(
      make_not_null(&get<Tags::BoundaryValue<Tags::W>>(*bondi_boundary_data)),
      d_r, inverse_null_metric, r);

  auto& bondi_j = get<Tags::BoundaryValue<Tags::J>>(*bondi_boundary_data);
  get(bondi_j).data() = ComplexDataVector{size};
  bondi_j_worldtube_data(make_not_null(&bondi_j), null_metric, r, up_dyad);

  auto& dr_j =
      get<Tags::BoundaryValue<Tags::Dr<Tags::J>>>(*bondi_boundary_data);
  get(dr_j).data() = ComplexDataVector{size};
  dr_bondi_j(make_not_null(&dr_j), dlambda_null_metric, d_r, bondi_j, r,
             up_dyad);

  Scalar<SpinWeighted<ComplexDataVector, 0>> d2lambda_r;
  get(d2lambda_r).data() = ComplexDataVector{size};
  d2lambda_bondi_r(make_not_null(&d2lambda_r), d_r, dr_j, bondi_j, r);

  tnsr::i<DataVector, 2> angular_d_dlambda_r{size, 0.0};
  buffer_for_derivatives.data() = std::complex<double>(1.0, 0.0) * get<1>(d_r);
  Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
      make_not_null(&eth_buffer), make_not_null(&buffer_for_derivatives),
      l_max);
  angular_d_dlambda_r.get(0) = -real(eth_buffer.data());
  angular_d_dlambda_r.get(1) = -imag(eth_buffer.data());

  // TODO need also dr_u = d_lambda_u / d_lambda_r

  bondi_q_worldtube_data(
      make_not_null(&get<Tags::BoundaryValue<Tags::Q>>(*bondi_boundary_data)),
      make_not_null(
          &get<Tags::BoundaryValue<Tags::Dr<Tags::U>>>(*bondi_boundary_data)),
      d2lambda_r, dlambda_inverse_null_metric, d_r, down_dyad,
      angular_d_dlambda_r, inverse_null_metric, bondi_j, r, bondi_u);

  auto& bondi_h = get<Tags::BoundaryValue<Tags::H>>(*bondi_boundary_data);
  bondi_h_worldtube_data(make_not_null(&bondi_h), d_r, bondi_j, du_null_metric,
                         r, up_dyad);

  get(get<Tags::BoundaryValue<Tags::SpecH>>(*bondi_boundary_data)).data() =
      get(bondi_h).data() - get<0>(d_r) * get(dr_j).data();
}
}  // namespace Cce
