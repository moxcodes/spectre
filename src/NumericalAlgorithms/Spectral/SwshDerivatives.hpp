// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/ComplexDiagonalModalOperator.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransformJob.hpp"

namespace Spectral {
namespace Swsh {
namespace detail {
// derivative factors needed for compute_spin_weighted_derivative_coefficient
template <typename DerivativeKind>
SPECTRE_ALWAYS_INLINE std::complex<double> derivative_factor(int l,
                                                             int s) noexcept;

template <>
SPECTRE_ALWAYS_INLINE std::complex<double> derivative_factor<Tags::Eth>(
    int l, int s) noexcept {
  return static_cast<std::complex<double>>((l - s) * (l + s + 1));
}

template <>
SPECTRE_ALWAYS_INLINE std::complex<double> derivative_factor<Tags::Ethbar>(
    int l, int s) noexcept {
  return static_cast<std::complex<double>>((l + s) * (l - s + 1));
}

template <>
SPECTRE_ALWAYS_INLINE std::complex<double> derivative_factor<Tags::EthEth>(
    int l, int s) noexcept {
  return static_cast<std::complex<double>>((l - s - 1) * (l + s + 2) * (l - s) *
                                           (l + s + 1));
}

template <>
SPECTRE_ALWAYS_INLINE std::complex<double>
derivative_factor<Tags::EthbarEthbar>(int l, int s) noexcept {
  return static_cast<std::complex<double>>((l + s - 1) * (l - s + 2) * (l + s) *
                                           (l - s + 1));
}

template <>
SPECTRE_ALWAYS_INLINE std::complex<double> derivative_factor<Tags::EthbarEth>(
    int l, int s) noexcept {
  return static_cast<std::complex<double>>(-(l + s) * (l - s + 1));
}

template <>
SPECTRE_ALWAYS_INLINE std::complex<double> derivative_factor<Tags::EthEthbar>(
    int l, int s) noexcept {
  return static_cast<std::complex<double>>(-(l - s) * (l + s + 1));
}

// function that is applied once the derivative factors have been put in the
// appropriate ComplexDiagonalModalOperator container. For most cases, a sqrt is
// applied afterward for efficiency.
template <typename DerivativeKind>
SPECTRE_ALWAYS_INLINE void derivative_factor_post_process(
    gsl::not_null<ComplexDiagonalModalOperator*> coeffs) noexcept;

template <>
SPECTRE_ALWAYS_INLINE void derivative_factor_post_process<Tags::Eth>(
    gsl::not_null<ComplexDiagonalModalOperator*> coeffs) noexcept {
  *coeffs = sqrt(*coeffs);
}

template <>
SPECTRE_ALWAYS_INLINE void derivative_factor_post_process<Tags::Ethbar>(
    gsl::not_null<ComplexDiagonalModalOperator*> coeffs) noexcept {
  *coeffs = -sqrt(*coeffs);
}

template <>
SPECTRE_ALWAYS_INLINE void derivative_factor_post_process<Tags::EthEth>(
    gsl::not_null<ComplexDiagonalModalOperator*> coeffs) noexcept {
  *coeffs = sqrt(*coeffs);
}

template <>
SPECTRE_ALWAYS_INLINE void derivative_factor_post_process<Tags::EthbarEthbar>(
    gsl::not_null<ComplexDiagonalModalOperator*> coeffs) noexcept {
  *coeffs = sqrt(*coeffs);
}

template <>
SPECTRE_ALWAYS_INLINE void derivative_factor_post_process<Tags::EthbarEth>(
    gsl::not_null<ComplexDiagonalModalOperator*> /*coeffs*/) noexcept {}

template <>
SPECTRE_ALWAYS_INLINE void derivative_factor_post_process<Tags::EthEthbar>(
    gsl::not_null<ComplexDiagonalModalOperator*> /*coeffs*/) noexcept {}
}  // namespace detail

/// \ingroup SpectralGroup
/// \brief Obtain the spin weighted spherical harmonic coefficients for the
/// transform of spin-weighted derivatives of fields from the spin weighted
/// spherical harmonic coefficients of the fields.
///
/// \tparam Spin The spin of the fields for which the derivative is taken
/// \tparam DerivativeTagList A `tmpl::list` of
/// `Tags::Coefficient<Tags::Derivative<...>>` Tags representing the set
/// of derivatives to compute.
///
/// \param l_max Maximum l-mode of the coefficients provided.
/// \param number_of_radial_grid_points Number of radial slices contained in the
/// three-dimensional data in the Variables.
/// \param input A Variables which must contain a
/// `Tags::CoefficientTag<tag_to_derive>` for each `tag_to_derive` base tags
/// used in any of the `DerivativeTagList` above.
/// \param output A Variables which must contain Tags for each tag in the
/// `DerivativeTagList`
template <int Spin, typename DerivativeTagList, typename InputVarTagList,
          typename OutputVarTagList>
void compute_derivative_coefficients(
    size_t l_max, size_t number_of_radial_grid_points,
    Variables<InputVarTagList>& input,
    gsl::not_null<Variables<OutputVarTagList>*> output) noexcept {
  // optimization note: this is probably a bit inefficient as the sharp data is
  // unavoidably in a format where l varies fastest, so chunks of the
  // derivatives cannot be assigned at once. It is unclear if it is more
  // efficient to copy, generate, and copy back. It may also be worth caching
  // these derivative vectors, but that may be costly in memory (~l^2 points per
  // spin per l_max)
  sharp_alm_info* alm_info =
      detail::precomputed_coefficients(l_max).get_sharp_alm_info();
  tmpl::for_each<DerivativeTagList>([&input, &output, &alm_info, &l_max,
                                     &number_of_radial_grid_points](auto x) {
    using derivative_tag = typename decltype(x)::type;
    ComplexModalVector& pre_derivative_factors =
        get(get<Tags::SwshTransform<typename derivative_tag::derived_from>>(
                input))
            .data();
    ComplexModalVector& derived_coeffs =
        get(get<Tags::SwshTransform<derivative_tag>>(*output)).data();
    // compute the derivative factors for each radial block
    ComplexDiagonalModalOperator derivative_factors{
        2 * number_of_swsh_coefficients(l_max)};
    for (size_t i = 0; i < 2; i++) {
      for (size_t m = 0; m < static_cast<size_t>(alm_info->nm); m++) {
        for (size_t l = m; l <= l_max; l++) {
          // clang-tidy do not use pointer arithmetic
          // pointer arithmetic is unavoidable due to the libsharp interface
          derivative_factors[static_cast<size_t>(
                                 alm_info->mvstart[m]) +  // NOLINT
                             l * static_cast<size_t>(alm_info->stride) +
                             i * number_of_swsh_coefficients(l_max)] =
              (detail::derivative_factor<
                  typename derivative_tag::derivative_kind>(l, Spin));
        }
      }
    }

    // apply the square-root if necessary
    detail::derivative_factor_post_process<
        typename derivative_tag::derivative_kind>(
        make_not_null(&derivative_factors));
    // apply the sign change as appropriate due to the adjusted spin
    for (size_t i = 0; i < 2; i++) {
      ComplexDiagonalModalOperator view_of_derivative_factor{
          derivative_factors.data() + i * number_of_swsh_coefficients(l_max),
          number_of_swsh_coefficients(l_max)};
      view_of_derivative_factor *=
          sharp_swsh_sign_change(Spin, derivative_tag::spin, i == 0);
    }

    // multiply each radial chunk by the appropriate derivative factors.
    for (size_t i = 0; i < number_of_radial_grid_points; ++i) {
      ComplexModalVector output_mode_chunk{
          derived_coeffs.data() + 2 * i * number_of_swsh_coefficients(l_max),
          2 * number_of_swsh_coefficients(l_max)};
      ComplexModalVector input_mode_chunk{
          pre_derivative_factors.data() +
              2 * i * number_of_swsh_coefficients(l_max),
          2 * number_of_swsh_coefficients(l_max)};
      output_mode_chunk = input_mode_chunk * derivative_factors;
    }
  });
}

/// \ingroup SpectralGroup
template <ComplexRepresentation Representation, typename DerivativeTagList,
          typename InputVarTagList, typename BufferVarTagList,
          typename OutputVarTagList>
void compute_derivatives(gsl::not_null<Variables<InputVarTagList>*> input,
                         gsl::not_null<Variables<BufferVarTagList>*> buffer,
                         gsl::not_null<Variables<OutputVarTagList>*> output,
                         size_t l_max) noexcept {
  size_t number_of_radial_grid_points =
      input->number_of_grid_points() / number_of_swsh_collocation_points(l_max);

  using ForwardJobList =
      make_swsh_transform_job_list_from_derivative_tags<Representation,
                                                   DerivativeTagList>;

  tmpl::for_each<ForwardJobList>(
      [&number_of_radial_grid_points, &l_max, &input, &buffer](auto x) {
        using transform_job_type = typename decltype(x)::type;
        auto transform_job =
            transform_job_type{l_max, number_of_radial_grid_points};
        transform_job.execute_transform(buffer, input);
        compute_derivative_coefficients<
            transform_job_type::spin,
            get_prefix_tags_that_wrap_tags_with_spin<transform_job_type::spin,
                                                     DerivativeTagList>>(
            l_max, number_of_radial_grid_points, *buffer, buffer);
      });

  using InverseJobList =
      make_swsh_transform_job_list<Representation, DerivativeTagList>;

  tmpl::for_each<InverseJobList>(
      [&number_of_radial_grid_points, &l_max, &buffer, &output](auto x) {
        using inverse_transform_job_type = typename decltype(x)::type;

        auto inverse_transform_job =
            inverse_transform_job_type{l_max, number_of_radial_grid_points};
        inverse_transform_job.execute_inverse_transform(output, buffer);
      });
};
}  // namespace Swsh
}  // namespace Spectral
