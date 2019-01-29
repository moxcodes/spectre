// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/ComplexDiagonalModalOperator.hpp"
#include "DataStructures/ComplexModalVector.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/TempBuffer.hpp"  // IWYU pragma: keep
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare ComplexDataVector
// IWYU pragma: no_forward_declare SpinWeighted
// IWYU pragma: no_forward_declare Variables

namespace Spectral {
namespace Swsh {
namespace detail {

const size_t swsh_derivative_maximum_l_max = 200;

// derivative factors needed for compute_spin_weighted_derivative_coefficient
template <typename DerivativeKind>
SPECTRE_ALWAYS_INLINE std::complex<double> derivative_factor(int l,
                                                             int s) noexcept;

template <>
SPECTRE_ALWAYS_INLINE std::complex<double> derivative_factor<Tags::Eth>(
    int l, int s) noexcept {
  return sqrt(static_cast<std::complex<double>>((l - s) * (l + s + 1)));
}

template <>
SPECTRE_ALWAYS_INLINE std::complex<double> derivative_factor<Tags::Ethbar>(
    int l, int s) noexcept {
  return -sqrt(static_cast<std::complex<double>>((l + s) * (l - s + 1)));
}

template <>
SPECTRE_ALWAYS_INLINE std::complex<double> derivative_factor<Tags::EthEth>(
    int l, int s) noexcept {
  return sqrt(static_cast<std::complex<double>>((l - s - 1) * (l + s + 2) *
                                                (l - s) * (l + s + 1)));
}

template <>
SPECTRE_ALWAYS_INLINE std::complex<double>
derivative_factor<Tags::EthbarEthbar>(int l, int s) noexcept {
  return sqrt(static_cast<std::complex<double>>((l + s - 1) * (l - s + 2) *
                                                (l + s) * (l - s + 1)));
}

template <>
SPECTRE_ALWAYS_INLINE std::complex<double> derivative_factor<Tags::EthbarEth>(
    int l, int s) noexcept {
  return static_cast<std::complex<double>>(-(l - s) * (l + s + 1));
}

template <>
SPECTRE_ALWAYS_INLINE std::complex<double> derivative_factor<Tags::EthEthbar>(
    int l, int s) noexcept {
  return static_cast<std::complex<double>>(-(l + s) * (l - s + 1));
}

// For a particular derivative represented by `DerivativeKind` and input spin
// `Spin`, multiplies the derivative spectral factors with
// `pre_derivative_modes`, returning by pointer via parameter `derivative_modes`
template <typename DerivativeKind, int Spin>
void compute_derivative_coefficients(
    size_t l_max, size_t number_of_radial_points,
    gsl::not_null<
        SpinWeighted<ComplexModalVector,
                     Spin + Tags::derivative_spin_weight<DerivativeKind>>*>
        derivative_modes,
    gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*>
        pre_derivative_modes) noexcept;

// Helper function for dealing with the parameter packs in the utilities which
// evaluate several spin-weighted derivatives at once. The `apply` function of
// this struct locates the appropriate mode buffer in the input tuple
// `pre_derivative_mode_tuple`, and calls `compute_derivative_coefficients`,
// deriving the coefficients for `DerivativeTag` and returning by pointer.
template <typename DerivativeTag, typename PreDerivativeTagList>
struct dispatch_to_compute_derivative_coefficients;

template <typename DerivativeTag, typename... PreDerivativeTags>
struct dispatch_to_compute_derivative_coefficients<
    DerivativeTag, tmpl::list<PreDerivativeTags...>> {
  template <int Spin, typename... ModalTypes>
  static void apply(
      const size_t l_max, const size_t number_of_radial_points,
      const gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*>
          derivative_modes,
      const std::tuple<ModalTypes...>& pre_derivative_mode_tuple) noexcept {
    compute_derivative_coefficients<typename DerivativeTag::derivative_kind>(
        l_max, number_of_radial_points, derivative_modes,
        get<tmpl::position_of_first<tmpl::list<PreDerivativeTags...>,
                                    typename DerivativeTag::derived_from>>(
            pre_derivative_mode_tuple));
  }
};

// Helper function for dealing with the parameter packs in the utilities which
// evaluate several spin-weighted derivatives at once. The `apply` function of
// this struct locates the like-spin set of paired modes and nodes, and calls
// the member function of `SwshTransform` or `InverseSwshTransform` to perform
// the spin-weighted transform. This function is a friend of both
// `SwshTransform` and `InverseSwshTransform` to gain access to the private
// `apply_to_vectors`.
template <typename Transform, typename FullTagList>
struct dispatch_to_transform;

template <ComplexRepresentation Representation, typename... TransformTags,
          typename... FullTags>
struct dispatch_to_transform<
    SwshTransform<tmpl::list<TransformTags...>, Representation>,
    tmpl::list<FullTags...>> {
  template <typename... ModalTypes, typename... NodalTypes>
  static void apply(const size_t l_max, const size_t number_of_radial_points,
                    const gsl::not_null<std::tuple<ModalTypes...>*> modal_tuple,
                    const std::tuple<NodalTypes...>& nodal_tuple) noexcept {
    SwshTransform<tmpl::list<TransformTags...>, Representation>::
        apply_to_vectors(
            get<tmpl::position_of_first<tmpl::list<FullTags...>,
                                        TransformTags>>(*modal_tuple)...,
            get<tmpl::position_of_first<tmpl::list<FullTags...>,
                                        TransformTags>>(nodal_tuple)...,
            l_max, number_of_radial_points);
  }
};

template <ComplexRepresentation Representation, typename... TransformTags,
          typename... FullTags>
struct dispatch_to_transform<
    InverseSwshTransform<tmpl::list<TransformTags...>, Representation>,
    tmpl::list<FullTags...>> {
  template <typename... NodalTypes, typename... ModalTypes>
  static void apply(const size_t l_max, const size_t number_of_radial_points,
                    const gsl::not_null<std::tuple<NodalTypes...>*> nodal_tuple,
                    const std::tuple<ModalTypes...>& modal_tuple) noexcept {
    InverseSwshTransform<tmpl::list<TransformTags...>, Representation>::
        apply_to_vectors(
            get<tmpl::position_of_first<tmpl::list<FullTags...>,
                                        TransformTags>>(*nodal_tuple)...,
            *get<tmpl::position_of_first<tmpl::list<FullTags...>,
                                         TransformTags>>(modal_tuple)...,
            l_max, number_of_radial_points);
  }
};

// template 'implementation' for the DataBox mutate-compatible interface to
// spin-weighted derivative evaluation. This impl version is needed to have easy
// access to the `DeDuplicatedDerivedFromTagList` as a parameter pack
template <typename DerivativeTagList, typename DeDuplicatedDerivedFromTagList,
          ComplexRepresentation Representation>
struct SwshDerivativesImpl;

template <typename... DerivativeTags, typename... DeDuplicatedDerivedFromTags,
          ComplexRepresentation Representation>
struct SwshDerivativesImpl<tmpl::list<DerivativeTags...>,
                           tmpl::list<DeDuplicatedDerivedFromTags...>,
                           Representation> {
  using return_tags =
      tmpl::list<DerivativeTags..., Tags::SwshTransform<DerivativeTags>...,
                 Tags::SwshTransform<DeDuplicatedDerivedFromTags>...>;
  using argument_tags = tmpl::list<DeDuplicatedDerivedFromTags..., Tags::LMax,
                                   Tags::NumberOfRadialPoints>;

  static void apply(
      const gsl::not_null<db::item_type<DerivativeTags>*>... derivative_scalars,
      const gsl::not_null<db::item_type<
          Tags::SwshTransform<DerivativeTags>>*>... derivative_buffer_scalars,
      const gsl::not_null<db::item_type<Tags::SwshTransform<
          DeDuplicatedDerivedFromTags>>*>... input_buffer_scalars,
      const db::item_type<DeDuplicatedDerivedFromTags>&... input_scalars,
      const size_t l_max, const size_t number_of_radial_points) noexcept {
    apply_to_vectors(make_not_null(&get(*derivative_buffer_scalars))...,
                     make_not_null(&get(*input_buffer_scalars))...,
                     make_not_null(&get(*derivative_scalars))...,
                     get(input_scalars)..., l_max, number_of_radial_points);
  }

  template <ComplexRepresentation FriendRepresentation,
            typename... DerivativeKinds, typename... ArgumentTypes,
            size_t... Is>
  // NOLINTNEXTLINE(readability-redundant-declaratioon)
  friend void swsh_derivatives_impl(
      size_t, size_t, std::index_sequence<Is...>,
      tmpl::list<DerivativeKinds...>, cpp17::bool_constant<true>,
      const std::tuple<ArgumentTypes...>&) noexcept;

 private:
  // note inputs reordered to accommodate the alternative tag-free functions
  // which call into this function.
  static void apply_to_vectors(
      const gsl::not_null<typename db::item_type<
          Tags::SwshTransform<DerivativeTags>>::type*>... derivative_buffers,
      const gsl::not_null<typename db::item_type<Tags::SwshTransform<
          DeDuplicatedDerivedFromTags>>::type*>... input_buffers,
      const gsl::not_null<
          typename db::item_type<DerivativeTags>::type*>... derivatives,
      const typename db::item_type<
          DeDuplicatedDerivedFromTags>::type&... inputs,
      const size_t l_max, const size_t number_of_radial_points) noexcept {
    // perform the forward transform on the minimal set of input nodal
    // quantities to obtain all of the requested derivatives
    using ForwardTransformList =
        make_transform_list_from_derivative_tags<Representation,
                                                 tmpl::list<DerivativeTags...>>;

    tmpl::for_each<ForwardTransformList>([&number_of_radial_points, &l_max,
                                          &inputs...,
                                          &input_buffers...](auto transform_v) {
      using transform = typename decltype(transform_v)::type;
      auto input_transforms = std::make_tuple(input_buffers...);
      dispatch_to_transform<transform,
                            tmpl::list<DeDuplicatedDerivedFromTags...>>::
          apply(l_max, number_of_radial_points,
                make_not_null(&input_transforms), std::make_tuple(inputs...));
    });

    // apply the modal derivative factors and place the result in the
    // `derivative_buffers`
    EXPAND_PACK_LEFT_TO_RIGHT(
        dispatch_to_compute_derivative_coefficients<
            DerivativeTags, tmpl::list<DeDuplicatedDerivedFromTags...>>::
            apply(l_max, number_of_radial_points, derivative_buffers,
                  std::make_tuple(input_buffers...)));

    // perform the inverse transform on the derivative results, placing the
    // result in the nodal `derivatives` passed by pointer.
    using InverseTransformList =
        make_inverse_transform_list<Representation,
                                    tmpl::list<DerivativeTags...>>;

    tmpl::for_each<InverseTransformList>([&number_of_radial_points, &l_max,
                                          &derivatives...,
                                          &derivative_buffers...](
                                             auto transform_v) {
      using transform = typename decltype(transform_v)::type;
      auto derivative_tuple = std::make_tuple(derivatives...);
      dispatch_to_transform<transform, tmpl::list<DerivativeTags...>>::apply(
          l_max, number_of_radial_points, make_not_null(&derivative_tuple),
          std::make_tuple(derivative_buffers...));
    });
  }
};

// metafunction for determining the tags needed to evaluate the derivative tags
// in `DerivativeTagList`, removing all duplicate tags.
template <typename DerivativeTagList>
struct unique_derived_from_list;

template <typename... DerivativeTags>
struct unique_derived_from_list<tmpl::list<DerivativeTags...>> {
  using type = tmpl::remove_duplicates<
      tmpl::list<typename DerivativeTags::derived_from...>>;
};
}  // namespace detail

/*!
 * \ingroup SpectralGroup
 * \brief A \ref DataBoxGroup mutate-compatible computational struct for
 * computing a set of spin-weighted spherical harmonic derivatives by
 * grouping and batch-computing spin-weighted spherical harmonic transforms.
 *
 * \details A derivative is evaluated for each tag in `DerivativeTagList`. All
 * entries in `DerivativeTagList` must be the tag
 * `Spectral::Swsh::Tags::Derivative<Tag, DerivativeKind>` prefixing the `Tag`
 * to be differentiated, and indicating the spin-weighted derivative
 * `DerivativeKind` to be taken. A \ref DataBoxGroup on which this struct is
 * invoked must contain:
 * - each of the tags in `DerivativeTagList` (the 'returns' of the computation)
 * - each of the tags `Tag` prefixed by `Spectral::Swsh::Tags::Derivative` in
 * `DerivativeTagList` (the 'inputs' of the computation).
 * - each of the tags `Spectral::Swsh::Tags::SwshTransform<DerivativeTag>` for
 *  `DerivativeTag`in `DerivativeTagList` (the 'buffers' for the derivative
 * applied to the modes)
 * - each of the tags `Spectral::Swsh::Tags::SwshTransform<Tag>` for `Tag`
 * prefixed by any `DerivativeTag` in `DerivativeTagList` (the 'buffers' for the
 * transforms of the input data).
 *
 * This function optimizes the derivative taking process by clustering like
 * spins of tags, forward-transforming each spin cluster together, applying the
 * factor for the derivative to each modal vector, re-clustering according to
 * the new spin weights (the derivatives alter the spin weights), and finally
 * inverse-transforming in clusters.
 */
template <typename DerivativeTagList, ComplexRepresentation Representation =
                                          ComplexRepresentation::Interleaved>
using SwshDerivatives = detail::SwshDerivativesImpl<
    DerivativeTagList,
    typename detail::unique_derived_from_list<DerivativeTagList>::type,
    Representation>;

/*!
 * \ingroup SpectralGroup
 * \brief Produces a `SpinWeighted<ComplexModalVector, Spin>` of the appropriate
 * size to be used as a modal buffer for `Spectral::Swsh::SwshDerivatives` or
 * `Spectral::Swsh::swsh_derivatives`.
 *
 * \details The `Spectral::Swsh::swsh_derivatives` and
 * `Spectral::Swsh::SwshDerivatives` interfaces require that calling code
 * provides a buffer for the intermediate transform results, to ensure that
 * callers are aware of the allocations and can suitably reuse buffers if
 * possible. This utility eases the creation of those buffers.
 */
template <int Spin>
auto swsh_buffer(const size_t l_max,
                 const size_t number_of_radial_points) noexcept {
  return SpinWeighted<ComplexModalVector, Spin>{
      size_of_libsharp_coefficient_vector(l_max) * number_of_radial_points};
}

namespace detail {
// template 'implementation' for the `swsh_derivatives` function below which
// evaluates an arbitrary number of derivatives, and places them in the set of
// nodal containers passed by pointer.
template <ComplexRepresentation Representation, typename... DerivativeKinds,
          typename... ArgumentTypes, size_t... Is>
void swsh_derivatives_impl(
    const size_t l_max, const size_t number_of_radial_points,
    std::index_sequence<Is...> /*meta*/,
    tmpl::list<DerivativeKinds...> /*meta*/,
    cpp17::bool_constant<true> /*buffers_included_in_arguments*/,
    const std::tuple<ArgumentTypes...>& argument_tuple) noexcept {
  SwshDerivatives<
      tmpl::list<Tags::Derivative<
          ::Tags::SpinWeighted<
              ::Tags::TempScalar<Is, ComplexDataVector>,
              std::integral_constant<
                  int, std::decay_t<decltype(get<Is + 3 * sizeof...(Is)>(
                           argument_tuple))>::spin>>,
          DerivativeKinds>...>,
      Representation>::apply_to_vectors(get<Is>(argument_tuple)...,
                                        get<Is + sizeof...(Is)>(
                                            argument_tuple)...,
                                        get<Is + 2 * sizeof...(Is)>(
                                            argument_tuple)...,
                                        get<Is + 3 * sizeof...(Is)>(
                                            argument_tuple)...,
                                        l_max, number_of_radial_points);
}

template <ComplexRepresentation Representation, typename... DerivativeKinds,
          typename... ArgumentTypes, size_t... Is>
void swsh_derivatives_impl(
    const size_t l_max, const size_t number_of_radial_points,
    std::index_sequence<Is...> index_sequence,
    tmpl::list<DerivativeKinds...> derivative_kinds,
    cpp17::bool_constant<false> /*buffers_included_in_arguments*/,
    const std::tuple<ArgumentTypes...>& argument_tuple) noexcept {
  auto derivative_buffer_tuple = std::make_tuple(
      swsh_buffer<std::decay_t<decltype(*get<Is>(argument_tuple))>::spin>(
          l_max, number_of_radial_points)...);
  auto input_buffer_tuple = std::make_tuple(
      swsh_buffer<std::decay_t<decltype(get<Is + sizeof...(Is)>(
          argument_tuple))>::spin>(l_max, number_of_radial_points)...);
  swsh_derivatives_impl<Representation>(
      l_max, number_of_radial_points, index_sequence, derivative_kinds,
      cpp17::bool_constant<true>{},
      std::make_tuple(make_not_null(&get<Is>(derivative_buffer_tuple))...,
                      make_not_null(&get<Is>(input_buffer_tuple))...,
                      get<Is>(argument_tuple)...,
                      get<Is + sizeof...(Is)>(argument_tuple)...));
}
}  // namespace detail

/*!
 * \ingroup SpectralGroup
 * \brief Evaluate all of the spin-weighted derivatives in `DerivKindList` on
 * input `SpinWeighted<ComplexDataVector, Spin>` collocation data, returning by
 * pointer.
 *
 * \details This function provides two interfaces, one in which the caller
 * provides the intermediate coefficient buffers needed during the computation
 * of the derivatives, and one in which those buffers are temporarily allocated
 * during the derivative function calls.
 *
 * For the interface in which the caller does not provide buffers, the arguments
 * must take the following structure (enforced by internal function calls):
 *
 * - `size_t l_max` : angular resolution for the spherical representation
 * - `size_t number_of_radial_points` : radial resolution (number of consecutive
 * blocks to evaluate derivatives, for each input vector )
 * - for each `DerivKind` in `DerivKindList`, a
 * `gsl::not_null<SpinWeighted<ComplexDataVector, Spin +
 * Tags::derivative_spin_weight<DerivKind>>>` : the output of the derivative
 * evaluation
 * - for each `DerivKind` in `DerivKindList`, a `const
 * SpinWeighted<ComplexDataVector, Spin>` (where the `Spin` for these arguments
 * matches the corresponding vector from the previous set) : the input to the
 * derivative evaluation.
 *
 * For the interface in which the caller does provide buffers, the arguments
 * must take the following structure (enforced by internal function calls):
 *
 * - `size_t l_max` : angular resolution for the spherical representation
 * - `size_t number_of_radial_points` : radial resolution (number of consecutive
 * blocks to evaluate derivatives, for each input vector )
 * - for each `DerivKind` in `DerivKindList`, a
 * `gsl::not_null<SpinWeighted<ComplexModalVector, Spin +
 * Tags::derivative_spin_weight<DerivKind>>>` : the buffer for the spin-weighted
 * spherical harmonic modes of the derivative quantities.
 * - for each `DerivKind` in `DerivKindList`, a `const
 * SpinWeighted<ComplexModalVector, Spin>` (where the `Spin` for these arguments
 * matches the corresponding vector from the previous set) : the buffer for the
 * spin-weighted spherical harmonic modes of the input quantities.
 * - for each `DerivKind` in `DerivKindList`, a
 * `gsl::not_null<SpinWeighted<ComplexDataVector, Spin +
 * Tags::derivative_spin_weight<DerivKind>>>` : the output of the derivative
 * evaluation
 * - for each `DerivKind` in `DerivKindList`, a `const
 * SpinWeighted<ComplexDataVector, Spin>` (where the `Spin` for these arguments
 * matches the corresponding vector from the previous set) : the input to the
 * derivative evaluation.
 *
 * The function `swsh_buffer` assists in generating the modal buffers of
 * appropriate size.
 */
template <
    typename DerivativeKindList,
    ComplexRepresentation Representation = ComplexRepresentation::Interleaved,
    typename... ArgumentTypes>
void swsh_derivatives(const size_t l_max, const size_t number_of_radial_points,
                      const ArgumentTypes&... arguments) noexcept {
  static_assert(
      tmpl::size<DerivativeKindList>::value * 2 == sizeof...(ArgumentTypes) or
          tmpl::size<DerivativeKindList>::value * 4 == sizeof...(ArgumentTypes),
      "When using the tagless `swsh_derivatives` interface, you must "
      "provide either one nodal input and one nodal output per derivative "
      "or one nodal input, one nodal output, and two appropriate "
      "intermediate transform buffers per derivative.");

  detail::swsh_derivatives_impl<Representation>(
      l_max, number_of_radial_points,
      std::make_index_sequence<tmpl::size<DerivativeKindList>::value>{},
      DerivativeKindList{},
      cpp17::bool_constant<tmpl::size<DerivativeKindList>::value * 4 ==
                           sizeof...(ArgumentTypes)>{},
      std::make_tuple(arguments...));
}


/*!
 * \ingroup SpectralGroup
 * \brief evaluate the spin-weighted derivative `DerivKind` on the provided
 * `SpinWeighted<ComplexDataVector, Spin>` collocation data, returning by value.
 */
template <
    typename DerivKind,
    ComplexRepresentation Representation = ComplexRepresentation::Interleaved,
    int Spin>
SpinWeighted<ComplexDataVector, Tags::derivative_spin_weight<DerivKind> + Spin>
swsh_derivative(
    size_t l_max, size_t number_of_radial_points,
    const SpinWeighted<ComplexDataVector, Spin>& to_derive) noexcept;
}  // namespace Swsh
}  // namespace Spectral
