// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"

namespace Cce {

// TODO dox
using bondi_hypersurface_step_tags =
    tmpl::list<Tags::BondiBeta, Tags::BondiQ, Tags::BondiU, Tags::BondiW,
               Tags::BondiH>;

namespace detail {
template <typename Tag>
struct TagsToComputeForImpl {
  using pre_swsh_derivative_tags = tmpl::list<>;
  using second_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
};

template <>
struct TagsToComputeForImpl<Tags::BondiBeta> {
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Dy<Tags::BondiJ>, Tags::Dy<Tags::Dy<Tags::BondiJ>>>;
  using second_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
};

// Note: Due to the order in which Jacobians for the conversion between
// numerical and Bondi spin-weighted derivatives, all of the higher (second)
// spin-weighted derivatives must be computed AFTER the eth(dy(bondi)) values
// (which act as inputs to those conversions), so must appear later in the
// `swsh_derivative_tags` typelists.
template <>
struct TagsToComputeForImpl<Tags::BondiQ> {
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Dy<Tags::BondiBeta>, Tags::Dy<Tags::Dy<Tags::BondiBeta>>,
                 ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
                 ::Tags::Multiplies<Tags::BondiJbar, Tags::Dy<Tags::BondiJ>>>;
  using swsh_derivative_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiBeta>,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJbar, Tags::Dy<Tags::BondiJ>>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiJ>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                       Spectral::Swsh::Tags::Ethbar>>;
  using second_swsh_derivative_tags = tmpl::list<>;
};

template <>
struct TagsToComputeForImpl<Tags::BondiU> {
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Exp2Beta, Tags::Dy<Tags::BondiQ>,
                 Tags::Dy<Tags::Dy<Tags::BondiQ>>>;
  using second_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
};

template <>
struct TagsToComputeForImpl<Tags::BondiW> {
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Dy<Tags::BondiU>, Tags::Dy<Tags::Dy<Tags::BondiU>>,
                 Tags::Dy<::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>>,
                 Tags::Dy<Spectral::Swsh::Tags::Derivative<
                     Tags::BondiJ, Spectral::Swsh::Tags::Ethbar>>>;
  using swsh_derivative_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiBeta>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<
          Tags::Dy<::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>>,
          Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<
          Tags::Dy<::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiU>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::EthEth>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::EthEthbar>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
          Spectral::Swsh::Tags::EthEthbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                       Spectral::Swsh::Tags::EthbarEthbar>>;
  // Currently, the `eth_ethbar_j` term is the single instance of a swsh
  // derivative needing nested `Spectral::Swsh::Tags::Derivatives` steps to
  // compute. The reason is that if we do not do this in two steps, there are
  // intermediate terms in the Jacobian which depend on eth_j, which as a spin 3
  // quantity is not computable with libsharp. If `eth_ethbar_j` becomes not
  // needed, the remaining `second_swsh_derivative_tags` can be merged to the
  // end of `swsh_derivative_tags` and the corresponding computational steps
  // from `ComputeSwshDerivatives.hpp` removed.
  using second_swsh_derivative_tags =
      tmpl::list<Spectral::Swsh::Tags::Derivative<
          Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                           Spectral::Swsh::Tags::Ethbar>,
          Spectral::Swsh::Tags::Eth>>;
};

template <>
struct TagsToComputeForImpl<Tags::BondiH> {
  using pre_swsh_derivative_tags =
      tmpl::list<::Tags::Multiplies<Tags::BondiJbar, Tags::BondiU>,
                 ::Tags::Multiplies<Tags::BondiUbar, Tags::Dy<Tags::BondiJ>>,
                 Tags::JbarQMinus2EthBeta, Tags::Dy<Tags::BondiW>>;
  using swsh_derivative_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<Tags::BondiQ, Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiU, Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiUbar, Tags::Dy<Tags::BondiJ>>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJbar, Tags::Dy<Tags::BondiJ>>,
          Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::JbarQMinus2EthBeta,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJbar, Tags::BondiU>,
          Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiQ,
                                       Spectral::Swsh::Tags::Ethbar>>;
  using second_swsh_derivative_tags = tmpl::list<>;
};
}  // namespace detail

/*!
 * \brief A type list for the set of `BoundaryValue` tags needed as an input to
 * any of the template specializations of
 * `PrecomputeCceDependencies`. This is provided for easy and maintainable
 * construction of a `Variables` or `DataBox` with all of the quantities
 * necessary for a Cce computation or component thereof.
 * \details A container of these tags should have size
 * `Spectral::Swsh::number_of_swsh_collocation_points(l_max)`.
 */
using pre_computation_boundary_tags =
    tmpl::list<Tags::BoundaryValue<Tags::BondiR>,
               Tags::BoundaryValue<Tags::DuRDividedByR>>;

/*!
 * \brief A type list for the set of tags computed by the set of
 * template specializations of `PrecomputeCceDepedencies`. This is provided for
 * easy and maintainable construction of a `Variables` or `DataBox` with all of
 * the quantities needed for a Cce computation or component.
 *
 * \details This set of tags is the union of the `integration_independent_tags`
 * typelists in the individual specialization, and if the Cce computation is
 * using the storage model of a collection of `Variables` in a `DataBox, these
 * should be the set of tags in the `integration_independent` `Variables` and
 * should have size `number_of_radial_points *
 * number_of_swsh_collocation_points(l_max)`. All of these tags may be computed
 * at once if using a `DataBox` using the template
 * `mutate_all_precompute_cce_dependencies` or individually using
 * the template specializations `PrecomputeCceDependencies`.
 */
using pre_computation_tags =
    tmpl::list<Tags::DuRDividedByR, Tags::EthRDividedByR,
               Tags::EthEthRDividedByR, Tags::EthEthbarRDividedByR,
               Tags::BondiK, Tags::OneMinusY, Tags::BondiR>;

/*!
 * \brief A type list for a set of tags needed to be computed as input to the
 *  template specialization of `ComputeBondiIntegrand<Tag>`. The
 * union of the sets `pre_swsh_derivative_tags_to_compute_for<Tag>`,
 * `swsh_derivative_tags_to_compute_for<Tag>`, and
 * `second_swsh_derivative_tags_to_compute_for<Tag>` is the full set of inputs
 * for `ComputeBondiIntegrand<Tag>`.
 *
 * \details All tags in this typelist can be computed by the
 * template specialization `ComputePreSwshDerivatives`. In the storage model for
 * which a collection of `Variables` is stored in a `DataBox`, these tags should
 * be in the tag list attributed to the `pre_swsh_derivatives` `Variables` and
 * should have size `number_of_radial_points *
 * number_of_swsh_collocation_points(l_max)`. For the full set of tags that
 * should be put in the `pre_swsh_derivatives` `Variables`, see
 * `all_pre_swsh_derivative_tags`.
 */
template <typename Tag>
using pre_swsh_derivative_tags_to_compute_for =
    typename detail::TagsToComputeForImpl<Tag>::pre_swsh_derivative_tags;

/*!
 * \brief A type list for a set of tags needed to be computed as input to the
 *  template specialization of `ComputeBondiIntegrand<Tag>`. The
 * union of the sets `pre_swsh_derivative_tags_to_compute_for<Tag>`,
 * `swsh_derivative_tags_to_compute_for<Tag>`, and
 * `second_swsh_derivative_tags_to_compute_for<Tag>` is the full set of inputs
 * for `ComputeBondiIntegrand<Tag>`.
 *
 * \details All tags in this typelist can be computed by the
 * template specialization `ComputeSwshDerivatives`. In the storage model for
 * which a collection of `Variables` is stored in a `DataBox`, these tags should
 * be in the tag list attributed to the `swsh_derivatives` `Variables` and
 * should have size `number_of_radial_points *
 * number_of_swsh_collocation_points(l_max)`. For the full set of tags that
 * should be put in the `swsh_derivatives` `Variables`, see
 * `all_swsh_derivative_tags`.
 */
template <typename Tag>
using swsh_derivative_tags_to_compute_for =
    typename detail::TagsToComputeForImpl<Tag>::swsh_derivative_tags;

/*!
 * \brief A type list for a set of tags needed to be computed as input to the
 * template specialization of `ComputeBondiIntegrand<Tag>`. The
 * union of the sets `pre_swsh_derivative_tags_to_compute_for<Tag>`,
 * `swsh_derivative_tags_to_compute_for<Tag>`, and
 * `second_swsh_derivative_tags_to_compute_for<Tag>` is the full set of inputs
 * for `ComputeBondiIntegrand<Tag>`.
 *
 * \details All tags in this typelist can be computed by the
 * template specialization `ComputeSwshDerivatives`. In the storage model for
 * which a collection of `Variables` is stored in a `DataBox`, these tags should
 * be in the tag list attributed to the `swsh_derivatives` `Variables` and
 * should have size `number_of_radial_points *
 * number_of_swsh_collocation_points(l_max)`. For the full set of tags that
 * should be put in the `swsh_derivatives` `Variables`, see
 * `all_swsh_derivative_tags`. \note This set tags must be computed after the
 * set of tags `swsh_derivative_tags_to_compute_for<Tag>`, due to dependencies
 * of Jacobian factors.
 */
template <typename Tag>
using second_swsh_derivative_tags_to_compute_for =
    typename detail::TagsToComputeForImpl<Tag>::second_swsh_derivative_tags;

/*!
 * \brief A type list for the full set of tags needed by any specialization of
 * the template `ComputeBondiIntegrand` that are computed by specializations of
 * `ComputeSwshDerivatives`.
 *
 * \details In the storage model for which a collection of `Variables` is stored
 * in a `DataBox`, this is the full set of tags that should be attributed to
 * the `swsh_derivatives` `Variables` and should have size
 * `number_of_radial_points * number_of_swsh_collocation_points(l_max)`. All of
 * the swsh derivatives for a given tag (e.g. both the first and second
 * derivatives for `Tags::BondiBeta`) may be computed at once if using a
 * `DataBox` via `mutate_all_swsh_derivatives_for_tag`.
 */
using all_swsh_derivative_tags = tmpl::remove_duplicates<
    tmpl::append<swsh_derivative_tags_to_compute_for<Tags::BondiBeta>,
                 second_swsh_derivative_tags_to_compute_for<Tags::BondiBeta>,
                 swsh_derivative_tags_to_compute_for<Tags::BondiQ>,
                 second_swsh_derivative_tags_to_compute_for<Tags::BondiQ>,
                 swsh_derivative_tags_to_compute_for<Tags::BondiU>,
                 second_swsh_derivative_tags_to_compute_for<Tags::BondiU>,
                 swsh_derivative_tags_to_compute_for<Tags::BondiW>,
                 second_swsh_derivative_tags_to_compute_for<Tags::BondiW>,
                 swsh_derivative_tags_to_compute_for<Tags::BondiH>,
                 second_swsh_derivative_tags_to_compute_for<Tags::BondiH>>>;

/*!
 * \brief A type list for the full set of coefficient buffers needed to process
 * all of the tags in `all_swsh_derivative_tags` using batch processing provided
 * in `Spectral::Swsh::compute_swsh_derivatives`.
 *
 * \details In the storage model for which a collection of `Variables` is stored
 * in a `DataBox`, this is the full set of tags that should be attributed to the
 * `swsh_coefficients_buffers` `Variables` and should have size `2 *
 * number_of_radial_points * number_of_swsh_coefficients(l_max)` (the factor of
 * 2 is necessary for the libsharp representation of spin-weighted spherical
 * harmonic coefficients, for reference see the documentation in
 * `TransformJob`). Providing a `Variables` of this form is necessary for the
 * use of the aggregated computation `mutate_all_swsh_derivatives_for_tag`.
 */
using all_transform_buffer_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        all_swsh_derivative_tags,
        tmpl::bind<Spectral::Swsh::coefficient_buffer_tags_for_derivative_tag,
                   tmpl::_1>>>>;

namespace detail {
template <typename BondiTag>
struct AllPreSwshTagsForTagImpl {
  // A convenience utility for including the additional tags computed by the
  // linear solvers in the full typelist as well as the set of tags computed by
  // the `ComputePreSwshDerivatives`
  using type = tmpl::flatten<
      tmpl::list<BondiTag, Tags::Dy<BondiTag>,
                 pre_swsh_derivative_tags_to_compute_for<BondiTag>>>;
};
}  // namespace detail

/*!
 * \brief A type list for the set of tags needed as input to
 * `ComputeBondiIntegrand<Tag>` either directly or because they are needed for
 * `ComputeSwshDerivatives` needed by `ComputeBondiIntegrand<Tag>`.
 */
template <typename Tag>
using all_pre_swsh_derivative_tags_for_tag =
    typename detail::AllPreSwshTagsForTagImpl<Tag>::type;

/*!
 * \brief A type list for the full set of tags needed as input to either
 * `ComputeBondiIntegrand` or `ComputeSwshDerivatives` that are computed by
 * any specialization of `ComputePreSwshDerivatives`.
 *
 * \details In the storage model for which a collection of `Variables` is stored
 * in a `DataBox`,  this is the full set of tags that should be attributed to
 * the `pre_swsh_derivatives` `Variables` and should have size
 * `number_of_radial_points * number_of_swsh_collocation_points(l_max)`. All of
 * the tags for a given Bondi quantity (e.g. the set of tags in
 * `all_pre_swsh_derivative_tags_for_tag<Tags::BondiBeta>`) tags may be computed
 * at once if using a `DataBox` via `mutate_all_pre_swsh_derivatives_for_tag`.
 */
using all_pre_swsh_derivative_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::list<
        all_pre_swsh_derivative_tags_for_tag<Tags::BondiJ>,
        all_pre_swsh_derivative_tags_for_tag<Tags::BondiBeta>,
        all_pre_swsh_derivative_tags_for_tag<Tags::BondiQ>,
        all_pre_swsh_derivative_tags_for_tag<Tags::BondiU>,
        all_pre_swsh_derivative_tags_for_tag<Tags::BondiW>,
        pre_swsh_derivative_tags_to_compute_for<Tags::BondiH>, Tags::BondiH>>>;

}  // namespace Cce
