// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
template <typename StepChooserRegistrars>
class StepChooser;
/// \endcond

/// \ingroup TimeSteppersGroup
///
/// Holds all the StepChoosers
namespace StepChoosers {
/// Holds all the StepChooser registrars
///
/// These can be passed in a list to the template argument of
/// StepChooser to choose which StepChoosers can be constructed.
namespace Registrars {}

CREATE_HAS_TYPE_ALIAS(slab_choosers)
CREATE_HAS_TYPE_ALIAS_V(slab_choosers)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(slab_choosers)
CREATE_HAS_TYPE_ALIAS(step_choosers)
CREATE_HAS_TYPE_ALIAS_V(step_choosers)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(step_choosers)
CREATE_HAS_TYPE_ALIAS(compute_tags)
CREATE_HAS_TYPE_ALIAS_V(compute_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(compute_tags)

template <typename Metavariables>
using step_chooser_compute_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        typename StepChooser<tmpl::remove_duplicates<tmpl::append<
            get_step_choosers_or_default_t<Metavariables, tmpl::list<>>,
            get_slab_choosers_or_default_t<Metavariables, tmpl::list<>>>>>::
            creatable_classes,
        tmpl::bind<get_compute_tags_or_default_t, tmpl::_1,
                   tmpl::pin<tmpl::list<>>>>>>;
}  // namespace StepChoosers

/// \ingroup TimeSteppersGroup
///
/// StepChoosers suggest upper bounds on step sizes.  Concrete
/// StepChoosers should define operator() returning the magnitude of
/// the desired step (as a double).
///
/// The step choosers valid for the integration being controlled are
/// specified by passing a `tmpl::list` of the corresponding
/// registrars.
template <typename StepChooserRegistrars>
class StepChooser : public PUP::able {
 protected:
  /// \cond HIDDEN_SYMBOLS
  StepChooser() = default;
  StepChooser(const StepChooser&) = default;
  StepChooser(StepChooser&&) = default;
  StepChooser& operator=(const StepChooser&) = default;
  StepChooser& operator=(StepChooser&&) = default;
  /// \endcond

 public:
  ~StepChooser() override = default;

  WRAPPED_PUPable_abstract(StepChooser);  // NOLINT

  using creatable_classes = Registration::registrants<StepChooserRegistrars>;

  /// The `last_step_magnitude` parameter describes the step size to be
  /// adjusted.  It may be the step size or the slab size, or may be
  /// infinite if the appropriate size cannot be determined.
  template <typename Metavariables, typename DbTags>
  std::pair<double, bool> desired_step(
      const gsl::not_null<db::DataBox<DbTags>*> box,
      const double last_step_magnitude,
      const Parallel::GlobalCache<Metavariables>& cache) const noexcept {
    ASSERT(last_step_magnitude > 0.,
           "Passed non-positive step magnitude: " << last_step_magnitude);
    const auto result =
        call_with_dynamic_type<std::pair<double, bool>, creatable_classes>(
            this, [&last_step_magnitude, &box,
                   &cache](const auto* const chooser) noexcept {
              using chooser_type = typename std::decay_t<decltype(*chooser)>;
              return db::mutate_apply<typename chooser_type::return_tags,
                                      typename chooser_type::argument_tags>(
                  *chooser, box, last_step_magnitude, cache);
            });
    ASSERT(
        result > 0.,
        "StepChoosers should always return positive values.  Got " << result);
    return result;
  }

  /// The `last_step_magnitude` parameter describes the slab size to be
  /// adjusted; It may be infinite if the appropriate size cannot be determined.
  ///
  /// This function is distinct from `desired_step` because the slab change
  /// decision must be callable from an event (so cannot store state information
  /// in the \ref DataBoxGroup "DataBox"), and we do not have the capability to
  /// reject a slab so this function returns only a `double` indicating the
  /// desired slab size.
  template <typename Metavariables, typename DbTags>
  double desired_slab(
      const double last_step_magnitude,
      const db::DataBox<DbTags>& box,
      const Parallel::GlobalCache<Metavariables>& cache) const noexcept {
    ASSERT(last_step_magnitude > 0.,
           "Passed non-positive step magnitude: " << last_step_magnitude);
    const auto result =
        call_with_dynamic_type<std::pair<double, bool>, creatable_classes>(
            this, [&last_step_magnitude, &box,
                   &cache](const auto* const chooser) noexcept {
              return db::apply(*chooser, box, last_step_magnitude, cache);
            });
    ASSERT(
        result.first > 0.,
        "StepChoosers should always return positive values.  Got " << result);
    return result.first;
  }
};
