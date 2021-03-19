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
template <typename StepChooserRegistrars>
class StepChooser;
// IWYU pragma: no_forward_declare db::DataBox
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
CREATE_HAS_TYPE_ALIAS(simple_tags)
CREATE_HAS_TYPE_ALIAS_V(simple_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(simple_tags)

/// Designation for the context in which a step chooser may be used
enum class UsableFor { OnlyLtsStepChoice, OnlySlabChoice, AnyStepChoice };

template <typename Metavariables>
using step_chooser_compute_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        typename StepChooser<tmpl::remove_duplicates<tmpl::append<
            get_step_choosers_or_default_t<Metavariables, tmpl::list<>>,
            get_slab_choosers_or_default_t<Metavariables, tmpl::list<>>>>>::
            creatable_classes,
        tmpl::bind<get_compute_tags_or_default_t, tmpl::_1,
                   tmpl::pin<tmpl::list<>>>>>>;

template <typename Metavariables>
using step_chooser_simple_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        typename StepChooser<tmpl::remove_duplicates<tmpl::append<
            get_step_choosers_or_default_t<Metavariables, tmpl::list<>>,
            get_slab_choosers_or_default_t<Metavariables, tmpl::list<>>>>>::
            creatable_classes,
        tmpl::bind<get_simple_tags_or_default_t, tmpl::_1,
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
  ///
  /// The return value of this function contains the desired step size
  /// and a `bool` indicating whether the step should be accepted. If the `bool`
  /// is `false`, the current time step will be recomputed with a step size
  /// informed by the desired step value returned by this function. The
  /// implementations of the call operator in derived classes should always
  /// return a strictly smaller step than the `last_step_magnitude` when they
  /// return `false` for the second member of the pair (indicating step
  /// rejection).
  template <typename Metavariables, typename DbTags>
  std::pair<double, bool> desired_step(
      const double last_step_magnitude, const db::DataBox<DbTags>& box,
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
    return result;
  }
};
