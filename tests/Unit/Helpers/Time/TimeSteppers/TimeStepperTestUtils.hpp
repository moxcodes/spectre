// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"

/// \cond
class LtsTimeStepper;
class TimeStepper;
/// \endcond

namespace TimeStepperTestUtils {

void check_multistep_properties(const TimeStepper& stepper) noexcept;

void check_substep_properties(const TimeStepper& stepper) noexcept;

void integrate_test(const TimeStepper& stepper, size_t number_of_past_steps,
                    double integration_time, double epsilon) noexcept;

void integrate_test_explicit_time_dependence(const TimeStepper& stepper,
                                             size_t number_of_past_steps,
                                             double integration_time,
                                             double epsilon) noexcept;

void integrate_variable_test(const TimeStepper& stepper,
                             size_t number_of_past_steps,
                             double epsilon) noexcept;

void integrate_error_test(const TimeStepper& stepper,
                          size_t number_of_past_steps, double integration_time,
                          double epsilon, size_t num_steps,
                          double error_factor) noexcept;

template <typename F1, typename F2, typename EvolvedType>
void initialize_history(
    Time time,
    const gsl::not_null<TimeSteppers::History<EvolvedType, EvolvedType>*>
        history,
    F1&& analytic, F2&& rhs, TimeDelta step_size,
    const size_t number_of_past_steps) noexcept {
  int64_t slab_number = -1;
  for (size_t j = 0; j < number_of_past_steps; ++j) {
    ASSERT(time.slab() == step_size.slab(), "Slab mismatch");
    if ((step_size.is_positive() and time.is_at_slab_start()) or
        (not step_size.is_positive() and time.is_at_slab_end())) {
      const Slab new_slab = time.slab().advance_towards(-step_size);
      time = time.with_slab(new_slab);
      step_size = step_size.with_slab(new_slab);
      --slab_number;
    }
    time -= step_size;
    history->insert_initial(
        TimeStepId(step_size.is_positive(), slab_number, time),
        analytic(time.value()), rhs(analytic(time.value()), time.value()));
  }
}

void stability_test(const TimeStepper& stepper) noexcept;

void equal_rate_boundary(const LtsTimeStepper& stepper,
                         size_t number_of_past_steps,
                         double epsilon, bool forward) noexcept;

void check_convergence_order(const TimeStepper& stepper,
                             int expected_order) noexcept;

void check_dense_output(const TimeStepper& stepper,
                        int expected_order) noexcept;

void check_boundary_dense_output(const LtsTimeStepper& stepper) noexcept;
}  // namespace TimeStepperTestUtils
