// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <deque>

#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"

// currently assumes ToInterpolate is a vector type
template <typename Interpolator, typename ToInterpolate>
struct ScriPlusInterpolationManager {
 public:
  ScriPlusInterpolationManager(const size_t target_number_of_points,
                               const size_t vector_size) noexcept
      : vector_size_{vector_size},
        target_number_of_points_{target_number_of_points} {}

  void insert_data(const DataVector& u_bondi,
                   const ToInterpolate& to_interpolate) noexcept {
    ASSERT(to_interpolate.size() == vector_size_,
           "Inserted data must be of size specified at construction: "
               << vector_size_
               << " and provided data is of size: " << to_interpolate.size());
    u_bondi_values_.push_back(u_bondi);
    to_interpolate_values_.push_back(to_interpolate);
    u_bondi_ranges_.push_back(std::make_pair(min(u_bondi), max(u_bondi)));
    // printf("range inserted : %f, %f\n", min(u_bondi), max(u_bondi));
  }

  // for optimization, we assume that these are inserted in ascending order.
  void insert_target_time(const double time) noexcept {
    target_times_.push_back(time);
    if (target_times_.size() == 1) {
      cull_unneeded_early_times();
    }
  }

  bool first_time_is_ready_to_interpolate() const noexcept {
    size_t maxes_below = 0;
    size_t mins_above = 0;
    for (auto a : u_bondi_ranges_) {
      if (a.first >= target_times_.front()) {
        ++mins_above;
      }
      if (a.second <= target_times_.front()) {
        ++maxes_below;
      }
    }
    // printf("%f, %zu %zu\n", target_times_.front(), maxes_below, mins_above);
    // we might ask for a time that's too close to the end or the beginning of
    // our data, in which case we will settle for at least one point below and
    // above and a sufficient number of total points.
    // This class will never be 'ready' to extrapolate.
    return ((maxes_below > target_number_of_points_) and
            (mins_above > target_number_of_points_)) or
           (maxes_below + mins_above > 2 * target_number_of_points_ and
            maxes_below > 0 and mins_above > 0);
  }

  // note: this will work regardless of whether it is 'ready' to interpolate
  // (has the target number of points before and after). This functionality is
  // required to interpolate the values at the end of a simulation before
  // exiting.
  std::pair<ToInterpolate, double> interpolate_peek_first_time() noexcept {
    if (target_times_.size() == 0) {
      return std::make_pair(ToInterpolate{}, 0.0);
    }
    // note that because we demand at least a certain number before and at least
    // a certain number after, we are likely to have a surfeit of points for the
    // interpolator, but this should not cause significant trouble for a
    // reasonable method.
    ToInterpolate result{vector_size_};

    auto interpolation_values_begin = to_interpolate_values_.begin();
    auto interpolation_values_end = to_interpolate_values_.begin();

    auto interpolation_times_begin = u_bondi_values_.begin();
    auto interpolation_times_end = u_bondi_values_.begin();

    auto ranges_begin = u_bondi_ranges_.begin();
    auto ranges_end = u_bondi_ranges_.begin();

    size_t maxes_below = 0;
    size_t mid_points = 0;
    size_t mins_above = 0;
    // scan until the iterators represent the begins and ends that we'll use for
    // interpolation
    while (ranges_end != u_bondi_ranges_.end() and
           (*ranges_end).second < target_times_.front()) {
      if (maxes_below >= target_number_of_points_ and
          u_bondi_ranges_.size() - maxes_below > target_number_of_points_) {
        ++ranges_begin;
        ++interpolation_times_begin;
        ++interpolation_values_begin;
      } else {
        ++maxes_below;
      }
      ++ranges_end;
      ++interpolation_times_end;
      ++interpolation_values_end;
    }

    // for the set of bondi times which 'straddle' the time to interpolate, they
    // guarantee neither additional points above nor additional points below, so
    // add them to the range but do not count them toward the target.
    while (ranges_end != u_bondi_ranges_.end() and
           (*ranges_end).second >= target_times_.front() and
           (*ranges_end).first <= target_times_.front()) {
      ++mid_points;
      ++ranges_end;
      ++interpolation_times_end;
      ++interpolation_values_end;
    }
    // If we suffer from not having the data to reach the above and below
    // targets, instead settle for having twice the target number in some
    // lopsided configuration.
    while (ranges_end != u_bondi_ranges_.end() and
           (*ranges_end).first > target_times_.front() and
           (mins_above < target_number_of_points_ or
            maxes_below + mid_points + mins_above <
                2 * target_number_of_points_)) {
      ++mins_above;
      ++ranges_end;
      ++interpolation_times_end;
      ++interpolation_values_end;
    }

    // interpolate using the data sets in the restricted iterators
    ToInterpolate interpolation_values{maxes_below + mid_points + mins_above};
    DataVector interpolation_times{maxes_below + mid_points + mins_above};
    for (size_t i = 0; i < vector_size_; ++i) {
      auto value_it = interpolation_values_begin;
      size_t vector_position = 0;
      for (auto time_it = interpolation_times_begin;
           time_it != interpolation_times_end;
           ++time_it, ++value_it, ++vector_position) {
        interpolation_values[vector_position] = (*value_it)[i];
        interpolation_times[vector_position] = (*time_it)[i];
      }
      result[i] = Interpolator::interpolate(
          interpolation_times, interpolation_values, target_times_.front());
    }
    return std::make_pair(result, target_times_.front());
  }

  // note: this will work regardless of whether it is 'ready' to interpolate
  // (has the target number of points before and after). This functionality is
  // required to interpolate the values at the end of a simulation before
  // exiting.
  std::pair<ToInterpolate, double> interpolate_and_pop_first_time() noexcept {
    std::pair<ToInterpolate, double> interpolated =
        interpolate_peek_first_time();
    target_times_.pop_front();

    if (target_times_.size() > 0) {
      cull_unneeded_early_times();
    }
    return interpolated;
  }

  size_t target_times_size() noexcept { return target_times_.size(); }

  size_t data_sizes() noexcept { return u_bondi_ranges_.size(); }

 private:
  void cull_unneeded_early_times() noexcept {
    // pop times we no longer need because their maxes are too far in the past
    auto time_it = u_bondi_ranges_.begin();
    size_t times_counter = 0;
    while ((*time_it).second < target_times_.front()) {
      if (times_counter > target_number_of_points_ and
          u_bondi_ranges_.size() >= 2 * target_number_of_points_) {
        u_bondi_ranges_.pop_front();
        u_bondi_values_.pop_front();
        to_interpolate_values_.pop_front();
      } else {
        ++times_counter;
      }
      ++time_it;
    }
  }

  std::deque<DataVector> u_bondi_values_;
  std::deque<ToInterpolate> to_interpolate_values_;
  std::deque<std::pair<double, double>> u_bondi_ranges_;
  std::deque<double> target_times_;
  size_t vector_size_;
  size_t target_number_of_points_;
};

// // this can probably be merged with the above interpolator, as much of the
// // point-management code is very similar. currently assumes ToInterpolate is
// a
// // vector type
// template <typename Interpolator, typename ToDerive>
// struct ScriPlusDerivativeManager {
//  public:
//   ScriPlusDerivativeManager(const size_t target_number_of_points,
//                             const size_t vector_size) noexcept
//       : vector_size_{vector_size},
//         target_number_of_points_{target_number_of_points} {}

//   void insert_data(const DataVector& u_bondi,
//                    const ToDerive& to_interpolate) noexcept {
//     ASSERT(to_interpolate.size() == vector_size_,
//            "Inserted data must be of size specified at construction: "
//                << vector_size_
//                << " and provided data is of size: " <<
//                to_interpolate.size());
//     u_bondi_values_.push_back(u_bondi);
//     to_interpolate_values_.push_back(to_interpolate);
//     u_bondi_ranges_.push_back(std::make_pair(min(u_bondi), max(u_bondi)));
//   }

//   // for optimization, we assume that these are inserted in ascending order.
//   void insert_target_time(const double time) noexcept {
//     target_times_.push_back(time);
//     if(target_times_.size() == 1) {
//       cull_unneeded_early_times();
//     }
//   }

//   bool first_time_is_ready_to_derive() noexcept {
//     size_t maxes_below = 0;
//     size_t mins_above = 0;
//     for(auto a : u_bondi_ranges_) {
//       if(a.first >= target_times_.front()) {
//         ++mins_above;
//       }
//       if(a.second <= target_times_.front()) {
//         ++maxes_below;
//       }
//     }
//     printf("%f, %zu %zu\n", target_times_.front(), maxes_below, mins_above);
//     // we might ask for a time that's too close to the end or the beginning
//     of
//     // our data, in which case we will settle for at least one point below
//     and
//     // above and a sufficient number of total points.
//     // This class will never be 'ready' to extrapolate.
//     return ((maxes_below > target_number_of_points_) and
//             (mins_above > target_number_of_points_)) or
//            (maxes_below + mins_above > 2 * target_number_of_points_ and
//             maxes_below > 0 and mins_above > 0);
//   }

//   // note: this will work regardless of whether it is 'ready' to interpolate
//   // (has the target number of points before and after). This functionality
//   is
//   // required to interpolate the values at the end of a simulation before
//   // exiting.
//   std::pair<ToDerive, double> derive_peek_first_time() noexcept {
//     if(target_times_.size() == 0) {
//       return std::make_pair(ToDerive{}, 0.0);
//     }
//     // note that because we demand at least a certain number before and at
//     least
//     // a certain number after, we are likely to have a surfeit of points for
//     the
//     // interpolator, but this should not cause significant trouble for a
//     // reasonable method.
//     ToDerive result{vector_size_};

//     auto interpolation_values_begin = to_interpolate_values_.begin();
//     auto interpolation_values_end = to_interpolate_values_.begin();

//     auto interpolation_times_begin = u_bondi_values_.begin();
//     auto interpolation_times_end = u_bondi_values_.begin();

//     auto ranges_begin = u_bondi_ranges_.begin();
//     auto ranges_end = u_bondi_ranges_.begin();

//     size_t maxes_below = 0;
//     size_t mid_points = 0;
//     size_t mins_above = 0;
//     // scan until the iterators represent the begins and ends that we'll use
//     for
//     // interpolation
//     while (ranges_end != u_bondi_ranges_.end() and
//            (*ranges_end).second < target_times_.front()) {
//       if(maxes_below >= target_number_of_points_
//          and u_bondi_ranges_.size() - maxes_below > target_number_of_points_)
//          {
//         ++ranges_begin;
//         ++interpolation_times_begin;
//         ++interpolation_values_begin;
//       } else {
//         ++maxes_below;
//       }
//       ++ranges_end;
//       ++interpolation_times_end;
//       ++interpolation_values_end;
//     }

//     // for the set of bondi times which 'straddle' the time to interpolate,
//     they
//     // guarantee neither additional points above nor additional points below,
//     so
//     // add them to the range but do not count them toward the target.
//     while (ranges_end != u_bondi_ranges_.end() and
//            (*ranges_end).second >= target_times_.front() and
//            (*ranges_end).first <= target_times_.front()) {
//       ++mid_points;
//       ++ranges_end;
//       ++interpolation_times_end;
//       ++interpolation_values_end;
//     }
//     // If we suffer from not having the data to reach the above and below
//     // targets, instead settle for having twice the target number in some
//     // lopsided configuration.
//     while (ranges_end != u_bondi_ranges_.end() and
//            (*ranges_end).first > target_times_.front() and
//            (mins_above < target_number_of_points_ or
//             maxes_below + mid_points + mins_above <
//                 2 * target_number_of_points_)) {
//       ++mins_above;
//       ++ranges_end;
//       ++interpolation_times_end;
//       ++interpolation_values_end;
//     }

//     // interpolate using the data sets in the restricted iterators
//     ToDerive interpolation_values{maxes_below + mid_points + mins_above};
//     DataVector interpolation_times{maxes_below + mid_points + mins_above};
//     for(size_t i = 0; i < vector_size_; ++i) {
//       auto value_it = interpolation_values_begin;
//       size_t vector_position = 0;
//       for (auto time_it = interpolation_times_begin;
//            time_it != interpolation_times_end;
//            ++time_it, ++value_it, ++vector_position) {
//         interpolation_values[vector_position] = (*value_it)[i];
//         interpolation_times[vector_position] = (*time_it)[i];
//       }
//       result[i] = Interpolator::interpolate(
//           interpolation_times, interpolation_values, target_times_.front());
//     }
//     return std::make_pair(result, target_times_.front());
//   }

//   // note: this will work regardless of whether it is 'ready' to interpolate
//   // (has the target number of points before and after). This functionality
//   is
//   // required to interpolate the values at the end of a simulation before
//   // exiting.
//   std::pair<ToDerive, double> interpolate_and_pop_first_time() noexcept {
//     std::pair<ToDerive, double> interpolated =
//         interpolate_peek_first_time();
//     target_times_.pop_front();

//     if (target_times_.size() > 0) {
//       cull_unneeded_early_times();
//     }
//     return interpolated;
//   }

//   size_t target_times_size() noexcept { return target_times_.size(); }

//   size_t data_sizes() noexcept { return u_bondi_ranges_.size(); }

//  private:
//   void cull_unneeded_early_times() noexcept {
//     // pop times we no longer need because their maxes are too far in the
//     past auto time_it = u_bondi_ranges_.begin(); size_t times_counter = 0;
//     while ((*time_it).second < target_times_.front()) {
//       if (times_counter > target_number_of_points_ and
//           u_bondi_ranges_.size() >= 2 * target_number_of_points_) {
//         u_bondi_ranges_.pop_front();
//         u_bondi_values_.pop_front();
//         to_interpolate_values_.pop_front();
//       } else {
//         ++times_counter;
//       }
//       ++time_it;
//     }
//   }

//   std::deque<DataVector> u_bondi_values_;
//   std::deque<ToDerive> to_interpolate_values_;
//   std::deque<std::pair<double, double>> u_bondi_ranges_;
//   std::deque<double> target_times_;
//   size_t vector_size_;
//   size_t target_number_of_points_;
// };
