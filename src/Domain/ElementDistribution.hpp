// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "Domain/Structure/ElementId.hpp"

namespace domain {

/*!
 * \brief Distribution strategy based on grouping elements within blocks
 * together, and using a 'Z-curve' to determine distribution within blocks.
 *
 * \details This class suggests starting placements of elements on processors by
 * first greedily assigning portions of blocks to each processor, then dividing
 * up the blocks among processors that share it according to a 'Z-curve'.
 * Z-curves are a simple and easily-computed space-filling curve that (unlike
 * hilbert curves) permit diagonal traversal.
 *
 * For example, here is a sketch of a 4x2 2D block, with each element labelled
 * according to the order on the Z-curve:
 * ```
 *          x-->
 *          0   1   2   3
 *        ----------------
 *  y  0 |  0   2   4   6
 *  |    |  | / | / | / |
 *  v  1 |  1   3   5   7
 * ```
 * (forming a zig-zag path, that under some rotation/reflection has a 'Z'
 * shape).
 *
 * Then, the first processor for a block takes as many elements have been
 * assigned to it along the first part of the Z-curve, then the second processor
 * takes its elements from the next part of the Z-curve, and so on.
 * This is a quick way of getting acceptaable spatial locality -- usually, for
 * approximately even distributions, this will ensure that elements are assigned
 * in large volume chunks, and in no more than two large clusters within the
 * block. In principle, a Hilbert curve could potentially improve upon the gains
 * obtained by this class by guaranteeing that all elements within each block
 * are orthogonally adjacent.
 */
template <size_t Dim>
struct BlockZCurveProcDistribution {
  BlockZCurveProcDistribution(size_t number_of_procs,
                              const std::vector<std::array<size_t, Dim>>&
                                  refinements_by_block) noexcept;

  /// Gets the suggested processor number for a particular block and element,
  /// determined by the greedy block assignment and Z-curve element assignment
  /// described in detail in the parent class documentation.
  size_t get_proc_for_element(size_t block_id,
                              const ElementId<Dim>& element_id) const noexcept;

 private:
  // in this nested data structure:
  // - The block id is the first index:
  //   There is an arbitrary number of nodes per block, each with an element
  //   cluster
  //   - Each element cluster is represented by a pair of proc number, number of
  //     elements in the cluster
  std::vector<std::vector<std::pair<size_t, size_t>>>
      block_element_distribution_;
  std::vector<size_t> elements_per_proc_;
};
}  // namespace domain
