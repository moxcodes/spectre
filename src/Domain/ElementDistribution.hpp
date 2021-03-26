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
 * together, and using a Morton ('Z-order') space-filling curve to determine
 * distribution within blocks.
 *
 * \details This class suggests an element placement based on a Z-curve/Morton
 * space filling curves.
 * Morton curves are a simple and easily-computed space-filling curve that
 * (unlike Hilbert curves) permit diagonal traversal. See, for instance,
 * \cite Borrell2018 for a discussion of mesh partitioning using space-filling
 * curves.
 *
 * First, a curve is computed through each block. Then the elements are assigned
 * to a processor in chunks such that each processor has approximately the same
 * number of elements. This is done by starting with block zero, assigning the
 * first chunk to processor 0 on node 0, the second to processor 1 on node 0,
 * and so on until node 0 is filled. The process then repeats on the next node.
 * If the number of elements in a block is not a multiple of the element chunk
 * size, then part of the next block is also added to the processor. For
 * example, processor 1 may receive half its elements from block 0 and half
 * from block 1. A concrete example in 2d is given below.
 *
 * A sketch of a 4x2 2D block, with each element labeled according to the order
 * on the Morton curve:
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
 * The Morton curve method is a quick way of getting acceptable spatial locality
 * -- usually, for approximately even distributions, it will ensure that
 * elements are assigned in large volume chunks, and the structure of the Morton
 * curve ensures that for a given processor and block, the elements will be
 * assigned in no more than two orthogonally connected clusters. In principle, a
 * Hilbert curve could potentially improve upon the gains obtained by this class
 * by guaranteeing that all elements within each block form a single orthognally
 * connected cluster.
 *
 * The assignment of portions of blocks to processors may use partial blocks,
 * and/or multiple blocks to ensure an even distribution of elements to
 * processors.
 * We currently make no distinction between dividing elements between processors
 * within a node and dividing elements between processors across nodes. The
 * current technique aims to have a simple method of reducing communication
 * globally, though it would likely be more efficient to prioritize minimization
 * of inter-node communication, because communication across interconnects is
 * the primary cost of communication in charm++ runs.
 */
template <size_t Dim>
struct BlockZCurveProcDistribution {
  BlockZCurveProcDistribution(size_t number_of_procs,
                              const std::vector<std::array<size_t, Dim>>&
                                  refinements_by_block) noexcept;

  /// Gets the suggested processor number for a particular block and element,
  /// determined by the greedy block assignment and Morton curve element
  /// assignment described in detail in the parent class documentation.
  size_t get_proc_for_element(size_t block_id,
                              const ElementId<Dim>& element_id) const noexcept;

 private:
  // in this nested data structure:
  // - The block id is the first index:
  //   There is an arbitrary number of nodes per block, each with an element
  //   cluster
  // - Each element cluster is represented by a pair of proc number, number of
  //   elements in the cluster
  std::vector<std::vector<std::pair<size_t, size_t>>>
      block_element_distribution_;
  std::vector<size_t> elements_per_proc_;
};
}  // namespace domain
