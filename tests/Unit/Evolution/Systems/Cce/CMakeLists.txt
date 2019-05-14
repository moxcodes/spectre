# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Cce")

set(LIBRARY_SOURCES
  Test_Equations.cpp
  CceComputationTestHelpers.cpp
  Test_ComputePreSwshDerivatives.cpp
  Test_InitializeCce.cpp
  Test_LinearOperators.cpp
  Test_PrecomputeCceDependencies.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Cce/"
  "${LIBRARY_SOURCES}"
  "Cce"
  )
