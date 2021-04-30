# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Actions/EmulateLoad.hpp
  Actions/InitializeGraphDumpLabel.hpp
  Actions/InitializeLoadBalancingTestArray.hpp
  Actions/LoadBalancingCommunication.hpp
  Actions/StepManagement.hpp
  LoadBalancingTestArray.hpp
  StepTriggers.hpp
  Tags.hpp
  )
