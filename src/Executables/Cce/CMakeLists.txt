# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE cce_extract)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  Cce.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  Cce
  ${SPECTRE_LIBRARIES}
  ${YAMLCPP_LIBRARIES}
  )


set_target_properties(
  ${EXECUTABLE}
  PROPERTIES LINK_FLAGS "-nomain-module -nomain"
  )
