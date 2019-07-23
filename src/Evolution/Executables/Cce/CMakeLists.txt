# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Cce
  ${SPECTRE_LIBRARIES}
  ${YAMLCPP_LIBRARIES}
  )

add_spectre_parallel_executable(
  CharacteristicExtract
  CharacteristicExtract
  Evolution/Executables/Cce
  EvolutionMetavars
  "{LIBS_TO_LINK}"
  )
