# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Cce
  Time
  Interpolation
  Informer
  Spectral
  LinearOperators
  Options
  )

add_spectre_parallel_executable(
  BondiCharacteristicExtract
  CharacteristicExtract
  Evolution/Executables/Cce
  EvolutionMetavars<Cce::ReducedWorldtubeDataManager>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  MetricCharacteristicExtract
  CharacteristicExtract
  Evolution/Executables/Cce
  EvolutionMetavars<Cce::WorldtubeDataManager>
  "${LIBS_TO_LINK}"
  )
