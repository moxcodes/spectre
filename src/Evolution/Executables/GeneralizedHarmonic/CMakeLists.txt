# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  ApparentHorizons
  Cce
  CoordinateMaps
  DiscontinuousGalerkin
  Domain
  DomainCreators
  Evolution
  GeneralRelativity
  GeneralRelativitySolutions
  GeneralizedHarmonic
  GeneralizedHarmonicGaugeSourceFunctions
  IO
  Informer
  Interpolation
  LinearOperators
  MathFunctions
  Options
  Spectral
  Time
  Utilities
  )

add_spectre_parallel_executable(
  EvolveGeneralizedHarmonic
  EvolveGeneralizedHarmonic
  Evolution/Executables/GeneralizedHarmonic
  EvolutionMetavars
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  EvolveGeneralizedHarmonicCce
  EvolveGeneralizedHarmonicCce
  Evolution/Executables/GeneralizedHarmonic
  EvolutionMetavarsWithCce
  "${LIBS_TO_LINK}"
  )
