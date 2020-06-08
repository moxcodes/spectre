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
  GeneralizedHarmonic
  GeneralizedHarmonicGaugeSourceFunctions
  GhConstraintDamping
  IO
  Informer
  Interpolation
  LinearOperators
  MathFunctions
  Options
  Parallel
  Spectral
  Time
  Utilities
  )

function(add_generalized_harmonic_executable
    INITIAL_DATA_NAME INITIAL_DATA BOUNDARY_CONDITIONS)
  add_spectre_parallel_executable(
    "EvolveGh${INITIAL_DATA_NAME}"
    EvolveGeneralizedHarmonic
    Evolution/Executables/GeneralizedHarmonic
    "EvolutionMetavars<${INITIAL_DATA}, ${BOUNDARY_CONDITIONS}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_generalized_harmonic_executable)

function(add_generalized_harmonic_cce_executable
  EXECUTABLE_NAME INITIAL_DATA BOUNDARY_CONDITIONS)
  add_spectre_parallel_executable(
    "EvolveGh${EXECUTABLE_NAME}WithCce"
    EvolveGeneralizedHarmonicCce
    Evolution/Executables/GeneralizedHarmonic
    "EvolutionMetavars<${INITIAL_DATA}, ${BOUNDARY_CONDITIONS}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_generalized_harmonic_cce_executable)

add_generalized_harmonic_executable(
  KerrSchild
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
)
target_link_libraries(
  EvolveGhKerrSchild
  PRIVATE
  GeneralRelativitySolutions
  )
add_generalized_harmonic_cce_executable(
  KerrSchild
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
)
target_link_libraries(
  EvolveGhKerrSchildWithCce
  PRIVATE
  GeneralRelativitySolutions
  )

add_generalized_harmonic_executable(
  KerrSchildNumericInitialData
  evolution::NumericInitialData
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
)
target_link_libraries(
  EvolveGhKerrSchildNumericInitialData
  PRIVATE
  GeneralRelativitySolutions
  Importers
  )
add_generalized_harmonic_cce_executable(
  KerrSchildNumericInitialData
  evolution::NumericInitialData
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
)
target_link_libraries(
  EvolveGhKerrSchildNumericInitialDataWithCce
  PRIVATE
  GeneralRelativitySolutions
  )
