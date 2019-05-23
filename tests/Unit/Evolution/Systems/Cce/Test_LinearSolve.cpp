// Distributed under the MIT License.
// See LICENSE.txt for details

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "Evolution/Systems/Cce/LinearSolve.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Utilities/VectorAlgebra.hpp"
#include "tests/Unit/Evolution/Systems/Cce/CceComputationTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {

namespace {

ComplexDataVector radial_vector_from_power_series(
    const ComplexModalVector& powers,
    const ComplexDataVector& one_minus_y) noexcept {
  ComplexDataVector result{one_minus_y.size(), powers[0]};
  for (size_t i = 1; i < powers.size(); ++i) {
    // use of TestHelpers::power due to an internal bug in blaze powers of
    // Complex vectors
    result += powers[i] * TestHelpers::power(one_minus_y, i);
  }
  return result;
}

template <typename BondiValueTag, typename DataBoxType>
void make_boundary_data(const gsl::not_null<DataBoxType*> box,
                        const gsl::not_null<ComplexDataVector*> expected,
                        const size_t l_max) noexcept {
  db::mutate<Tags::BoundaryValue<BondiValueTag>>(
      box,
      [
        &expected, &l_max
      ](const gsl::not_null<db::item_type<Tags::BoundaryValue<BondiValueTag>>*>
            boundary) noexcept {
        get(*boundary).data() = ComplexDataVector{
            expected->data(),
            Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
      });
}

template <typename DataBoxType, typename Generator, typename Distribution,
          typename... Tags>
void generate_powers_for(const gsl::not_null<DataBoxType*> box,
                         const gsl::not_null<Generator*> generator,
                         const gsl::not_null<Distribution*> distribution,
                         const size_t number_of_modes,
                         tmpl::list<Tags...> /*meta*/) noexcept {
  expand_pack([&generator, &distribution, &number_of_modes, &box ]() noexcept {
    db::mutate<TestHelpers::RadialPolyCoefficientsFor<Tags>>(
        box, [
          &generator, &distribution, &number_of_modes
        ](const gsl::not_null<Scalar<ComplexModalVector>*> modes) noexcept {
          get(*modes) = make_with_random_values<ComplexModalVector>(
              generator, distribution, number_of_modes);
        });
    // extra wrapper with return is required for the pack expansion
    return 0;
  }()...);
}

template <typename Tag, typename DataBoxType>
void zero_top_modes(const gsl::not_null<DataBoxType*> box,
                    const size_t number_of_modes_to_zero,
                    const size_t total_number_of_modes) noexcept {
  db::mutate<TestHelpers::RadialPolyCoefficientsFor<Tag>>(
      box, [
        &number_of_modes_to_zero, &total_number_of_modes
      ](const gsl::not_null<Scalar<ComplexModalVector>*> modes) noexcept {
        for (size_t i = total_number_of_modes - number_of_modes_to_zero;
             i < total_number_of_modes; ++i) {
          get(*modes)[i] = 0.0;
        }
      });
}

template <typename DataBoxType, typename... Tags>
void generate_volume_data_from_separable(
    const gsl::not_null<DataBoxType*> box,
    const ComplexDataVector& angular_data, const ComplexDataVector& one_minus_y,
    tmpl::list<Tags...> /*meta*/) noexcept {
  expand_pack([&one_minus_y, &angular_data, &box ]() noexcept {
    db::mutate<Tags>(
        box,
        [&one_minus_y, &
         angular_data ](const gsl::not_null<db::item_type<Tags>*> to_fill,
                        const Scalar<ComplexModalVector>& modes) noexcept {
          get(*to_fill).data() = outer_product(
              angular_data,
              radial_vector_from_power_series(get(modes), one_minus_y));
        },
        db::get<TestHelpers::RadialPolyCoefficientsFor<Tags>>(*box));
    // extra wrapper with return is required for the pack expansion
    return 0;
  }()...);
}

template <typename BondiValueTag, typename Generator>
void test_regular_integration(const gsl::not_null<Generator*> gen,
                              size_t number_of_radial_grid_points,
                              size_t l_max) noexcept {
  UniformCustomDistribution<double> dist(0.1, 5.0);
  size_t number_of_grid_points =
      number_of_radial_grid_points *
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  size_t number_of_radial_polynomials = 5;

  using integration_tags =
      tmpl::list<BondiValueTag, Tags::Integrand<BondiValueTag>>;
  using integration_variables_tag = ::Tags::Variables<integration_tags>;
  using integration_modes_variables_tag =
      ::Tags::Variables<db::wrap_tags_in<TestHelpers::RadialPolyCoefficientsFor,
                                         integration_tags>>;

  auto box = db::create<db::AddSimpleTags<
      integration_variables_tag, Tags::BoundaryValue<BondiValueTag>,
      integration_modes_variables_tag, Spectral::Swsh::Tags::LMax>>(
      db::item_type<integration_variables_tag>{number_of_grid_points},
      db::item_type<Tags::BoundaryValue<BondiValueTag>>{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)},
      db::item_type<integration_modes_variables_tag>{
          number_of_radial_polynomials},
      l_max);

  generate_powers_for(make_not_null(&box), gen, make_not_null(&dist),
                      number_of_radial_polynomials,
                      tmpl::list<BondiValueTag>{});

  // use the above powers to infer the powers for the derivative
  db::mutate<
      TestHelpers::RadialPolyCoefficientsFor<Tags::Integrand<BondiValueTag>>>(
      make_not_null(&box),
      [](const gsl::not_null<Scalar<ComplexModalVector>*> integrand_modes,
         const Scalar<ComplexModalVector>& bondi_value_modes) noexcept {
        for (size_t i = 0; i < get(bondi_value_modes).size() - 1; ++i) {
          // sign change because these are modes of 1 - y.
          get(*integrand_modes)[i] =
              -static_cast<double>(i + 1) * get(bondi_value_modes)[i + 1];
        }
        get(*integrand_modes)[get(bondi_value_modes).size() - 1] = 0.0;
      },
      db::get<TestHelpers::RadialPolyCoefficientsFor<BondiValueTag>>(box));

  const ComplexDataVector one_minus_y =
      std::complex<double>(1.0, 0.0) *
      (1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto>(
                 number_of_radial_grid_points));

  auto random_angular_data = make_with_random_values<ComplexDataVector>(
      gen, make_not_null(&dist),
      Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  ComplexDataVector expected = outer_product(
      random_angular_data,
      radial_vector_from_power_series(
          get(db::get<TestHelpers::RadialPolyCoefficientsFor<BondiValueTag>>(
              box)),
          one_minus_y));

  generate_volume_data_from_separable(
      make_not_null(&box), random_angular_data, one_minus_y,
      tmpl::list<Tags::Integrand<BondiValueTag>>{});

  make_boundary_data<BondiValueTag>(make_not_null(&box),
                                    make_not_null(&expected), l_max);

  db::mutate_apply<RadialIntegrateBondi<Tags::BoundaryValue, BondiValueTag>>(
      make_not_null(&box));

  Approx numerical_differentiation_approximation =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);

  CHECK_ITERABLE_CUSTOM_APPROX(expected,
                               get(db::get<BondiValueTag>(box)).data(),
                               numerical_differentiation_approximation);
}

template <typename BondiValueTag, typename Generator>
void test_pole_integration(const gsl::not_null<Generator*> gen,
                           size_t number_of_radial_grid_points,
                           size_t l_max) noexcept {
  UniformCustomDistribution<double> dist(0.1, 5.0);
  size_t number_of_grid_points =
      number_of_radial_grid_points *
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  size_t number_of_radial_polynomials = 5;

  using integration_tags =
      tmpl::list<BondiValueTag, Tags::PoleOfIntegrand<BondiValueTag>,
                 Tags::RegularIntegrand<BondiValueTag>>;
  using integration_variables_tag = ::Tags::Variables<integration_tags>;
  using integration_modes_variables_tag =
      ::Tags::Variables<db::wrap_tags_in<TestHelpers::RadialPolyCoefficientsFor,
                                         integration_tags>>;

  auto box = db::create<db::AddSimpleTags<
      integration_variables_tag, Tags::BoundaryValue<BondiValueTag>,
      integration_modes_variables_tag, Spectral::Swsh::Tags::LMax,
      Tags::OneMinusY>>(
      db::item_type<integration_variables_tag>{number_of_grid_points},
      db::item_type<Tags::BoundaryValue<BondiValueTag>>{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)},
      db::item_type<integration_modes_variables_tag>{
          number_of_radial_polynomials},
      l_max, Scalar<SpinWeighted<ComplexDataVector, 0>>{number_of_grid_points});

  generate_powers_for(
      make_not_null(&box), gen, make_not_null(&dist),
      number_of_radial_polynomials,
      tmpl::list<BondiValueTag, Tags::PoleOfIntegrand<BondiValueTag>>{});

  // use the above powers to infer the powers for regular part of the integrand
  db::mutate<TestHelpers::RadialPolyCoefficientsFor<
      Tags::PoleOfIntegrand<BondiValueTag>>>(
      make_not_null(&box),
      [](const gsl::not_null<Scalar<ComplexModalVector>*>
             pole_of_integrand_modes,
         const Scalar<ComplexModalVector>& bondi_value_modes) noexcept {
        get(*pole_of_integrand_modes)[0] = 2.0 * get(bondi_value_modes)[0];
      },
      db::get<TestHelpers::RadialPolyCoefficientsFor<BondiValueTag>>(box));

  db::mutate<TestHelpers::RadialPolyCoefficientsFor<
      Tags::RegularIntegrand<BondiValueTag>>>(
      make_not_null(&box),
      [](const gsl::not_null<Scalar<ComplexModalVector>*>
             regular_integrand_modes,
         const Scalar<ComplexModalVector>& bondi_value_modes,
         const Scalar<ComplexModalVector>& pole_integrand_modes) noexcept {
        for (size_t i = 0; i < get(bondi_value_modes).size() - 1; ++i) {
          // sign change because these are modes of 1 - y.
          get(*regular_integrand_modes)[i] =
              -get(pole_integrand_modes)[i + 1] +
              (1.0 - static_cast<double>(i)) * get(bondi_value_modes)[i + 1];
        }
        get(*regular_integrand_modes)[get(bondi_value_modes).size() - 1] = 0.0;
      },
      db::get<TestHelpers::RadialPolyCoefficientsFor<BondiValueTag>>(box),
      db::get<TestHelpers::RadialPolyCoefficientsFor<
          Tags::PoleOfIntegrand<BondiValueTag>>>(box));

  const ComplexDataVector one_minus_y =
      std::complex<double>(1.0, 0.0) *
      (1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto>(
                 number_of_radial_grid_points));

  auto random_angular_data = make_with_random_values<ComplexDataVector>(
      gen, make_not_null(&dist),
      Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  ComplexDataVector expected = outer_product(
      random_angular_data,
      radial_vector_from_power_series(
          get(db::get<TestHelpers::RadialPolyCoefficientsFor<BondiValueTag>>(
              box)),
          one_minus_y));

  generate_volume_data_from_separable(
      make_not_null(&box), random_angular_data, one_minus_y,
      tmpl::list<Tags::PoleOfIntegrand<BondiValueTag>,
                 Tags::RegularIntegrand<BondiValueTag>>{});

  make_boundary_data<BondiValueTag>(make_not_null(&box),
                                    make_not_null(&expected), l_max);

  db::mutate_apply<
      PrecomputeCceDependencies<Tags::BoundaryValue, Tags::OneMinusY>>(
      make_not_null(&box));

  db::mutate_apply<RadialIntegrateBondi<Tags::BoundaryValue, BondiValueTag>>(
      make_not_null(&box));

  Approx numerical_differentiation_approximation =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);

  CHECK_ITERABLE_CUSTOM_APPROX(expected,
                               get(db::get<BondiValueTag>(box)).data(),
                               numerical_differentiation_approximation);
}

// TODO JM please find a way to make this less gross

template <typename BondiValueTag, typename Generator>
void test_pole_integration_with_linear_operator(
    const gsl::not_null<Generator*> gen, size_t number_of_radial_grid_points,
    size_t l_max) noexcept {
  UniformCustomDistribution<double> dist(0.1, 5.0);
  size_t number_of_grid_points =
      number_of_radial_grid_points *
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  size_t number_of_radial_polynomials = 6;

  using integration_tags =
      tmpl::list<BondiValueTag, Tags::PoleOfIntegrand<BondiValueTag>,
                 Tags::RegularIntegrand<BondiValueTag>,
                 Tags::LinearFactor<BondiValueTag>,
                 Tags::LinearFactorForConjugate<BondiValueTag>>;
  using integration_variables_tag = ::Tags::Variables<integration_tags>;
  using integration_modes_variables_tag =
      ::Tags::Variables<db::wrap_tags_in<TestHelpers::RadialPolyCoefficientsFor,
                                         integration_tags>>;

  auto box = db::create<db::AddSimpleTags<
      integration_variables_tag, Tags::BoundaryValue<BondiValueTag>,
      integration_modes_variables_tag, Spectral::Swsh::Tags::LMax,
      Tags::OneMinusY>>(
      db::item_type<integration_variables_tag>{number_of_grid_points},
      db::item_type<Tags::BoundaryValue<BondiValueTag>>{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)},
      db::item_type<integration_modes_variables_tag>{
          number_of_radial_polynomials},
      l_max, Scalar<SpinWeighted<ComplexDataVector, 0>>{number_of_grid_points});

  generate_powers_for(
      make_not_null(&box), gen, make_not_null(&dist),
      number_of_radial_polynomials,
      tmpl::list<BondiValueTag, Tags::PoleOfIntegrand<BondiValueTag>,
                 Tags::LinearFactor<BondiValueTag>,
                 Tags::LinearFactorForConjugate<BondiValueTag>>{});
  zero_top_modes<BondiValueTag>(make_not_null(&box), 2,
                                number_of_radial_polynomials);
  zero_top_modes<Tags::LinearFactor<BondiValueTag>>(
      make_not_null(&box), 3, number_of_radial_polynomials);
  zero_top_modes<Tags::LinearFactorForConjugate<BondiValueTag>>(
      make_not_null(&box), 3, number_of_radial_polynomials);

  const ComplexDataVector one_minus_y =
      std::complex<double>(1.0, 0.0) *
      (1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto>(
                 number_of_radial_grid_points));

  auto random_angular_data = make_with_random_values<ComplexDataVector>(
      gen, make_not_null(&dist),
      Spectral::Swsh::number_of_swsh_collocation_points(l_max));
  generate_volume_data_from_separable(
      make_not_null(&box), random_angular_data, one_minus_y,
      tmpl::list<Tags::PoleOfIntegrand<BondiValueTag>,
                 Tags::LinearFactor<BondiValueTag>,
                 Tags::LinearFactorForConjugate<BondiValueTag>>{});
  // unlike the above tests, the nonlinear operators in the H equation ensures
  // that we have to actually manually build up the last operator to ensure
  // consistency (from the above data, the pole of integrand is not separable).
  db::mutate<Tags::PoleOfIntegrand<BondiValueTag>,
             Tags::RegularIntegrand<BondiValueTag>>(
      make_not_null(&box),
      [
        &random_angular_data, &l_max, &one_minus_y, &
        number_of_radial_grid_points
      ](const gsl::not_null<
            db::item_type<Tags::PoleOfIntegrand<BondiValueTag>>*>
            pole_integrand,
        const gsl::not_null<
            db::item_type<Tags::RegularIntegrand<BondiValueTag>>*>
            regular_integrand,
        const Scalar<ComplexModalVector>& bondi_value_modes,
        const Scalar<ComplexModalVector>& pole_integrand_modes,
        const Scalar<ComplexModalVector>& linear_factor_modes,
        const Scalar<ComplexModalVector>&
            linear_factor_of_conjugate_modes) noexcept {
        for (size_t i = 0; i < number_of_radial_grid_points; ++i) {
          ComplexDataVector angular_view_for_pole_integrand{
              get(*pole_integrand).data().data() +
                  i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
              Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
          ComplexDataVector angular_view_for_regular_integrand{
              get(*regular_integrand).data().data() +
                  i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
              Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

          angular_view_for_pole_integrand +=
              square(random_angular_data) * get(bondi_value_modes)[0] *
                  get(linear_factor_modes)[0] +
              random_angular_data * conj(random_angular_data) *
                  conj(get(bondi_value_modes)[0]) *
                  get(linear_factor_of_conjugate_modes)[0] -
              random_angular_data * get(pole_integrand_modes)[0];
          angular_view_for_regular_integrand = 0.0;

          for (size_t j = 0; j < 5; ++j) {
            angular_view_for_regular_integrand +=
                (-static_cast<double>(j + 1) * get(bondi_value_modes)[j + 1] -
                 get(pole_integrand_modes)[j + 1]) *
                random_angular_data *
                (j == 0
                     ? 1.0
                     : (one_minus_y[i] == 0.0 ? 0.0 : pow(one_minus_y[i], j)));
            for (size_t k =
                     static_cast<size_t>(std::max(0, static_cast<int>(j) - 2));
                 k < std::min(j + 2, size_t{3}); ++k) {
              angular_view_for_regular_integrand +=
                  (square(random_angular_data) *
                       (get(linear_factor_modes)[k] *
                        get(bondi_value_modes)[(j + 1) - k]) +
                   random_angular_data * conj(random_angular_data) *
                       (get(linear_factor_of_conjugate_modes)[k] *
                        conj(get(bondi_value_modes)[(j + 1) - k]))) *
                  (j == 0 ? 1.0
                          : (real(one_minus_y[i]) == 0.0
                                 ? 0.0
                                 : pow(one_minus_y[i], j)));
            }
          }
        }
      },
      db::get<TestHelpers::RadialPolyCoefficientsFor<BondiValueTag>>(box),
      db::get<TestHelpers::RadialPolyCoefficientsFor<
          Tags::PoleOfIntegrand<BondiValueTag>>>(box),
      db::get<TestHelpers::RadialPolyCoefficientsFor<
          Tags::LinearFactor<BondiValueTag>>>(box),
      db::get<TestHelpers::RadialPolyCoefficientsFor<
          Tags::LinearFactorForConjugate<BondiValueTag>>>(box));
  ComplexDataVector expected = outer_product(
      random_angular_data,
      radial_vector_from_power_series(
          get(db::get<TestHelpers::RadialPolyCoefficientsFor<BondiValueTag>>(
              box)),
          one_minus_y));

  make_boundary_data<BondiValueTag>(make_not_null(&box),
                                    make_not_null(&expected), l_max);

  db::mutate_apply<
      PrecomputeCceDependencies<Tags::BoundaryValue, Tags::OneMinusY>>(
      make_not_null(&box));

  db::mutate_apply<RadialIntegrateBondi<Tags::BoundaryValue, BondiValueTag>>(
      make_not_null(&box));

  Approx numerical_differentiation_approximation =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);

  CHECK_ITERABLE_CUSTOM_APPROX(expected,
                               get(db::get<BondiValueTag>(box)).data(),
                               numerical_differentiation_approximation);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.LinearSolve",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{3, 6};
  const size_t l_max = sdist(gen);
  const size_t number_of_radial_grid_points = 2 * sdist(gen);

  test_regular_integration<Tags::BondiBeta>(
      make_not_null(&gen), number_of_radial_grid_points, l_max);
  test_regular_integration<Tags::BondiU>(make_not_null(&gen),
                                         number_of_radial_grid_points, l_max);
  test_pole_integration<Tags::BondiQ>(make_not_null(&gen),
                                      number_of_radial_grid_points, l_max);
  test_pole_integration<Tags::BondiW>(make_not_null(&gen),
                                      number_of_radial_grid_points, l_max);
  test_pole_integration_with_linear_operator<Tags::BondiH>(
      make_not_null(&gen), number_of_radial_grid_points, l_max);
}
}  // namespace
}  // namespace Cce
