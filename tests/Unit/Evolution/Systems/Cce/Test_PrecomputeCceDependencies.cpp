// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "tests/Unit/Evolution/Systems/Cce/CceComputationTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {

template <typename Generator, typename PrecomputationBox, typename ExpectedBox>
void generate_boundary_values_and_expected(
    const gsl::not_null<Generator*> generator,
    const gsl::not_null<PrecomputationBox*> precomputation_box,
    const gsl::not_null<ExpectedBox*> expected_box, const size_t l_max,
    const size_t number_of_radial_grid_points) {
  ComplexDataVector y = outer_product(
      ComplexDataVector{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max), 1.0},
      Spectral::collocation_points<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto>(
          number_of_radial_grid_points));

  UniformCustomDistribution<double> dist(0.1, 1.0);
  db::mutate<Tags::BoundaryValue<Tags::BondiR>,
             Tags::BoundaryValue<Tags::DuRDividedByR>, Tags::BondiR,
             Tags::DuRDividedByR, Tags::OneMinusY, Tags::BondiJ, Tags::BondiK>(
      expected_box,
      [&generator, &dist, &l_max, &number_of_radial_grid_points, &y](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              boundary_r,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              boundary_du_r_divided_by_r,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> r,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              du_r_divided_by_r,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              one_minus_y,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> k) {
        get(*boundary_r).data() =
            (10.0 +
             std::complex<double>(1.0, 0.0) *
                 make_with_random_values<DataVector>(
                     generator, make_not_null(&dist),
                     Spectral::Swsh::number_of_swsh_collocation_points(l_max)));
        get(*boundary_du_r_divided_by_r).data() =
            std::complex<double>(1.0, 0.0) *
            make_with_random_values<DataVector>(
                generator, make_not_null(&dist),
                Spectral::Swsh::number_of_swsh_collocation_points(l_max));
        // prevent aliasing; some terms are nonlinear in R
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&get(*boundary_r)), l_max, l_max - 3);
        repeat(make_not_null(&get(*r).data()), get(*boundary_r).data(),
               number_of_radial_grid_points);
        repeat(make_not_null(&get(*du_r_divided_by_r).data()),
               get(*boundary_du_r_divided_by_r).data(),
               number_of_radial_grid_points);
        get(*one_minus_y).data() = 1.0 - y;
        get(*j).data() = make_with_random_values<ComplexDataVector>(
            generator, make_not_null(&dist),
            number_of_radial_grid_points *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max));
        get(*k).data() = sqrt(1.0 + get(*j).data() * conj(get(*j).data()));
      });

  TestHelpers::CopyDataBoxTags<Tags::BoundaryValue<Tags::BondiR>,
                               Tags::BoundaryValue<Tags::DuRDividedByR>,
                               Tags::BondiJ>::apply(precomputation_box,
                                                    *expected_box);

  db::mutate<Tags::EthRDividedByR, Tags::EthEthbarRDividedByR,
             Tags::EthEthRDividedByR>(
      expected_box,
      [&l_max, &number_of_radial_grid_points](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
              eth_r_divided_by_r,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              eth_ethbar_r_divided_by_r,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              eth_eth_r_divided_by_r,
          const Scalar<SpinWeighted<ComplexDataVector, 0>>& r) {
        SpinWeighted<ComplexDataVector, 0> r_buffer;
        r_buffer = get(r);
        get(*eth_r_divided_by_r) =
            Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::Eth>(
                l_max, number_of_radial_grid_points, r_buffer) /
            (get(r));
        get(*eth_ethbar_r_divided_by_r) =
            Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::EthEthbar>(
                l_max, number_of_radial_grid_points, r_buffer) /
            (get(r));
        get(*eth_eth_r_divided_by_r) =
            Spectral::Swsh::swsh_derivative<Spectral::Swsh::Tags::EthEth>(
                l_max, number_of_radial_grid_points, r_buffer) /
            (get(r));
      },
      db::get<Tags::BondiR>(*expected_box));
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.PrecomputeCceDependencies",
                  "[Unit][Evolution]") {
  // Initialize the Variables that we need
  MAKE_GENERATOR(generator);
  UniformCustomDistribution<size_t> sdist{5, 8};
  const size_t l_max = sdist(generator);
  const size_t number_of_radial_grid_points = sdist(generator);

  using boundary_variables_tag =
      ::Tags::Variables<pre_computation_boundary_tags>;
  using independent_of_integration_variables_tag =
      ::Tags::Variables<pre_computation_tags>;
  using pre_swsh_derivatives_variables_tag =
      ::Tags::Variables<tmpl::list<Tags::BondiJ>>;

  const size_t number_of_boundary_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_volume_points =
      number_of_boundary_points * number_of_radial_grid_points;
  auto precomputation_box = db::create<db::AddSimpleTags<
      boundary_variables_tag, independent_of_integration_variables_tag,
      pre_swsh_derivatives_variables_tag, Spectral::Swsh::Tags::LMax>>(
      typename boundary_variables_tag::type{number_of_boundary_points},
      typename independent_of_integration_variables_tag::type{
          number_of_volume_points},
      typename pre_swsh_derivatives_variables_tag::type{
          number_of_volume_points},
      l_max);

  auto expected_box =
      db::create<db::AddSimpleTags<boundary_variables_tag,
                                   independent_of_integration_variables_tag,
                                   pre_swsh_derivatives_variables_tag>>(
          typename boundary_variables_tag::type{number_of_boundary_points},
          typename independent_of_integration_variables_tag::type{
              number_of_volume_points},
          typename pre_swsh_derivatives_variables_tag::type{
              number_of_volume_points});

  generate_boundary_values_and_expected(
      make_not_null(&generator), make_not_null(&precomputation_box),
      make_not_null(&expected_box), l_max, number_of_radial_grid_points);

  mutate_all_precompute_cce_dependencies<Tags::BoundaryValue>(
      make_not_null(&precomputation_box));

  CHECK_VARIABLES_APPROX(db::get<boundary_variables_tag>(precomputation_box),
                         db::get<boundary_variables_tag>(expected_box));

  CHECK_VARIABLES_APPROX(
      db::get<pre_swsh_derivatives_variables_tag>(precomputation_box),
      db::get<pre_swsh_derivatives_variables_tag>(expected_box));

  CHECK_VARIABLES_APPROX(
      db::get<independent_of_integration_variables_tag>(precomputation_box),
      db::get<independent_of_integration_variables_tag>(expected_box));
}
}  // namespace Cce
