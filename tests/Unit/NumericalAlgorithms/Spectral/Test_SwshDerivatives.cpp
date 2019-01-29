// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>
#include <string>

#include "DataStructures/ComplexDataVector.hpp"   // IWYU pragma: keep
#include "DataStructures/ComplexModalVector.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransformJob.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare ComplexDataVector
// IWYU pragma: no_forward_declare ComplexModalVector
// IWYU pragma: no_forward_declare SpinWeighted

namespace Spectral {
namespace Swsh {
namespace {

template <int Spin>
struct TestTag : db::SimpleTag {
  static std::string name() noexcept { return "TestTag"; }
  using type = Scalar<SpinWeighted<ComplexDataVector, Spin>>;
};

template <int Spin>
struct ExpectedTestTag : db::SimpleTag {
  static std::string name() noexcept { return "ExpectedTestTag"; }
  using type = Scalar<SpinWeighted<ComplexDataVector, Spin>>;
};

template <int Spin>
struct GeneratedTestTag : db::SimpleTag {
  static std::string name() noexcept { return "ExpectedTestTag"; }
  using type = Scalar<SpinWeighted<ComplexDataVector, Spin>>;
};

// TODO:JM more cleanup and documentation
template <typename DerivativeKind, ComplexRepresentation Representation,
          int Spin>
void test_derivative_via_transforms() noexcept {
  // generate coefficients for the transformation
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> size_distribution{2, 7};
  const size_t l_max = size_distribution(gen);
  const size_t number_of_radial_points = 2;

  UniformCustomDistribution<double> coefficient_distribution{-10.0, 10.0};

  Variables<
      tmpl::list<TestTag<Spin>, Tags::Derivative<TestTag<Spin>, DerivativeKind>,
                 Tags::Derivative<ExpectedTestTag<Spin>, DerivativeKind>>>
      collocation_data{
          number_of_radial_points * number_of_swsh_collocation_points(l_max),
          0.0};
  Variables<tmpl::list<
      Tags::SwshTransform<ExpectedTestTag<Spin>>,
      Tags::SwshTransform<TestTag<Spin>>,
      Tags::SwshTransform<Tags::Derivative<TestTag<Spin>, DerivativeKind>>>>
      coefficient_data{
          number_of_swsh_coefficients(l_max) * 2 * number_of_radial_points,
          0.0};

  ComplexModalVector& generated_modes =
      get(get<Tags::SwshTransform<ExpectedTestTag<Spin>>>(coefficient_data))
          .data();
  TestHelpers::generate_swsh_modes<Spin>(
      make_not_null(&generated_modes), make_not_null(&gen),
      make_not_null(&coefficient_distribution), number_of_radial_points, l_max);

  // fill the expected collocation point data by evaluating the analytic
  // functions. This is very slow and rough (due to factorial division), but
  // comparatively simple to formulate.
  ComplexDataVector& computed_collocation =
      get(get<TestTag<Spin>>(collocation_data)).data();
  ComplexDataVector& expected_derived_collocation =
      get(get<Tags::Derivative<ExpectedTestTag<Spin>, DerivativeKind>>(
              collocation_data))
          .data();
  TestHelpers::swsh_collocation_from_coefficients_and_basis_func<
      Spin, Representation>(&computed_collocation, &generated_modes, l_max,
                            number_of_radial_points,
                            TestHelpers::spin_weighted_spherical_harmonic);
  TestHelpers::swsh_collocation_from_coefficients_and_basis_func<
      Spin, Representation>(
      &expected_derived_collocation, &generated_modes, l_max,
      number_of_radial_points,
      TestHelpers::derivative_of_spin_weighted_spherical_harmonic<
          DerivativeKind>);

  using JobTags = tmpl::list<TestTag<Spin>>;
  TransformJob<Spin, Representation, JobTags> job{l_max,
                                                  number_of_radial_points};
  job.execute_transform(make_not_null(&coefficient_data),
                        make_not_null(&collocation_data));

  compute_derivative_coefficients<
      Spin, tmpl::list<Tags::Derivative<TestTag<Spin>, DerivativeKind>>>(
      l_max, number_of_radial_points, coefficient_data,
      make_not_null(&coefficient_data));

  using InverseJobTags =
      tmpl::list<Tags::Derivative<TestTag<Spin>, DerivativeKind>>;
  TransformJob<Tags::Derivative<TestTag<Spin>, DerivativeKind>::spin,
               Representation, InverseJobTags>
      inverse_job{l_max, number_of_radial_points};
  inverse_job.execute_inverse_transform(make_not_null(&collocation_data),
                                        make_not_null(&coefficient_data));

  // approximation needs to be a little loose to consistently accomodate the
  // ratios of factorials in the analytic form
  Approx sht_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);

  ComplexDataVector& transform_derived_collocation =
      get(get<Tags::Derivative<TestTag<Spin>, DerivativeKind>>(
              collocation_data))
          .data();

  CHECK_ITERABLE_CUSTOM_APPROX(transform_derived_collocation,
                               expected_derived_collocation, sht_approx);
}

template <ComplexRepresentation Representation, int Spin1, int Spin2,
          typename DerivativeKind1, typename DerivativeKind2>
void test_compute_swsh_derivatives() noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> size_distribution{2, 7};
  const size_t l_max = size_distribution(gen);
  const size_t number_of_radial_points = 2;
  UniformCustomDistribution<double> coefficient_distribution{-10.0, 10.0};

  using collocation_tag_list = tmpl::list<TestTag<Spin1>, TestTag<Spin2>>;
  using coefficient_tag_list =
      db::wrap_tags_in<Tags::SwshTransform, collocation_tag_list>;
  using derivative_collocation_tag_list =
      tmpl::list<Tags::Derivative<TestTag<Spin1>, DerivativeKind1>,
                 Tags::Derivative<TestTag<Spin2>, DerivativeKind1>,
                 Tags::Derivative<TestTag<Spin1>, DerivativeKind2>,
                 Tags::Derivative<TestTag<Spin2>, DerivativeKind2>>;
  using derivative_coefficient_tag_list =
      db::wrap_tags_in<Tags::SwshTransform, derivative_collocation_tag_list>;

  // The Variables for the randomly generated coefficients to take derivatives
  // of and transform
  Variables<coefficient_tag_list> generated_coefficient_data{
      number_of_swsh_coefficients(l_max) * 2 * number_of_radial_points, 0.0};
  // The Variables for the derivatives evaluated from the generated coefficients
  // multiplied by the derivative of analytic basis functions
  Variables<derivative_collocation_tag_list>
      expected_derivative_collocation_data{
          number_of_radial_points * number_of_swsh_collocation_points(l_max),
          0.0};
  // The Variables for collocation points prior to tranformation. These will be
  // created from generated coefficients, with analytic basis functions.
  Variables<collocation_tag_list> collocation_data{
      number_of_radial_points * number_of_swsh_collocation_points(l_max), 0.0};
  // The Variables for the coefficients, both before and after the derivative is
  // taken. These will function as an intermediate 'buffer' passed to the
  // derivative-taking function.
  Variables<tmpl::append<coefficient_tag_list, derivative_coefficient_tag_list>>
      coefficient_data{
          number_of_swsh_coefficients(l_max) * 2 * number_of_radial_points,
          0.0};
  // The Variables for the final output derivatives from the derivative-taking
  // operation
  Variables<derivative_collocation_tag_list> derivative_collocation_data{
      number_of_radial_points * number_of_swsh_collocation_points(l_max), 0.0};

  // generate the modes to be used for the test
  tmpl::for_each<coefficient_tag_list>(
      [&gen, &coefficient_distribution, &generated_coefficient_data,
       &number_of_radial_points, &l_max](auto x) {
        using coefficient_tag = typename decltype(x)::type;
        TestHelpers::generate_swsh_modes<coefficient_tag::spin>(
            make_not_null(
                &get(get<coefficient_tag>(generated_coefficient_data)).data()),
            make_not_null(&gen), make_not_null(&coefficient_distribution),
            number_of_radial_points, l_max);
      });
  // Put the collocation information associated with the generated modes in the
  // appropriate collocation data
  tmpl::for_each<collocation_tag_list>([&collocation_data,
                                        &generated_coefficient_data, &l_max,
                                        &number_of_radial_points](auto x) {
    using collocation_tag = typename decltype(x)::type;
    TestHelpers::swsh_collocation_from_coefficients_and_basis_func<
        collocation_tag::type::type::spin, Representation>(
        &get(get<collocation_tag>(collocation_data)).data(),
        &get(get<Tags::SwshTransform<collocation_tag>>(
                 generated_coefficient_data))
             .data(),
        l_max, number_of_radial_points,
        TestHelpers::spin_weighted_spherical_harmonic);
  });
  // put the collocation data from the analytic solution for the derivatives in
  // the appropriate locations in expected_derivative_collocation_data
  tmpl::for_each<derivative_collocation_tag_list>(
      [&expected_derivative_collocation_data, &generated_coefficient_data,
       &l_max, &number_of_radial_points](auto x) {
        using derivative_collocation_tag = typename decltype(x)::type;
        TestHelpers::swsh_collocation_from_coefficients_and_basis_func<
            derivative_collocation_tag::derived_from::type::type::spin,
            Representation>(
            &get(get<derivative_collocation_tag>(
                     expected_derivative_collocation_data))
                 .data(),
            &get(get<Tags::SwshTransform<
                     typename derivative_collocation_tag::derived_from>>(
                     generated_coefficient_data))
                 .data(),
            l_max, number_of_radial_points,
            TestHelpers::derivative_of_spin_weighted_spherical_harmonic<
                typename derivative_collocation_tag::derivative_kind>);
      });
  // the actual derivative call
  compute_derivatives<Representation, derivative_collocation_tag_list>(
      make_not_null(&collocation_data), make_not_null(&coefficient_data),
      make_not_null(&derivative_collocation_data), l_max);
  // check the result
  Approx sht_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);
  {
    INFO("Check the coefficient data intermediate step");
    tmpl::for_each<coefficient_tag_list>(
        [&coefficient_data, &generated_coefficient_data, &sht_approx](auto x) {
          using coefficient_tag = typename decltype(x)::type;
          CHECK_ITERABLE_CUSTOM_APPROX(
              get(get<coefficient_tag>(generated_coefficient_data)).data(),
              get(get<coefficient_tag>(coefficient_data)).data(), sht_approx);
        });
  }
  {
    INFO("Check the collocation derivatives final result");
    CHECK_VARIABLES_CUSTOM_APPROX(derivative_collocation_data,
                                  expected_derivative_collocation_data,
                                  sht_approx);
  }
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.Derivatives",
                  "[Unit][NumericalAlgorithms]") {
  // we do not test the full set of combinations of derivatives, spins, and
  // slice kinds due to the slow execution time. We test a handful of each spin,
  // each derivative, and of each slice type.
  {
    INFO("Test evaluation of Eth using generated values");
    test_derivative_via_transforms<Tags::Eth,
                                   ComplexRepresentation::Interleaved, -2>();
    test_derivative_via_transforms<Tags::Eth,
                                   ComplexRepresentation::RealsThenImags, 0>();
  }
  {
    INFO("Test evaluation of Ethbar using generated values");
    test_derivative_via_transforms<Tags::Ethbar,
                                   ComplexRepresentation::RealsThenImags, -1>();
    test_derivative_via_transforms<Tags::Ethbar,
                                   ComplexRepresentation::Interleaved, 1>();
  }
  {
    INFO("Test evaluation of EthEth using generated values");
    test_derivative_via_transforms<Tags::EthEth,
                                   ComplexRepresentation::Interleaved, -2>();
    test_derivative_via_transforms<Tags::EthEth,
                                   ComplexRepresentation::RealsThenImags, 0>();
  }
  {
    INFO("Test evaluation of EthbarEthbar using generated values");
    test_derivative_via_transforms<Tags::EthbarEthbar,
                                   ComplexRepresentation::Interleaved, 0>();
    test_derivative_via_transforms<Tags::EthbarEthbar,
                                   ComplexRepresentation::RealsThenImags, 2>();
  }
  {
    INFO("Test evaluation of EthEthbar using generated values");
    test_derivative_via_transforms<Tags::EthEthbar,
                                   ComplexRepresentation::Interleaved, -2>();
    test_derivative_via_transforms<Tags::EthEthbar,
                                   ComplexRepresentation::RealsThenImags, 0>();
    test_derivative_via_transforms<Tags::EthEthbar,
                                   ComplexRepresentation::Interleaved, 2>();
  }
  {
    INFO("Test evaluation of EthbarEth using generated values");
    test_derivative_via_transforms<Tags::EthbarEth,
                                   ComplexRepresentation::RealsThenImags, -1>();
    test_derivative_via_transforms<Tags::EthbarEth,
                                   ComplexRepresentation::Interleaved, 0>();
    test_derivative_via_transforms<Tags::EthbarEth,
                                   ComplexRepresentation::RealsThenImags, 1>();
  }
  {
    INFO("Test compute_swsh_derivatives utility");
    test_compute_swsh_derivatives<ComplexRepresentation::Interleaved, -1, 1,
                                  Tags::Eth, Tags::Ethbar>();
    test_compute_swsh_derivatives<ComplexRepresentation::RealsThenImags, -1, 1,
                                  Tags::Eth, Tags::Ethbar>();
    test_compute_swsh_derivatives<ComplexRepresentation::Interleaved, -1, 1,
                                  Tags::EthEthbar, Tags::EthbarEth>();
    test_compute_swsh_derivatives<ComplexRepresentation::RealsThenImags, -1, 1,
                                  Tags::EthEthbar, Tags::EthbarEth>();
    test_compute_swsh_derivatives<ComplexRepresentation::Interleaved, 0, 2,
                                  Tags::Ethbar, Tags::EthbarEthbar>();
    test_compute_swsh_derivatives<ComplexRepresentation::RealsThenImags, -2, 0,
                                  Tags::Eth, Tags::EthEth>();
  }
}
}  // namespace
}  // namespace Swsh
}  // namespace Spectral
