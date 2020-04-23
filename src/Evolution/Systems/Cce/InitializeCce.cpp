// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/InitializeCce.hpp"

#include <complex>
#include <boost/optional.hpp>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Parallel/Printf.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

void radial_evolve_psi0_condition(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> volume_j_id,
    const SpinWeighted<ComplexDataVector, 2>& boundary_j,
    const SpinWeighted<ComplexDataVector, 2>& boundary_dr_j,
    const SpinWeighted<ComplexDataVector, 0>& r, const double radial_step,
    const size_t l_max, const size_t number_of_radial_points) noexcept {
  // initialize J and dy J = I with their boundary values
  // Optimization note: this could be slightly improved with aggregated
  // allocations via Variables
  SpinWeighted<ComplexDataVector, 2> radial_evolved_j = boundary_j;
  SpinWeighted<ComplexDataVector, 2> radial_evolved_i = 0.5 * boundary_dr_j * r;

  TimeSteppers::RungeKutta3 stepper{};
  TimeSteppers::History<ComplexDataVector, ComplexDataVector> j_history{};
  TimeSteppers::History<ComplexDataVector, ComplexDataVector> i_history{};
  Slab radial_slab{-1.0, 1.0};
  TimeDelta radial_delta{
      radial_slab, Time::rational_t{1, static_cast<int>(2.0 / radial_step)}};
  TimeStepId time{true, 0, Time{radial_slab, Time::rational_t{0, 1}}};
  auto y_collocation =
      Spectral::collocation_points<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto>(
          number_of_radial_points);
  size_t next_y_collocation_point = 0;
  SpinWeighted<ComplexDataVector, 2> radial_evolved_j_dense_step{
      boundary_j.size()};

  while (time.step_time().value() < 1.0) {
    ComplexDataVector dy_j = radial_evolved_i.data();
    const double one_minus_y = 1.0 - time.substep_time().value();
    ComplexDataVector dy_i =
        -0.0625 *
        (square(conj(radial_evolved_i.data()) * radial_evolved_j.data()) +
         square(conj(radial_evolved_j.data()) * radial_evolved_i.data()) -
         2.0 * radial_evolved_i.data() * conj(radial_evolved_i.data()) *
             (2.0 + radial_evolved_j.data() * conj(radial_evolved_j.data()))) *
        (4.0 * radial_evolved_j.data() +
         radial_evolved_i.data() * one_minus_y) /
        (1.0 + radial_evolved_j.data() * conj(radial_evolved_j.data()));
    i_history.insert(time, radial_evolved_i.data(), std::move(dy_i));
    j_history.insert(time, radial_evolved_j.data(), std::move(dy_j));

    if (time.substep() == 2 and
        time.step_time().value() + radial_step >=
            y_collocation[next_y_collocation_point] and
        time.step_time().value() <= y_collocation[next_y_collocation_point]) {
      radial_evolved_j_dense_step = radial_evolved_j;
      stepper.dense_update_u(make_not_null(&radial_evolved_j_dense_step.data()),
                             j_history,
                             y_collocation[next_y_collocation_point]);
      ComplexDataVector angular_view{
          volume_j_id->data().data() +
              next_y_collocation_point *
                  Spectral::Swsh::number_of_swsh_collocation_points(l_max),
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
      angular_view = radial_evolved_j_dense_step.data();
      ++next_y_collocation_point;
    }
    stepper.update_u(make_not_null(&radial_evolved_j.data()),
                     make_not_null(&j_history), radial_delta);
    stepper.update_u(make_not_null(&radial_evolved_i.data()),
                     make_not_null(&i_history), radial_delta);
    time = stepper.next_time_id(time, radial_delta);
  }
  if (time.substep() == 2 and
      time.step_time().value() + radial_step >=
          y_collocation[next_y_collocation_point] and
      time.step_time().value() <= y_collocation[next_y_collocation_point]) {
    radial_evolved_j_dense_step = radial_evolved_j;
    stepper.dense_update_u(make_not_null(&radial_evolved_j_dense_step.data()),
                           j_history, y_collocation[next_y_collocation_point]);
    radial_evolved_j_dense_step = radial_evolved_j;
    ComplexDataVector angular_view{
        volume_j_id->data().data() +
            next_y_collocation_point *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    angular_view = radial_evolved_j_dense_step.data();
    ++next_y_collocation_point;
  }
}

// perform an iterative solve for the set of angular coordinates necessary to
// set the gauge transformed version of `surface_j` to zero. This reliably
// converges eventually provided `surface_j` is initially reasonably small. As a
// comparatively primitive method, the convergence often takes several
// iterations (10-100) to reach roundoff; However, the iterations are fast, and
// the computation is for initial data that needs to be computed only once
// during a simulation, so it is not currently an optimization priority. If this
// function becomes a bottleneck, the numerical procedure of the iterative
// method should be revisited.
double adjust_angular_coordinates_for_j(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> volume_j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const SpinWeighted<ComplexDataVector, 2>& surface_j, const size_t l_max,
    const double tolerance, const size_t max_steps,
    const bool adjust_volume_gauge,
    boost::optional<const SpinWeighted<ComplexDataVector, 2>&> target_j =
        boost::none) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    get<0>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.theta;
    get<1>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.phi;
  }
  get<0>(*cartesian_cauchy_coordinates) =
      sin(get<0>(*angular_cauchy_coordinates)) *
      cos(get<1>(*angular_cauchy_coordinates));
  get<1>(*cartesian_cauchy_coordinates) =
      sin(get<0>(*angular_cauchy_coordinates)) *
      sin(get<1>(*angular_cauchy_coordinates));
  get<2>(*cartesian_cauchy_coordinates) =
      cos(get<0>(*angular_cauchy_coordinates));

  Variables<tmpl::list<
      // cartesian coordinates
      ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // eth of cartesian coordinates
      ::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<5, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      // eth of gauge-transformed cartesian coordinates
      ::Tags::SpinWeighted<::Tags::TempScalar<6, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<7, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<8, ComplexDataVector>,
                           std::integral_constant<int, 1>>,
      // iterated J
      ::Tags::SpinWeighted<::Tags::TempScalar<9, ComplexDataVector>,
                           std::integral_constant<int, 2>>,
      // intermediate J buffer
      ::Tags::SpinWeighted<::Tags::TempScalar<10, ComplexDataVector>,
                           std::integral_constant<int, 2>>,
      // K buffer
      ::Tags::SpinWeighted<::Tags::TempScalar<11, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // gauge Jacobians
      ::Tags::SpinWeighted<::Tags::TempScalar<12, ComplexDataVector>,
                           std::integral_constant<int, 2>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<13, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // gauge Jacobians on next iteration
      ::Tags::SpinWeighted<::Tags::TempScalar<14, ComplexDataVector>,
                           std::integral_constant<int, 2>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<15, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // gauge conformal factor
      ::Tags::SpinWeighted<::Tags::TempScalar<16, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      // cartesian coordinates steps
      ::Tags::SpinWeighted<::Tags::TempScalar<17, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<18, ComplexDataVector>,
                           std::integral_constant<int, 0>>,
      ::Tags::SpinWeighted<::Tags::TempScalar<19, ComplexDataVector>,
                           std::integral_constant<int, 0>>>>
      computation_buffers{number_of_angular_points};

  auto& x =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& y =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<1, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& z =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<2, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));

  x.data() =
      std::complex<double>(1.0, 0.0) * get<0>(*cartesian_cauchy_coordinates);
  y.data() =
      std::complex<double>(1.0, 0.0) * get<1>(*cartesian_cauchy_coordinates);
  z.data() =
      std::complex<double>(1.0, 0.0) * get<2>(*cartesian_cauchy_coordinates);

  auto& eth_x =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<3, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& eth_y =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<4, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& eth_z =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<5, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));

  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::Eth, Spectral::Swsh::Tags::Eth,
                 Spectral::Swsh::Tags::Eth>>(l_max, 1, make_not_null(&eth_x),
                                             make_not_null(&eth_y),
                                             make_not_null(&eth_z), x, y, z);

  auto& evolution_gauge_eth_x =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<6, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& evolution_gauge_eth_y =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<7, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));
  auto& evolution_gauge_eth_z =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<8, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          computation_buffers));

  auto& evolution_gauge_surface_j =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<9, ComplexDataVector>,
                                   std::integral_constant<int, 2>>>(
          computation_buffers));

  auto& interpolated_j =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<10, ComplexDataVector>,
                                   std::integral_constant<int, 2>>>(
          computation_buffers));
  auto& interpolated_k =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<11, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));

  auto& gauge_c =
      get<::Tags::SpinWeighted<::Tags::TempScalar<12, ComplexDataVector>,
                               std::integral_constant<int, 2>>>(
          computation_buffers);
  auto& gauge_d =
      get<::Tags::SpinWeighted<::Tags::TempScalar<13, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  auto& next_gauge_c =
      get<::Tags::SpinWeighted<::Tags::TempScalar<14, ComplexDataVector>,
                               std::integral_constant<int, 2>>>(
          computation_buffers);
  auto& next_gauge_d =
      get<::Tags::SpinWeighted<::Tags::TempScalar<15, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  auto& gauge_omega =
      get<::Tags::SpinWeighted<::Tags::TempScalar<16, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          computation_buffers);

  auto& x_step =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<17, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& y_step =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<18, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));
  auto& z_step =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<19, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          computation_buffers));

  double max_error = 1.0;
  size_t number_of_steps = 0;
  while (true) {
    GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(angular_cauchy_coordinates,
                                            cartesian_cauchy_coordinates);

    Spectral::Swsh::SwshInterpolator iteration_interpolator{
      get<0>(*angular_cauchy_coordinates),
      get<1>(*angular_cauchy_coordinates), l_max};

    GaugeUpdateJacobianFromCoordinates<
        Tags::GaugeC, Tags::GaugeD, Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(make_not_null(&gauge_c),
                                            make_not_null(&gauge_d),
                                            angular_cauchy_coordinates,
                                            *cartesian_cauchy_coordinates,
                                            l_max);

    iteration_interpolator.interpolate(make_not_null(&interpolated_j),
                                       surface_j);
    interpolated_k.data() =
        sqrt(1.0 + interpolated_j.data() * conj(interpolated_j.data()));

    get(gauge_omega).data() =
        0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                   get(gauge_c).data() * conj(get(gauge_c).data()));
    // the fully computed j in the coordinate system determined so far
    // (`previous_angular_cauchy_coordinates`)
    evolution_gauge_surface_j.data() =
        0.25 *
        (square(conj(get(gauge_d).data())) * interpolated_j.data() +
         square(get(gauge_c).data()) * conj(interpolated_j.data()) +
         2.0 * get(gauge_c).data() * conj(get(gauge_d).data()) *
             interpolated_k.data()) /
        square(get(gauge_omega).data());

    // check completion conditions
    if(not target_j) {
      max_error = max(abs(evolution_gauge_surface_j.data()));
    } else {
      max_error =
          max(abs(evolution_gauge_surface_j.data() - (*target_j).data()));
    }
    Parallel::printf("step %zu, error %e\n", number_of_steps, max_error);
    ++number_of_steps;
    if (max_error > 5.0e-3) {
      ERROR(
          "Iterative solve for surface coordinates of initial data failed. The "
          "strain is too large to be fully eliminated by a well-behaved "
          "alteration of the spherical mesh. For this data, please use an "
          "alternative initial data generator such as "
          "`InitializeJInverseCubic`.");
    }
    if (max_error < tolerance or number_of_steps > max_steps) {
      break;
    }
    // The alteration in each of the spin-weighted Jacobian factors determined
    // by linearizing the system in small J
    if (not target_j) {
      get(next_gauge_c).data() = -0.5 * evolution_gauge_surface_j.data() *
                                 square(get(gauge_omega).data()) /
                                 (get(gauge_d).data() * interpolated_k.data());
      get(next_gauge_d).data() = get(next_gauge_c).data() *
                                 conj(get(gauge_c).data()) /
                                 conj(get(gauge_d).data());
    } else {
      get(next_gauge_c).data() =
          -0.5 * (evolution_gauge_surface_j.data() - (*target_j).data()) *
          square(get(gauge_omega).data()) /
          (get(gauge_d).data() * interpolated_k.data());
      get(next_gauge_d).data() = get(next_gauge_c).data() *
          conj(get(gauge_c).data()) /
          conj(get(gauge_d).data());
    }

    iteration_interpolator.interpolate(make_not_null(&evolution_gauge_eth_x),
                                       eth_x);
    iteration_interpolator.interpolate(make_not_null(&evolution_gauge_eth_y),
                                       eth_y);
    iteration_interpolator.interpolate(make_not_null(&evolution_gauge_eth_z),
                                       eth_z);

    evolution_gauge_eth_x =
        0.5 * ((get(next_gauge_c)) * conj(evolution_gauge_eth_x) +
               conj((get(next_gauge_d))) * evolution_gauge_eth_x);
    evolution_gauge_eth_y =
        0.5 * ((get(next_gauge_c)) * conj(evolution_gauge_eth_y) +
               conj((get(next_gauge_d))) * evolution_gauge_eth_y);
    evolution_gauge_eth_z =
        0.5 * ((get(next_gauge_c)) * conj(evolution_gauge_eth_z) +
               conj((get(next_gauge_d))) * evolution_gauge_eth_z);

    // here we attempt to just update the current value according to the
    // alteration suggested. Ideally that's the `dominant` part of the needed
    // alteration
    Spectral::Swsh::angular_derivatives<tmpl::list<
        Spectral::Swsh::Tags::InverseEth, Spectral::Swsh::Tags::InverseEth,
        Spectral::Swsh::Tags::InverseEth>>(
        l_max, 1, make_not_null(&x_step), make_not_null(&y_step),
        make_not_null(&z_step), evolution_gauge_eth_x, evolution_gauge_eth_y,
        evolution_gauge_eth_z);

    get<0>(*cartesian_cauchy_coordinates) += real(x_step.data());
    get<1>(*cartesian_cauchy_coordinates) += real(y_step.data());
    get<2>(*cartesian_cauchy_coordinates) += real(z_step.data());
  }

  GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords,
      Tags::CauchyCartesianCoords>::apply(angular_cauchy_coordinates,
                                          cartesian_cauchy_coordinates);

  if (adjust_volume_gauge) {
    GaugeUpdateJacobianFromCoordinates<
        Tags::GaugeC, Tags::GaugeD, Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>::apply(make_not_null(&gauge_c),
                                            make_not_null(&gauge_d),
                                            angular_cauchy_coordinates,
                                            *cartesian_cauchy_coordinates,
                                            l_max);

    get(gauge_omega).data() =
        0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                   get(gauge_c).data() * conj(get(gauge_c).data()));

    GaugeAdjustInitialJ::apply(volume_j, gauge_c, gauge_d, gauge_omega,
                               *angular_cauchy_coordinates, l_max);
  }
  return max_error;
}

void GaugeAdjustInitialJ::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> volume_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_omega,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
        cauchy_angular_coordinates,
    const size_t l_max) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points =
      get(*volume_j).size() /
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Scalar<SpinWeighted<ComplexDataVector, 2>> evolution_coords_j_buffer{
      number_of_angular_points};
  Spectral::Swsh::SwshInterpolator interpolator{
      get<0>(cauchy_angular_coordinates), get<1>(cauchy_angular_coordinates),
      l_max};
  for (size_t i = 0; i < number_of_radial_points; ++i) {
    Scalar<SpinWeighted<ComplexDataVector, 2>> angular_view_j;
    get(angular_view_j)
        .set_data_ref(
            get(*volume_j).data().data() + i * number_of_angular_points,
            number_of_angular_points);
    get(evolution_coords_j_buffer) = get(angular_view_j);
    GaugeAdjustedBoundaryValue<Tags::BondiJ>::apply(
        make_not_null(&angular_view_j), evolution_coords_j_buffer, gauge_c,
        gauge_d, gauge_omega, interpolator);
  }
}

InitializeJImportModes::InitializeJImportModes(
    const std::string& mode_filename, const std::string& mode_dataset,
    const size_t file_l_max, const double angular_coordinate_tolerance,
    const size_t max_iterations, const double start_time) noexcept
    : buffer_updater_{mode_filename, mode_dataset, file_l_max, 2_st},
      filename_{mode_filename},
      dataset_name_{mode_dataset},
      file_l_max_{file_l_max},
      angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations},
      start_time_{start_time} {}

std::unique_ptr<InitializeJ> InitializeJImportModes::get_clone() const
    noexcept {
  return std::make_unique<InitializeJImportModes>(
      filename_, dataset_name_, file_l_max_, angular_coordinate_tolerance_,
      max_iterations_, start_time_);
}

void InitializeJImportModes::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, const size_t l_max,
    const size_t number_of_radial_points) const noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  intrp::BarycentricRationalSpanInterpolator interpolator{10_st, 10_st};
  // get the appropriately interpolated dataset to use for the angular
  // coordinate solve.
  ComplexModalVector buffer{
      square(l_max + 1) *
      (2_st + 2 * interpolator.required_number_of_points_before_and_after())};
  size_t time_span_start;
  size_t time_span_end;
  buffer_updater_.update_buffer_for_time(
      make_not_null(&buffer), make_not_null(&time_span_start),
      make_not_null(&time_span_end), start_time_, l_max, 6_st, 2_st);

  auto interpolation_time_span = detail::create_span_for_time_value(
      start_time_, 0, interpolator.required_number_of_points_before_and_after(),
      time_span_start, time_span_end, buffer_updater_.get_time_buffer());

  // search through and find the two interpolation points the time point is
  // between. If we can, put the range for the interpolation centered on the
  // desired point. If that can't be done (near the start or the end of the
  // simulation), make the range terminated at the start or end of the cached
  // data and extending for the desired range in the other direction.
  const size_t buffer_span_size = time_span_end - time_span_start;
  const size_t interpolation_span_size =
      interpolation_time_span.second - interpolation_time_span.first;

  DataVector time_points{
      buffer_updater_.get_time_buffer().data() + interpolation_time_span.first,
      interpolation_span_size};

  auto interpolate_from_column =
      [this, &time_points, &buffer_span_size, &interpolation_time_span,
       &interpolator, &time_span_start,
       &interpolation_span_size](auto data, size_t column) {
        const auto interp_val = interpolator.interpolate(
            gsl::span<const double>(time_points.data(), time_points.size()),
            gsl::span<const std::complex<double>>(
                data + column * (buffer_span_size) +
                    (interpolation_time_span.first - time_span_start),
                interpolation_span_size),
            start_time_);
        return interp_val;
      };

  SpinWeighted<ComplexModalVector, 2> target_j_transform{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};

  for (const auto& libsharp_mode :
       Spectral::Swsh::cached_coefficients_metadata(l_max)) {
    Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
        libsharp_mode, make_not_null(&target_j_transform), 0,
        interpolate_from_column(
            buffer.data(),
            Spectral::Swsh::goldberg_mode_index(
                l_max, libsharp_mode.l, static_cast<int>(libsharp_mode.m))),
        interpolate_from_column(
            buffer.data(),
            Spectral::Swsh::goldberg_mode_index(
                l_max, libsharp_mode.l, -static_cast<int>(libsharp_mode.m))));
  }
  SpinWeighted<ComplexDataVector, 2> target_j =
      Spectral::Swsh::inverse_swsh_transform(l_max, 1, target_j_transform);
  target_j = target_j / get(r);

  // now we have initialized the boundary data compatible with the inverse-cubic
  // state from the boundary, and we'll transform to the set of coordinates
  // necessary to fix the 1/r part of the initial J to the modes determined by
  // the input data.
  adjust_angular_coordinates_for_j(j, cartesian_cauchy_coordinates,
                                   angular_cauchy_coordinates, get(boundary_j),
                                   l_max, angular_coordinate_tolerance_,
                                   max_iterations_, false, target_j);

  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);

  const SpinWeighted<ComplexDataVector, 2> one_minus_y_coefficient = target_j;

  for (size_t i = 0; i < number_of_radial_points; i++) {
    ComplexDataVector angular_view_j{
        get(*j).data().data() + get(boundary_j).size() * i,
        number_of_angular_points};
    angular_view_j =
        0.5 * one_minus_y_collocation[i] * one_minus_y_coefficient.data();
  }
}

void InitializeJImportModes::pup(PUP::er& p) noexcept {
  p | buffer_updater_;
  p | filename_;
  p | dataset_name_;
  p | file_l_max_;
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
  p | start_time_;
}


InitializeJNoIncomingRadiation::InitializeJNoIncomingRadiation(
    const double angular_coordinate_tolerance, const size_t max_iterations,
    const double radial_step) noexcept
    : angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations},
      radial_step_{radial_step} {}

std::unique_ptr<InitializeJ> InitializeJNoIncomingRadiation::get_clone() const
    noexcept {
  return std::make_unique<InitializeJNoIncomingRadiation>(
      angular_coordinate_tolerance_, max_iterations_, radial_step_);
}

void InitializeJNoIncomingRadiation::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, const size_t l_max,
    const size_t number_of_radial_points) const noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  radial_evolve_psi0_condition(make_not_null(&get(*j)), get(boundary_j),
                               get(boundary_dr_j), get(r), radial_step_, l_max,
                               number_of_radial_points);
  const SpinWeighted<ComplexDataVector, 2> j_at_scri_view;
  make_const_view(make_not_null(&j_at_scri_view), get(*j),
                  (number_of_radial_points - 1) * number_of_angular_points,
                  number_of_angular_points);

  adjust_angular_coordinates_for_j(j, cartesian_cauchy_coordinates,
                                   angular_cauchy_coordinates, j_at_scri_view,
                                   l_max, angular_coordinate_tolerance_,
                                   max_iterations_, true);
}

void InitializeJNoIncomingRadiation::pup(PUP::er& p) noexcept {
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
  p | radial_step_;
}

InitializeJZeroNonSmooth::InitializeJZeroNonSmooth(
    const double angular_coordinate_tolerance,
    const size_t max_iterations) noexcept
    : angular_coordinate_tolerance_{angular_coordinate_tolerance},
      max_iterations_{max_iterations} {}

std::unique_ptr<InitializeJ> InitializeJZeroNonSmooth::get_clone() const
    noexcept {
  return std::make_unique<InitializeJZeroNonSmooth>(
      angular_coordinate_tolerance_, max_iterations_);
}

void InitializeJZeroNonSmooth::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& /* boundary_dr_j*/,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*r*/, const size_t l_max,
    const size_t /*number_of_radial_points*/) const noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  const SpinWeighted<ComplexDataVector, 2> j_boundary_view;
  make_const_view(make_not_null(&j_boundary_view), get(*j), 0,
                  number_of_angular_points);

  get(*j).data() = 0.0;
  adjust_angular_coordinates_for_j(j, cartesian_cauchy_coordinates,
                                   angular_cauchy_coordinates, get(boundary_j),
                                   l_max, angular_coordinate_tolerance_,
                                   max_iterations_, false);
}

void InitializeJZeroNonSmooth::pup(PUP::er& p) noexcept {
  p | angular_coordinate_tolerance_;
  p | max_iterations_;
}

std::unique_ptr<InitializeJ> InitializeJInverseCubic::get_clone() const
    noexcept {
  return std::make_unique<InitializeJInverseCubic>();
}

void InitializeJInverseCubic::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, const size_t l_max,
    const size_t number_of_radial_points) const noexcept {
  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);
  for (size_t i = 0; i < number_of_radial_points; i++) {
    ComplexDataVector angular_view_j{
        get(*j).data().data() + get(boundary_j).size() * i,
        get(boundary_j).size()};
    // auto is acceptable here as these two values are only used once in the
    // below computation. `auto` causes an expression template to be
    // generated, rather than allocating.
    const auto one_minus_y_coefficient =
        0.25 * (3.0 * get(boundary_j).data() +
                get(r).data() * get(boundary_dr_j).data());
    const auto one_minus_y_cubed_coefficient =
        -0.0625 *
        (get(boundary_j).data() + get(r).data() * get(boundary_dr_j).data());
    angular_view_j =
        one_minus_y_collocation[i] * one_minus_y_coefficient +
        pow<3>(one_minus_y_collocation[i]) * one_minus_y_cubed_coefficient;
  }
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    get<0>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.theta;
    get<1>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.phi;
  }
  get<0>(*cartesian_cauchy_coordinates) =
      sin(get<0>(*angular_cauchy_coordinates)) *
      cos(get<1>(*angular_cauchy_coordinates));
  get<1>(*cartesian_cauchy_coordinates) =
      sin(get<0>(*angular_cauchy_coordinates)) *
      sin(get<1>(*angular_cauchy_coordinates));
  get<2>(*cartesian_cauchy_coordinates) =
      cos(get<0>(*angular_cauchy_coordinates));
}

void InitializeJInverseCubic::pup(PUP::er& /*p*/) noexcept {}


std::unique_ptr<InitializeJ> InitializeJInverseQuartic::get_clone() const
    noexcept {
  return std::make_unique<InitializeJInverseQuartic>();
}

void InitializeJInverseQuartic::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, const size_t l_max,
    const size_t number_of_radial_points) const noexcept {
  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);
  for (size_t i = 0; i < number_of_radial_points; i++) {
    ComplexDataVector angular_view_j{
        get(*j).data().data() + get(boundary_j).size() * i,
        get(boundary_j).size()};
    // auto is acceptable here as these two values are only used once in the
    // below computation. `auto` causes an expression template to be
    // generated, rather than allocating.
    const auto one_minus_y_cubed_coefficient =
        0.125 * (4.0 * get(boundary_j).data() +
                 get(r).data() * get(boundary_dr_j).data());
    const auto one_minus_y_quartic_coefficient =
        -0.0625 * (3.0 * get(boundary_j).data() +
                   get(r).data() * get(boundary_dr_j).data());
    angular_view_j =
        pow<3>(one_minus_y_collocation[i]) * one_minus_y_cubed_coefficient +
        pow<4>(one_minus_y_collocation[i]) * one_minus_y_quartic_coefficient;
  }
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    get<0>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.theta;
    get<1>(*angular_cauchy_coordinates)[collocation_point.offset] =
        collocation_point.phi;
  }
  get<0>(*cartesian_cauchy_coordinates) =
      sin(get<0>(*angular_cauchy_coordinates)) *
      cos(get<1>(*angular_cauchy_coordinates));
  get<1>(*cartesian_cauchy_coordinates) =
      sin(get<0>(*angular_cauchy_coordinates)) *
      sin(get<1>(*angular_cauchy_coordinates));
  get<2>(*cartesian_cauchy_coordinates) =
      cos(get<0>(*angular_cauchy_coordinates));
}

void InitializeJInverseQuartic::pup(PUP::er& /*p*/) noexcept {}

/// \cond
PUP::able::PUP_ID InitializeJImportModes::my_PUP_ID = 0;
PUP::able::PUP_ID InitializeJNoIncomingRadiation::my_PUP_ID = 0;
PUP::able::PUP_ID InitializeJZeroNonSmooth::my_PUP_ID = 0;
PUP::able::PUP_ID InitializeJInverseCubic::my_PUP_ID = 0;
PUP::able::PUP_ID InitializeJInverseQuartic::my_PUP_ID = 0;
/// \endcond
}  // namespace Cce
