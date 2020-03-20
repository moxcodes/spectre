// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce {
namespace Tags {
/// \cond
struct LMax;
struct NumberOfRadialPoints;
/// \endcond
}  // namespace Tags

struct GaugeAdjustInitialJ {
  using boundary_tags =
      tmpl::list<Tags::GaugeC, Tags::GaugeD, Tags::GaugeOmega,
                 Tags::CauchyAngularCoords, Spectral::Swsh::Tags::LMax>;
  using return_tags = tmpl::list<Tags::BondiJ>;
  using argument_tags = tmpl::append<boundary_tags>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> volume_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_omega,
      const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
          cauchy_angular_coordinates,
      size_t l_max) noexcept;
};

struct InitializeJZeroNonSmooth;

struct InitializeJInverseCubic;

/*!
 * \brief Abstract base class for an initial hypersurface data generator for
 * Cce.
 *
 * \details The functions that are required to be overriden in the derived
 * classes are:
 * - `InitializeJ::get_clone()`: should return a
 * `std::unique_ptr<InitializeJ>` with cloned state.
 * - `InitializeJ::operator() const`: should take as arguments, first a set of
 * `gsl::not_null` pointers represented by `mutate_tags`, followed by a set of
 * `const` references to quantities represented by `argument_tags`.
 * \note The `InitializeJ::operator()` should be const, and therefore not alter
 * the internal state of the generator. This is compatible with all known
 * use-cases and permits the `InitializeJ` generator to be placed in the
 * `ConstGlobalCache`.
 */
struct InitializeJ : public PUP::able {
  using boundary_tags = tmpl::list<Tags::BoundaryValue<Tags::BondiJ>,
                                   Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
                                   Tags::BoundaryValue<Tags::BondiR>>;

  using mutate_tags = tmpl::list<Tags::BondiJ, Tags::CauchyCartesianCoords,
                                 Tags::CauchyAngularCoords>;
  using argument_tags =
      tmpl::push_back<boundary_tags, Tags::LMax, Tags::NumberOfRadialPoints>;

  using creatable_classes =
      tmpl::list<InitializeJZeroNonSmooth, InitializeJInverseCubic>;

  WRAPPED_PUPable_abstract(InitializeJ);  // NOLINT

  virtual std::unique_ptr<InitializeJ> get_clone() const noexcept = 0;

  virtual void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max,
      size_t number_of_radial_points) const noexcept = 0;
};

/*!
 * \brief Initialize \f$J\f$ on the first hypersurface to be vanishing, finding
 * the appropriate angular coordinates to be continuous with the provided
 * worldtube boundary data.
 *
 * \details Internally, this performs an iterative solve for the angular
 * coordinates necessary to give rise to a vanishing gauge-transformed J on the
 * worldtube boundary. The parameters for the iterative procedure are determined
 * by options `InitializeJZeroNonSmooth::AngularCoordinateTolerance` and
 * `InitializeJZeroNonSmooth::MaxIterations`. The resulting `J` will necessarily
 * have vanishing first radial derivative, and so will typically not be smooth
 * (only continuous) with the provided Cauchy data at the worldtube boundary.
 */
struct InitializeJZeroNonSmooth : InitializeJ {
  struct AngularCoordinateTolerance {
    using type = double;
    static std::string name() noexcept { return "AngularCoordTolerance"; }
    static constexpr OptionString help = {
      "Tolerance of initial angular coordinates for CCE"};
    static type lower_bound() noexcept { return 1.0e-14; }
    static type upper_bound() noexcept { return 1.0e-3; }
    static type default_value() noexcept { return 1.0e-10; }
  };

  struct MaxIterations {
    using type = size_t;
    static constexpr OptionString help = {
      "Number of linearized inversion iterations."};
    static type lower_bound() noexcept { return 10; }
    static type upper_bound() noexcept { return 1000; }
    static type default_value() noexcept { return 300; }
  };
  using options =
      tmpl::list<AngularCoordinateTolerance, MaxIterations>;

  static constexpr OptionString help = {
      "Initialization process where J is set so Psi0 is vanishing\n"
      "vanishing (roughly a no incoming radiation condition)"};

  static std::string name() noexcept { return "ZeroNonsmooth"; }

  WRAPPED_PUPable_decl_template(InitializeJZeroNonSmooth);  // NOLINT
  explicit InitializeJZeroNonSmooth(
      CkMigrateMessage* /*unused*/) noexcept {}

  InitializeJZeroNonSmooth(double angular_coordinate_tolerance,
                                 size_t max_iterations) noexcept;

  InitializeJZeroNonSmooth() = default;

  std::unique_ptr<InitializeJ> get_clone() const noexcept override;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max,
      size_t number_of_radial_points) const noexcept override;

  void pup(PUP::er& p) noexcept override;

 private:
  double angular_coordinate_tolerance_ = 1.0e-10;
  size_t max_iterations_ = 300;
};

/*!
 * \brief Initialize \f$J\f$ on the first hypersurface from provided boundary
 * values of \f$J\f$, \f$R\f$, and \f$\partial_r J\f$.
 *
 * \details This initial data is chosen to take the function:
 *
 * \f[ J = \frac{A}{r} + \frac{B}{r^3},\f]
 *
 * where
 *
 * \f{align*}{
 * A &= R \left( \frac{3}{2} J|_{r = R} + \frac{1}{2} R \partial_r J|_{r =
 * R}\right) \notag\\
 * B &= - \frac{1}{2} R^3 (J|_{r = R} + R \partial_r J|_{r = R})
 * \f}
 */
struct InitializeJInverseCubic : InitializeJ {
  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "Initialization process where J is set to a simple Ansatz with a\n"
      " A/r + B/r^3 piece such that it is smooth with the Cauchy data at the \n"
      "worldtube"};

  static std::string name() noexcept { return "InverseCubic"; }

  WRAPPED_PUPable_decl_template(InitializeJInverseCubic);  // NOLINT
  explicit InitializeJInverseCubic(CkMigrateMessage* /*unused*/) noexcept {}

  InitializeJInverseCubic() = default;

  std::unique_ptr<InitializeJ> get_clone() const noexcept override;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max,
      size_t number_of_radial_points) const noexcept override;

  void pup(PUP::er& /*p*/) noexcept override;
};
}  // namespace Cce
