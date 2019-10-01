// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"



namespace Cce{
namespace Tags{

struct NpAlpha : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

struct NpBeta : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

struct NpGamma : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

struct NpEpsilon : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

struct NpKappa : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

struct NpTau : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

struct NpSigma : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

struct NpRho : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

struct NpPi : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

struct NpNu : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

struct NpMu : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

struct NpLambda : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};

}  // namespace Tags
}  // namespace Cce
