// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: kaihong.huang11@google.com (Kaihong Huang)

#include "ceres/constraint_block.h"

#include <algorithm>
#include <cstddef>
#include <vector>
#include "ceres/corrector.h"
#include "ceres/gauss_helmert_constraint_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/fixed_array.h"
#include "ceres/local_parameterization.h"
#include "ceres/loss_function.h"
#include "ceres/parameter_block.h"
#include "ceres/residual_block_utils.h"
#include "ceres/small_blas.h"

using Eigen::Dynamic;

namespace ceres {
namespace internal {

using std::string;

void InvalidateEvaluation(const ConstraintBlock& block, double* cost, double* residuals, double** jacobians) {
  const int num_parameter_blocks = block.NumParameterBlocks();
  const int num_residuals = block.NumResiduals();

  InvalidateArray(1, cost);
  InvalidateArray(num_residuals, residuals);
  if (jacobians != NULL) {
    for (int i = 0; i < num_parameter_blocks; ++i) {
      const int parameter_block_size = block.parameter_blocks()[i]->Size();
      InvalidateArray(num_residuals * parameter_block_size, jacobians[i]);
    }
  }
}

string EvaluationToString(const ConstraintBlock& block, double const* const* parameters, double* cost,
                          double* residuals, double** jacobians) {
  CHECK_NOTNULL(cost);
  CHECK_NOTNULL(residuals);

  const int num_parameter_blocks = block.NumParameterBlocks();
  const int num_residuals = block.NumResiduals();
  string result = "";

  StringAppendF(&result, "Residual Block size: %d parameter blocks x %d residuals\n\n", num_parameter_blocks,
                num_residuals);
  result +=
      "For each parameter block, the value of the parameters are printed in the first column   \n"  // NOLINT
      "and the value of the jacobian under the corresponding residual. If a ParameterBlock was \n"  // NOLINT
      "held constant then the corresponding jacobian is printed as 'Not Computed'. If an entry \n"  // NOLINT
      "of the Jacobian/residual array was requested but was not written to by user code, it is \n"  // NOLINT
      "indicated by 'Uninitialized'. This is an error. Residuals or Jacobian values evaluating \n"  // NOLINT
      "to Inf or NaN is also an error.  \n\n";                                                      // NOLINT

  string space = "Residuals:     ";
  result += space;
  AppendArrayToString(num_residuals, residuals, &result);
  StringAppendF(&result, "\n\n");

  for (int i = 0; i < num_parameter_blocks; ++i) {
    const int parameter_block_size = block.parameter_blocks()[i]->Size();
    StringAppendF(&result, "Parameter Block %d, size: %d\n", i, parameter_block_size);
    StringAppendF(&result, "\n");
    for (int j = 0; j < parameter_block_size; ++j) {
      AppendArrayToString(1, parameters[i] + j, &result);
      StringAppendF(&result, "| ");
      for (int k = 0; k < num_residuals; ++k) {
        AppendArrayToString(
            1, (jacobians != NULL && jacobians[i] != NULL) ? jacobians[i] + k * parameter_block_size + j : NULL,
            &result);
      }
      StringAppendF(&result, "\n");
    }
    StringAppendF(&result, "\n");
  }
  StringAppendF(&result, "\n");
  return result;
}

bool IsEvaluationValid(const ConstraintBlock& block, double const* const* parameters, double* cost, double* residuals,
                       double** jacobians) {
  const int num_parameter_blocks = block.NumParameterBlocks();
  const int num_residuals = block.NumResiduals();

  if (!IsArrayValid(num_residuals, residuals)) {
    return false;
  }

  if (jacobians != NULL) {
    for (int i = 0; i < num_parameter_blocks; ++i) {
      const int parameter_block_size = block.parameter_blocks()[i]->Size();
      if (!IsArrayValid(num_residuals * parameter_block_size, jacobians[i])) {
        return false;
      }
    }
  }

  return true;
}

ConstraintBlock::ConstraintBlock(const GaussHelmertConstraintFunction* constraint_function,
                                 const LossFunction* loss_function,
                                 const std::vector<ParameterBlock*>& parameter_blocks,
                                 const std::vector<ObservationBlock*>& observation_blocks, int index)
    : constraint_function_(constraint_function),
      loss_function_(loss_function),
      parameter_blocks_(new ParameterBlock*[constraint_function->parameter_block_sizes().size()]),
      observation_blocks_(new ObservationBlock*[constraint_function->observation_block_sizes().size()]),
      index_(index) {
  std::copy(parameter_blocks.begin(), parameter_blocks.end(), parameter_blocks_.get());
  std::copy(observation_blocks.begin(), observation_blocks.end(), observation_blocks_.get());
}

bool ConstraintBlock::Evaluate(bool apply_loss_function, double* cost, double* residuals, double** jacobians_x,
                               double** jacobians_l, double* scratch) const {
  const int num_parameter_blocks = NumParameterBlocks();
  const int num_residuals = constraint_function_->num_residuals();
#if 0
  // Collect the parameters from their blocks. This will rarely allocate, since
  // residuals taking more than 8 parameter block arguments are rare.
  FixedArray<const double*, 8> parameters(num_parameter_blocks);
  for (int i = 0; i < num_parameter_blocks; ++i) {
    parameters[i] = parameter_blocks_[i]->state();
  }

  // Put pointers into the scratch space into global_jacobians as appropriate.
  FixedArray<double*, 8> global_jacobians(num_parameter_blocks);
  if (jacobians_x != NULL) {
    for (int i = 0; i < num_parameter_blocks; ++i) {
      const ParameterBlock* parameter_block = parameter_blocks_[i];
      if (jacobians_x[i] != NULL && parameter_block->LocalParameterizationJacobian() != NULL) {
        global_jacobians[i] = scratch;
        scratch += num_residuals * parameter_block->Size();
      } else {
        global_jacobians[i] = jacobians_x[i];
      }
    }
  }

  // If the caller didn't request residuals, use the scratch space for them.
  bool outputting_residuals = (residuals != NULL);
  if (!outputting_residuals) {
    residuals = scratch;
  }

  // Invalidate the evaluation buffers so that we can check them after
  // the CostFunction::Evaluate call, to see if all the return values
  // that were required were written to and that they are finite.
  double** eval_jacobians = (jacobians_x != NULL) ? global_jacobians.get() : NULL;

  InvalidateEvaluation(*this, cost, residuals, eval_jacobians);

  if (!constraint_function_->Evaluate(parameters.get(), residuals, eval_jacobians)) {
    return false;
  }

  if (!IsEvaluationValid(*this, parameters.get(), cost, residuals, eval_jacobians)) {
    std::string message =
        "\n\n"
        "Error in evaluating the ResidualBlock.\n\n"
        "There are two possible reasons. Either the CostFunction did not evaluate and fill all    \n"  // NOLINT
        "residual and jacobians that were requested or there was a non-finite value (nan/infinite)\n"  // NOLINT
        "generated during the or jacobian computation. \n\n" +
        EvaluationToString(*this, parameters.get(), cost, residuals, eval_jacobians);
    LOG(WARNING) << message;
    return false;
  }

  double squared_norm = VectorRef(residuals, num_residuals).squaredNorm();

  // Update the jacobians with the local parameterizations.
  if (jacobians != NULL) {
    for (int i = 0; i < num_parameter_blocks; ++i) {
      if (jacobians[i] != NULL) {
        const ParameterBlock* parameter_block = parameter_blocks_[i];

        // Apply local reparameterization to the jacobians.
        if (parameter_block->LocalParameterizationJacobian() != NULL) {
          // jacobians[i] = global_jacobians[i] * global_to_local_jacobian.
          MatrixMatrixMultiply<Dynamic, Dynamic, Dynamic, Dynamic, 0>(
              global_jacobians[i], num_residuals, parameter_block->Size(),
              parameter_block->LocalParameterizationJacobian(), parameter_block->Size(), parameter_block->LocalSize(),
              jacobians[i], 0, 0, num_residuals, parameter_block->LocalSize());
        }
      }
    }
  }

  if (loss_function_ == NULL || !apply_loss_function) {
    *cost = 0.5 * squared_norm;
    return true;
  }

  double rho[3];
  loss_function_->Evaluate(squared_norm, rho);
  *cost = 0.5 * rho[0];

  // No jacobians and not outputting residuals? All done. Doing an early exit
  // here avoids constructing the "Corrector" object below in a common case.
  if (jacobians == NULL && !outputting_residuals) {
    return true;
  }

  // Correct for the effects of the loss function. The jacobians need to be
  // corrected before the residuals, since they use the uncorrected residuals.
  Corrector correct(squared_norm, rho);
  if (jacobians != NULL) {
    for (int i = 0; i < num_parameter_blocks; ++i) {
      if (jacobians[i] != NULL) {
        const ParameterBlock* parameter_block = parameter_blocks_[i];

        // Correct the jacobians for the loss function.
        correct.CorrectJacobian(num_residuals, parameter_block->LocalSize(), residuals, jacobians[i]);
      }
    }
  }

  // Correct the residuals with the loss function.
  if (outputting_residuals) {
    correct.CorrectResiduals(num_residuals, residuals);
  }
#endif
  return true;
}

int ConstraintBlock::NumScratchDoublesForEvaluate() const {
  // Compute the amount of scratch space needed to store the full-sized
  // jacobians. For parameters that have no local parameterization  no storage
  // is needed and the passed-in jacobian array is used directly. Also include
  // space to store the residuals, which is needed for cost-only evaluations.
  // This is slightly pessimistic, since both won't be needed all the time, but
  // the amount of excess should not cause problems for the caller.
  int num_parameters = NumParameterBlocks();
  int scratch_doubles = 1;
  for (int i = 0; i < num_parameters; ++i) {
    const ParameterBlock* parameter_block = parameter_blocks_[i];
    if (!parameter_block->IsConstant() && parameter_block->LocalParameterizationJacobian() != NULL) {
      scratch_doubles += parameter_block->Size();
    }
  }
  scratch_doubles *= NumResiduals();
  return scratch_doubles;
}

}  // namespace internal
}  // namespace ceres
