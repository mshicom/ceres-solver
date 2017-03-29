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
// Author: keir@google.com (Keir Mierle)

#include "ceres/GH_scratch_evaluate_preparer.h"

#include "ceres/GH_parameter_block.h"
#include "ceres/GH_program.h"
#include "ceres/constraint_block.h"

namespace ceres {
namespace internal {

GHScratchEvaluatePreparer* GHScratchEvaluatePreparer::Create(
    const GHProgram &program,
    int num_threads) {
  GHScratchEvaluatePreparer* preparers = new GHScratchEvaluatePreparer[num_threads];
  int max_derivatives_per_constraint_block =
      program.MaxDerivativesPerConstraintBlock();

  for (int i = 0; i < num_threads; i++) {
    preparers[i].Init(max_derivatives_per_constraint_block);
  }
  return preparers;
}

void GHScratchEvaluatePreparer::Init(int max_derivatives_per_constraint_block) {
  jacobian_p_scratch_.reset(
      new double[max_derivatives_per_constraint_block]);
  jacobian_o_scratch_.reset(
      new double[max_derivatives_per_constraint_block]);
}

// Point the jacobian blocks into the scratch area of this evaluate preparer.
void GHScratchEvaluatePreparer::Prepare_p(const GHConstraintBlock* constraint_block,
                                      int /* residual_block_index */,
                                        SparseMatrix* /*jacobian_p*/,
                                        double** jacobians_p) {
  double* jacobian_block_cursor = jacobian_p_scratch_.get();
  int num_residuals = constraint_block->NumResiduals();
  int num_parameter_blocks = constraint_block->NumParameterBlocks();
  for (int j = 0; j < num_parameter_blocks; ++j) {
    const GHParameterBlock* parameter_block =
        constraint_block->parameter_blocks()[j];
    if (parameter_block->IsConstant()) {
      jacobians_p[j] = NULL;
    } else {
      jacobians_p[j] = jacobian_block_cursor;
      jacobian_block_cursor += num_residuals * parameter_block->LocalSize();
    }
  }
}

void GHScratchEvaluatePreparer::Prepare_o(const GHConstraintBlock* constraint_block,
                                      int /* residual_block_index */,
                                        SparseMatrix* /* jacobian_o */,
                                        double** jacobians_o) {
  double* jacobian_block_cursor = jacobian_o_scratch_.get();
  int num_residuals = constraint_block->NumResiduals();
  int num_observation_blocks = constraint_block->NumObservationBlocks();
  for (int j = 0; j < num_observation_blocks; ++j) {
    const GHObservationBlock* observation_block =
        constraint_block->observation_blocks()[j];
    if (observation_block->IsConstant()) {
      jacobians_o[j] = NULL;
    } else {
      jacobians_o[j] = jacobian_block_cursor;
      jacobian_block_cursor += num_residuals * observation_block->LocalSize();
    }
  }
}

}  // namespace internal
}  // namespace ceres
