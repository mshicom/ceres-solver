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

#include "ceres/block_evaluate_preparer.h"

#include <vector>
#include "ceres/block_sparse_matrix.h"
#include "ceres/casts.h"
#include "ceres/parameter_block.h"
#include "ceres/residual_block.h"
#include "ceres/sparse_matrix.h"

namespace ceres {
namespace internal {

void BlockEvaluatePreparer::Init(int const* const* jacobian_layout_p,
                                 const int * const *jacobian_layout_o,
                                 int max_derivatives_per_residual_block) {
  jacobian_layout_p_ = jacobian_layout_p;
  jacobian_layout_o_ = jacobian_layout_o;
  scratch_evaluate_preparer_.Init(max_derivatives_per_residual_block);
}

// Point the jacobian blocks directly into the block sparse matrix.
void BlockEvaluatePreparer::Prepare_p(const ResidualBlock* residual_block,
                                    int residual_block_index,
                                    SparseMatrix* jacobian_p,
                                    double** jacobians_p) {
  // If the overall jacobian is not available, use the scratch space.
  if (jacobian_p == NULL) {
    scratch_evaluate_preparer_.Prepare_p(residual_block,
                                       residual_block_index,
                                       jacobian_p,
                                       jacobians_p);
    return;
  }

  double* jacobian_values =
      down_cast<BlockSparseMatrix*>(jacobian_p)->mutable_values();

  const int* jacobian_block_offset = jacobian_layout_p_[residual_block_index];
  const int num_parameter_blocks = residual_block->NumParameterBlocks();
  for (int j = 0; j < num_parameter_blocks; ++j) {
    if (!residual_block->parameter_blocks()[j]->IsConstant()) {
      jacobians_p[j] = jacobian_values + *jacobian_block_offset;

      // The jacobian_block_offset can't be indexed with 'j' since the code
      // that creates the layout strips out any blocks for inactive
      // parameters. Instead, bump the pointer for active parameters only.
      jacobian_block_offset++;
    } else {
      jacobians_p[j] = NULL;
    }
  }
}

void BlockEvaluatePreparer::Prepare_o(const ResidualBlock* residual_block,
                                          int constraint_block_index,
                                          SparseMatrix* jacobian_o,
                                          double** jacobians_o) {
    // If the overall jacobian is not available, use the scratch space.
    if (jacobian_o == NULL) {
      scratch_evaluate_preparer_.Prepare_o(residual_block,
                                         constraint_block_index,
                                         jacobian_o,
                                         jacobians_o);
      return;
    }

    double* jacobian_values =
        down_cast<BlockSparseMatrix*>(jacobian_o)->mutable_values();

    const int* jacobian_block_offset = jacobian_layout_o_[constraint_block_index];
    const int num_observation_blocks = residual_block->NumObservationBlocks();
    for (int j = 0; j < num_observation_blocks; ++j) {
      if (!residual_block->observation_blocks()[j]->IsConstant()) {
        jacobians_o[j] = jacobian_values + *jacobian_block_offset;

        // The jacobian_block_offset can't be indexed with 'j' since the code
        // that creates the layout strips out any blocks for inactive
        // observations. Instead, bump the pointer for active observations only.
        jacobian_block_offset++;
      } else {
        jacobians_o[j] = NULL;
      }
    }
}
} // namespace internal
}  // namespace ceres
