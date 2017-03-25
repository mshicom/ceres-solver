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
//
// A jacobian writer that writes to dense Eigen matrices.

#ifndef CERES_INTERNAL_GH_DENSE_JACOBIAN_WRITER_H_
#define CERES_INTERNAL_GH_DENSE_JACOBIAN_WRITER_H_

#include "ceres/casts.h"
#include "ceres/dense_sparse_matrix.h"
#include "ceres/GH_parameter_block.h"
#include "ceres/GH_program.h"
#include "ceres/constraint_block.h"
#include "ceres/GH_scratch_evaluate_preparer.h"
#include "ceres/GH_evaluator.h"
#include "ceres/internal/eigen.h"

namespace ceres {
namespace internal {

class GHDenseJacobianWriter {
 public:
  GHDenseJacobianWriter(GHEvaluator::Options options/* ignored */,
                      GHProgram* program)
    : program_(program) {
  }

  // JacobianWriter interface.

  // Since the dense matrix has different layout than that assumed by the cost
  // functions, use scratch space to store the jacobians temporarily then copy
  // them over to the larger jacobian later.
  GHScratchEvaluatePreparer* CreateEvaluatePreparers(int num_threads) {
    return GHScratchEvaluatePreparer::Create(*program_, num_threads);
  }

  SparseMatrix* CreateJacobian_p() const {
    return new DenseSparseMatrix(program_->NumResiduals(),
                                 program_->NumEffectiveParameters(),
                                 true);
  }

  SparseMatrix* CreateJacobian_o() const {
    return new DenseSparseMatrix(program_->NumResiduals(),
                                 program_->NumEffectiveObservations(),
                                 true);
  }

  void Write_p(int constraint_id,
             int constraint_offset,
             double **jacobians,
             SparseMatrix* jacobian) {
    const GHConstraintBlock* constraint_block =
          program_->constraint_blocks()[constraint_id];
    int num_parameter_blocks = constraint_block->NumParameterBlocks();
    int num_residuals = constraint_block->NumResiduals();

    DenseSparseMatrix* dense_jacobian = down_cast<DenseSparseMatrix*>(jacobian);
    // Now copy the jacobians for each parameter into the dense jacobian matrix.
    for (int j = 0; j < num_parameter_blocks; ++j) {
      GHParameterBlock* parameter_block = constraint_block->parameter_blocks()[j];

      // If the parameter block is fixed, then there is nothing to do.
      if (parameter_block->IsConstant()) {
        continue;
      }

      const int parameter_block_size = parameter_block->LocalSize();
      ConstMatrixRef parameter_jacobian(jacobians[j],
                                        num_residuals,
                                        parameter_block_size);

      dense_jacobian->mutable_matrix().block(
          constraint_offset,
          parameter_block->delta_offset(),
          num_residuals,
          parameter_block_size) = parameter_jacobian;
    }
  }

  void Write_o(int constraint_id,
             int constraint_offset,
             double **jacobians,
             SparseMatrix* jacobian) {
    const GHConstraintBlock* constraint_block =
          program_->constraint_blocks()[constraint_id];
    int num_observation_blocks = constraint_block->NumObservationBlocks();
    int num_residuals = constraint_block->NumResiduals();

    DenseSparseMatrix* dense_jacobian = down_cast<DenseSparseMatrix*>(jacobian);
    for (int j = 0; j < num_observation_blocks; ++j) {
      GHObservationBlock* observation_block = constraint_block->observation_blocks()[j];

      // If the observation block is fixed, then there is nothing to do.
      if (observation_block->IsConstant()) {
        continue;
      }

      const int observation_block_size = observation_block->LocalSize();
      ConstMatrixRef observation_jacobian(jacobians[j],
                                        num_residuals,
                                        observation_block_size);

      dense_jacobian->mutable_matrix().block(
          constraint_offset,
          observation_block->delta_offset(),
          num_residuals,
          observation_block_size) = observation_jacobian;
    }
  }
 private:
  GHProgram* program_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_DENSE_JACOBIAN_WRITER_H_
