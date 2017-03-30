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
// Author: richie.stebbing@gmail.com (Richard Stebbing)

#include "ceres/GH_compressed_row_jacobian_writer.h"
#include "ceres/GH_dynamic_compressed_row_jacobian_writer.h"
#include "ceres/casts.h"
#include "ceres/dynamic_compressed_row_sparse_matrix.h"
#include "ceres/GH_parameter_block.h"
#include "ceres/GH_program.h"
#include "ceres/constraint_block.h"

namespace ceres {
namespace internal {

using std::pair;
using std::vector;

GHScratchEvaluatePreparer*
GHDynamicCompressedRowJacobianWriter::CreateEvaluatePreparers(int num_threads) {
  return GHScratchEvaluatePreparer::Create(*program_, num_threads);
}

SparseMatrix* GHDynamicCompressedRowJacobianWriter::CreateJacobian_p() const {
  // Initialize `jacobian` with zero number of `max_num_nonzeros`.
  const int num_residuals = program_->NumResiduals();
  const int num_effective_parameters = program_->NumEffectiveParameters();

  DynamicCompressedRowSparseMatrix* jacobian =
      new DynamicCompressedRowSparseMatrix(num_residuals,
                                           num_effective_parameters,
                                           0);

  vector<int>* row_blocks = jacobian->mutable_row_blocks();
  for (int i = 0; i < jacobian->num_rows(); ++i) {
    row_blocks->push_back(1);
  }

  vector<int>* col_blocks = jacobian->mutable_col_blocks();
  for (int i = 0; i < jacobian->num_cols(); ++i) {
    col_blocks->push_back(1);
  }

  return jacobian;
}

SparseMatrix* GHDynamicCompressedRowJacobianWriter::CreateJacobian_o() const {
  // Initialize `jacobian` with zero number of `max_num_nonzeros`.
  const int num_residuals = program_->NumResiduals();
  const int num_effective_observations = program_->NumEffectiveObservations();

  DynamicCompressedRowSparseMatrix* jacobian =
      new DynamicCompressedRowSparseMatrix(num_residuals,
                                           num_effective_observations,
                                           0);

  vector<int>* row_blocks = jacobian->mutable_row_blocks();
  for (int i = 0; i < jacobian->num_rows(); ++i) {
    row_blocks->push_back(1);
  }

  vector<int>* col_blocks = jacobian->mutable_col_blocks();
  for (int i = 0; i < jacobian->num_cols(); ++i) {
    col_blocks->push_back(1);
  }

  return jacobian;
}

void GHDynamicCompressedRowJacobianWriter::Write_p(int constraint_id,
                                               int constraint_offset,
                                               double **jacobians,
                                               SparseMatrix* base_jacobian) {
  DynamicCompressedRowSparseMatrix* jacobian =
    down_cast<DynamicCompressedRowSparseMatrix*>(base_jacobian);

  // Get the `constraint_block` of interest.
  const GHConstraintBlock* constraint_block =
      program_->constraint_blocks()[constraint_id];
  const int num_residuals = constraint_block->NumResiduals();

  vector<pair<int, int> > evaluated_jacobian_blocks;
  GHCompressedRowJacobianWriter::GetOrderedParameterBlocks(
    program_, constraint_id, &evaluated_jacobian_blocks);

  // `constraint_offset` is the constraint row in the global jacobian.
  // Empty the jacobian rows.
  jacobian->ClearRows(constraint_offset, num_residuals);

  // Iterate over each parameter block.
  for (int i = 0; i < evaluated_jacobian_blocks.size(); ++i) {
    const GHParameterBlock* parameter_block =
        program_->parameter_blocks()[evaluated_jacobian_blocks[i].first];
    const int parameter_block_jacobian_index =
        evaluated_jacobian_blocks[i].second;
    const int parameter_block_size = parameter_block->LocalSize();

    // For each parameter block only insert its non-zero entries.
    for (int r = 0; r < num_residuals; ++r) {
      for (int c = 0; c < parameter_block_size; ++c) {
        const double& v = jacobians[parameter_block_jacobian_index][
            r * parameter_block_size + c];
        // Only insert non-zero entries.
        if (v != 0.0) {
          jacobian->InsertEntry(
            constraint_offset + r, parameter_block->delta_offset() + c, v);
        }
      }
    }
  }
}

void GHDynamicCompressedRowJacobianWriter::Write_o(int constraint_id,
                                               int constraint_offset,
                                               double **jacobians,
                                               SparseMatrix* base_jacobian) {
  DynamicCompressedRowSparseMatrix* jacobian =
    down_cast<DynamicCompressedRowSparseMatrix*>(base_jacobian);

  // Get the `constraint_block` of interest.
  const GHConstraintBlock* constraint_block =
      program_->constraint_blocks()[constraint_id];
  const int num_residuals = constraint_block->NumResiduals();

  vector<pair<int, int> > evaluated_jacobian_blocks;
  GHCompressedRowJacobianWriter::GetOrderedObservationBlocks(
    program_, constraint_id, &evaluated_jacobian_blocks);

  // `constraint_offset` is the constraint row in the global jacobian.
  // Empty the jacobian rows.
  jacobian->ClearRows(constraint_offset, num_residuals);

  // Iterate over each observation block.
  for (int i = 0; i < evaluated_jacobian_blocks.size(); ++i) {
    const GHObservationBlock* observation_block =
        program_->observation_blocks()[evaluated_jacobian_blocks[i].first];
    const int observation_block_jacobian_index =
        evaluated_jacobian_blocks[i].second;
    const int observation_block_size = observation_block->LocalSize();

    // For each observation block only insert its non-zero entries.
    for (int r = 0; r < num_residuals; ++r) {
      for (int c = 0; c < observation_block_size; ++c) {
        const double& v = jacobians[observation_block_jacobian_index][
            r * observation_block_size + c];
        // Only insert non-zero entries.
        if (v != 0.0) {
          jacobian->InsertEntry(
            constraint_offset + r, observation_block->delta_offset() + c, v);
        }
      }
    }
  }
}
}  // namespace internal
}  // namespace ceres
