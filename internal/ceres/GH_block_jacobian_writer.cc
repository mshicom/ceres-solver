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

#include "ceres/GH_block_jacobian_writer.h"

#include "ceres/GH_block_evaluate_preparer.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/GH_parameter_block.h"
#include "ceres/GH_program.h"
#include "ceres/constraint_block.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/port.h"
#include "ceres/internal/scoped_ptr.h"

namespace ceres {
namespace internal {

using std::vector;

namespace {

// Given the residual block ordering, build a lookup table to determine which
// per-parameter jacobian goes where in the overall program jacobian.
//
// Since we expect to use a Schur type linear solver to solve the LM step, take
// extra care to place the E blocks and the F blocks contiguously. E blocks are
// the first num_eliminate_blocks parameter blocks as indicated by the parameter
// block ordering. The remaining parameter blocks are the F blocks.
//
// TODO(keir): Consider if we should use a boolean for each parameter block
// instead of num_eliminate_blocks.

// TODO(hkh): what about observations?
void BuildJacobianLayout(const GHProgram& program,
                         int num_eliminate_blocks,
                         vector<int*>* jacobian_layout,
                         vector<int>* jacobian_layout_storage) {
  const vector<GHConstraintBlock*>& constraint_blocks = program.constraint_blocks();

  // Iterate over all the active residual blocks and determine how many E blocks
  // are there. This will determine where the F blocks start in the jacobian
  // matrix. Also compute the number of jacobian blocks.
  int f_block_pos = 0;
  int num_jacobian_blocks = 0;
  for (int i = 0; i < constraint_blocks.size(); ++i) {
    GHConstraintBlock* constraint_block = constraint_blocks[i];
    const int num_residuals = constraint_block->NumResiduals();
    const int num_parameter_blocks = constraint_block->NumParameterBlocks();

    // Advance f_block_pos over each E block for this residual.
    for (int j = 0; j < num_parameter_blocks; ++j) {
      GHParameterBlock* parameter_block = constraint_block->parameter_blocks()[j];
      if (!parameter_block->IsConstant()) {
        // Only count blocks for active parameters.
        num_jacobian_blocks++;
        if (parameter_block->index() < num_eliminate_blocks) {
          f_block_pos += num_residuals * parameter_block->LocalSize();
        }
      }
    }
  }

  // We now know that the E blocks are laid out starting at zero, and the F
  // blocks are laid out starting at f_block_pos. Iterate over the residual
  // blocks again, and this time fill the jacobian_layout array with the
  // position information.

  jacobian_layout->resize(program.NumConstraintBlocks());
  jacobian_layout_storage->resize(num_jacobian_blocks);

  int e_block_pos = 0;
  int* jacobian_pos = &(*jacobian_layout_storage)[0];
  for (int i = 0; i < constraint_blocks.size(); ++i) {
    const GHConstraintBlock* constraint_block = constraint_blocks[i];
    const int num_residuals = constraint_block->NumResiduals();
    const int num_parameter_blocks = constraint_block->NumParameterBlocks();

    (*jacobian_layout)[i] = jacobian_pos;
    for (int j = 0; j < num_parameter_blocks; ++j) {
      GHParameterBlock* parameter_block = constraint_block->parameter_blocks()[j];
      const int parameter_block_index = parameter_block->index();
      if (parameter_block->IsConstant()) {
        continue;
      }
      const int jacobian_block_size =
          num_residuals * parameter_block->LocalSize();
      if (parameter_block_index < num_eliminate_blocks) {
        *jacobian_pos = e_block_pos;
        e_block_pos += jacobian_block_size;
      } else {
        *jacobian_pos = f_block_pos;
        f_block_pos += jacobian_block_size;
      }
      jacobian_pos++;
    }
  }
}

void BuildJacobianLayoutNoEliminate(const GHProgram& program,
                         vector<int*>* jacobian_layout,
                         vector<int>* jacobian_layout_storage) {
  const vector<GHConstraintBlock*>& constraint_blocks = program.constraint_blocks();

  int num_jacobian_blocks = 0;
  for (int i = 0; i < constraint_blocks.size(); ++i) {
    GHConstraintBlock* constraint_block = constraint_blocks[i];
    const int num_residuals = constraint_block->NumResiduals();
    const int num_observation_blocks = constraint_block->NumObservationBlocks();

    // Advance f_block_pos over each E block for this residual.
    for (int j = 0; j < num_observation_blocks; ++j) {
      GHObservationBlock* observation_block = constraint_block->observation_blocks()[j];
      if (!observation_block->IsConstant()) {
        // Only count blocks for active observations.
        num_jacobian_blocks++;
      }
    }
  }

  jacobian_layout->resize(program.NumConstraintBlocks());
  jacobian_layout_storage->resize(num_jacobian_blocks);

  int block_pos = 0;
  int* jacobian_pos = &(*jacobian_layout_storage)[0];
  for (int i = 0; i < constraint_blocks.size(); ++i) {
    const GHConstraintBlock* constraint_block = constraint_blocks[i];
    const int num_residuals = constraint_block->NumResiduals();
    const int num_observation_blocks = constraint_block->NumObservationBlocks();

    (*jacobian_layout)[i] = jacobian_pos;
    for (int j = 0; j < num_observation_blocks; ++j) {
      GHObservationBlock* observation_block = constraint_block->observation_blocks()[j];
      const int observation_block_index = observation_block->index();
      if (observation_block->IsConstant()) {
        continue;
      }
      const int jacobian_block_size =
          num_residuals * observation_block->LocalSize();

      *jacobian_pos++ = block_pos;
      block_pos += jacobian_block_size;
    }
  }
}

}  // namespace

GHBlockJacobianWriter::GHBlockJacobianWriter(const GHEvaluator::Options& options,
                                         GHProgram* program)
    : program_(program) {
  CHECK_GE(options.num_eliminate_blocks, 0)
      << "num_eliminate_blocks must be greater than 0.";

  BuildJacobianLayout(*program,
                      options.num_eliminate_blocks,
                      &jacobian_layout_p_,
                      &jacobian_layout_storage_p_);

  BuildJacobianLayoutNoEliminate(*program,
                      &jacobian_layout_o_,
                      &jacobian_layout_storage_o_);
}

// Create evaluate prepareres that point directly into the final jacobian. This
// makes the final Write() a nop.
GHBlockEvaluatePreparer* GHBlockJacobianWriter::CreateEvaluatePreparers(
    int num_threads) {
  int max_derivatives_per_residual_block =
      program_->MaxDerivativesPerConstraintBlock();

  GHBlockEvaluatePreparer* preparers = new GHBlockEvaluatePreparer[num_threads];
  for (int i = 0; i < num_threads; i++) {
    preparers[i].Init(&jacobian_layout_p_[0], &jacobian_layout_o_[0], max_derivatives_per_residual_block);
  }
  return preparers;
}

SparseMatrix* GHBlockJacobianWriter::CreateJacobian_p() const {
  CompressedRowBlockStructure* bs = new CompressedRowBlockStructure;

  const vector<GHParameterBlock*>& parameter_blocks =
      program_->parameter_blocks();

  // Construct the column blocks.
  bs->cols.resize(parameter_blocks.size());
  for (int i = 0, cursor = 0; i < parameter_blocks.size(); ++i) {
    CHECK_NE(parameter_blocks[i]->index(), -1);
    CHECK(!parameter_blocks[i]->IsConstant());
    bs->cols[i].size = parameter_blocks[i]->LocalSize();
    bs->cols[i].position = cursor;
    cursor += bs->cols[i].size;
  }

  // Construct the cells in each row.
  const vector<GHConstraintBlock*>& constraint_blocks = program_->constraint_blocks();
  int row_block_position = 0;
  bs->rows.resize(constraint_blocks.size());
  for (int i = 0; i < constraint_blocks.size(); ++i) {
    const GHConstraintBlock* constraint_block = constraint_blocks[i];
    CompressedRow* row = &bs->rows[i];

    row->block.size = constraint_block->NumResiduals();
    row->block.position = row_block_position;
    row_block_position += row->block.size;

    // Size the row by the number of active parameters in this residual.
    const int num_parameter_blocks = constraint_block->NumParameterBlocks();
    int num_active_parameter_blocks = 0;
    for (int j = 0; j < num_parameter_blocks; ++j) {
      if (constraint_block->parameter_blocks()[j]->index() != -1) {
        num_active_parameter_blocks++;
      }
    }
    row->cells.resize(num_active_parameter_blocks);

    // Add layout information for the active parameters in this row.
    for (int j = 0, k = 0; j < num_parameter_blocks; ++j) {
      const GHParameterBlock* parameter_block =
          constraint_block->parameter_blocks()[j];
      if (!parameter_block->IsConstant()) {
        Cell& cell = row->cells[k];
        cell.block_id = parameter_block->index();
        cell.position = jacobian_layout_p_[i][k];

        // Only increment k for active parameters, since there is only layout
        // information for active parameters.
        k++;
      }
    }

    sort(row->cells.begin(), row->cells.end(), CellLessThan);
  }

  BlockSparseMatrix* jacobian = new BlockSparseMatrix(bs);
  CHECK_NOTNULL(jacobian);
  return jacobian;
}

SparseMatrix* GHBlockJacobianWriter::CreateJacobian_o() const {
  CompressedRowBlockStructure* bs = new CompressedRowBlockStructure;

  const vector<GHObservationBlock*>& observation_blocks =
      program_->observation_blocks();

  // Construct the column blocks.
  bs->cols.resize(observation_blocks.size());
  for (int i = 0, cursor = 0; i < observation_blocks.size(); ++i) {
    CHECK_NE(observation_blocks[i]->index(), -1);
    CHECK(!observation_blocks[i]->IsConstant());
    bs->cols[i].size = observation_blocks[i]->LocalSize();
    bs->cols[i].position = cursor;
    cursor += bs->cols[i].size;
  }

  // Construct the cells in each row.
  const vector<GHConstraintBlock*>& constraint_blocks = program_->constraint_blocks();
  int row_block_position = 0;
  bs->rows.resize(constraint_blocks.size());
  for (int i = 0; i < constraint_blocks.size(); ++i) {
    const GHConstraintBlock* constraint_block = constraint_blocks[i];
    CompressedRow* row = &bs->rows[i];

    row->block.size = constraint_block->NumResiduals();
    row->block.position = row_block_position;
    row_block_position += row->block.size;

    // Size the row by the number of active observations in this residual.
    const int num_observation_blocks = constraint_block->NumObservationBlocks();
    int num_active_observation_blocks = 0;
    for (int j = 0; j < num_observation_blocks; ++j) {
      if (constraint_block->observation_blocks()[j]->index() != -1) {
        num_active_observation_blocks++;
      }
    }
    row->cells.resize(num_active_observation_blocks);

    // Add layout information for the active observations in this row.
    for (int j = 0, k = 0; j < num_observation_blocks; ++j) {
      const GHObservationBlock* observation_block =
          constraint_block->observation_blocks()[j];
      if (!observation_block->IsConstant()) {
        Cell& cell = row->cells[k];
        cell.block_id = observation_block->index();
        cell.position = jacobian_layout_p_[i][k];

        // Only increment k for active observations, since there is only layout
        // information for active observations.
        k++;
      }
    }

    sort(row->cells.begin(), row->cells.end(), CellLessThan);
  }

  BlockSparseMatrix* jacobian = new BlockSparseMatrix(bs);
  CHECK_NOTNULL(jacobian);
  return jacobian;
}

}  // namespace internal
}  // namespace ceres
