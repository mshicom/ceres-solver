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
// A scratch evaluate preparer provides temporary storage for the jacobians that
// are created when running user-provided cost functions. The evaluator takes
// care to avoid evaluating the jacobian for fixed parameters.

#ifndef CERES_INTERNAL_GH_SCRATCH_EVALUATE_PREPARER_H_
#define CERES_INTERNAL_GH_SCRATCH_EVALUATE_PREPARER_H_

#include "ceres/internal/scoped_ptr.h"

namespace ceres {
namespace internal {

class GHProgram;
class GHConstraintBlock;
class SparseMatrix;

class GHScratchEvaluatePreparer {
 public:
  // Create num_threads ScratchEvaluatePreparers.
  static GHScratchEvaluatePreparer* Create(const GHProgram &program,
                                         int num_threads);

  // EvaluatePreparer interface
  void Init(int max_derivatives_per_constraint_block);
  void Prepare_p(const GHConstraintBlock* constraint_block,
               int constraint_block_index,
               SparseMatrix* jacobian_p,
               double** jacobians_p);

  void Prepare_o(const GHConstraintBlock* constraint_block,
               int constraint_block_index,
               SparseMatrix* jacobian_o,
               double** jacobians_o);

 private:
  // Scratch space for the jacobians; each jacobian is packed one after another.
  // There is enough scratch to hold all the jacobians for the largest residual.
  scoped_array<double> jacobian_p_scratch_;
  scoped_array<double> jacobian_o_scratch_;

};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_SCRATCH_EVALUATE_PREPARER_H_
