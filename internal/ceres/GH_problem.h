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
// This is the implementation of the public Problem API. The pointer to
// implementation (PIMPL) idiom makes it possible for Ceres internal code to
// refer to the private data members without needing to exposing it to the
// world. An alternative to PIMPL is to have a factory which returns instances
// of a virtual base class; while that approach would work, it requires clients
// to always put a Problem object into a scoped pointer; this needlessly muddies
// client code for little benefit. Therefore, the PIMPL comprise was chosen.

#ifndef CERES_PUBLIC_GH_PROBLEM_IMPL_H_
#define CERES_PUBLIC_GH_PROBLEM_IMPL_H_

#include <map>
#include <vector>

#include "ceres/internal/macros.h"
#include "ceres/internal/port.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/collections_port.h"
#include "ceres/types.h"
#include "ceres/problem.h"

namespace ceres {

class GaussHelmertConstraintFunction;
class LossFunction;
class LocalParameterization;
struct CRSMatrix;

namespace internal {

class GHProgram;
class GHConstraintBlock;
class GHParameterBlock;

class GHObservationBlock;
typedef GHConstraintBlock* ConstraintBlockId;

class GHProblem {
 public:
  typedef std::map<double*, GHParameterBlock*> GHParameterMap;
  typedef std::map<double*, GHObservationBlock*> GHObservationMap;

  typedef HashSet<GHConstraintBlock*> ConstraintBlockSet;

  GHProblem();
  explicit GHProblem(const Problem::Options& options);

  ~GHProblem();

  // See the public problem.h file for description of these methods.
  GHConstraintBlock* AddConstraintBlock(
      GaussHelmertConstraintFunction* constraint_function,
      LossFunction* loss_function,
      const std::vector<double*>& parameter_blocks,
      const std::vector<double*>& observation_blocks);
//  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
//                                   LossFunction* loss_function,
//                                   double* x0);
//  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
//                                   LossFunction* loss_function,
//                                   double* x0, double* x1);
//  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
//                                   LossFunction* loss_function,
//                                   double* x0, double* x1, double* x2);
//  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
//                                   LossFunction* loss_function,
//                                   double* x0, double* x1, double* x2,
//                                   double* x3);
//  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
//                                   LossFunction* loss_function,
//                                   double* x0, double* x1, double* x2,
//                                   double* x3, double* x4);
//  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
//                                   LossFunction* loss_function,
//                                   double* x0, double* x1, double* x2,
//                                   double* x3, double* x4, double* x5);
//  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
//                                   LossFunction* loss_function,
//                                   double* x0, double* x1, double* x2,
//                                   double* x3, double* x4, double* x5,
//                                   double* x6);
//  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
//                                   LossFunction* loss_function,
//                                   double* x0, double* x1, double* x2,
//                                   double* x3, double* x4, double* x5,
//                                   double* x6, double* x7);
//  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
//                                   LossFunction* loss_function,
//                                   double* x0, double* x1, double* x2,
//                                   double* x3, double* x4, double* x5,
//                                   double* x6, double* x7, double* x8);
//  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
//                                   LossFunction* loss_function,
//                                   double* x0, double* x1, double* x2,
//                                   double* x3, double* x4, double* x5,
//                                   double* x6, double* x7, double* x8,
//                                   double* x9);
  void AddParameterBlock(double* values, int size);
  void AddParameterBlock(double* values,
                         int size,
                         LocalParameterization* local_parameterization);
  void AddObservationBlock(double* values, int size);
  void AddObservationBlock(double* values,
                         int size,
                         LocalParameterization* local_parameterization);

  void RemoveConstraintBlock(GHConstraintBlock* constraint_block);
  void RemoveParameterBlock(double* values);
  void RemoveObservationBlock(double* values);

  void SetParameterBlockConstant(double* values);
  void SetParameterBlockVariable(double* values);
  bool IsParameterBlockConstant(double* values) const;

  void SetObservationBlockConstant(double* values);
  void SetObservationBlockVariable(double* values);
  bool IsObservationBlockConstant(double* values) const;

  void SetParameterization(double* values,
                           LocalParameterization* local_parameterization);
  const LocalParameterization* GetParameterization(double* values) const;

  void SetParameterLowerBound(double* values, int index, double lower_bound);
  void SetParameterUpperBound(double* values, int index, double upper_bound);
  void SetObservationLowerBound(double* values, int index, double lower_bound);
  void SetObservationUpperBound(double* values, int index, double upper_bound);

  // Options struct to control Problem::Evaluate.
  struct EvaluateOptions {
    EvaluateOptions()
        : apply_loss_function(true),
          num_threads(1) {
    }

    // The set of parameter blocks for which evaluation should be
    // performed. This vector determines the order that parameter
    // blocks occur in the gradient vector and in the columns of the
    // jacobian matrix. If parameter_blocks is empty, then it is
    // assumed to be equal to vector containing ALL the parameter
    // blocks.  Generally speaking the parameter blocks will occur in
    // the order in which they were added to the problem. But, this
    // may change if the user removes any parameter blocks from the
    // problem.
    //
    // NOTE: This vector should contain the same pointers as the ones
    // used to add parameter blocks to the Problem. These parameter
    // block should NOT point to new memory locations. Bad things will
    // happen otherwise.
    std::vector<double*> parameter_blocks;
    std::vector<double*> observation_blocks;

    // The set of residual blocks to evaluate. This vector determines
    // the order in which the residuals occur, and how the rows of the
    // jacobian are ordered. If residual_blocks is empty, then it is
    // assumed to be equal to the vector containing ALL the residual
    // blocks. Generally speaking the residual blocks will occur in
    // the order in which they were added to the problem. But, this
    // may change if the user removes any residual blocks from the
    // problem.
    std::vector<ConstraintBlockId> constraint_blocks;

    // Even though the residual blocks in the problem may contain loss
    // functions, setting apply_loss_function to false will turn off
    // the application of the loss function to the output of the cost
    // function. This is of use for example if the user wishes to
    // analyse the solution quality by studying the distribution of
    // residuals before and after the solve.
    bool apply_loss_function;

    int num_threads;
  };

  bool Evaluate(const GHProblem::EvaluateOptions& options,
                double* cost,
                std::vector<double>* residuals,
                std::vector<double>* gradient_p, std::vector<double>* gradient,
                CRSMatrix* jacobian_p, CRSMatrix* jacobian_o);

  int NumParameterBlocks() const;
  int NumParameters() const;
  int NumObservationBlocks() const;
  int NumObservations() const;
  int NumConstraintBlocks() const;
  int NumResiduals() const;

  int ParameterBlockSize(const double* parameter_block) const;
  int ParameterBlockLocalSize(const double* parameter_block) const;
  bool HasParameterBlock(const double* parameter_block) const;
  void GetParameterBlocks(std::vector<double*>* parameter_blocks) const;

  int ObservationBlockSize(const double* observation_block) const;
  int ObservationBlockLocalSize(const double* observation_block) const;
  bool HasObservationBlock(const double* observation_block) const;
  void GetObservationBlocks(std::vector<double*>* observation_blocks) const;

  void GetConstraintBlocks(std::vector<ConstraintBlockId>* constraint_blocks) const;

  void GetParameterBlocksForConstraintBlock(
      const ConstraintBlockId constraint_block,
      std::vector<double*>* parameter_blocks) const;

  void GetObservationBlocksForConstraintBlock(
      const ConstraintBlockId constraint_block,
      std::vector<double*>* observation_blocks) const;

  const GaussHelmertConstraintFunction* GetConstraintFunctionForConstraintBlock(
      const ConstraintBlockId constraint_block) const;
  const LossFunction* GetLossFunctionForConstraintBlock(
      const ConstraintBlockId constraint_block) const;

  void GetConstraintBlockBlocksForParameterBlock(
      const double* values,
      std::vector<ConstraintBlockId>* constraint_block) const;
  void GetConstraintBlockBlocksForObservationBlock(
      const double* values,
      std::vector<ConstraintBlockId>* constraint_block) const;

  const GHProgram& program() const { return *program_; }
  GHProgram* mutable_program() { return program_.get(); }

  const GHParameterMap& parameter_map() const { return parameter_block_map_; }
  const GHObservationMap& observation_map() const { return observation_block_map_; }

  const ConstraintBlockSet& constraint_block_set() const {
    CHECK(options_.enable_fast_removal)
        << "Fast removal not enabled, constraint_block_set is not maintained.";
    return constraint_block_set_;
  }

 private:
  GHParameterBlock* InternalAddParameterBlock(double* values, int size);
  GHObservationBlock* InternalAddObservationBlock(double* values, int size);

  void InternalRemoveConstraintBlock(GHConstraintBlock* constraint_block);

  bool InternalEvaluate(GHProgram* program,
                        double* cost,
                        std::vector<double>* residuals,
                        std::vector<double>* gradient_p, std::vector<double>* gradient_o,
                        CRSMatrix* jacobian_p, CRSMatrix* jacobian_o);

  // Delete the arguments in question. These differ from the Remove* functions
  // in that they do not clean up references to the block to delete; they
  // merely delete them.
  template<typename Block>
  void DeleteBlockInVector(std::vector<Block*>* mutable_blocks,
                           Block* block_to_remove);
  void DeleteBlock(GHConstraintBlock* constraint_block);
  void DeleteBlock(GHParameterBlock* parameter_block);
  void DeleteBlock(GHObservationBlock* observation_block);

  const Problem::Options options_;

  // The mapping from user pointers to parameter blocks.
  std::map<double*, GHParameterBlock*> parameter_block_map_;
  std::map<double*, GHObservationBlock*> observation_block_map_;

  // Iff enable_fast_removal is enabled, contains the current residual blocks.
  ConstraintBlockSet constraint_block_set_;

  // The actual parameter and residual blocks.
  internal::scoped_ptr<internal::GHProgram> program_;

  // When removing residual and parameter blocks, cost/loss functions and
  // parameterizations have ambiguous ownership. Instead of scanning the entire
  // problem to see if the cost/loss/parameterization is shared with other
  // residual or parameter blocks, buffer them until destruction.
  //
  // TODO(keir): See if it makes sense to use sets instead.
  std::vector<GaussHelmertConstraintFunction*> constraint_functions_to_delete_;
  std::vector<LossFunction*> loss_functions_to_delete_;
  std::vector<LocalParameterization*> local_parameterizations_to_delete_;

  CERES_DISALLOW_COPY_AND_ASSIGN(GHProblem);
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_GH_PROBLEM_IMPL_H_
