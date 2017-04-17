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
// Author: sameeragarwal@google.com (Sameer Agarwal)
//         mierle@gmail.com (Keir Mierle)

#include "ceres/GH_problem.h"

#include <cstdarg>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "ceres/casts.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/cost_function.h"
#include "ceres/crs_matrix.h"
#include "ceres/GH_evaluator.h"
#include "ceres/loss_function.h"
#include "ceres/map_util.h"
#include "ceres/GH_parameter_block.h"
#include "ceres/GH_program.h"
#include "ceres/constraint_block.h"
#include "ceres/stl_util.h"
#include "ceres/stringprintf.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

using std::map;
using std::string;
using std::vector;
typedef std::map<double*, internal::GHParameterBlock*> GHParameterMap;
typedef std::map<double*, internal::GHObservationBlock*> GHObservationMap;

namespace {
// Returns true if two regions of memory, a and b, with sizes size_a and size_b
// respectively, overlap.
bool RegionsAlias(const double* a, int size_a,
                  const double* b, int size_b) {
  return (a < b) ? b < (a + size_a)
                 : a < (b + size_b);
}

void CheckForNoAliasing(double* existing_block,
                        int existing_block_size,
                        double* new_block,
                        int new_block_size) {
  CHECK(!RegionsAlias(existing_block, existing_block_size,
                      new_block, new_block_size))
      << "Aliasing detected between existing parameter block at memory "
      << "location " << existing_block
      << " and has size " << existing_block_size << " with new parameter "
      << "block that has memory address " << new_block << " and would have "
      << "size " << new_block_size << ".";
}

}  // namespace

GHParameterBlock* GHProblem::InternalAddParameterBlock(double* values,
                                                       int size) {
  CHECK(values != NULL) << "Null pointer passed to AddParameterBlock "
                        << "for a parameter with size " << size;

  // Ignore the request if there is a block for the given pointer already.
  GHParameterMap::iterator it = parameter_block_map_.find(values);
  if (it != parameter_block_map_.end()) {
    if (!options_.disable_all_safety_checks) {
      int existing_size = it->second->Size();
      CHECK(size == existing_size)
          << "Tried adding a parameter block with the same double pointer, "
          << values << ", twice, but with different block sizes. Original "
          << "size was " << existing_size << " but new size is "
          << size;
    }
    return it->second;
  }

  if (!options_.disable_all_safety_checks) {
    // Before adding the parameter block, also check that it doesn't alias any
    // other parameter blocks.
    if (!parameter_block_map_.empty()) {
      GHParameterMap::iterator lb = parameter_block_map_.lower_bound(values);

      // If lb is not the first block, check the previous block for aliasing.
      if (lb != parameter_block_map_.begin()) {
        GHParameterMap::iterator previous = lb;
        --previous;
        CheckForNoAliasing(previous->first,
                           previous->second->Size(),
                           values,
                           size);
      }

      // If lb is not off the end, check lb for aliasing.
      if (lb != parameter_block_map_.end()) {
        CheckForNoAliasing(lb->first,
                           lb->second->Size(),
                           values,
                           size);
      }
    }
  }

  // Pass the index of the new parameter block as well to keep the index in
  // sync with the position of the parameter in the program's parameter vector.
  GHParameterBlock* new_parameter_block =
      new GHParameterBlock(values, size, program_->parameter_blocks_.size());

  // For dynamic problems, add the list of dependent residual blocks, which is
  // empty to start.
  if (options_.enable_fast_removal) {
    new_parameter_block->EnableConstraintBlockDependencies();
  }
  parameter_block_map_[values] = new_parameter_block;
  program_->parameter_blocks_.push_back(new_parameter_block);
  return new_parameter_block;
}

GHObservationBlock* GHProblem::InternalAddObservationBlock(double* values,
                                                       int size) {
  CHECK(values != NULL) << "Null pointer passed to AddParameterBlock "
                        << "for a parameter with size " << size;

  // Ignore the request if there is a block for the given pointer already.
  GHObservationMap::iterator it = observation_block_map_.find(values);
  if (it != observation_block_map_.end()) {
    if (!options_.disable_all_safety_checks) {
      int existing_size = it->second->Size();
      CHECK(size == existing_size)
          << "Tried adding a observation block with the same double pointer, "
          << values << ", twice, but with different block sizes. Original "
          << "size was " << existing_size << " but new size is "
          << size;
    }
    return it->second;
  }

  if (!options_.disable_all_safety_checks) {
    // Before adding the parameter block, also check that it doesn't alias any
    // other parameter blocks.
    if (!observation_block_map_.empty()) {
      GHObservationMap::iterator lb = observation_block_map_.lower_bound(values);

      // If lb is not the first block, check the previous block for aliasing.
      if (lb != observation_block_map_.begin()) {
        GHObservationMap::iterator previous = lb;
        --previous;
        CheckForNoAliasing(previous->first,
                           previous->second->Size(),
                           values,
                           size);
      }

      // If lb is not off the end, check lb for aliasing.
      if (lb != observation_block_map_.end()) {
        CheckForNoAliasing(lb->first,
                           lb->second->Size(),
                           values,
                           size);
      }
    }
  }

  // Pass the index of the new parameter block as well to keep the index in
  // sync with the position of the parameter in the program's parameter vector.
  GHObservationBlock* new_observation_block =
      new GHObservationBlock(values, size, program_->observation_blocks_.size());

  // For dynamic problems, add the list of dependent residual blocks, which is
  // empty to start.
  if (options_.enable_fast_removal) {
    new_observation_block->EnableConstraintBlockDependencies();
  }
  observation_block_map_[values] = new_observation_block;
  program_->observation_blocks_.push_back(new_observation_block);
  return new_observation_block;
}

void GHProblem::InternalRemoveConstraintBlock(GHConstraintBlock* constraint_block) {
  CHECK_NOTNULL(constraint_block);
  // Perform no check on the validity of constraint_block, that is handled in
  // the public method: RemoveConstraintBlock().

  // If needed, remove the parameter dependencies on this residual block.
  if (options_.enable_fast_removal) {
    const int num_parameter_blocks_for_residual =
        constraint_block->NumParameterBlocks();
    for (int i = 0; i < num_parameter_blocks_for_residual; ++i) {
      constraint_block->parameter_blocks()[i]
          ->RemoveConstraintBlock(constraint_block);
    }

    ConstraintBlockSet::iterator it = constraint_block_set_.find(constraint_block);
    constraint_block_set_.erase(it);
  }
  DeleteBlockInVector(program_->mutable_constraint_blocks(), constraint_block);
}

// Deletes the residual block in question, assuming there are no other
// references to it inside the problem (e.g. by another parameter). Referenced
// cost and loss functions are tucked away for future deletion, since it is not
// possible to know whether other parts of the problem depend on them without
// doing a full scan.
void GHProblem::DeleteBlock(GHConstraintBlock* constraint_block) {
  // The const casts here are legit, since GHConstraintBlock holds these
  // pointers as const pointers but we have ownership of them and
  // have the right to destroy them when the destructor is called.
  if (options_.cost_function_ownership == TAKE_OWNERSHIP &&
      constraint_block->constraint_function() != NULL) {
    constraint_functions_to_delete_.push_back(
        const_cast<RelationFunction*>(constraint_block->constraint_function()));
  }
  if (options_.loss_function_ownership == TAKE_OWNERSHIP &&
      constraint_block->loss_function() != NULL) {
    loss_functions_to_delete_.push_back(
        const_cast<LossFunction*>(constraint_block->loss_function()));
  }
  delete constraint_block;
}

// Deletes the parameter block in question, assuming there are no other
// references to it inside the problem (e.g. by any residual blocks).
// Referenced parameterizations are tucked away for future deletion, since it
// is not possible to know whether other parts of the problem depend on them
// without doing a full scan.
void GHProblem::DeleteBlock(GHParameterBlock* parameter_block) {
  if (options_.local_parameterization_ownership == TAKE_OWNERSHIP &&
      parameter_block->local_parameterization() != NULL) {
    local_parameterizations_to_delete_.push_back(
        parameter_block->mutable_local_parameterization());
  }
  parameter_block_map_.erase(parameter_block->mutable_user_state());
  delete parameter_block;
}

void GHProblem::DeleteBlock(GHObservationBlock* observation_block) {
  if (options_.local_parameterization_ownership == TAKE_OWNERSHIP &&
      observation_block->local_parameterization() != NULL) {
    local_parameterizations_to_delete_.push_back(
        observation_block->mutable_local_parameterization());
  }
  observation_block_map_.erase(observation_block->mutable_user_state());
  delete observation_block;
}

GHProblem::GHProblem() : program_(new internal::GHProgram) {}
GHProblem::GHProblem(const Problem::Options& options)
    : options_(options),
      program_(new internal::GHProgram) {}

GHProblem::~GHProblem() {
  // Collect the unique cost/loss functions and delete the residuals.
  const int num_constraint_blocks = program_->constraint_blocks_.size();
  constraint_functions_to_delete_.reserve(num_constraint_blocks);
  loss_functions_to_delete_.reserve(num_constraint_blocks);
  for (int i = 0; i < program_->constraint_blocks_.size(); ++i) {
    DeleteBlock(program_->constraint_blocks_[i]);
  }

  // Collect the unique parameterizations and delete the parameters.
  for (int i = 0; i < program_->parameter_blocks_.size(); ++i) {
    DeleteBlock(program_->parameter_blocks_[i]);
  }
  for (int i = 0; i < program_->observation_blocks_.size(); ++i) {
    DeleteBlock(program_->observation_blocks_[i]);
  }

  // Delete the owned cost/loss functions and parameterizations.
  STLDeleteUniqueContainerPointers(local_parameterizations_to_delete_.begin(),
                                   local_parameterizations_to_delete_.end());
  STLDeleteUniqueContainerPointers(constraint_functions_to_delete_.begin(),
                                   constraint_functions_to_delete_.end());
  STLDeleteUniqueContainerPointers(loss_functions_to_delete_.begin(),
                                   loss_functions_to_delete_.end());
}

GHConstraintBlock* GHProblem::AddConstraintBlock(
    RelationFunction* constraint_function,
    LossFunction* loss_function,
    const vector<double*>& parameter_blocks,
    const vector<double*>& observation_blocks) {
  CHECK_NOTNULL(constraint_function);
  CHECK_EQ(parameter_blocks.size(),
           constraint_function->parameter_block_sizes().size());
  CHECK_EQ(observation_blocks.size(),
           constraint_function->observation_block_sizes().size());

  // Check the sizes match.
  const vector<int32>& parameter_block_sizes =
      constraint_function->parameter_block_sizes();

  if (!options_.disable_all_safety_checks) {
    CHECK_EQ(parameter_block_sizes.size(), parameter_blocks.size())
        << "Number of blocks input is different than the number of blocks "
        << "that the cost function expects.";

    // Check for duplicate parameter blocks.
    vector<double*> sorted_parameter_blocks(parameter_blocks);
    sort(sorted_parameter_blocks.begin(), sorted_parameter_blocks.end());
    const bool has_duplicate_items =
        (std::adjacent_find(sorted_parameter_blocks.begin(),
                            sorted_parameter_blocks.end())
         != sorted_parameter_blocks.end());
    if (has_duplicate_items) {
      string blocks;
      for (int i = 0; i < parameter_blocks.size(); ++i) {
        blocks += StringPrintf(" %p ", parameter_blocks[i]);
      }

      LOG(FATAL) << "Duplicate parameter blocks in a residual parameter "
                 << "are not allowed. Parameter block pointers: ["
                 << blocks << "]";
    }
  }

  const vector<int32>& observation_block_sizes =
      constraint_function->observation_block_sizes();

  if (!options_.disable_all_safety_checks) {
    CHECK_EQ(observation_block_sizes.size(), observation_blocks.size())
        << "Number of blocks input is different than the number of blocks "
        << "that the cost function expects.";

    // Check for duplicate observation blocks.
    vector<double*> sorted_observation_blocks(observation_blocks);
    sort(sorted_observation_blocks.begin(), sorted_observation_blocks.end());
    const bool has_duplicate_items =
        (std::adjacent_find(sorted_observation_blocks.begin(),
                            sorted_observation_blocks.end())
         != sorted_observation_blocks.end());
    if (has_duplicate_items) {
      string blocks;
      for (int i = 0; i < observation_blocks.size(); ++i) {
        blocks += StringPrintf(" %p ", observation_blocks[i]);
      }

      LOG(FATAL) << "Duplicate observation blocks in a residual observation "
                 << "are not allowed. observation block pointers: ["
                 << blocks << "]";
    }
  }

  // Add parameter blocks and convert the double*'s to parameter blocks.
  vector<GHParameterBlock*> parameter_block_ptrs(parameter_blocks.size());
  for (int i = 0; i < parameter_blocks.size(); ++i) {
    parameter_block_ptrs[i] =
        InternalAddParameterBlock(parameter_blocks[i],
                                  parameter_block_sizes[i]);
  }

  vector<GHObservationBlock*> observation_block_ptrs(observation_blocks.size());
  for (int i = 0; i < observation_blocks.size(); ++i) {
    observation_block_ptrs[i] =
        InternalAddObservationBlock(observation_blocks[i],
                                  observation_block_sizes[i]);
  }

  if (!options_.disable_all_safety_checks) {
    // Check that the block sizes match the block sizes expected by the
    // cost_function.
    for (int i = 0; i < parameter_block_ptrs.size(); ++i) {
      CHECK_EQ(constraint_function->parameter_block_sizes()[i],
               parameter_block_ptrs[i]->Size())
          << "The cost function expects parameter block " << i
          << " of size " << constraint_function->parameter_block_sizes()[i]
          << " but was given a block of size "
          << parameter_block_ptrs[i]->Size();
    }

    for (int i = 0; i < observation_block_ptrs.size(); ++i) {
      CHECK_EQ(constraint_function->observation_block_sizes()[i],
               observation_block_ptrs[i]->Size())
          << "The cost function expects observation block " << i
          << " of size " << constraint_function->observation_block_sizes()[i]
          << " but was given a block of size "
          << observation_block_ptrs[i]->Size();
    }
  }

  GHConstraintBlock* new_constraint_block =
      new GHConstraintBlock(constraint_function,
                        loss_function,
                        parameter_block_ptrs,
                        observation_block_ptrs,
                        program_->constraint_blocks_.size());

  // Add dependencies on the residual to the parameter blocks.
  if (options_.enable_fast_removal) {
    for (int i = 0; i < parameter_blocks.size(); ++i) {
      parameter_block_ptrs[i]->AddConstraintBlock(new_constraint_block);
    }
    for (int i = 0; i < observation_blocks.size(); ++i) {
      observation_block_ptrs[i]->AddConstraintBlock(new_constraint_block);
    }
  }

  program_->constraint_blocks_.push_back(new_constraint_block);

  if (options_.enable_fast_removal) {
    constraint_block_set_.insert(new_constraint_block);
  }

  return new_constraint_block;
}


GHConstraintBlock* GHProblem::AddConstraintBlock(
    RelationFunction* constraint_function,
    LossFunction* loss_function,
    ...) {

    const size_t num_parameter_blocks = constraint_function->parameter_block_sizes().size();
    const size_t num_observation_blocks = constraint_function->observation_block_sizes().size();

    vector<double*> parameters(num_parameter_blocks);
    vector<double*> observations(num_observation_blocks);

    va_list arguments;                     // A place to store the list of arguments
    va_start ( arguments, loss_function );           // Initializing arguments to store all values after loss_function
    for(size_t i=0; i < num_parameter_blocks; i++) {
        parameters[i] = va_arg ( arguments, double* );
    }

    for(size_t i=0; i < num_observation_blocks; i++) {
        observations[i] = va_arg ( arguments, double* );
    }
    va_end ( arguments );                  // Cleans up the list

    return AddConstraintBlock(constraint_function, loss_function, parameters, observations);
}


void GHProblem::AddParameterBlock(double* values, int size) {
  InternalAddParameterBlock(values, size);
}

void GHProblem::AddParameterBlock(
    double* values,
    int size,
    LocalParameterization* local_parameterization) {
  GHParameterBlock* parameter_block =
      InternalAddParameterBlock(values, size);
  if (local_parameterization != NULL) {
    parameter_block->SetParameterization(local_parameterization);
  }
}

void GHProblem::AddObservationBlock(double* values, int size) {
  InternalAddObservationBlock(values, size);
}

void GHProblem::AddObservationBlock(
    double* values,
    int size,
    LocalParameterization* local_parameterization) {
  GHObservationBlock* observation_block =
      InternalAddObservationBlock(values, size);
  if (local_parameterization != NULL) {
    observation_block->SetParameterization(local_parameterization);
  }
}

// Delete a block from a vector of blocks, maintaining the indexing invariant.
// This is done in constant time by moving an element from the end of the
// vector over the element to remove, then popping the last element. It
// destroys the ordering in the interest of speed.
template<typename Block>
void GHProblem::DeleteBlockInVector(vector<Block*>* mutable_blocks,
                                      Block* block_to_remove) {
  CHECK_EQ((*mutable_blocks)[block_to_remove->index()], block_to_remove)
      << "You found a Ceres bug! \n"
      << "Block requested: "
      << block_to_remove->ToString() << "\n"
      << "Block present: "
      << (*mutable_blocks)[block_to_remove->index()]->ToString();

  // Prepare the to-be-moved block for the new, lower-in-index position by
  // setting the index to the blocks final location.
  Block* tmp = mutable_blocks->back();
  tmp->set_index(block_to_remove->index());

  // Overwrite the to-be-deleted residual block with the one at the end.
  (*mutable_blocks)[block_to_remove->index()] = tmp;

  DeleteBlock(block_to_remove);

  // The block is gone so shrink the vector of blocks accordingly.
  mutable_blocks->pop_back();
}

void GHProblem::RemoveConstraintBlock(GHConstraintBlock* constraint_block) {
  CHECK_NOTNULL(constraint_block);

  // Verify that constraint_block identifies a residual in the current problem.
  const string residual_not_found_message =
      StringPrintf("Residual block to remove: %p not found. This usually means "
                   "one of three things have happened:\n"
                   " 1) constraint_block is uninitialised and points to a random "
                   "area in memory.\n"
                   " 2) constraint_block represented a residual that was added to"
                   " the problem, but referred to a parameter block which has "
                   "since been removed, which removes all residuals which "
                   "depend on that parameter block, and was thus removed.\n"
                   " 3) constraint_block referred to a residual that has already "
                   "been removed from the problem (by the user).",
                   constraint_block);
  if (options_.enable_fast_removal) {
    CHECK(constraint_block_set_.find(constraint_block) !=
          constraint_block_set_.end())
        << residual_not_found_message;
  } else {
    // Perform a full search over all current residuals.
    CHECK(std::find(program_->constraint_blocks().begin(),
                    program_->constraint_blocks().end(),
                    constraint_block) != program_->constraint_blocks().end())
        << residual_not_found_message;
  }

  InternalRemoveConstraintBlock(constraint_block);
}

void GHProblem::RemoveParameterBlock(double* values) {
  GHParameterBlock* parameter_block =
      FindWithDefault(parameter_block_map_, values, NULL);
  if (parameter_block == NULL) {
    LOG(FATAL) << "Parameter block not found: " << values
               << ". You must add the parameter block to the problem before "
               << "it can be removed.";
  }

  if (options_.enable_fast_removal) {
    // Copy the dependent residuals from the parameter block because the set of
    // dependents will change after each call to RemoveResidualBlock().
    vector<GHConstraintBlock*> constraint_blocks_to_remove(
        parameter_block->mutable_constraint_blocks()->begin(),
        parameter_block->mutable_constraint_blocks()->end());
    for (int i = 0; i < constraint_blocks_to_remove.size(); ++i) {
      InternalRemoveConstraintBlock(constraint_blocks_to_remove[i]);
    }
  } else {
    // Scan all the residual blocks to remove ones that depend on the parameter
    // block. Do the scan backwards since the vector changes while iterating.
    const int num_constraint_blocks = NumConstraintBlocks();
    for (int i = num_constraint_blocks - 1; i >= 0; --i) {
      GHConstraintBlock* constraint_block =
          (*(program_->mutable_constraint_blocks()))[i];
      const int num_parameter_blocks = constraint_block->NumParameterBlocks();
      for (int j = 0; j < num_parameter_blocks; ++j) {
        if (constraint_block->parameter_blocks()[j] == parameter_block) {
          InternalRemoveConstraintBlock(constraint_block);
          // The parameter blocks are guaranteed unique.
          break;
        }
      }
    }
  }
  DeleteBlockInVector(program_->mutable_parameter_blocks(), parameter_block);
}

void GHProblem::RemoveObservationBlock(double* values) {
  GHObservationBlock* observation_block =
      FindWithDefault(observation_block_map_, values, NULL);
  if (observation_block == NULL) {
    LOG(FATAL) << "Observation block not found: " << values
               << ". You must add the observation block to the problem before "
               << "it can be removed.";
  }

  if (options_.enable_fast_removal) {
    // Copy the dependent residuals from the parameter block because the set of
    // dependents will change after each call to RemoveResidualBlock().
    vector<GHConstraintBlock*> constraint_blocks_to_remove(
        observation_block->mutable_constraint_blocks()->begin(),
        observation_block->mutable_constraint_blocks()->end());
    for (int i = 0; i < constraint_blocks_to_remove.size(); ++i) {
      InternalRemoveConstraintBlock(constraint_blocks_to_remove[i]);
    }
  } else {
    // Scan all the residual blocks to remove ones that depend on the parameter
    // block. Do the scan backwards since the vector changes while iterating.
    const int num_constraint_blocks = NumConstraintBlocks();
    for (int i = num_constraint_blocks - 1; i >= 0; --i) {
      GHConstraintBlock* constraint_block =
          (*(program_->mutable_constraint_blocks()))[i];
      const int num_observation_blocks = constraint_block->NumParameterBlocks();
      for (int j = 0; j < num_observation_blocks; ++j) {
        if (constraint_block->observation_blocks()[j] == observation_block) {
          InternalRemoveConstraintBlock(constraint_block);
          // The observation blocks are guaranteed unique.
          break;
        }
      }
    }
  }
  DeleteBlockInVector(program_->mutable_observation_blocks(), observation_block);
}

void GHProblem::SetParameterBlockConstant(double* values) {
  GHParameterBlock* parameter_block =
      FindWithDefault(parameter_block_map_, values, NULL);
  if (parameter_block == NULL) {
    LOG(FATAL) << "Parameter block not found: " << values
               << ". You must add the parameter block to the problem before "
               << "it can be set constant.";
  }

  parameter_block->SetConstant();
}

bool GHProblem::IsParameterBlockConstant(double* values) const {
  const GHParameterBlock* parameter_block =
      FindWithDefault(parameter_block_map_, values, NULL);
  CHECK(parameter_block != NULL)
    << "Parameter block not found: " << values << ". You must add the "
    << "parameter block to the problem before it can be queried.";

  return parameter_block->IsConstant();
}

void GHProblem::SetParameterBlockVariable(double* values) {
  GHParameterBlock* parameter_block =
      FindWithDefault(parameter_block_map_, values, NULL);
  if (parameter_block == NULL) {
    LOG(FATAL) << "Parameter block not found: " << values
               << ". You must add the parameter block to the problem before "
               << "it can be set varying.";
  }

  parameter_block->SetVarying();
}

void GHProblem::SetObservationBlockConstant(double* values) {
  GHObservationBlock* observation_block =
      FindWithDefault(observation_block_map_, values, NULL);
  if (observation_block == NULL) {
    LOG(FATAL) << "Observation block not found: " << values
               << ". You must add the observation block to the problem before "
               << "it can be set constant.";
  }

  observation_block->SetConstant();
}

bool GHProblem::IsObservationBlockConstant(double* values) const {
  GHObservationBlock* observation_block =
      FindWithDefault(observation_block_map_, values, NULL);
  CHECK(observation_block != NULL)
    << "Observation block not found: " << values << ". You must add the "
    << "Observation block to the problem before it can be queried.";

  return observation_block->IsConstant();
}

void GHProblem::SetObservationBlockVariable(double* values) {
  GHObservationBlock* observation_block =
      FindWithDefault(observation_block_map_, values, NULL);
  if (observation_block == NULL) {
    LOG(FATAL) << "Observation block not found: " << values
               << ". You must add the parameter block to the problem before "
               << "it can be set varying.";
  }

  observation_block->SetVarying();
}

void GHProblem::SetParameterization(
    double* values,
    LocalParameterization* local_parameterization) {
  GHParameterBlock* parameter_block =
      FindWithDefault(parameter_block_map_, values, NULL);

  if (parameter_block != NULL) {
      parameter_block->SetParameterization(local_parameterization);
      return;
  }

  GHObservationBlock* observation_block =
      FindWithDefault(observation_block_map_, values, NULL);

  if (observation_block != NULL) {
      observation_block->SetParameterization(local_parameterization);
      return;
  }

  LOG(FATAL) << "Parameter or Observation block not found: " << values
             << ". You must add the block to the problem before "
             << "you can set its local parameterization.";
}

const LocalParameterization* GHProblem::GetParameterization(
    double* values) const {
  GHParameterBlock* parameter_block =
      FindWithDefault(parameter_block_map_, values, NULL);
  if (parameter_block != NULL) {
      return parameter_block->local_parameterization();
  }

  GHObservationBlock* observation_block =
      FindWithDefault(observation_block_map_, values, NULL);
  if (observation_block != NULL) {
      return observation_block->local_parameterization();;
  }

  LOG(FATAL) << "Parameter block not found: " << values
             << ". You must add the parameter block to the problem before "
             << "you can get its local parameterization.";
}

void GHProblem::SetParameterLowerBound(double* values,
                                         int index,
                                         double lower_bound) {
  GHParameterBlock* parameter_block =
      FindWithDefault(parameter_block_map_, values, NULL);
  if (parameter_block == NULL) {
    LOG(FATAL) << "Parameter block not found: " << values
               << ". You must add the parameter block to the problem before "
               << "you can set a lower bound on one of its components.";
  }

  parameter_block->SetLowerBound(index, lower_bound);
}

void GHProblem::SetParameterUpperBound(double* values,
                                         int index,
                                         double upper_bound) {
  GHParameterBlock* parameter_block =
      FindWithDefault(parameter_block_map_, values, NULL);
  if (parameter_block == NULL) {
    LOG(FATAL) << "Parameter block not found: " << values
               << ". You must add the parameter block to the problem before "
               << "you can set an upper bound on one of its components.";
  }
  parameter_block->SetUpperBound(index, upper_bound);
}

bool GHProblem::Evaluate(const GHProblem::EvaluateOptions& evaluate_options,
                           double* cost,
                           vector<double>* residuals,
                           vector<double>* gradient_p,vector<double>* gradient_o,
                           CRSMatrix* jacobian_p, CRSMatrix* jacobian_o) {
  if (cost == NULL &&
      residuals == NULL &&
      gradient_p == NULL && gradient_o == NULL &&
      jacobian_p == NULL && jacobian_o == NULL) {
    LOG(INFO) << "Nothing to do.";
    return true;
  }

  // If the user supplied residual blocks, then use them, otherwise
  // take the residual blocks from the underlying program.
  GHProgram program;
  *program.mutable_constraint_blocks() =
      ((evaluate_options.constraint_blocks.size() > 0)
       ? evaluate_options.constraint_blocks : program_->constraint_blocks());

  const vector<double*>& parameter_block_ptrs =
      evaluate_options.parameter_blocks;

  vector<GHParameterBlock*> variable_parameter_blocks;
  vector<GHParameterBlock*>& parameter_blocks =
      *program.mutable_parameter_blocks();

  if (parameter_block_ptrs.size() == 0) {
    // The user did not provide any parameter blocks, so default to
    // using all the parameter blocks in the order that they are in
    // the underlying program object.
    parameter_blocks = program_->parameter_blocks();
  } else {
    // The user supplied a vector of parameter blocks. Using this list
    // requires a number of steps.

    // 1. Convert double* into ParameterBlock*
    parameter_blocks.resize(parameter_block_ptrs.size());
    for (int i = 0; i < parameter_block_ptrs.size(); ++i) {
      parameter_blocks[i] = FindWithDefault(parameter_block_map_,
                                            parameter_block_ptrs[i],
                                            NULL);
      if (parameter_blocks[i] == NULL) {
        LOG(FATAL) << "No known parameter block for "
                   << "Problem::Evaluate::Options.parameter_blocks[" << i << "]"
                   << " = " << parameter_block_ptrs[i];
      }
    }

    // 2. The user may have only supplied a subset of parameter
    // blocks, so identify the ones that are not supplied by the user
    // and are NOT constant. These parameter blocks are stored in
    // variable_parameter_blocks.
    //
    // To ensure that the parameter blocks are not included in the
    // columns of the jacobian, we need to make sure that they are
    // constant during evaluation and then make them variable again
    // after we are done.
    vector<GHParameterBlock*> all_parameter_blocks(program_->parameter_blocks());
    vector<GHParameterBlock*> included_parameter_blocks(
        program.parameter_blocks());

    vector<GHParameterBlock*> excluded_parameter_blocks;
    sort(all_parameter_blocks.begin(), all_parameter_blocks.end());
    sort(included_parameter_blocks.begin(), included_parameter_blocks.end());
    set_difference(all_parameter_blocks.begin(),
                   all_parameter_blocks.end(),
                   included_parameter_blocks.begin(),
                   included_parameter_blocks.end(),
                   back_inserter(excluded_parameter_blocks));

    variable_parameter_blocks.reserve(excluded_parameter_blocks.size());
    for (int i = 0; i < excluded_parameter_blocks.size(); ++i) {
      GHParameterBlock* parameter_block = excluded_parameter_blocks[i];
      if (!parameter_block->IsConstant()) {
        variable_parameter_blocks.push_back(parameter_block);
        parameter_block->SetConstant();
      }
    }
  }
  // Setup the Parameter indices and offsets before an evaluator can
  // be constructed and used.
  program.SetParameterOffsetsAndIndex();

  const vector<double*>& observation_block_ptrs =
      evaluate_options.observation_blocks;
  vector<GHObservationBlock*> variable_observation_blocks;
  vector<GHObservationBlock*>& observation_blocks =
      *program.mutable_observation_blocks();

  if (observation_block_ptrs.size() == 0) {
    observation_blocks = program_->observation_blocks();
  } else {
    observation_blocks.resize(observation_block_ptrs.size());
    for (int i = 0; i < observation_block_ptrs.size(); ++i) {
      observation_blocks[i] = FindWithDefault(observation_block_map_,
                                            observation_block_ptrs[i],
                                            NULL);
      if (observation_blocks[i] == NULL) {
        LOG(FATAL) << "No known observation block for "
                   << "Problem::Evaluate::Options.observation_blocks[" << i << "]"
                   << " = " << observation_block_ptrs[i];
      }
    }

    vector<GHObservationBlock*> all_observation_blocks(program_->observation_blocks());
    vector<GHObservationBlock*> included_observation_blocks(
        program.observation_blocks());

    vector<GHObservationBlock*> excluded_observation_blocks;
    sort(all_observation_blocks.begin(), all_observation_blocks.end());
    sort(included_observation_blocks.begin(), included_observation_blocks.end());
    set_difference(all_observation_blocks.begin(),
                   all_observation_blocks.end(),
                   included_observation_blocks.begin(),
                   included_observation_blocks.end(),
                   back_inserter(excluded_observation_blocks));

    variable_observation_blocks.reserve(excluded_observation_blocks.size());
    for (int i = 0; i < excluded_observation_blocks.size(); ++i) {
      GHObservationBlock* observation_block = excluded_observation_blocks[i];
      if (!observation_block->IsConstant()) {
        variable_observation_blocks.push_back(observation_block);
        observation_block->SetConstant();
      }
    }
  }
  program.SetObservationOffsetsAndIndex();

  GHEvaluator::Options evaluator_options;

  // Even though using SPARSE_NORMAL_CHOLESKY requires SuiteSparse or
  // CXSparse, here it just being used for telling the evaluator to
  // use a SparseRowCompressedMatrix for the jacobian. This is because
  // the Evaluator decides the storage for the Jacobian based on the
  // type of linear solver being used.
  evaluator_options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
#ifndef CERES_USE_OPENMP
  LOG_IF(WARNING, evaluate_options.num_threads > 1)
      << "OpenMP support is not compiled into this binary; "
      << "only evaluate_options.num_threads = 1 is supported. Switching "
      << "to single threaded mode.";
  evaluator_options.num_threads = 1;
#else
  evaluator_options.num_threads = evaluate_options.num_threads;
#endif  // CERES_USE_OPENMP

  string error;
  scoped_ptr<GHEvaluator> evaluator(
      GHEvaluator::Create(evaluator_options, &program, &error));
  if (evaluator.get() == NULL) {
    LOG(ERROR) << "Unable to create an Evaluator object. "
               << "Error: " << error
               << "This is a Ceres bug; please contact the developers!";

    // Make the parameter blocks that were temporarily marked
    // constant, variable again.
    for (int i = 0; i < variable_parameter_blocks.size(); ++i) {
      variable_parameter_blocks[i]->SetVarying();
    }
    program_->SetParameterBlockStatePtrsToUserStatePtrs();
    program_->SetParameterOffsetsAndIndex();

    for (int i = 0; i < variable_observation_blocks.size(); ++i) {
      variable_observation_blocks[i]->SetVarying();
    }
    program_->SetObservationBlockStatePtrsToUserStatePtrs();
    program_->SetObservationOffsetsAndIndex();

    return false;
  }

  if (residuals !=NULL) {
    residuals->resize(evaluator->NumResiduals());
  }

  if (gradient_p != NULL) {
    gradient_p->resize(evaluator->NumEffectiveParameters());
  }
  if (gradient_o != NULL) {
    gradient_o->resize(evaluator->NumEffectiveObservations());
  }

  scoped_ptr<CompressedRowSparseMatrix> tmp_jacobian_p;
  if (jacobian_p != NULL) {
    tmp_jacobian_p.reset(
        down_cast<CompressedRowSparseMatrix*>(evaluator->CreateJacobian_p()));
  }

  scoped_ptr<CompressedRowSparseMatrix> tmp_jacobian_o;
  if (jacobian_o != NULL) {
    tmp_jacobian_o.reset(
        down_cast<CompressedRowSparseMatrix*>(evaluator->CreateJacobian_o()));
  }

  // Point the state pointers to the user state pointers. This is
  // needed so that we can extract a parameter vector which is then
  // passed to Evaluator::Evaluate.
  program.SetParameterBlockStatePtrsToUserStatePtrs();
  program.SetObservationBlockStatePtrsToUserStatePtrs();

  // Copy the value of the parameter blocks into a vector, since the
  // Evaluate::Evaluate method needs its input as such. The previous
  // call to SetParameterBlockStatePtrsToUserStatePtrs ensures that
  // these values are the ones corresponding to the actual state of
  // the parameter blocks, rather than the temporary state pointer
  // used for evaluation.
  Vector parameters(program.NumParameters());
  program.ParameterBlocksToStateVector(parameters.data());
  Vector observations(program.NumObservations());
  program.ObservationBlocksToStateVector(observations.data());

  double tmp_cost = 0;

  GHEvaluator::EvaluateOptions evaluator_evaluate_options;
  evaluator_evaluate_options.apply_loss_function =
      evaluate_options.apply_loss_function;
  bool status = evaluator->Evaluate(evaluator_evaluate_options,
                                    parameters.data(),observations.data(),
                                    &tmp_cost,
                                    residuals != NULL ? &(*residuals)[0] : NULL,
                                    gradient_p != NULL ? &(*gradient_p)[0] : NULL,
                                    gradient_o != NULL ? &(*gradient_o)[0] : NULL,
                                    tmp_jacobian_p.get(),tmp_jacobian_o.get());

  // Make the parameter blocks that were temporarily marked constant,
  // variable again.
  for (int i = 0; i < variable_parameter_blocks.size(); ++i) {
    variable_parameter_blocks[i]->SetVarying();
  }

  for (int i = 0; i < variable_observation_blocks.size(); ++i) {
    variable_observation_blocks[i]->SetVarying();
  }

  if (status) {
    if (cost != NULL) {
      *cost = tmp_cost;
    }
    if (jacobian_p != NULL) {
      tmp_jacobian_p->ToCRSMatrix(jacobian_p);
    }
    if (jacobian_o != NULL) {
      tmp_jacobian_o->ToCRSMatrix(jacobian_o);
    }
  }

  program_->SetParameterBlockStatePtrsToUserStatePtrs();
  program_->SetParameterOffsetsAndIndex();
  program_->SetObservationBlockStatePtrsToUserStatePtrs();
  program_->SetObservationOffsetsAndIndex();
  return status;
}

int GHProblem::NumParameterBlocks() const {
  return program_->NumParameterBlocks();
}

int GHProblem::NumParameters() const {
  return program_->NumParameters();
}

int GHProblem::NumConstraintBlocks() const {
  return program_->NumConstraintBlocks();
}

int GHProblem::NumResiduals() const {
  return program_->NumResiduals();
}

int GHProblem::ParameterBlockSize(const double* values) const {
  GHParameterBlock* parameter_block =
      FindWithDefault(parameter_block_map_, const_cast<double*>(values), NULL);
  if (parameter_block == NULL) {
    LOG(FATAL) << "Parameter block not found: " << values
               << ". You must add the parameter block to the problem before "
               << "you can get its size.";
  }

  return parameter_block->Size();
}

int GHProblem::ParameterBlockLocalSize(const double* values) const {
  GHParameterBlock* parameter_block =
      FindWithDefault(parameter_block_map_, const_cast<double*>(values), NULL);
  if (parameter_block == NULL) {
    LOG(FATAL) << "Parameter block not found: " << values
               << ". You must add the parameter block to the problem before "
               << "you can get its local size.";
  }

  return parameter_block->LocalSize();
}

bool GHProblem::HasParameterBlock(const double* parameter_block) const {
  return (parameter_block_map_.find(const_cast<double*>(parameter_block)) !=
          parameter_block_map_.end());
}

void GHProblem::GetParameterBlocks(vector<double*>* parameter_blocks) const {
  CHECK_NOTNULL(parameter_blocks);
  parameter_blocks->resize(0);
  for (GHParameterMap::const_iterator it = parameter_block_map_.begin();
       it != parameter_block_map_.end();
       ++it) {
    parameter_blocks->push_back(it->first);
  }
}

void GHProblem::GetConstraintBlocks(
    vector<GHConstraintBlock*>* constraint_blocks) const {
  CHECK_NOTNULL(constraint_blocks);
  *constraint_blocks = program().constraint_blocks();
}

void GHProblem::GetParameterBlocksForConstraintBlock(
    const ConstraintBlockId constraint_block,
    std::vector<double*>* parameter_blocks) const {
  int num_parameter_blocks = constraint_block->NumParameterBlocks();
  CHECK_NOTNULL(parameter_blocks)->resize(num_parameter_blocks);
  for (int i = 0; i < num_parameter_blocks; ++i) {
    (*parameter_blocks)[i] =
        constraint_block->parameter_blocks()[i]->mutable_user_state();
  }
}

void GHProblem::GetObservationBlocksForConstraintBlock(
    const ConstraintBlockId constraint_block,
    std::vector<double*>* observation_blocks) const {
    int num_observation_blocks = constraint_block->NumObservationBlocks();
    CHECK_NOTNULL(observation_blocks)->resize(num_observation_blocks);
    for (int i = 0; i < num_observation_blocks; ++i) {
      (*observation_blocks)[i] =
          constraint_block->observation_blocks()[i]->mutable_user_state();
    }
  }

const RelationFunction* GHProblem::GetConstraintFunctionForConstraintBlock(
    const ConstraintBlockId constraint_block) const {
  return constraint_block->constraint_function();
}

const LossFunction* GHProblem::GetLossFunctionForConstraintBlock(
    const ConstraintBlockId constraint_block) const {
  return constraint_block->loss_function();
}

void GHProblem::GetConstraintBlockBlocksForParameterBlock(
    const double* values,
    std::vector<ConstraintBlockId>* constraint_blocks) const {
  GHParameterBlock* parameter_block =
      FindWithDefault(parameter_block_map_, const_cast<double*>(values), NULL);
  if (parameter_block == NULL) {
    LOG(FATAL) << "Parameter block not found: " << values
               << ". You must add the parameter block to the problem before "
               << "you can get the residual blocks that depend on it.";
  }

  if (options_.enable_fast_removal) {
    // In this case the residual blocks that depend on the parameter block are
    // stored in the parameter block already, so just copy them out.
    CHECK_NOTNULL(constraint_blocks)->resize(
        parameter_block->mutable_constraint_blocks()->size());
    std::copy(parameter_block->mutable_constraint_blocks()->begin(),
              parameter_block->mutable_constraint_blocks()->end(),
              constraint_blocks->begin());
    return;
  }

  // Find residual blocks that depend on the parameter block.
  CHECK_NOTNULL(constraint_blocks)->clear();
  const int num_constraint_blocks = NumConstraintBlocks();
  for (int i = 0; i < num_constraint_blocks; ++i) {
    GHConstraintBlock* constraint_block =
        (*(program_->mutable_constraint_blocks()))[i];
    const int num_parameter_blocks = constraint_block->NumParameterBlocks();
    for (int j = 0; j < num_parameter_blocks; ++j) {
      if (constraint_block->parameter_blocks()[j] == parameter_block) {
        constraint_blocks->push_back(constraint_block);
        // The parameter blocks are guaranteed unique.
        break;
      }
    }
  }
}

void GHProblem::GetConstraintBlockBlocksForObservationBlock(
    const double* values,
    std::vector<ConstraintBlockId>* constraint_blocks) const {
  GHObservationBlock* observation_block =
      FindWithDefault(observation_block_map_, const_cast<double*>(values), NULL);
  if (observation_block == NULL) {
    LOG(FATAL) << "observation block not found: " << values
               << ". You must add the observation block to the problem before "
               << "you can get the residual blocks that depend on it.";
  }

  if (options_.enable_fast_removal) {
    // In this case the residual blocks that depend on the observation block are
    // stored in the observation block already, so just copy them out.
    CHECK_NOTNULL(constraint_blocks)->resize(
        observation_block->mutable_constraint_blocks()->size());
    std::copy(observation_block->mutable_constraint_blocks()->begin(),
              observation_block->mutable_constraint_blocks()->end(),
              constraint_blocks->begin());
    return;
  }

  // Find residual blocks that depend on the observation block.
  CHECK_NOTNULL(constraint_blocks)->clear();
  const int num_constraint_blocks = NumConstraintBlocks();
  for (int i = 0; i < num_constraint_blocks; ++i) {
    GHConstraintBlock* constraint_block =
        (*(program_->mutable_constraint_blocks()))[i];
    const int num_observation_blocks = constraint_block->NumObservationBlocks();
    for (int j = 0; j < num_observation_blocks; ++j) {
      if (constraint_block->observation_blocks()[j] == observation_block) {
        constraint_blocks->push_back(constraint_block);
        // The observation blocks are guaranteed unique.
        break;
      }
    }
  }
}

}  // namespace internal
}  // namespace ceres
