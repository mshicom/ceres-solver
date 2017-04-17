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
//         keir@google.m (Keir Mierle)
// modifed: kaihong.huang11@google.com (Kaihong Huang)
//
// This is the interface through which the least squares solver accesses the
// residual and Jacobian of the least squares problem. Users are expected to
// subclass CostFunction or RelationFunction to define their own terms in
// the least squares problem. Use CostFunction to model explict functions
// like f(x)=l, where x are unkown parameters and l are known observation.
// In case of implicit functions like g(x,l)=0, use RelationFunction.
// The CostFunction provides only Jx (the Jacobian of x) while RelationFunction
// provides both Jx and Jl. For more info about implicit functions, please wiki
// the term "Gauss-Helmert model", in comparision, explict function referes to
// "Gauss-Markov model"

// It is recommended that users define templated residual functors for use as
// arguments for AutoDiffCostFunction (see autodiff_cost_function.h), instead of
// directly implementing the CostFunction interface. This often results in both
// shorter code and faster execution than hand-coded derivatives. However,
// specialized cases may demand direct implementation of the lower-level
// CostFunction interface; for example, this is true when calling legacy code
// which is not templated on numeric types.

#ifndef CERES_PUBLIC_COST_FUNCTION_H_
#define CERES_PUBLIC_COST_FUNCTION_H_

#include <vector>
#include "ceres/internal/macros.h"
#include "ceres/internal/port.h"
#include "ceres/types.h"
#include "ceres/internal/disable_warnings.h"

namespace ceres {

// This class models the implicit relations like g(x,l)=0 between the (unkown)
// parameters x and the (known) observations l, which serves as a constraint
// function in the Gauss-Helmert least squares problem.
class CERES_EXPORT RelationFunction {
 public:
  RelationFunction() : num_residuals_(0) {}

  virtual ~RelationFunction() {}


  virtual bool Evaluate(double const* const* parameters,
                        double const* const* observations,
                        double* residuals,
                        double** jacobians_p,
                        double** jacobians_o) const = 0;

  // Same as above, but the Jacobian of the observations is not needed.
  // Subclasses can overwrite this function to provide a faster Evaluate
  // function when the Jacobian of the observations is not needed. Default
  // behavior is simply pass a NULL pointer.
  virtual bool Evaluate(double const* const* parameters,
                        double const* const* observations,
                        double* residuals,
                        double** jacobians_p) const {
    return Evaluate(parameters, observations, residuals, jacobians_p, NULL);
  }

  const std::vector<int32>& parameter_block_sizes() const {
    return parameter_block_sizes_;
  }
  const std::vector<int32>& observation_block_sizes() const {
    return observation_block_sizes_;
  }
  int num_residuals() const {
    return num_residuals_;
  }

 protected:
  std::vector<int32>* mutable_parameter_block_sizes() {
    return &parameter_block_sizes_;
  }
  std::vector<int32>* mutable_observation_block_sizes() {
    return &observation_block_sizes_;
  }
  void set_num_residuals(int num_residuals) {
    num_residuals_ = num_residuals;
  }

 private:
  // Cost function signature metadata: number of inputs & their sizes,
  // number of outputs (residuals).
  std::vector<int32> parameter_block_sizes_;
  std::vector<int32> observation_block_sizes_;

  int num_residuals_;
  CERES_DISALLOW_COPY_AND_ASSIGN(RelationFunction);
};



// This class implements the computation of the cost (a.k.a. residual) terms as
// a function of the input (control) variables, and is the interface for users
// to describe their least squares problem to Ceres. In other words, this is the
// modelling layer between users and the Ceres optimizer. The signature of the
// function (number and sizes of input parameter blocks and number of outputs)
// is stored in parameter_block_sizes_ and num_residuals_ respectively. User
// code inheriting from this class is expected to set these two members with the
// corresponding accessors. This information will be verified by the Problem
// when added with AddResidualBlock().
//
// CostFunction "is-a" ConstraintFunction but has no explicit observation blocks.
// Here use protected inheritance to inherit the functionality and hide the functions
// related to observations.
class CERES_EXPORT CostFunction : protected RelationFunction {
public:
  CostFunction():RelationFunction() {}
  virtual ~CostFunction() {}

  using RelationFunction::parameter_block_sizes;
  using RelationFunction::num_residuals;

  // Inputs:
  //
  // parameters is an array of pointers to arrays containing the
  // various parameter blocks. parameters has the same number of
  // elements as parameter_block_sizes_.  Parameter blocks are in the
  // same order as parameter_block_sizes_.i.e.,
  //
  //   parameters_[i] = double[parameter_block_sizes_[i]]
  //
  // Outputs:
  //
  // residuals is an array of size num_residuals_.
  //
  // jacobians is an array of size parameter_block_sizes_ containing
  // pointers to storage for jacobian blocks corresponding to each
  // parameter block. Jacobian blocks are in the same order as
  // parameter_block_sizes, i.e. jacobians[i], is an
  // array that contains num_residuals_* parameter_block_sizes_[i]
  // elements. Each jacobian block is stored in row-major order, i.e.,
  //
  //   jacobians[i][r*parameter_block_size_[i] + c] =
  //                              d residual[r] / d parameters[i][c]
  //
  // If jacobians is NULL, then no derivatives are returned; this is
  // the case when computing cost only. If jacobians[i] is NULL, then
  // the jacobian block corresponding to the i'th parameter block must
  // not to be returned.
  //
  // The return value indicates whether the computation of the
  // residuals and/or jacobians was successful or not.
  //
  // This can be used to communicate numerical failures in jacobian
  // computations for instance.
  //
  // A more interesting and common use is to impose constraints on the
  // parameters. If the initial values of the parameter blocks satisfy
  // the constraints, then returning false whenever the constraints
  // are not satisfied will prevent the solver from moving into the
  // infeasible region. This is not a very sophisticated mechanism for
  // enforcing constraints, but is often good enough.
  //
  // Note that it is important that the initial values of the
  // parameter block must be feasible, otherwise the solver will
  // declare a numerical problem at iteration 0.
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians_p) const = 0;

  virtual bool Evaluate(double const* const* parameters,
                        double const* const* /*observations*/,
                        double* residuals,
                        double** jacobians_p,
                        double** /*jacobians_o*/) const {
    return Evaluate(parameters, residuals, jacobians_p);
  }

protected:
  using RelationFunction::mutable_parameter_block_sizes;
  using RelationFunction::set_num_residuals;

private:
  CERES_DISALLOW_COPY_AND_ASSIGN(CostFunction);
};
}  // namespace ceres

#include "ceres/internal/reenable_warnings.h"

#endif  // CERES_PUBLIC_COST_FUNCTION_H_
