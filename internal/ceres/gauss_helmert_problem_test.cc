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

#include "ceres/autodiff_gauss_helmert_constraint_function.h"
#include "ceres/gauss_helmert_constraint_function.h"
#include "ceres/gauss_helmert_problem_impl.h"
#include "ceres/problem.h"
#include "ceres/problem_impl.h"

#include "ceres/casts.h"
#include "ceres/cost_function.h"
#include "ceres/crs_matrix.h"
#include "ceres/evaluator_test_utils.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/local_parameterization.h"
#include "ceres/loss_function.h"
#include "ceres/map_util.h"
#include "ceres/parameter_block.h"
#include "ceres/program.h"
#include "ceres/sized_cost_function.h"
#include "ceres/sparse_matrix.h"
#include "ceres/types.h"
#include "gtest/gtest.h"

#include "eigen3/Eigen/Eigen"

namespace ceres {
namespace internal {

using std::vector;

// Trivial constraint function that accepts one single argument and one single observation.
class EqualityConstraintFunction : public GaussHelmertConstraintFunction {
 public:
  EqualityConstraintFunction(int num_residuals, int32 parameter_block_size, int32 observation_block_size) {
    set_num_residuals(num_residuals);
    mutable_parameter_block_sizes()->push_back(parameter_block_size);
    mutable_observation_block_sizes()->push_back(observation_block_size);
  }
  virtual ~EqualityConstraintFunction() {}

  virtual bool Evaluate(double const* const* parameters,    //
                        double const* const* observations,  //
                        double* residuals,                  //
                        double** jacobians_p, double** jacobians_o) const {
    for (int i = 0; i < num_residuals(); ++i) {
      residuals[i] = 1;
    }
    return true;
  }
};

struct EqualityConstraintFunctor {
  template <typename T>
  bool operator()(T const X[4], T const L[4], T cost[4]) const {
    for (size_t i = 0; i < 4; i++)
      cost[i] = X[i] - L[i];
    return true;
  }

  static GaussHelmertConstraintFunction* create() {
    return new AutoDiffGaussHelmertConstraintFunction<EqualityConstraintFunctor, 4, 1, 4, 4>(
        new EqualityConstraintFunctor());
  }
};

TEST(GHProblem, AutoDiffGaussHelmertConstraintFunction) {
  double x[4] = {0, 0, 0, 0};
  double l[4] = {1, 2, 3, 4};
  double err[4];

  double a[4][4];
  double* jacobians_p[4] = {a[0], a[1], a[2], a[3]};
  Eigen::Map<Eigen::Matrix4d> A(&a[0][0]);

  double b[4][4];
  double* jacobians_o[4] = {b[0], b[1], b[2], b[3]};
  Eigen::Map<Eigen::Matrix4d> B(&b[0][0]);

  double* parameters[1] = {x};
  double* observations[1] = {l};

  GaussHelmertConstraintFunction* constraint_function = EqualityConstraintFunctor::create();

  constraint_function->Evaluate(parameters, observations, err, jacobians_p, NULL);
  std::cout << A << std::endl;

  constraint_function->Evaluate(parameters, observations, err, NULL, jacobians_o);
  std::cout << B << std::endl;

  std::cout << Eigen::Map<Eigen::Vector4d>(err) << std::endl;

  constraint_function->Evaluate(parameters, observations, err, jacobians_p, jacobians_o);

  EXPECT_TRUE(A.isApprox(Eigen::Matrix4d::Identity(), 1e-4));
  EXPECT_TRUE(B.isApprox(-Eigen::Matrix4d::Identity(), 1e-4));
}

//TEST(GHProblem, dev) {
//  GaussHelmertProblemImpl problem;
//  double x = 0;
//  double l = 1;
//  std::vector<double*> parameters;
//  parameters.push_back(&x);
//  std::vector<double*> observations;
//  observations.push_back(&l);
//  problem.AddConstraintBlock(new EqualityConstraintFunction(1, 1, 1), NULL, parameters, observations);
//}

}  // namespace internal
}  // namespace ceres
