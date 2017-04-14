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

#include "ceres/GH_parameter_block.h"
#include "ceres/constraint_block.h"
#include "ceres/GH_program.h"
#include "ceres/GH_problem.h"
#include "ceres/GH_program_evaluator.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/GH_block_jacobian_writer.h"

#include "ceres/local_parameterization.h"

#include <omp.h>
#include "Eigen/SparseCore"
#include "Eigen/SparseQR"
#include "Eigen/SVD"
#include "ceres/suitesparse.h"
#include "ceres/compressed_col_sparse_matrix_utils.h"
#include "ceres/compressed_row_sparse_matrix.h"

#include <glog/logging.h>

namespace ceres {

using namespace ceres::internal;
using std::vector;

// Trivial constraint function that accepts one single argument and one single observation.
class EqualityConstraintFunction : public GaussHelmertConstraintFunction {
 public:
  EqualityConstraintFunction() {
    set_num_residuals(4);
    mutable_parameter_block_sizes()->push_back(4);
    mutable_observation_block_sizes()->push_back(4);
  }
  virtual ~EqualityConstraintFunction() {}

  virtual bool Evaluate(double const* const* parameters,    //
                        double const* const* observations,  //
                        double* residuals,                  //
                        double** jacobians_p, double** jacobians_o) const {
    CHECK_NOTNULL(residuals);
    VectorRef(residuals, 4) = ConstVectorRef(parameters[0], 4) - ConstVectorRef(observations[0], 4);

    const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();
    if (jacobians_p != NULL) {
        for(int i=0; i<4; i++) {
            if (jacobians_p[i] != NULL){
                VectorRef(jacobians_p[i], 4) = I4.row(i);
            }
        }
    }
    if (jacobians_o != NULL) {
        for(int i=0; i<4; i++) {
            if (jacobians_o[i] != NULL){
                VectorRef(jacobians_o[i], 4) = -( I4.row(i) );
            }
        }
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

struct AffineConstraintFunctor {
  AffineConstraintFunctor(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
      :A_(A), B_(B) {
      CHECK( A_.rows()==4 & A_.cols()==4 & B_.rows()==4 & B_.cols()==4)
              << "Wrong matrix size, should be 4-by-4";
   }

  template <typename T>
  bool operator()(T const X[4], T const L[4], T cost[4]) const {
    Eigen::Map<const Eigen::Matrix<T,4,1> > X_(X), L_(L);
    Eigen::Map<Eigen::Matrix<T,4,1> >cost_(cost);
    cost_ = A_.cast<T>() * X_ + B_.cast<T>() * L_;
    return true;
  }

  static GaussHelmertConstraintFunction* create(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) {
    return new AutoDiffGaussHelmertConstraintFunction<AffineConstraintFunctor, 4, 1, 4, 4>(
        new AffineConstraintFunctor(A, B));
  }
  protected:
  Eigen::MatrixXd A_, B_; //

};

class GHProblemTest : public ::testing::Test {
  protected:

  virtual void SetUp() {
    VectorRef(x[0], 4) << 0, 0, 0, 0;
    VectorRef(l[0], 4) << 1, 1, 1, 1;
    VectorRef(l[1], 4) << 2, 2, 2, 2;

    p.push_back(x[0]);
    o1.push_back(l[0]);
    o2.push_back(l[1]);

    affine_A = Eigen::MatrixXd(4,4);
    affine_A << 1, 0, 0, 0,
                0, 2, 0, 0,
                0, 0, 3, 0,
                0, 0, 0, 4;

    affine_B = Eigen::MatrixXd(4,4);
    affine_B << 1, 2, 3, 4,
                0, 5, 6, 7,
                0, 0, 8, 9,
                0, 0, 0, 10;
    expect_residual_affine = (Vector(4)<< 10, 18, 17, 10).finished();
    expect_cost_affine  =  0.5*expect_residual_affine.squaredNorm();
  }
    double x[1][4];
    double l[2][4];
    std::vector<double*> p;
    std::vector<double*> o1;
    std::vector<double*> o2;
    Matrix affine_A, affine_B;
    Eigen::VectorXd expect_residual_affine;
    double expect_cost_affine;
};

TEST_F(GHProblemTest, GaussHelmertConstraintFunction)
{
  double residuals[4];

  double a[4][4];
  double* jacobians_p[4] = {a[0], a[1], a[2], a[3]};
  Eigen::Map<Matrix> A(&a[0][0],4,4);

  double b[4][4];
  double* jacobians_o[4] = {b[0], b[1], b[2], b[3]};
  Eigen::Map<Matrix> B(&b[0][0],4,4);

  double* parameters[1] = {x[0]};
  double* observations[1] = {l[0]};

  GaussHelmertConstraintFunction* constraint_function = new EqualityConstraintFunction();

  constraint_function->Evaluate(parameters, observations, residuals, NULL, NULL);
  EXPECT_TRUE( ( ConstVectorRef(residuals, 4).array() == -ConstVectorRef(l[0], 4).array() ).all() );

  constraint_function->Evaluate(parameters, observations, residuals, jacobians_p, NULL);
  EXPECT_TRUE( ( A.array() == Eigen::Matrix4d::Identity().array() ).all() );

  constraint_function->Evaluate(parameters, observations, residuals, NULL, jacobians_o);
  EXPECT_TRUE( ( B.array() == -Eigen::Matrix4d::Identity().array() ).all() );

  VectorRef(residuals, 4).setZero(); A.setZero(); B.setZero();
  constraint_function->Evaluate(parameters, observations, residuals, jacobians_p, jacobians_o);
  EXPECT_TRUE( ( ConstVectorRef(residuals, 4).array() == -ConstVectorRef(l[0], 4).array() ).all() );
  EXPECT_TRUE( ( A.array() == Eigen::Matrix4d::Identity().array() ).all() );
  EXPECT_TRUE( ( B.array() == -Eigen::Matrix4d::Identity().array() ).all() );
}


TEST_F(GHProblemTest, AutoDiffGaussHelmertConstraintFunction)
{
  double residuals[4];

  double a[4][4];
  double* jacobians_p[4] = {a[0], a[1], a[2], a[3]};
  MatrixRef A(&a[0][0],4,4);

  double b[4][4];
  double* jacobians_o[4] = {b[0], b[1], b[2], b[3]};
  MatrixRef B(&b[0][0],4,4);

  double* parameters[1] = {x[0]};
  double* observations[1] = {l[0]};

  GaussHelmertConstraintFunction* constraint_function = AffineConstraintFunctor::create(affine_A, affine_B);

  constraint_function->Evaluate(parameters, observations, residuals, NULL, NULL);
  EXPECT_TRUE( ( ConstVectorRef(residuals, 4)- expect_residual_affine ).squaredNorm() < 1e-10 );

  constraint_function->Evaluate(parameters, observations, residuals, jacobians_p, NULL);
  EXPECT_TRUE( ( A - affine_A ).squaredNorm() < 1e-10 );

  constraint_function->Evaluate(parameters, observations, residuals, NULL, jacobians_o);
  EXPECT_TRUE( ( B - affine_B ).squaredNorm() < 1e-10 );

  VectorRef(residuals, 4).setZero(); A.setZero(); B.setZero();
  constraint_function->Evaluate(parameters, observations, residuals, jacobians_p, jacobians_o);
  EXPECT_TRUE( ( ConstVectorRef(residuals, 4)- expect_residual_affine ).squaredNorm() < 1e-10 );
  EXPECT_TRUE( ( A - affine_A ).squaredNorm() < 1e-10 );
  EXPECT_TRUE( ( B - affine_B ).squaredNorm() < 1e-10 );
}

TEST_F(GHProblemTest, GHConstraintBlock)
{
    double residuals[4];
    double cost;

    GHParameterBlock   p0(x[0], 4,0);
    GHObservationBlock o0(l[0], 4,0);

    std::vector<GHParameterBlock*> p;
    p.push_back(&p0);

    std::vector<GHObservationBlock*> o;
    o.push_back(&o0);

    GHConstraintBlock block(AffineConstraintFunctor::create(affine_A, affine_B), NULL, p, o, 0);
    EXPECT_EQ(block.NumObservationBlocks(),  1);
    EXPECT_EQ(block.NumParameterBlocks(),  1);
    EXPECT_EQ(block.NumResiduals(),  4);
    EXPECT_EQ(block.NumScratchDoublesForEvaluate(),4);
    double a[4][4];
    double* jacobians_p[4] = {a[0], a[1], a[2], a[3]};
    MatrixRef A(&a[0][0],4,4);

    double b[4][4];
    double* jacobians_o[4] = {b[0], b[1], b[2], b[3]};
    MatrixRef B(&b[0][0],4,4);
    scoped_ptr<double> scratch(new double[block.NumScratchDoublesForEvaluate()]);

    block.Evaluate(false, &cost, residuals, jacobians_p, jacobians_o, scratch.get());
    EXPECT_EQ(cost, expect_cost_affine);
    EXPECT_TRUE( ( ConstVectorRef(residuals, 4)- expect_residual_affine ).squaredNorm() < 1e-10 );
    EXPECT_TRUE( ( A - affine_A ).squaredNorm() < 1e-10 );
    EXPECT_TRUE( ( B - affine_B ).squaredNorm() < 1e-10 );
}

TEST_F(GHProblemTest, GHObservationBlock)
{
    GHObservationBlock o0(l[0],4,0);
    EXPECT_FALSE(o0.HasCovariance());

    o0.SetCovariance(Eigen::Matrix4d::Identity());
    EXPECT_TRUE(o0.HasCovariance());
    EXPECT_TRUE((o0.Covariance().array()==Eigen::Matrix4d::Identity().array()).all());

    EXPECT_DEATH_IF_SUPPORTED( o0.SetCovariance( Eigen::Matrix3d::Identity() ), "the size of covariance matrix does not match" );
}

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> EigenSparseMatrix;

void CRSMatrixToDenseMatrix(const CRSMatrix& csr_matrix, Matrix* dense_matrix) {

  CHECK_NOTNULL(dense_matrix);
  dense_matrix->resize(csr_matrix.num_rows, csr_matrix.num_cols);
  dense_matrix->setZero();

  for (int r = 0; r < csr_matrix.num_rows; ++r) {
    for (int idx = csr_matrix.rows[r]; idx < csr_matrix.rows[r + 1]; ++idx) {
      (*dense_matrix)(r, csr_matrix.cols[idx]) = csr_matrix.values[idx];
    }
  }
}

TEST_F(GHProblemTest, GHProblemEvaluate)
{
    GHProblem problem;
    problem.AddConstraintBlock(EqualityConstraintFunctor::create(),NULL, p, o1);
    problem.AddConstraintBlock(EqualityConstraintFunctor::create(),NULL, p, o2);

    double cost;
    std::vector<double> residual, gradient_p, gradient_o;
    CRSMatrix A,B;
    Matrix A_dense, B_dense;

    problem.Evaluate(GHProblem::EvaluateOptions(),
                     &cost, &residual,
                     &gradient_p, &gradient_o,
                     &A, &B);
    EXPECT_EQ(cost, 10);
    CRSMatrixToDenseMatrix(A, &A_dense);
    CRSMatrixToDenseMatrix(B, &B_dense);
    Eigen::MatrixXd EXP_A(8,4);
    EXP_A << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1,
             1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;
    EXPECT_TRUE( ( A_dense.array() == EXP_A.array() ).all() );
    EXPECT_TRUE( ( B_dense.array() == -Eigen::Matrix<double,8,8>::Identity().array() ).all() );


    problem.SetParameterBlockConstant(x[0]);
    problem.Evaluate(GHProblem::EvaluateOptions(),
                     NULL, NULL,
                     NULL, NULL,
                     &A, NULL);
    CRSMatrixToDenseMatrix(A, &A_dense);
    EXPECT_TRUE( ( A_dense.array() == Eigen::Matrix<double,8,4>::Zero().array() ).all() );

    problem.SetObservationBlockConstant(l[1]);
    problem.Evaluate(GHProblem::EvaluateOptions(),
                     NULL, NULL,
                     NULL, NULL,
                     NULL, &B );
    CRSMatrixToDenseMatrix(B, &B_dense);
    Eigen::MatrixXd EXP_B(8,8);
    EXP_B << -1,  0,  0,  0,  0,  0,  0,  0,
              0, -1,  0,  0,  0,  0,  0,  0,
              0,  0, -1,  0,  0,  0,  0,  0,
              0,  0,  0, -1,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0;
    EXPECT_TRUE( ( B_dense.array() == EXP_B.array() ).all() );
}

TEST_F(GHProblemTest, GHProblemEvaluateWithParameterization)
{
    GHProblem problem;
    problem.AddConstraintBlock(AffineConstraintFunctor::create(affine_A, affine_B),NULL, p, o1);
    problem.AddConstraintBlock(AffineConstraintFunctor::create(affine_A, affine_B),NULL, p, o2);

    std::vector<int> const_x_idx;
    const_x_idx.push_back(2);
    problem.SetParameterization(x[0], new SubsetParameterization(4, const_x_idx));

    double cost;
    std::vector<double> residual, gradient_p, gradient_o;
    CRSMatrix A,B;
    Matrix A_dense, B_dense;

    problem.Evaluate(GHProblem::EvaluateOptions(),
                     &cost, &residual,
                     &gradient_p, &gradient_o,
                     &A, &B);

    CRSMatrixToDenseMatrix(A, &A_dense);
    CRSMatrixToDenseMatrix(B, &B_dense);
    std::cout<< A_dense << std::endl;
    std::cout<< B_dense << std::endl;
}

// Given a upper triangular matrix R in compressed column form, solve
// the linear system,
//
//  R x = e_i
//
// which is used to calculate the i^th column of R^{-1}. e_i is a vector with
// e(rhs_nonzero_index) = 1 and all other entries zero.
inline void SolveInverseOfUpperTriangularInPlace(int num_cols,
                      const int* rows,
                      const int* cols,
                      const double* values,
                      const int rhs_nonzero_index,      // denoted as n in the following comment
                      double* solution) {
    std::fill(solution, solution + num_cols, 0.0);
    // Note: The n-th diagonal element in compressed column matrix is in the end of this n-th column,
    // i.e. head of next column minus one (idx = cols[n+1]-1).

    // solve x[n] = 1/R[n,n]
    solution[rhs_nonzero_index] = 1.0/values[cols[rhs_nonzero_index + 1] - 1];

    // Solve x[r], r={n-1,...,0}
    // x[r] = -sum(R[r,c]*x[c])/R[r,r], c={r+1,...,n}
    for(int c = rhs_nonzero_index; c > 0; --c) {
        for (int idx = cols[c]; idx < cols[c + 1] - 1; ++idx) {
          const int r = rows[idx];
          const double v = values[idx];
          solution[r] -= v * solution[c];
        }
        solution[c-1] = solution[c-1] / values[cols[c] - 1]; // diagonal element
    }
}


TEST_F(GHProblemTest, dev)
{
    GHProblem problem;
    problem.AddConstraintBlock(AffineConstraintFunctor::create(affine_A, affine_B),NULL, p, o1);
    problem.AddConstraintBlock(AffineConstraintFunctor::create(affine_A, affine_B),NULL, p, o2);

    GHProgram* program = problem.mutable_program();
    program->SetParameterOffsetsAndIndex();
    program->SetObservationOffsetsAndIndex();

    GHEvaluator::Options options;
    std::string error;
    options.linear_solver_type = DENSE_QR;
    scoped_ptr<GHEvaluator> evaluator(GHEvaluator::Create(options, program, &error));
    scoped_ptr<DenseSparseMatrix> A(down_cast<DenseSparseMatrix*>(evaluator->CreateJacobian_p()));
    scoped_ptr<DenseSparseMatrix> B(down_cast<DenseSparseMatrix*>(evaluator->CreateJacobian_o()));
    double cost;

    Vector residual(program->NumResiduals());
    Vector correction(program->NumEffectiveObservations());
    correction.setZero();
    for(int i=0; i<5; i++ ) {
        evaluator->Evaluate(&x[0][0], &l[0][0],
                &cost, residual.data(),
                NULL, NULL,
                A.get() , B.get());
        std::cout << "iter:" << i << " cost:" << cost << std::endl;

//        std::cout<< A->mutable_matrix() << std::endl;
//        std::cout<< B->mutable_matrix() << std::endl;
        Eigen::HouseholderQR< ColMajorMatrix > qr_solver(B->mutable_matrix().transpose());

        Vector c = B->mutable_matrix() * correction - residual;
//        std::cout<< c << std::endl;

        ColMajorMatrix J = qr_solver.matrixQR().template triangularView<Eigen::Upper>().transpose().solve(A->mutable_matrix());
//        std::cout<< J << std::endl;

        DenseSparseMatrix J_op(J);

        Vector cp = qr_solver.matrixQR().template triangularView<Eigen::Upper>().transpose().solve(c);
//        std::cout<< cp << std::endl;

        LinearSolver::Options solver_options;
        LinearSolver::PerSolveOptions pre_opt;
        solver_options.type = DENSE_QR;
        scoped_ptr<LinearSolver> solver(LinearSolver::Create(solver_options));
        Vector dx(program->NumEffectiveParameters());

        solver->Solve(&J_op, cp.data(), pre_opt, dx.data());
        LOG(INFO) << dx << std::endl;

        Vector dl = qr_solver.householderQ() * (cp - J*dx) - correction;
        std::cout << dl << std::endl;

        program->Plus_p(&x[0][0], dx.data(), &x[0][0]);
        program->Plus_o(&l[0][0], dl.data(), &l[0][0]);
        program->Plus_o(correction.data(), dl.data(), correction.data());
    }

#if 0
    options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
    scoped_ptr<GHEvaluator> evaluator(GHEvaluator::Create(options, program, &error));
    scoped_ptr<CompressedRowSparseMatrix> A(down_cast<CompressedRowSparseMatrix*>(evaluator->CreateJacobian_p()));
    scoped_ptr<CompressedRowSparseMatrix> B(down_cast<CompressedRowSparseMatrix*>(evaluator->CreateJacobian_o()));

    double cost;
    evaluator->Evaluate(&x[0][0], &l[0][0],
                        &cost, NULL,
                        NULL, NULL,
                        A.get() , B.get());

    typedef Eigen::SparseMatrix<double, Eigen::ColMajor> EigenSparseMatrix;

    // Convert the matrix to column major order as required by SparseQR.
    EigenSparseMatrix sparse_jacobian_transpose =
        Eigen::MappedSparseMatrix<double, Eigen::RowMajor>(
            B->num_rows(), B->num_cols(), B->num_nonzeros(),
            B->mutable_rows(), B->mutable_cols(), B->mutable_values()).transpose();

    // A*P = Q*R,
    // P:=colsPermutation() for fill-reducing,
    // Q:=matrixQ() orthogonal matrix,
    // R:=matrixR().topLeftCorner(rank(), rank())
    Eigen::SparseQR<EigenSparseMatrix, Eigen::COLAMDOrdering<int> >
        qr_solver(sparse_jacobian_transpose);

    if (qr_solver.info() != Eigen::Success) {
      LOG(ERROR) << "Eigen::SparseQR decomposition failed.";
    }

    if (qr_solver.rank() < B->num_cols()) {
      LOG(ERROR) << "Jacobian matrix is rank deficient. "
                 << "Number of columns: " << B->num_cols()
                 << " rank: " << qr_solver.rank();
    }
    // Compute the inverse column permutation used by QR factorization.
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> inverse_permutation =
        qr_solver.colsPermutation().inverse();

    const int num_cols = qr_solver.matrixR().cols();
    Eigen::MatrixXd R_inv(num_cols, num_cols);

    // The following loop exploits the fact that the i^th column of A^{-1}
    // is given by the solution to the linear system
    //
    //  A x = e_i
    //
    // where e_i is a vector with e(i) = 1 and all other entries zero.
//  #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int c = 0; c < num_cols; ++c) {
      double* solution_colum = R_inv.col(c).data();
      SolveInverseOfUpperTriangularInPlace(
          num_cols,
          qr_solver.matrixR().innerIndexPtr(),  // rows
          qr_solver.matrixR().outerIndexPtr(),  // cols
          &qr_solver.matrixR().data().value(0), // values
          c,
          solution_colum);
    }
    std::cout<< R_inv << std::endl;
    std::cout<< qr_solver.matrixR().toDense()*R_inv << std::endl;
#endif
}


}  // namespace ceres
