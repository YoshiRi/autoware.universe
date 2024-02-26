// Copyright 2018-2021 The Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MPC_LATERAL_CONTROLLER__QP_SOLVER__QP_SOLVER_UNCONSTRAINT_FAST_HPP_
#define MPC_LATERAL_CONTROLLER__QP_SOLVER__QP_SOLVER_UNCONSTRAINT_FAST_HPP_

#include "mpc_lateral_controller/qp_solver/qp_solver_interface.hpp"

namespace autoware::motion::control::mpc_lateral_controller
{

/**
 * Solver for QP problems using least square decomposition
 * implement with Eigen's standard Cholesky decomposition (LLT)
 */
class QPSolverEigenLeastSquareLLT : public QPSolverInterface
{
public:
  /**
   * @brief constructor
   */
  QPSolverEigenLeastSquareLLT();

  /**
   * @brief destructor
   */
  ~QPSolverEigenLeastSquareLLT() = default;

  /**
   * @brief solve QP problem : minimize j = u' * h_mat * u + f_vec' * u without constraint
   * @param [in] h_mat parameter matrix in object function
   * @param [in] f_vec parameter matrix in object function
   * @param [in] a parameter matrix for constraint lb_a < a*u < ub_a (not used here)
   * @param [in] lb parameter matrix for constraint lb < U < ub (not used here)
   * @param [in] ub parameter matrix for constraint lb < U < ub (not used here)
   * @param [in] lb_a parameter matrix for constraint lb_a < a*u < ub_a (not used here)
   * @param [in] ub_a parameter matrix for constraint lb_a < a*u < ub_a (not used here)
   * @param [out] u optimal variable vector
   * @return true if the problem was solved
   */
  bool solve(
    const Eigen::MatrixXd & h_mat, const Eigen::MatrixXd & f_vec, const Eigen::MatrixXd & a,
    const Eigen::VectorXd & lb, const Eigen::VectorXd & ub, const Eigen::VectorXd & lb_a,
    const Eigen::VectorXd & ub_a, Eigen::VectorXd & u) override;

  int64_t getTakenIter() const override { return 1; };
  // TODO(someone): calculate runtime and object value, otherwise just return 0 by base class
  // double getRunTime() const override { return 0.0; }
  // double getObjVal() const override { return 0.0; }

private:
};
}  // namespace autoware::motion::control::mpc_lateral_controller
#endif  // MPC_LATERAL_CONTROLLER__QP_SOLVER__QP_SOLVER_UNCONSTRAINT_FAST_HPP_
