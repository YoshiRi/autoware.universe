// Copyright 2026 TIER IV, Inc.
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
//
// Unit tests for `autoware/multi_object_tracker/kalman_filter_template.hpp`:
// KalmanFilterTemplate<StateSize,MeasurementSize> — predict, update, getter/setter, guards.
//
// All test fixtures use a 2-state (position, velocity) / 1-measurement (position) system:
//   x = [pos, vel],  A = [[1,dt],[0,1]],  C = [1,0],  dt = 1.0
// with a small process noise Q and measurement noise R so that the filter converges predictably.

#include "autoware/multi_object_tracker/kalman_filter_template.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace
{
using KF = autoware::multi_object_tracker::KalmanFilterTemplate<2, 1>;
using StateVec = KF::StateVec;
using StateMat = KF::StateMat;
using MeasVec = KF::MeasVec;
using MeasMat = KF::MeasMat;
using MeasModelMat = KF::MeasModelMat;

// Build a constant-velocity state-transition matrix for time step dt.
StateMat makeA(double dt = 1.0)
{
  StateMat A;
  A << 1.0, dt, 0.0, 1.0;
  return A;
}

// Build a diagonal process-noise matrix.
StateMat makeQ(double q_pos = 0.001, double q_vel = 0.001)
{
  StateMat Q = StateMat::Zero();
  Q(0, 0) = q_pos;
  Q(1, 1) = q_vel;
  return Q;
}

// Observation matrix: observe position only.
MeasModelMat makeC()
{
  MeasModelMat C;
  C << 1.0, 0.0;
  return C;
}

// Measurement noise.
MeasMat makeR(double r = 0.1)
{
  MeasMat R;
  R << r;
  return R;
}

// Build a fully initialized KF (initial state [0,0], identity covariance).
KF makeKF(const StateVec & x0 = StateVec::Zero(), const StateMat & P0 = StateMat::Identity())
{
  KF kf;
  kf.init(x0, makeA(), KF::ControlMat{}, makeC(), makeQ(), makeR(), P0);
  return kf;
}

}  // namespace

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

TEST(KalmanFilterTemplate, InitSetsStateAndCovariance)
{
  StateVec x0;
  x0 << 5.0, 2.0;
  StateMat P0 = 3.0 * StateMat::Identity();
  auto kf = makeKF(x0, P0);

  StateVec x_out;
  StateMat P_out;
  kf.getX(x_out);
  kf.getP(P_out);
  EXPECT_NEAR(x_out(0), 5.0, 1e-12);
  EXPECT_NEAR(x_out(1), 2.0, 1e-12);
  EXPECT_NEAR(P_out(0, 0), 3.0, 1e-12);
  EXPECT_NEAR(P_out(1, 1), 3.0, 1e-12);
}

TEST(KalmanFilterTemplate, GetXelementMatchesGetX)
{
  StateVec x0;
  x0 << 7.0, -3.0;
  auto kf = makeKF(x0);
  EXPECT_NEAR(kf.getXelement(0), 7.0, 1e-12);
  EXPECT_NEAR(kf.getXelement(1), -3.0, 1e-12);
}

// ---------------------------------------------------------------------------
// predict(): state and covariance propagation
// ---------------------------------------------------------------------------

TEST(KalmanFilterTemplate, PredictAdvancesStateByA)
{
  // x = [2, 1], A = [[1,1],[0,1]]: after one step x_new = [3, 1].
  StateVec x0;
  x0 << 2.0, 1.0;
  auto kf = makeKF(x0);
  ASSERT_TRUE(kf.predict());

  StateVec x_new;
  kf.getX(x_new);
  EXPECT_NEAR(x_new(0), 3.0, 1e-12);
  EXPECT_NEAR(x_new(1), 1.0, 1e-12);
}

TEST(KalmanFilterTemplate, PredictIncreasesUncertainty)
{
  // Without an update, P grows at each prediction step (A*P*A.T + Q ≥ P entry-wise for typical A).
  auto kf = makeKF(StateVec::Zero(), StateMat::Identity());
  StateMat P_before;
  kf.getP(P_before);

  ASSERT_TRUE(kf.predict());

  StateMat P_after;
  kf.getP(P_after);
  // The X_X variance must grow because the position uncertainty accumulates velocity uncertainty.
  EXPECT_GT(P_after(0, 0), P_before(0, 0));
}

TEST(KalmanFilterTemplate, PredictWithExplicitNextStateOverridesDefaultAdvance)
{
  // predict(x_next, A): uses the caller-supplied x_next rather than computing A*x.
  StateVec x0;
  x0 << 1.0, 0.5;
  auto kf = makeKF(x0);

  StateVec x_next;
  x_next << 99.0, 99.0;
  ASSERT_TRUE(kf.predict(x_next, makeA()));

  StateVec x_out;
  kf.getX(x_out);
  EXPECT_NEAR(x_out(0), 99.0, 1e-12);
  EXPECT_NEAR(x_out(1), 99.0, 1e-12);
}

// ---------------------------------------------------------------------------
// update(): Kalman measurement correction
// ---------------------------------------------------------------------------

TEST(KalmanFilterTemplate, UpdateReturnsTrueForValidInputs)
{
  auto kf = makeKF();
  ASSERT_TRUE(kf.predict());
  MeasVec y;
  y << 1.0;
  EXPECT_TRUE(kf.update(y));
}

TEST(KalmanFilterTemplate, UpdateWithExactMeasurementLeavesStateUnchanged)
{
  // If y == C * x (measurement matches prediction exactly), the Kalman correction is zero.
  StateVec x0;
  x0 << 3.0, 1.0;
  auto kf = makeKF(x0);
  ASSERT_TRUE(kf.predict());  // x_pred = [4, 1]

  StateVec x_pred;
  kf.getX(x_pred);
  MeasVec y = makeC() * x_pred;  // exact measurement
  ASSERT_TRUE(kf.update(y));

  StateVec x_out;
  kf.getX(x_out);
  EXPECT_NEAR(x_out(0), x_pred(0), 1e-10);
  EXPECT_NEAR(x_out(1), x_pred(1), 1e-10);
}

TEST(KalmanFilterTemplate, UpdateMovesStateTowardMeasurement)
{
  // A measurement below the predicted state must pull the estimate downward.
  StateVec x0;
  x0 << 5.0, 0.0;
  auto kf = makeKF(x0);
  ASSERT_TRUE(kf.predict());  // x_pred = [5, 0]

  MeasVec y;
  y << 3.0;  // measured position below predicted
  ASSERT_TRUE(kf.update(y));

  StateVec x_out;
  kf.getX(x_out);
  // After the update, position estimate must be between 3 and 5 (pulled toward measurement).
  EXPECT_LT(x_out(0), 5.0);
  EXPECT_GT(x_out(0), 3.0);
}

TEST(KalmanFilterTemplate, UpdateDecreasesPosteriorUncertainty)
{
  // Adding a measurement must reduce (or maintain) the covariance trace relative to the prior.
  auto kf = makeKF(StateVec::Zero(), StateMat::Identity());
  ASSERT_TRUE(kf.predict());

  StateMat P_prior;
  kf.getP(P_prior);

  MeasVec y;
  y << 1.0;
  ASSERT_TRUE(kf.update(y));

  StateMat P_posterior;
  kf.getP(P_posterior);
  EXPECT_LT(P_posterior.trace(), P_prior.trace());
}

TEST(KalmanFilterTemplate, UpdatePreservesSymmetryOfCovariance)
{
  // The Joseph-form update must keep P numerically symmetric.
  auto kf = makeKF(StateVec::Zero(), StateMat::Identity());
  ASSERT_TRUE(kf.predict());
  MeasVec y;
  y << 0.5;
  ASSERT_TRUE(kf.update(y));

  StateMat P_out;
  kf.getP(P_out);
  EXPECT_NEAR(P_out(0, 1), P_out(1, 0), 1e-14);
}

TEST(KalmanFilterTemplate, UpdateReturnsFalseForNaNState)
{
  // If any state element is NaN, update must fail gracefully and return false.
  auto kf = makeKF();
  StateVec x_nan;
  x_nan << std::numeric_limits<double>::quiet_NaN(), 0.0;
  StateMat P_id = StateMat::Identity();
  kf.init(x_nan, P_id);

  MeasVec y;
  y << 1.0;
  EXPECT_FALSE(kf.update(y));
}

TEST(KalmanFilterTemplate, UpdateReturnsFalseForNaNMeasurement)
{
  auto kf = makeKF();
  ASSERT_TRUE(kf.predict());
  MeasVec y_nan;
  y_nan << std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(kf.update(y_nan));
}

// ---------------------------------------------------------------------------
// setA / setQ / setC / setR
// ---------------------------------------------------------------------------

TEST(KalmanFilterTemplate, SetAReplacesPreviousTransitionMatrix)
{
  // Override A with a "stay still" matrix ([[1,0],[0,1]] with dt=0): prediction must not advance
  // position.
  StateVec x0_stay;
  x0_stay << 2.0, 1.0;
  auto kf = makeKF(x0_stay);
  StateMat A_still = StateMat::Identity();
  kf.setA(A_still);
  ASSERT_TRUE(kf.predict());

  StateVec x_out;
  kf.getX(x_out);
  EXPECT_NEAR(x_out(0), 2.0, 1e-12);
  EXPECT_NEAR(x_out(1), 1.0, 1e-12);
}

TEST(KalmanFilterTemplate, SetQLargerProcessNoiseIncreasesCovarianceGrowthRate)
{
  // After one prediction step, the covariance must be larger when Q is large than when Q is small.
  StateVec x0 = StateVec::Zero();
  StateMat P0 = StateMat::Identity();

  auto kf_small_q = makeKF(x0, P0);
  StateMat Q_large = 100.0 * StateMat::Identity();
  auto kf_large_q = makeKF(x0, P0);
  kf_large_q.setQ(Q_large);

  ASSERT_TRUE(kf_small_q.predict());
  ASSERT_TRUE(kf_large_q.predict());

  StateMat P_small, P_large;
  kf_small_q.getP(P_small);
  kf_large_q.getP(P_large);
  EXPECT_LT(P_small.trace(), P_large.trace());
}

// ---------------------------------------------------------------------------
// Multi-step convergence
// ---------------------------------------------------------------------------

TEST(KalmanFilterTemplate, RepeatedPredictUpdateConvergesPositionEstimate)
{
  // Simulate 30 steps of a stationary object at position 0. After convergence, the position
  // estimate and its variance should both be small.
  constexpr double true_position = 0.0;
  constexpr int num_steps = 30;

  StateVec x0;
  x0 << 5.0, 0.0;  // start off at position 5, but truth is 0
  auto kf = makeKF(x0, 10.0 * StateMat::Identity());

  // Use a nearly-zero-velocity transition (object is stationary).
  StateMat A_stationary = StateMat::Identity();
  kf.setA(A_stationary);

  for (int i = 0; i < num_steps; ++i) {
    ASSERT_TRUE(kf.predict());
    MeasVec y;
    y << true_position;
    ASSERT_TRUE(kf.update(y));
  }

  StateVec x_final;
  StateMat P_final;
  kf.getX(x_final);
  kf.getP(P_final);

  EXPECT_NEAR(x_final(0), true_position, 0.1);
  EXPECT_LT(P_final(0, 0), 0.1);  // uncertainty has collapsed
}
