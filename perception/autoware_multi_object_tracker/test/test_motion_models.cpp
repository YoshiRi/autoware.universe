// Copyright 2024 TIER IV, Inc.
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

#define EIGEN_MPL2_ONLY

#include "autoware/multi_object_tracker/tracker/motion_model/bicycle_motion_model.hpp"
#include "autoware/multi_object_tracker/tracker/motion_model/ctrv_motion_model.hpp"
#include "autoware/multi_object_tracker/tracker/motion_model/cv_motion_model.hpp"
#include "autoware/multi_object_tracker/tracker/motion_model/static_motion_model.hpp"

#include <autoware_utils_geometry/msg/covariance.hpp>
#include <rclcpp/rclcpp.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace
{
using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
using autoware::multi_object_tracker::BicycleMotionModel;
using autoware::multi_object_tracker::CTRVMotionModel;
using autoware::multi_object_tracker::CVMotionModel;
using autoware::multi_object_tracker::StaticMotionModel;

std::array<double, 36> zeroCov()
{
  std::array<double, 36> c{};
  return c;
}

std::array<double, 36> diagPoseCov(double xx = 1.0, double yy = 1.0, double yawyaw = 0.1)
{
  auto c = zeroCov();
  c[XYZRPY_COV_IDX::X_X] = xx;
  c[XYZRPY_COV_IDX::Y_Y] = yy;
  c[XYZRPY_COV_IDX::YAW_YAW] = yawyaw;
  return c;
}

rclcpp::Time toTime(double seconds)
{
  return rclcpp::Time(static_cast<int64_t>(seconds * 1e9));
}
}  // namespace

// =============================================================================
// StaticMotionModel
// =============================================================================

TEST(StaticMotionModel, CheckInitializedFalseBeforeInit)
{
  StaticMotionModel m;
  EXPECT_FALSE(m.checkInitialized());
}

TEST(StaticMotionModel, InitializeSetsXY)
{
  StaticMotionModel m;
  ASSERT_TRUE(m.initialize(toTime(0), 3.0, -2.0, diagPoseCov()));
  EXPECT_DOUBLE_EQ(m.getStateElement(StaticMotionModel::IDX::X), 3.0);
  EXPECT_DOUBLE_EQ(m.getStateElement(StaticMotionModel::IDX::Y), -2.0);
}

TEST(StaticMotionModel, PredictDoesNotChangePosition)
{
  StaticMotionModel m;
  m.initialize(toTime(0), 5.0, 7.0, diagPoseCov());
  m.predictState(toTime(10.0));
  EXPECT_DOUBLE_EQ(m.getStateElement(StaticMotionModel::IDX::X), 5.0);
  EXPECT_DOUBLE_EQ(m.getStateElement(StaticMotionModel::IDX::Y), 7.0);
}

TEST(StaticMotionModel, PredictReturnsFalseBeforeInit)
{
  StaticMotionModel m;
  EXPECT_FALSE(m.predictState(toTime(1.0)));
}

TEST(StaticMotionModel, AdjustPositionShiftsXY)
{
  StaticMotionModel m;
  m.initialize(toTime(0), 1.0, 2.0, diagPoseCov());
  m.adjustPosition(3.0, -5.0);
  EXPECT_DOUBLE_EQ(m.getStateElement(StaticMotionModel::IDX::X), 4.0);
  EXPECT_DOUBLE_EQ(m.getStateElement(StaticMotionModel::IDX::Y), -3.0);
}

// =============================================================================
// CVMotionModel
// =============================================================================

TEST(CVMotionModel, CheckInitializedFalseBeforeInit)
{
  CVMotionModel m;
  EXPECT_FALSE(m.checkInitialized());
}

TEST(CVMotionModel, InitializeSetsState)
{
  CVMotionModel m;
  ASSERT_TRUE(m.initialize(toTime(0), 1.0, 2.0, diagPoseCov(), 3.0, -1.0, diagPoseCov()));
  EXPECT_DOUBLE_EQ(m.getStateElement(CVMotionModel::IDX::X), 1.0);
  EXPECT_DOUBLE_EQ(m.getStateElement(CVMotionModel::IDX::Y), 2.0);
  EXPECT_DOUBLE_EQ(m.getStateElement(CVMotionModel::IDX::VX), 3.0);
  EXPECT_DOUBLE_EQ(m.getStateElement(CVMotionModel::IDX::VY), -1.0);
}

TEST(CVMotionModel, PredictAdvancesPositionByVelocity)
{
  CVMotionModel m;
  // vx=5, vy=2; after 1.0s: x=5, y=2 (multi-step with dt_max=0.11 gives exact result)
  m.initialize(toTime(0), 0.0, 0.0, diagPoseCov(), 5.0, 2.0, diagPoseCov());
  m.predictState(toTime(1.0));
  EXPECT_NEAR(m.getStateElement(CVMotionModel::IDX::X), 5.0, 1e-9);
  EXPECT_NEAR(m.getStateElement(CVMotionModel::IDX::Y), 2.0, 1e-9);
}

TEST(CVMotionModel, PredictDoesNotChangeVelocity)
{
  CVMotionModel m;
  m.initialize(toTime(0), 0.0, 0.0, diagPoseCov(), 5.0, 2.0, diagPoseCov());
  m.predictState(toTime(1.0));
  EXPECT_NEAR(m.getStateElement(CVMotionModel::IDX::VX), 5.0, 1e-9);
  EXPECT_NEAR(m.getStateElement(CVMotionModel::IDX::VY), 2.0, 1e-9);
}

TEST(CVMotionModel, LimitStatesClampsTooHighVx)
{
  CVMotionModel m;
  // Default max_vx = 16.67 m/s
  m.initialize(toTime(0), 0.0, 0.0, diagPoseCov(), 100.0, 0.0, diagPoseCov());
  m.limitStates();
  EXPECT_NEAR(m.getStateElement(CVMotionModel::IDX::VX), 16.67, 1e-9);
}

TEST(CVMotionModel, LimitStatesClampsTooNegativeVy)
{
  CVMotionModel m;
  // Default max_vy = 16.67 m/s
  m.initialize(toTime(0), 0.0, 0.0, diagPoseCov(), 0.0, -100.0, diagPoseCov());
  m.limitStates();
  EXPECT_NEAR(m.getStateElement(CVMotionModel::IDX::VY), -16.67, 1e-9);
}

TEST(CVMotionModel, AdjustPositionShiftsXY)
{
  CVMotionModel m;
  m.initialize(toTime(0), 1.0, 2.0, diagPoseCov(), 0.0, 0.0, diagPoseCov());
  m.adjustPosition(3.0, -1.0);
  EXPECT_DOUBLE_EQ(m.getStateElement(CVMotionModel::IDX::X), 4.0);
  EXPECT_DOUBLE_EQ(m.getStateElement(CVMotionModel::IDX::Y), 1.0);
}

TEST(CVMotionModel, UpdateStatePoseMovesStateTowardObservation)
{
  CVMotionModel m;
  // P(X,X) = 4.0, observe x=2.0 with R=1.0
  // K_x = P_x / (P_x + R_x) = 4.0 / 5.0 = 0.8 → x_updated = 0 + 0.8 * 2.0 = 1.6
  auto init_cov = zeroCov();
  init_cov[XYZRPY_COV_IDX::X_X] = 4.0;
  init_cov[XYZRPY_COV_IDX::Y_Y] = 4.0;
  m.initialize(toTime(0), 0.0, 0.0, init_cov, 0.0, 0.0, zeroCov());

  auto obs_cov = zeroCov();
  obs_cov[XYZRPY_COV_IDX::X_X] = 1.0;
  obs_cov[XYZRPY_COV_IDX::Y_Y] = 1.0;
  ASSERT_TRUE(m.updateStatePose(2.0, 0.0, obs_cov));
  EXPECT_NEAR(m.getStateElement(CVMotionModel::IDX::X), 1.6, 1e-6);
  EXPECT_NEAR(m.getStateElement(CVMotionModel::IDX::Y), 0.0, 1e-9);
}

// =============================================================================
// CTRVMotionModel
// =============================================================================

TEST(CTRVMotionModel, CheckInitializedFalseBeforeInit)
{
  CTRVMotionModel m;
  EXPECT_FALSE(m.checkInitialized());
}

TEST(CTRVMotionModel, InitializeSetsState)
{
  CTRVMotionModel m;
  ASSERT_TRUE(m.initialize(toTime(0), 1.0, 2.0, M_PI / 4, diagPoseCov(), 3.0, 0.1, 0.05, 0.01));
  EXPECT_DOUBLE_EQ(m.getStateElement(CTRVMotionModel::IDX::X), 1.0);
  EXPECT_DOUBLE_EQ(m.getStateElement(CTRVMotionModel::IDX::Y), 2.0);
  EXPECT_DOUBLE_EQ(m.getStateElement(CTRVMotionModel::IDX::YAW), M_PI / 4);
  EXPECT_DOUBLE_EQ(m.getStateElement(CTRVMotionModel::IDX::VEL), 3.0);
  EXPECT_DOUBLE_EQ(m.getStateElement(CTRVMotionModel::IDX::WZ), 0.05);
}

TEST(CTRVMotionModel, PredictStraightLineAtZeroYawRate)
{
  CTRVMotionModel m;
  // yaw=0, wz=0, vel=5 → x += 5*1, y=0 unchanged after 1s
  m.initialize(toTime(0), 0.0, 0.0, 0.0, diagPoseCov(), 5.0, 0.1, 0.0, 0.01);
  m.predictState(toTime(1.0));
  EXPECT_NEAR(m.getStateElement(CTRVMotionModel::IDX::X), 5.0, 1e-9);
  EXPECT_NEAR(m.getStateElement(CTRVMotionModel::IDX::Y), 0.0, 1e-9);
  EXPECT_NEAR(m.getStateElement(CTRVMotionModel::IDX::YAW), 0.0, 1e-9);
}

TEST(CTRVMotionModel, PredictChangesYawByWzTimesDt)
{
  CTRVMotionModel m;
  // wz=0.1 rad/s, predict 1.0s; dt_max=0.11 → 10 steps of 0.1s
  // yaw = 0 + 0.1 * 0.1 * 10 = 0.10 rad (linear update, exact)
  m.initialize(toTime(0), 0.0, 0.0, 0.0, diagPoseCov(), 0.0, 0.1, 0.1, 0.01);
  m.predictState(toTime(1.0));
  EXPECT_NEAR(m.getStateElement(CTRVMotionModel::IDX::YAW), 0.1, 1e-9);
}

TEST(CTRVMotionModel, LimitStatesNormalizesYawToWithinPiRange)
{
  CTRVMotionModel m;
  m.initialize(toTime(0), 0.0, 0.0, M_PI + 0.5, diagPoseCov(), 0.0, 0.1, 0.0, 0.01);
  m.limitStates();
  const double yaw = m.getStateElement(CTRVMotionModel::IDX::YAW);
  EXPECT_GE(yaw, -M_PI);
  EXPECT_LE(yaw, M_PI);
}

TEST(CTRVMotionModel, LimitStatesClampsTooHighVelocity)
{
  CTRVMotionModel m;
  // Default max_vel = 2.78 m/s
  m.initialize(toTime(0), 0.0, 0.0, 0.0, diagPoseCov(), 100.0, 0.1, 0.0, 0.01);
  m.limitStates();
  EXPECT_NEAR(m.getStateElement(CTRVMotionModel::IDX::VEL), 2.78, 1e-9);
}

TEST(CTRVMotionModel, LimitStatesFlipsExcessiveReverseVelocity)
{
  // When vel < 0 AND vel < max_reverse_vel (-1.38), the tracker flips orientation:
  // vel = -vel, yaw += π. Assumes the object is actually moving forward with wrong heading.
  CTRVMotionModel m;
  m.initialize(toTime(0), 0.0, 0.0, 0.0, diagPoseCov(), -2.0, 0.1, 0.0, 0.01);
  m.limitStates();
  EXPECT_NEAR(m.getStateElement(CTRVMotionModel::IDX::VEL), 2.0, 1e-9);
  EXPECT_NEAR(std::abs(m.getStateElement(CTRVMotionModel::IDX::YAW)), M_PI, 1e-9);
}

TEST(CTRVMotionModel, LimitStatesDoesNotFlipModerateReverseVelocity)
{
  // vel = -1.0 is between 0 and max_reverse_vel (-1.38), so no flip
  CTRVMotionModel m;
  m.initialize(toTime(0), 0.0, 0.0, 0.0, diagPoseCov(), -1.0, 0.1, 0.0, 0.01);
  m.limitStates();
  EXPECT_NEAR(m.getStateElement(CTRVMotionModel::IDX::VEL), -1.0, 1e-9);
  EXPECT_NEAR(m.getStateElement(CTRVMotionModel::IDX::YAW), 0.0, 1e-9);
}

TEST(CTRVMotionModel, UpdateStatePoseHeadWrapsYawAcrossBoundary)
{
  // Estimated yaw=0, observation yaw = -(π-0.1) ≈ -3.04 rad.
  // The wrap loop brings fixed_yaw from -(π-0.1) to +(0.1) so the update
  // uses measurement ≈ +0.1 rather than the raw -3.04.
  CTRVMotionModel m;
  m.initialize(toTime(0), 0.0, 0.0, 0.0, diagPoseCov(), 1.0, 0.1, 0.0, 0.01);
  auto cov = diagPoseCov(0.01, 0.01, 0.01);
  ASSERT_TRUE(m.updateStatePoseHead(0.0, 0.0, -(M_PI - 0.1), cov));
  // After wrapping and KF update the yaw stays close to 0, not near -π
  EXPECT_NEAR(m.getStateElement(CTRVMotionModel::IDX::YAW), 0.0, 0.5);
}

// =============================================================================
// BicycleMotionModel
// =============================================================================

TEST(BicycleMotionModel, InitializeSetsWheelPositions)
{
  BicycleMotionModel m;
  // Default lr_ratio=0.25, lf_ratio=0.3; length=4 → lr=1.0, lf=1.2
  // center=(0,0), yaw=0 → rear x1=-1.0, front x2=1.2
  ASSERT_TRUE(m.initialize(toTime(0), 0.0, 0.0, 0.0, diagPoseCov(), 0.0, 0.1, 0.0, 0.01, 4.0));
  EXPECT_NEAR(m.getStateElement(BicycleMotionModel::IDX::X1), -1.0, 1e-9);
  EXPECT_NEAR(m.getStateElement(BicycleMotionModel::IDX::Y1), 0.0, 1e-9);
  EXPECT_NEAR(m.getStateElement(BicycleMotionModel::IDX::X2), 1.2, 1e-9);
  EXPECT_NEAR(m.getStateElement(BicycleMotionModel::IDX::Y2), 0.0, 1e-9);
}

TEST(BicycleMotionModel, GetYawStateReturnsHeadingAngle)
{
  BicycleMotionModel m;
  m.initialize(toTime(0), 0.0, 0.0, 0.0, diagPoseCov(), 0.0, 0.1, 0.0, 0.01, 4.0);
  EXPECT_NEAR(m.getYawState(), 0.0, 1e-9);
}

TEST(BicycleMotionModel, PredictAdvancesWheelPositionsAlongHeading)
{
  BicycleMotionModel m;
  // yaw=0, vel_long=5, vel_lat=0 → both wheels advance by 5*1=5 in x over 1s
  m.initialize(toTime(0), 0.0, 0.0, 0.0, diagPoseCov(), 5.0, 0.1, 0.0, 0.01, 4.0);
  m.predictState(toTime(1.0));
  EXPECT_NEAR(m.getStateElement(BicycleMotionModel::IDX::X1), -1.0 + 5.0, 1e-9);
  EXPECT_NEAR(m.getStateElement(BicycleMotionModel::IDX::X2), 1.2 + 5.0, 1e-9);
}

TEST(BicycleMotionModel, LateralVelocityDecaysWithHalfLife1Point5Seconds)
{
  // The predictStateStep comment says:
  //   "vel_lat * exp(-dt / 2.0)  // with a half-life of 2 seconds"
  // and the Jacobian comment uses "exp(-dt / 2.0)" as well.
  // However the actual implementation uses: constexpr double half_life = 1.5 (not 2.0).
  // This is current implementation behavior; spec is undecided.
  //
  // After exactly 1.5 s, the V state (vel_lat * wheel_pos_ratio) must be halved:
  //   gamma = ln(2) / 1.5, decay over 1.5 s = exp(-gamma * 1.5) = 0.5 (exact)
  BicycleMotionModel m;
  // vel_long=0 so wheel positions stay fixed, isolating the V-state decay
  m.initialize(toTime(0), 0.0, 0.0, 0.0, diagPoseCov(), 0.0, 0.1, 1.0, 0.01, 4.0);
  const double v0 = m.getStateElement(BicycleMotionModel::IDX::V);
  m.predictState(toTime(1.5));
  const double v1 = m.getStateElement(BicycleMotionModel::IDX::V);
  EXPECT_NEAR(v1, v0 * 0.5, 1e-7);
}
