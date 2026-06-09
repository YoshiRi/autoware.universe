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
// Unit tests for `autoware/multi_object_tracker/uncertainty/uncertainty_processor.hpp`:
// - modelUncertainty: passthrough vs. class-based covariance injection
// - normalizeUncertainty: floor on each covariance diagonal
// - addOdometryUncertainty: propagation of ego pose/twist noise onto object covariances

#include "autoware/multi_object_tracker/uncertainty/uncertainty_processor.hpp"
#include "test_object_factory.hpp"

#include <autoware_utils_geometry/msg/covariance.hpp>
#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace
{
namespace uncertainty = autoware::multi_object_tracker::uncertainty;
namespace types = autoware::multi_object_tracker::types;
namespace classes = autoware::multi_object_tracker::classes;
using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
using test_object_factory::DynamicObjectBuilder;
using test_object_factory::makeBoundingBox;
using test_object_factory::makeDiagonalCovariance;
using test_object_factory::makePose;

// Build a minimal DynamicObjectList containing a single object.
types::DynamicObjectList makeSingleObjectList(const types::DynamicObject & obj)
{
  types::DynamicObjectList list;
  list.header.stamp = obj.time;
  list.objects.push_back(obj);
  list.buildUuidIndex();
  return list;
}

// Build an Odometry message with all-zero covariance and velocity, timestamped to `stamp`.
nav_msgs::msg::Odometry makeZeroOdometry(const rclcpp::Time & stamp)
{
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = stamp;
  for (auto & v : odom.pose.covariance) v = 0.0;
  for (auto & v : odom.twist.covariance) v = 0.0;
  return odom;
}

}  // namespace

// ---------------------------------------------------------------------------
// modelUncertainty: passthrough for objects with has_position_covariance = true
// ---------------------------------------------------------------------------

TEST(ModelUncertainty, PassthroughIfHasPositionCovariance)
{
  // Objects that already carry a position covariance must NOT be modified.
  const auto cov = makeDiagonalCovariance(0.5, 0.3);
  const auto obj = DynamicObjectBuilder().withHasPositionCovariance(true).withPoseCovariance(cov).build();
  const auto list = makeSingleObjectList(obj);
  const auto result = uncertainty::modelUncertainty(list);
  ASSERT_EQ(result.objects.size(), 1U);
  EXPECT_EQ(result.objects[0].pose_covariance, obj.pose_covariance);
}

// ---------------------------------------------------------------------------
// modelUncertainty: fills covariance when has_position_covariance = false
// ---------------------------------------------------------------------------

TEST(ModelUncertainty, FillsPositionCovarianceForCarAtZeroYaw)
{
  // CAR at yaw=0 → X_X = pos_x (=0.16), Y_Y = pos_y (=0.16), off-diag X_Y = 0.
  // (For normal_vehicle: pos_x = pos_y = 0.4^2 = 0.16, so rotation doesn't change the values.)
  const auto obj = DynamicObjectBuilder()
                     .withHasPositionCovariance(false)
                     .withLabel(classes::Label::CAR)
                     .withPose(makePose(1.0, 2.0, 0.0))
                     .build();
  const auto list = makeSingleObjectList(obj);
  const auto result = uncertainty::modelUncertainty(list);
  ASSERT_EQ(result.objects.size(), 1U);

  const auto & cov = result.objects[0].pose_covariance;
  EXPECT_GT(cov[XYZRPY_COV_IDX::X_X], 0.0);
  EXPECT_GT(cov[XYZRPY_COV_IDX::Y_Y], 0.0);
  EXPECT_NEAR(cov[XYZRPY_COV_IDX::X_Y], 0.0, 1e-12);  // sym, off-diag zero when pos_x == pos_y
  EXPECT_GT(cov[XYZRPY_COV_IDX::YAW_YAW], 0.0);
  // Twist covariance: vel_long (X_X) and vel_lat (Y_Y) must both be positive.
  const auto & tcov = result.objects[0].twist_covariance;
  EXPECT_GT(tcov[XYZRPY_COV_IDX::X_X], 0.0);
  EXPECT_GT(tcov[XYZRPY_COV_IDX::Y_Y], 0.0);
}

TEST(ModelUncertainty, InflatesYawCovarianceWhenOrientationUnavailable)
{
  // When orientation is UNAVAILABLE, YAW_YAW is multiplied by 1e3 relative to the AVAILABLE case.
  const auto available = DynamicObjectBuilder()
                           .withHasPositionCovariance(false)
                           .withLabel(classes::Label::CAR)
                           .withOrientationAvailability(types::OrientationAvailability::AVAILABLE)
                           .build();
  const auto unavailable = DynamicObjectBuilder()
                             .withHasPositionCovariance(false)
                             .withLabel(classes::Label::CAR)
                             .withOrientationAvailability(types::OrientationAvailability::UNAVAILABLE)
                             .build();

  const auto r_avail = uncertainty::modelUncertainty(makeSingleObjectList(available));
  const auto r_unavail = uncertainty::modelUncertainty(makeSingleObjectList(unavailable));
  ASSERT_EQ(r_avail.objects.size(), 1U);
  ASSERT_EQ(r_unavail.objects.size(), 1U);

  const double yaw_avail = r_avail.objects[0].pose_covariance[XYZRPY_COV_IDX::YAW_YAW];
  const double yaw_unavail = r_unavail.objects[0].pose_covariance[XYZRPY_COV_IDX::YAW_YAW];
  EXPECT_NEAR(yaw_unavail, yaw_avail * 1e3, 1e-12);
}

TEST(ModelUncertainty, BuildsUuidIndexAfterProcessing)
{
  // modelUncertainty() always calls buildUuidIndex(), so even newly processed objects are indexed.
  const auto obj = DynamicObjectBuilder().withHasPositionCovariance(false).build();
  const auto list = makeSingleObjectList(obj);
  const auto result = uncertainty::modelUncertainty(list);
  ASSERT_EQ(result.objects.size(), 1U);
  const auto idx = result.getObjectIndexByUuid(result.objects[0].uuid);
  ASSERT_TRUE(idx.has_value());
  EXPECT_EQ(*idx, 0U);
}

TEST(ModelUncertainty, PedestrianGetsDifferentCovarianceThanCar)
{
  // CAR and PEDESTRIAN use different object models; their injected covariances must differ.
  auto make = [](classes::Label label) {
    return uncertainty::modelUncertainty(makeSingleObjectList(
      DynamicObjectBuilder().withHasPositionCovariance(false).withLabel(label).build()));
  };
  const auto car = make(classes::Label::CAR);
  const auto ped = make(classes::Label::PEDESTRIAN);
  ASSERT_EQ(car.objects.size(), 1U);
  ASSERT_EQ(ped.objects.size(), 1U);
  // Yaw covariance differs: pedestrian has much larger yaw uncertainty than a car.
  EXPECT_NE(
    car.objects[0].pose_covariance[XYZRPY_COV_IDX::YAW_YAW],
    ped.objects[0].pose_covariance[XYZRPY_COV_IDX::YAW_YAW]);
}

// ---------------------------------------------------------------------------
// normalizeUncertainty: floor on diagonal covariance entries
// ---------------------------------------------------------------------------

TEST(NormalizeUncertainty, ClampsVerySmallCovariancesToMinimumValues)
{
  // All diagonal covariance entries that are smaller than the internal minimums must be raised.
  auto obj = DynamicObjectBuilder().build();
  obj.pose_covariance.fill(0.0);
  obj.twist_covariance.fill(0.0);
  obj.pose_covariance[XYZRPY_COV_IDX::X_X] = 1e-10;
  obj.pose_covariance[XYZRPY_COV_IDX::Y_Y] = 1e-10;
  obj.pose_covariance[XYZRPY_COV_IDX::Z_Z] = 1e-10;
  obj.pose_covariance[XYZRPY_COV_IDX::YAW_YAW] = 1e-10;
  obj.twist_covariance[XYZRPY_COV_IDX::X_X] = 1e-10;
  obj.twist_covariance[XYZRPY_COV_IDX::Y_Y] = 1e-10;

  auto list = makeSingleObjectList(obj);
  uncertainty::normalizeUncertainty(list);
  ASSERT_EQ(list.objects.size(), 1U);
  const auto & cov = list.objects[0].pose_covariance;
  const auto & tcov = list.objects[0].twist_covariance;
  constexpr double min_cov_dist = 1e-4;
  constexpr double min_cov_rad = 1e-6;
  constexpr double min_cov_vel = 1e-4;
  EXPECT_GE(cov[XYZRPY_COV_IDX::X_X], min_cov_dist);
  EXPECT_GE(cov[XYZRPY_COV_IDX::Y_Y], min_cov_dist);
  EXPECT_GE(cov[XYZRPY_COV_IDX::Z_Z], min_cov_dist);
  EXPECT_GE(cov[XYZRPY_COV_IDX::YAW_YAW], min_cov_rad);
  EXPECT_GE(tcov[XYZRPY_COV_IDX::X_X], min_cov_vel);
  EXPECT_GE(tcov[XYZRPY_COV_IDX::Y_Y], min_cov_vel);
}

TEST(NormalizeUncertainty, LeavesLargeCovariancesUnchanged)
{
  // Entries already above the minimums must NOT be reduced.
  auto obj = DynamicObjectBuilder().build();
  constexpr double large = 100.0;
  obj.pose_covariance[XYZRPY_COV_IDX::X_X] = large;
  obj.pose_covariance[XYZRPY_COV_IDX::Y_Y] = large;
  obj.pose_covariance[XYZRPY_COV_IDX::Z_Z] = large;
  obj.pose_covariance[XYZRPY_COV_IDX::YAW_YAW] = large;
  obj.twist_covariance[XYZRPY_COV_IDX::X_X] = large;
  obj.twist_covariance[XYZRPY_COV_IDX::Y_Y] = large;

  auto list = makeSingleObjectList(obj);
  uncertainty::normalizeUncertainty(list);
  ASSERT_EQ(list.objects.size(), 1U);
  const auto & cov = list.objects[0].pose_covariance;
  const auto & tcov = list.objects[0].twist_covariance;
  EXPECT_DOUBLE_EQ(cov[XYZRPY_COV_IDX::X_X], large);
  EXPECT_DOUBLE_EQ(cov[XYZRPY_COV_IDX::Y_Y], large);
  EXPECT_DOUBLE_EQ(cov[XYZRPY_COV_IDX::Z_Z], large);
  EXPECT_DOUBLE_EQ(cov[XYZRPY_COV_IDX::YAW_YAW], large);
  EXPECT_DOUBLE_EQ(tcov[XYZRPY_COV_IDX::X_X], large);
  EXPECT_DOUBLE_EQ(tcov[XYZRPY_COV_IDX::Y_Y], large);
}

// ---------------------------------------------------------------------------
// addOdometryUncertainty: ego uncertainty propagation onto object covariances
// ---------------------------------------------------------------------------

TEST(AddOdometryUncertainty, ZeroEgoUncertaintyAndZeroVelocityLeavesObjectCovarianceUnchanged)
{
  // Odometry with all-zero covariance and zero velocity: no additional uncertainty from ego motion.
  // Object and odometry share the same timestamp (dt = 0), so there is no motion term either.
  const rclcpp::Time t(1000, 0, RCL_ROS_TIME);
  auto obj = DynamicObjectBuilder().withTime(t).withPose(makePose(10.0, 0.0)).build();
  auto list = makeSingleObjectList(obj);

  auto odom = makeZeroOdometry(t);  // dt = 0, all covariances = 0
  const auto cov_before = list.objects[0].pose_covariance;
  uncertainty::addOdometryUncertainty(odom, list);
  EXPECT_EQ(list.objects[0].pose_covariance, cov_before);
}

TEST(AddOdometryUncertainty, EgoPositionUncertaintyAddsToObjectUncertainty)
{
  // When the ego pose has nonzero position covariance, it must add to the object position
  // covariance. Here we use a diagonal ego pose covariance (no rotation, zero velocity, dt=0) so
  // the addition is simply pose_cov_object += ego_pose_cov (for the 2x2 xy block).
  const rclcpp::Time t(1000, 0, RCL_ROS_TIME);
  constexpr double ego_cov_xx = 0.25, ego_cov_yy = 0.09;

  auto obj = DynamicObjectBuilder().withTime(t).withPose(makePose(10.0, 0.0)).build();
  auto list = makeSingleObjectList(obj);
  const double xx_before = list.objects[0].pose_covariance[XYZRPY_COV_IDX::X_X];
  const double yy_before = list.objects[0].pose_covariance[XYZRPY_COV_IDX::Y_Y];

  auto odom = makeZeroOdometry(t);  // dt = 0, zero velocity
  odom.pose.covariance[0] = ego_cov_xx;  // odom pose cov [0][0] = X_X
  odom.pose.covariance[7] = ego_cov_yy;  // odom pose cov [1][1] = Y_Y

  uncertainty::addOdometryUncertainty(odom, list);

  // Since ego yaw = 0 and dt = 0, m_cov_ego_pose = [[ego_cov_xx, 0],[0, ego_cov_yy]].
  // The yaw-uncertainty rotation term adds 0 (cov_ego_yaw = 0). So:
  EXPECT_NEAR(
    list.objects[0].pose_covariance[XYZRPY_COV_IDX::X_X], xx_before + ego_cov_xx, 1e-10);
  EXPECT_NEAR(
    list.objects[0].pose_covariance[XYZRPY_COV_IDX::Y_Y], yy_before + ego_cov_yy, 1e-10);
}

TEST(AddOdometryUncertainty, DistantObjectGetsLargerYawUncertaintyContribution)
{
  // The ego-yaw uncertainty term adds `cov_ego_yaw * r²` to the perpendicular axis of the object
  // position covariance (see step 1-b in the implementation). Objects farther from ego accumulate
  // more positional uncertainty from the same ego yaw variance.
  const rclcpp::Time t(1000, 0, RCL_ROS_TIME);

  auto make_obj_at = [&](double x) {
    return DynamicObjectBuilder().withTime(t).withPose(makePose(x, 0.0)).build();
  };

  auto list_near = makeSingleObjectList(make_obj_at(5.0));
  auto list_far = makeSingleObjectList(make_obj_at(20.0));

  auto odom = makeZeroOdometry(t);
  odom.pose.covariance[35] = 0.01;  // YAW_YAW ego orientation uncertainty

  uncertainty::addOdometryUncertainty(odom, list_near);
  uncertainty::addOdometryUncertainty(odom, list_far);

  // For objects on the x-axis from origin ego, the yaw contribution adds to the Y_Y direction.
  const double yy_near = list_near.objects[0].pose_covariance[XYZRPY_COV_IDX::Y_Y];
  const double yy_far = list_far.objects[0].pose_covariance[XYZRPY_COV_IDX::Y_Y];
  EXPECT_GT(yy_far, yy_near);
}
