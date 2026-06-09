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
// Deterministic factories for `types::DynamicObject` and its building blocks (pose, shape,
// covariance, classification), shared across the unit test files. Every factory returns a fully
// populated, self-consistent value with documented defaults so that tests only need to override
// the fields that matter for the behavior under test.
#ifndef TEST_OBJECT_FACTORY_HPP_
#define TEST_OBJECT_FACTORY_HPP_

#include <autoware/multi_object_tracker/object_model/classes.hpp>
#include <autoware/multi_object_tracker/object_model/uuid.hpp>
#include <autoware/multi_object_tracker/types.hpp>

#include <autoware_utils_geometry/msg/covariance.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/shape.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace test_object_factory
{

using autoware::multi_object_tracker::classes::Classification;
using autoware::multi_object_tracker::classes::Label;
using autoware::multi_object_tracker::types::DynamicObject;
using autoware::multi_object_tracker::types::ObjectKinematics;
using autoware::multi_object_tracker::types::OrientationAvailability;
using ShapeMsg = autoware_perception_msgs::msg::Shape;
using CovIdx = autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;

// Fixed reference time so that tests do not depend on wall-clock / ROS clock state.
inline rclcpp::Time defaultTime()
{
  return rclcpp::Time(1700000000, 0, RCL_ROS_TIME);
}

// Yaw-only rotation about Z, matching how DetectedObject poses are produced upstream.
inline geometry_msgs::msg::Pose makePose(double x, double y, double yaw = 0.0, double z = 0.0)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  pose.orientation.z = std::sin(yaw * 0.5);
  pose.orientation.w = std::cos(yaw * 0.5);
  return pose;
}

// Diagonal-only 6x6 (xyzrpy) covariance. Off-diagonal terms stay zero; pass them in explicitly via
// `cov[CovIdx::...]` after construction when a test needs cross terms (e.g. odometry rotation).
inline std::array<double, 36> makeDiagonalCovariance(
  double var_x, double var_y, double var_z = 1.0, double var_roll = 0.1, double var_pitch = 0.1,
  double var_yaw = 0.1)
{
  std::array<double, 36> cov{};
  cov.fill(0.0);
  cov[CovIdx::X_X] = var_x;
  cov[CovIdx::Y_Y] = var_y;
  cov[CovIdx::Z_Z] = var_z;
  cov[CovIdx::ROLL_ROLL] = var_roll;
  cov[CovIdx::PITCH_PITCH] = var_pitch;
  cov[CovIdx::YAW_YAW] = var_yaw;
  return cov;
}

inline ShapeMsg makeBoundingBox(double length, double width, double height = 1.5)
{
  ShapeMsg shape;
  shape.type = ShapeMsg::BOUNDING_BOX;
  shape.dimensions.x = length;
  shape.dimensions.y = width;
  shape.dimensions.z = height;
  return shape;
}

inline ShapeMsg makeCylinder(double diameter, double height = 1.5)
{
  ShapeMsg shape;
  shape.type = ShapeMsg::CYLINDER;
  shape.dimensions.x = diameter;
  shape.dimensions.y = diameter;
  shape.dimensions.z = height;
  return shape;
}

// `points` are (x, y) pairs in the object's local frame, stored in the given order without any
// winding-order normalization -- callers control CW vs. CCW deliberately (e.g. to pin down the
// signed-area behavior of `types::getArea(POLYGON)`).
inline ShapeMsg makePolygon(const std::vector<std::pair<double, double>> & points, double height = 1.5)
{
  ShapeMsg shape;
  shape.type = ShapeMsg::POLYGON;
  shape.dimensions.z = height;
  shape.footprint.points.reserve(points.size());
  for (const auto & xy : points) {
    geometry_msgs::msg::Point32 point;
    point.x = static_cast<float>(xy.first);
    point.y = static_cast<float>(xy.second);
    point.z = 0.0F;
    shape.footprint.points.push_back(point);
  }
  return shape;
}

// Counter-clockwise unit square footprint centered at the origin, side length `side`.
inline ShapeMsg makeSquarePolygonCCW(double side = 1.0, double height = 1.5)
{
  const double h = side * 0.5;
  return makePolygon({{h, h}, {-h, h}, {-h, -h}, {h, -h}}, height);
}

// Same square, traversed clockwise -- the mirror winding order of `makeSquarePolygonCCW`.
inline ShapeMsg makeSquarePolygonCW(double side = 1.0, double height = 1.5)
{
  const double h = side * 0.5;
  return makePolygon({{h, h}, {h, -h}, {-h, -h}, {-h, h}}, height);
}

inline Classification makeClassification(Label label, float probability = 1.0F)
{
  return Classification{label, probability};
}

// Builder for `types::DynamicObject`. Defaults describe a confidently-classified, fully-observed
// car-sized bounding box at the origin so that tests only need to override what they care about.
//
//   const auto object = test_object_factory::DynamicObjectBuilder()
//                         .withLabel(Label::PEDESTRIAN)
//                         .withPose(makePose(1.0, 2.0))
//                         .withShape(makeCylinder(0.5))
//                         .build();
class DynamicObjectBuilder
{
public:
  DynamicObjectBuilder()
  : time_(defaultTime()),
    channel_index_(0),
    existence_probability_(0.8F),
    classification_({makeClassification(Label::CAR, 1.0F)}),
    pose_(makePose(0.0, 0.0, 0.0)),
    pose_covariance_(makeDiagonalCovariance(0.1, 0.1, 0.1, 0.01, 0.01, 0.05)),
    twist_(),
    twist_covariance_(makeDiagonalCovariance(0.1, 0.05, 0.05, 0.01, 0.01, 0.05)),
    shape_(makeBoundingBox(4.0, 2.0, 1.5)),
    trust_extension_(true),
    has_position_covariance_(true),
    has_twist_(true),
    has_twist_covariance_(true),
    orientation_availability_(OrientationAvailability::AVAILABLE)
  {
  }

  DynamicObjectBuilder & withTime(const rclcpp::Time & time)
  {
    time_ = time;
    return *this;
  }

  DynamicObjectBuilder & withChannel(uint channel_index)
  {
    channel_index_ = channel_index;
    return *this;
  }

  DynamicObjectBuilder & withExistenceProbability(float probability)
  {
    existence_probability_ = probability;
    return *this;
  }

  DynamicObjectBuilder & withLabel(Label label, float probability = 1.0F)
  {
    classification_ = {makeClassification(label, probability)};
    return *this;
  }

  DynamicObjectBuilder & withClassification(std::vector<Classification> classification)
  {
    classification_ = std::move(classification);
    return *this;
  }

  DynamicObjectBuilder & withPose(const geometry_msgs::msg::Pose & pose)
  {
    pose_ = pose;
    return *this;
  }

  DynamicObjectBuilder & withPose(double x, double y, double yaw = 0.0, double z = 0.0)
  {
    pose_ = makePose(x, y, yaw, z);
    return *this;
  }

  DynamicObjectBuilder & withPoseCovariance(const std::array<double, 36> & covariance)
  {
    pose_covariance_ = covariance;
    return *this;
  }

  DynamicObjectBuilder & withTwist(double vx, double vy = 0.0, double yaw_rate = 0.0)
  {
    twist_.linear.x = vx;
    twist_.linear.y = vy;
    twist_.angular.z = yaw_rate;
    return *this;
  }

  DynamicObjectBuilder & withTwistCovariance(const std::array<double, 36> & covariance)
  {
    twist_covariance_ = covariance;
    return *this;
  }

  DynamicObjectBuilder & withShape(const ShapeMsg & shape)
  {
    shape_ = shape;
    return *this;
  }

  DynamicObjectBuilder & withTrustExtension(bool trust)
  {
    trust_extension_ = trust;
    return *this;
  }

  DynamicObjectBuilder & withHasPositionCovariance(bool has_covariance)
  {
    has_position_covariance_ = has_covariance;
    return *this;
  }

  DynamicObjectBuilder & withHasTwist(bool has_twist)
  {
    has_twist_ = has_twist;
    return *this;
  }

  DynamicObjectBuilder & withHasTwistCovariance(bool has_covariance)
  {
    has_twist_covariance_ = has_covariance;
    return *this;
  }

  DynamicObjectBuilder & withOrientationAvailability(OrientationAvailability availability)
  {
    orientation_availability_ = availability;
    return *this;
  }

  DynamicObject build() const
  {
    DynamicObject object;
    object.time = time_;
    object.uuid = autoware::multi_object_tracker::object_model::generate_uuid();
    object.channel_index = channel_index_;
    object.existence_probability = existence_probability_;
    object.existence_probabilities = {{channel_index_, existence_probability_}};
    object.classification = classification_;
    object.kinematics.has_position_covariance = has_position_covariance_;
    object.kinematics.orientation_availability = orientation_availability_;
    object.kinematics.has_twist = has_twist_;
    object.kinematics.has_twist_covariance = has_twist_covariance_;
    object.kinematics.is_stationary = false;
    object.pose = pose_;
    object.pose_covariance = pose_covariance_;
    object.twist = twist_;
    object.twist_covariance = twist_covariance_;
    object.shape = shape_;
    object.trust_extension = trust_extension_;
    object.area = autoware::multi_object_tracker::types::getArea(shape_);
    return object;
  }

private:
  rclcpp::Time time_;
  uint channel_index_;
  float existence_probability_;
  std::vector<Classification> classification_;
  geometry_msgs::msg::Pose pose_;
  std::array<double, 36> pose_covariance_;
  geometry_msgs::msg::Twist twist_;
  std::array<double, 36> twist_covariance_;
  ShapeMsg shape_;
  bool trust_extension_;
  bool has_position_covariance_;
  bool has_twist_;
  bool has_twist_covariance_;
  OrientationAvailability orientation_availability_;
};

}  // namespace test_object_factory

#endif  // TEST_OBJECT_FACTORY_HPP_
