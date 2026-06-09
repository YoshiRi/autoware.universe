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
// Unit tests for type/classification conversions and DynamicObject(List) helpers in
// `autoware/multi_object_tracker/types.hpp` and `lib/types.cpp`.

#include "autoware/multi_object_tracker/object_model/classes.hpp"
#include "autoware/multi_object_tracker/types.hpp"
#include "test_object_factory.hpp"

#include <autoware_perception_msgs/msg/detected_object.hpp>
#include <autoware_perception_msgs/msg/detected_object_kinematics.hpp>
#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/shape.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace
{
using autoware::multi_object_tracker::ShapeType;
using autoware::multi_object_tracker::toShapeType;
using autoware::multi_object_tracker::toShortString;
using autoware::multi_object_tracker::toString;
using autoware::multi_object_tracker::toTrackerType;
using autoware::multi_object_tracker::TrackerType;
namespace classes = autoware::multi_object_tracker::classes;
namespace types = autoware::multi_object_tracker::types;
using ClassificationMsg = autoware_perception_msgs::msg::ObjectClassification;
using DetectedObjectMsg = autoware_perception_msgs::msg::DetectedObject;
using ShapeMsg = autoware_perception_msgs::msg::Shape;
using test_object_factory::DynamicObjectBuilder;
using test_object_factory::makeBoundingBox;
using test_object_factory::makeClassification;
using test_object_factory::makeCylinder;
using test_object_factory::makePolygon;
using test_object_factory::makePose;
using test_object_factory::makeSquarePolygonCCW;
using test_object_factory::makeSquarePolygonCW;

}  // namespace

// ---------------------------------------------------------------------------
// TrackerType <-> string
// ---------------------------------------------------------------------------

TEST(TrackerTypeConversion, StringRoundTripForAllKnownTypes)
{
  for (const auto tracker_type : autoware::multi_object_tracker::allTrackerTypes()) {
    const auto name = toString(tracker_type);
    const auto round_tripped = toTrackerType(name);
    ASSERT_TRUE(round_tripped.has_value()) << "name=" << name;
    EXPECT_EQ(*round_tripped, tracker_type) << "name=" << name;
  }
}

TEST(TrackerTypeConversion, InvalidStringReturnsNullopt)
{
  EXPECT_EQ(toTrackerType("not_a_tracker"), std::nullopt);
  EXPECT_EQ(toTrackerType(""), std::nullopt);
}

TEST(TrackerTypeConversion, UnknownEnumValueFallsBackToPolygonTrackerName)
{
  // This pins down the current fallback behavior of `toString(TrackerType)` for values outside the
  // declared enumerators (e.g. produced by `static_cast`). Whether this should instead be treated
  // as an error is an open design question noted in test_design.md.
  const auto bogus = static_cast<TrackerType>(0);
  EXPECT_EQ(toString(bogus), "polygon_tracker");
  EXPECT_EQ(toShortString(bogus), "default");
}

// ---------------------------------------------------------------------------
// ShapeType <-> string / msg
// ---------------------------------------------------------------------------

TEST(ShapeTypeConversion, StringRoundTripForAllKnownTypes)
{
  for (const auto shape_type : autoware::multi_object_tracker::ALL_SHAPE_TYPES) {
    const auto name = toString(shape_type);
    const auto round_tripped = toShapeType(name);
    ASSERT_TRUE(round_tripped.has_value()) << "name=" << name;
    EXPECT_EQ(*round_tripped, shape_type) << "name=" << name;
  }
}

TEST(ShapeTypeConversion, InvalidStringReturnsNullopt)
{
  EXPECT_EQ(toShapeType("not_a_shape"), std::nullopt);
  EXPECT_EQ(toShapeType(""), std::nullopt);
}

TEST(ShapeTypeConversion, UnknownMsgShapeTypeIsTreatedAsBoundingBox)
{
  // `toShapeType(uint8_t)` maps any value outside {BOUNDING_BOX, CYLINDER, POLYGON} to bounding
  // box; pin this fallback down explicitly since downstream code relies on it.
  constexpr uint8_t unknown_msg_shape_type = 255;
  EXPECT_EQ(toShapeType(unknown_msg_shape_type), ShapeType::BOUNDING_BOX);
}

// ---------------------------------------------------------------------------
// Label <-> string / msg
// ---------------------------------------------------------------------------

TEST(LabelConversion, StringRoundTripForAllTrackedLabels)
{
  for (const auto label : classes::trackedLabels()) {
    const auto name = classes::toString(label);
    const auto round_tripped = classes::toLabel(name);
    ASSERT_TRUE(round_tripped.has_value()) << "name=" << name;
    EXPECT_EQ(*round_tripped, label) << "name=" << name;
  }
}

TEST(LabelConversion, InvalidStringReturnsNullopt)
{
  EXPECT_EQ(classes::toLabel("not_a_label"), std::nullopt);
}

TEST(LabelConversion, MsgRoundTripForAllTrackedLabels)
{
  for (const auto label : classes::trackedLabels()) {
    const auto msg_label = classes::toMsgLabel(label);
    EXPECT_EQ(classes::toLabel(msg_label), label);
  }
}

TEST(LabelConversion, UnknownMsgLabelFallsBackToUnknown)
{
  constexpr uint8_t bogus_msg_label = 200;
  EXPECT_EQ(classes::toLabel(bogus_msg_label), classes::Label::UNKNOWN);
}

// ---------------------------------------------------------------------------
// classes::getHighestProbClassification
// ---------------------------------------------------------------------------

TEST(GetHighestProbClassification, EmptyVectorReturnsUnknownWithZeroProbability)
{
  const auto result = classes::getHighestProbClassification({});
  EXPECT_EQ(result.label, classes::Label::UNKNOWN);
  EXPECT_FLOAT_EQ(result.probability, 0.0F);
}

TEST(GetHighestProbClassification, ReturnsLabelWithMaxProbability)
{
  const std::vector<classes::Classification> classification = {
    makeClassification(classes::Label::CAR, 0.2F),
    makeClassification(classes::Label::PEDESTRIAN, 0.7F),
    makeClassification(classes::Label::BICYCLE, 0.1F),
  };
  const auto result = classes::getHighestProbClassification(classification);
  EXPECT_EQ(result.label, classes::Label::PEDESTRIAN);
  EXPECT_FLOAT_EQ(result.probability, 0.7F);
}

TEST(GetHighestProbClassification, TiesResolveToTheFirstMaxElement)
{
  // `std::max_element` returns the first of equal maximal elements; pin this down since the
  // iteration order of `classification` therefore matters for tie-breaking.
  const std::vector<classes::Classification> classification = {
    makeClassification(classes::Label::CAR, 0.5F),
    makeClassification(classes::Label::TRUCK, 0.5F),
  };
  const auto result = classes::getHighestProbClassification(classification);
  EXPECT_EQ(result.label, classes::Label::CAR);
}

// ---------------------------------------------------------------------------
// types::toDynamicObject
// ---------------------------------------------------------------------------

namespace
{
DetectedObjectMsg makeDetectedObjectMsg(
  float existence_probability, const geometry_msgs::msg::Pose & pose = makePose(1.0, 2.0, 0.3))
{
  DetectedObjectMsg msg;
  msg.existence_probability = existence_probability;
  msg.classification = {[] {
    ClassificationMsg c;
    c.label = ClassificationMsg::CAR;
    c.probability = 0.9F;
    return c;
  }()};
  msg.kinematics.pose_with_covariance.pose = pose;
  msg.kinematics.pose_with_covariance.covariance =
    test_object_factory::makeDiagonalCovariance(0.2, 0.3);
  msg.kinematics.twist_with_covariance.twist.linear.x = 1.5;
  msg.kinematics.twist_with_covariance.twist.angular.z = 0.1;
  msg.kinematics.twist_with_covariance.covariance =
    test_object_factory::makeDiagonalCovariance(0.05, 0.05);
  msg.kinematics.has_position_covariance = true;
  msg.kinematics.orientation_availability =
    autoware_perception_msgs::msg::DetectedObjectKinematics::AVAILABLE;
  msg.kinematics.has_twist = true;
  msg.kinematics.has_twist_covariance = true;
  msg.shape = makeBoundingBox(4.0, 2.0, 1.5);
  return msg;
}
}  // namespace

TEST(ToDynamicObject, AlwaysGeneratesAFreshUuid)
{
  // DetectedObject carries no UUID of its own; `toDynamicObject` must always mint a new one so
  // that two conversions of the same message never collide.
  const auto msg = makeDetectedObjectMsg(0.5F);
  const auto a = types::toDynamicObject(msg);
  const auto b = types::toDynamicObject(msg);

  const auto all_zero = [](const unique_identifier_msgs::msg::UUID & uuid) {
    return std::all_of(uuid.uuid.begin(), uuid.uuid.end(), [](uint8_t b) { return b == 0; });
  };
  EXPECT_FALSE(all_zero(a.uuid));
  EXPECT_FALSE(all_zero(b.uuid));
  EXPECT_NE(a.uuid.uuid, b.uuid.uuid);
}

TEST(ToDynamicObject, ReplacesExistenceProbabilityBelowThresholdWithDefault)
{
  const auto msg = makeDetectedObjectMsg(1e-7F);
  const auto object = types::toDynamicObject(msg);
  EXPECT_FLOAT_EQ(object.existence_probability, types::default_existence_probability);
}

TEST(ToDynamicObject, ClampsExistenceProbabilityAboveUpperBoundTo0999)
{
  const auto msg = makeDetectedObjectMsg(1.0F);
  const auto object = types::toDynamicObject(msg);
  EXPECT_FLOAT_EQ(object.existence_probability, 0.999F);
}

TEST(ToDynamicObject, PassesThroughMidRangeExistenceProbabilityUnchanged)
{
  const auto msg = makeDetectedObjectMsg(0.42F);
  const auto object = types::toDynamicObject(msg);
  EXPECT_FLOAT_EQ(object.existence_probability, 0.42F);
}

TEST(ToDynamicObject, SyncsChannelIndexWithExistenceProbabilities)
{
  constexpr uint channel_index = 3;
  const auto msg = makeDetectedObjectMsg(0.6F);
  const auto object = types::toDynamicObject(msg, channel_index);

  EXPECT_EQ(object.channel_index, channel_index);
  ASSERT_EQ(object.existence_probabilities.size(), 1U);
  EXPECT_EQ(object.existence_probabilities.front().channel_index, channel_index);
  EXPECT_FLOAT_EQ(object.existence_probabilities.front().existence_probability, 0.6F);
}

TEST(ToDynamicObject, PreservesKinematicsPoseTwistClassificationAndShape)
{
  const auto pose = makePose(1.0, 2.0, 0.3);
  const auto msg = makeDetectedObjectMsg(0.6F, pose);
  const auto object = types::toDynamicObject(msg);

  EXPECT_TRUE(object.kinematics.has_position_covariance);
  EXPECT_EQ(object.kinematics.orientation_availability, types::OrientationAvailability::AVAILABLE);
  EXPECT_TRUE(object.kinematics.has_twist);
  EXPECT_TRUE(object.kinematics.has_twist_covariance);

  EXPECT_DOUBLE_EQ(object.pose.position.x, pose.position.x);
  EXPECT_DOUBLE_EQ(object.pose.position.y, pose.position.y);
  EXPECT_EQ(object.pose_covariance, msg.kinematics.pose_with_covariance.covariance);

  EXPECT_DOUBLE_EQ(object.twist.linear.x, msg.kinematics.twist_with_covariance.twist.linear.x);
  EXPECT_EQ(object.twist_covariance, msg.kinematics.twist_with_covariance.covariance);

  ASSERT_EQ(object.classification.size(), 1U);
  EXPECT_EQ(object.classification.front().label, classes::Label::CAR);
  EXPECT_FLOAT_EQ(object.classification.front().probability, 0.9F);

  EXPECT_EQ(object.shape.type, ShapeMsg::BOUNDING_BOX);
  EXPECT_DOUBLE_EQ(object.shape.dimensions.x, msg.shape.dimensions.x);
  EXPECT_DOUBLE_EQ(object.shape.dimensions.y, msg.shape.dimensions.y);
}

// ---------------------------------------------------------------------------
// DynamicObjectList::getObjectIndexByUuid
// ---------------------------------------------------------------------------

TEST(DynamicObjectListUuidIndex, ReturnsIndexAfterExplicitBuild)
{
  types::DynamicObjectList list;
  list.objects = {
    DynamicObjectBuilder().build(),
    DynamicObjectBuilder().build(),
  };
  list.buildUuidIndex();

  const auto index0 = list.getObjectIndexByUuid(list.objects[0].uuid);
  const auto index1 = list.getObjectIndexByUuid(list.objects[1].uuid);
  ASSERT_TRUE(index0.has_value());
  ASSERT_TRUE(index1.has_value());
  EXPECT_EQ(*index0, 0U);
  EXPECT_EQ(*index1, 1U);
}

TEST(DynamicObjectListUuidIndex, RebuildsLazilyWhenObjectCountChanges)
{
  types::DynamicObjectList list;
  list.objects = {DynamicObjectBuilder().build()};
  list.buildUuidIndex();

  // Append without rebuilding the index: `uuid_to_index_.size() != objects.size()` should trigger
  // a lazy rebuild on the next lookup.
  const auto appended = DynamicObjectBuilder().build();
  list.objects.push_back(appended);

  const auto index = list.getObjectIndexByUuid(appended.uuid);
  ASSERT_TRUE(index.has_value());
  EXPECT_EQ(*index, 1U);
}

TEST(DynamicObjectListUuidIndex, ReturnsNulloptForUnknownUuid)
{
  types::DynamicObjectList list;
  list.objects = {DynamicObjectBuilder().build()};
  list.buildUuidIndex();

  const auto unknown = DynamicObjectBuilder().build();
  EXPECT_EQ(list.getObjectIndexByUuid(unknown.uuid), std::nullopt);
}

// ---------------------------------------------------------------------------
// toTrackedObjectMsg / toDetectedObjectMsg
// ---------------------------------------------------------------------------

TEST(ToTrackedObjectMsg, UsesTrackerUuidAsObjectId)
{
  const auto object = DynamicObjectBuilder().build();
  const auto msg = types::toTrackedObjectMsg(object);
  EXPECT_EQ(msg.object_id.uuid, object.uuid.uuid);
}

TEST(ToDetectedObjectMsg, SetsHasCovarianceAndTwistFlagsToTrue)
{
  // Output detected objects always advertise covariance/twist availability, regardless of the
  // source DynamicObject's `kinematics` flags.
  const auto object = DynamicObjectBuilder().withHasPositionCovariance(false).withHasTwist(false)
    .withHasTwistCovariance(false).build();
  const auto msg = types::toDetectedObjectMsg(object);
  EXPECT_TRUE(msg.kinematics.has_position_covariance);
  EXPECT_TRUE(msg.kinematics.has_twist);
  EXPECT_TRUE(msg.kinematics.has_twist_covariance);
}

// ---------------------------------------------------------------------------
// types::getArea
// ---------------------------------------------------------------------------

TEST(GetArea, BoundingBoxIsLengthTimesWidth)
{
  EXPECT_DOUBLE_EQ(types::getArea(makeBoundingBox(4.0, 2.0)), 8.0);
}

TEST(GetArea, CylinderIsPiTimesRadiusSquared)
{
  const double diameter = 2.0;
  const double expected = diameter * diameter * M_PI * 0.25;
  EXPECT_DOUBLE_EQ(types::getArea(makeCylinder(diameter)), expected);
}

TEST(GetArea, PolygonUsesShoelaceFormulaAndKeepsItsSign)
{
  // `getArea(POLYGON)` does not take an absolute value: a CCW footprint yields a positive area and
  // the mirrored CW footprint yields the same magnitude but negative. This sign is significant
  // because association area gates consume `object.area` directly (see test_design.md section 1).
  const double side = 2.0;
  const double expected_magnitude = side * side;

  const double ccw_area = types::getArea(makeSquarePolygonCCW(side));
  const double cw_area = types::getArea(makeSquarePolygonCW(side));

  EXPECT_NEAR(ccw_area, expected_magnitude, 1e-9);
  EXPECT_NEAR(cw_area, -expected_magnitude, 1e-9);
}

TEST(GetArea, UnknownShapeTypeIsZero)
{
  ShapeMsg shape = makeBoundingBox(4.0, 2.0);
  shape.type = 255;  // outside {BOUNDING_BOX, CYLINDER, POLYGON}
  EXPECT_DOUBLE_EQ(types::getArea(shape), 0.0);
}
