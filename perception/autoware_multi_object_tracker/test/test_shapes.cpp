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
// Unit tests for `autoware/multi_object_tracker/object_model/shapes.hpp`: IoU family scoring,
// convex-hull-to-bounding-box conversion, cluster orientation alignment and 3D z-range / GIoU.

#include "autoware/multi_object_tracker/object_model/shapes.hpp"
#include "autoware/multi_object_tracker/types.hpp"
#include "test_object_factory.hpp"

#include <autoware_perception_msgs/msg/shape.hpp>
#include <tf2/utils.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace
{
namespace shapes = autoware::multi_object_tracker::shapes;
namespace types = autoware::multi_object_tracker::types;
using ShapeMsg = autoware_perception_msgs::msg::Shape;
using test_object_factory::DynamicObjectBuilder;
using test_object_factory::makeBoundingBox;
using test_object_factory::makeCylinder;
using test_object_factory::makePolygon;
using test_object_factory::makePose;

double pointDistance(const geometry_msgs::msg::Point32 & a, const geometry_msgs::msg::Point32 & b)
{
  return std::hypot(a.x - b.x, a.y - b.y);
}

std::vector<std::pair<double, double>> rotatePoints(
  const std::vector<std::pair<double, double>> & points, double angle)
{
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  std::vector<std::pair<double, double>> rotated;
  rotated.reserve(points.size());
  for (const auto & p : points) {
    rotated.emplace_back(p.first * c - p.second * s, p.first * s + p.second * c);
  }
  return rotated;
}

// CCW unit-rectangle (4 wide x 2 tall) ordered so that the first edge runs along +x: this makes
// `convertConvexHullToBoundingBox` deterministically pick that edge first (all four candidate
// edges of a rectangle yield the same bbox area, and ties resolve to the first edge encountered),
// giving a yaw-zero, swap-free expected output.
const std::vector<std::pair<double, double>> kRectangleCCW = {{-2.0, -1.0}, {2.0, -1.0},
                                                               {2.0, 1.0}, {-2.0, 1.0}};

double normalizeAngle(double angle)
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle <= -M_PI) angle += 2.0 * M_PI;
  return angle;
}

}  // namespace

// ---------------------------------------------------------------------------
// get1dIoU
// ---------------------------------------------------------------------------

TEST(Get1dIoU, IdenticalCircleLikeObjectsReturnsOne)
{
  const auto a = DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeCylinder(2.0)).build();
  const auto b = DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeCylinder(2.0)).build();
  EXPECT_NEAR(shapes::get1dIoU(a, b), 1.0, 1e-9);
}

TEST(Get1dIoU, FarApartObjectsReturnZero)
{
  const auto a = DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeCylinder(2.0)).build();
  const auto b = DynamicObjectBuilder().withPose(100.0, 0.0).withShape(makeCylinder(2.0)).build();
  EXPECT_DOUBLE_EQ(shapes::get1dIoU(a, b), 0.0);
}

TEST(Get1dIoU, RadiusBelowMinimumLengthReturnsZero)
{
  const auto a = DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeCylinder(0.001)).build();
  const auto b = DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeCylinder(2.0)).build();
  EXPECT_DOUBLE_EQ(shapes::get1dIoU(a, b), 0.0);
}

TEST(Get1dIoU, ContainedObjectReturnsSquaredRadiusRatio)
{
  // source radius 2, target radius 1, concentric: dist(0) < r1 - r2(=1), so the implementation
  // returns the squared-radius ratio (r2/r1)^2 as a 2D-area-like approximation.
  const auto source =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeCylinder(4.0)).build();
  const auto target =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeCylinder(2.0)).build();
  EXPECT_NEAR(shapes::get1dIoU(source, target), 0.25, 1e-9);
}

// ---------------------------------------------------------------------------
// get2dIoU
// ---------------------------------------------------------------------------

TEST(Get2dIoU, IdenticalBoundingBoxesReturnOne)
{
  const auto a =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(4.0, 2.0)).build();
  const auto b =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(4.0, 2.0)).build();
  EXPECT_NEAR(shapes::get2dIoU(a, b), 1.0, 1e-9);
}

TEST(Get2dIoU, NonOverlappingBoundingBoxesReturnZero)
{
  const auto a =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(1.0, 1.0)).build();
  const auto b =
    DynamicObjectBuilder().withPose(100.0, 0.0).withShape(makeBoundingBox(1.0, 1.0)).build();
  EXPECT_DOUBLE_EQ(shapes::get2dIoU(a, b), 0.0);
}

TEST(Get2dIoU, PartialOverlapReturnsIntersectionOverUnion)
{
  // Source spans x in [-2, 2], target spans x in [0, 4] (both 4x2): intersection 2x2=4,
  // union 8 + 8 - 4 = 12, iou = 4 / 12.
  const auto source =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(4.0, 2.0)).build();
  const auto target =
    DynamicObjectBuilder().withPose(2.0, 0.0).withShape(makeBoundingBox(4.0, 2.0)).build();
  EXPECT_NEAR(shapes::get2dIoU(source, target), 4.0 / 12.0, 1e-6);
}

TEST(Get2dIoU, UnionAreaBelowMinimumReturnsZero)
{
  // Two coincident tiny boxes: intersection/union area (0.0025) clears MIN_AREA but is below the
  // default `min_union_area` (0.01), so the function reports an invalid score of 0.
  const auto a =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(0.05, 0.05)).build();
  const auto b =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(0.05, 0.05)).build();
  EXPECT_DOUBLE_EQ(shapes::get2dIoU(a, b), 0.0);
}

TEST(Get2dIoU, DegeneratePolygonReturnsZero)
{
  const auto degenerate = DynamicObjectBuilder()
                            .withPose(0.0, 0.0)
                            .withShape(makePolygon({{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}}))
                            .build();
  const auto normal =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(4.0, 2.0)).build();
  EXPECT_DOUBLE_EQ(shapes::get2dIoU(degenerate, normal), 0.0);
}

// ---------------------------------------------------------------------------
// get2dGeneralizedIoU
// ---------------------------------------------------------------------------

TEST(Get2dGeneralizedIoU, IdenticalBoundingBoxesReturnOne)
{
  const auto a =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(4.0, 2.0)).build();
  const auto b =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(4.0, 2.0)).build();
  EXPECT_NEAR(shapes::get2dGeneralizedIoU(a, b), 1.0, 1e-9);
}

TEST(Get2dGeneralizedIoU, NonOverlappingBoundingBoxesReturnNegativeValue)
{
  // iou = 0 for disjoint shapes, and the convex hull of the union strictly contains empty space
  // between them, so the penalty term is positive and the GIoU becomes negative.
  const auto a =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(1.0, 1.0)).build();
  const auto b =
    DynamicObjectBuilder().withPose(10.0, 0.0).withShape(makeBoundingBox(1.0, 1.0)).build();
  EXPECT_LT(shapes::get2dGeneralizedIoU(a, b), 0.0);
}

TEST(Get2dGeneralizedIoU, BothObjectsDegenerateReturnsMinusOne)
{
  const auto degenerate_shape = makePolygon({{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}});
  const auto a = DynamicObjectBuilder().withPose(0.0, 0.0).withShape(degenerate_shape).build();
  const auto b = DynamicObjectBuilder().withPose(5.0, 0.0).withShape(degenerate_shape).build();
  EXPECT_DOUBLE_EQ(shapes::get2dGeneralizedIoU(a, b), -1.0);
}

// ---------------------------------------------------------------------------
// get2dPrecisionRecallGIoU
// ---------------------------------------------------------------------------

TEST(Get2dPrecisionRecallGIoU, ContainedSourceYieldsExpectedPrecisionAndRecall)
{
  // 1x1 source fully inside a 4x2 target, both centered at the origin: intersection = source area
  // (1), so precision = 1 / 1 = 1.0 and recall = 1 / 8 = 0.125.
  const auto source =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(1.0, 1.0)).build();
  const auto target =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(4.0, 2.0)).build();

  double precision = 0.0;
  double recall = 0.0;
  double generalized_iou = 0.0;
  ASSERT_TRUE(shapes::get2dPrecisionRecallGIoU(source, target, precision, recall, generalized_iou));
  EXPECT_NEAR(precision, 1.0, 1e-6);
  EXPECT_NEAR(recall, 0.125, 1e-6);
  EXPECT_LE(generalized_iou, 1.0);
}

TEST(Get2dPrecisionRecallGIoU, ReturnsFalseWhenSourceIsDegenerate)
{
  const auto degenerate = DynamicObjectBuilder()
                            .withPose(0.0, 0.0)
                            .withShape(makePolygon({{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}}))
                            .build();
  const auto normal =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(4.0, 2.0)).build();

  double precision = 0.0;
  double recall = 0.0;
  double generalized_iou = 0.0;
  EXPECT_FALSE(
    shapes::get2dPrecisionRecallGIoU(degenerate, normal, precision, recall, generalized_iou));
}

TEST(Get2dPrecisionRecallGIoU, ReturnsFalseWhenTargetIsDegenerate)
{
  const auto normal =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(4.0, 2.0)).build();
  const auto degenerate = DynamicObjectBuilder()
                            .withPose(0.0, 0.0)
                            .withShape(makePolygon({{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}}))
                            .build();

  double precision = 0.0;
  double recall = 0.0;
  double generalized_iou = 0.0;
  EXPECT_FALSE(
    shapes::get2dPrecisionRecallGIoU(normal, degenerate, precision, recall, generalized_iou));
}

// ---------------------------------------------------------------------------
// convertConvexHullToBoundingBox
// ---------------------------------------------------------------------------

TEST(ConvertConvexHullToBoundingBox, FewerThanThreePointsReturnsFalse)
{
  const auto input = DynamicObjectBuilder()
                       .withPose(0.0, 0.0)
                       .withShape(makePolygon({{0.0, 0.0}, {1.0, 0.0}}))
                       .build();
  types::DynamicObject output;
  EXPECT_FALSE(shapes::convertConvexHullToBoundingBox(input, output));
}

TEST(ConvertConvexHullToBoundingBox, PreservesFootprintPointCountAndPairwiseGeometry)
{
  const std::vector<std::pair<double, double>> trapezoid = {{0.0, 0.0}, {4.0, 0.0}, {3.0, 2.0},
                                                             {1.0, 2.0}};
  const auto input =
    DynamicObjectBuilder().withPose(makePose(2.0, -1.0, 0.4)).withShape(makePolygon(trapezoid)).build();

  types::DynamicObject output;
  ASSERT_TRUE(shapes::convertConvexHullToBoundingBox(input, output));

  EXPECT_EQ(output.shape.type, ShapeMsg::BOUNDING_BOX);
  ASSERT_EQ(output.shape.footprint.points.size(), input.shape.footprint.points.size());

  // A rigid rotation + translation into the new object frame must preserve all pairwise distances
  // between corresponding footprint points (this is the "footprint geometry is preserved"
  // guarantee that downstream polygon-based publishing relies on).
  const auto n = input.shape.footprint.points.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      EXPECT_NEAR(
        pointDistance(input.shape.footprint.points[i], input.shape.footprint.points[j]),
        pointDistance(output.shape.footprint.points[i], output.shape.footprint.points[j]), 1e-5)
        << "i=" << i << " j=" << j;
    }
  }
}

TEST(ConvertConvexHullToBoundingBox, AxisAlignedRectangleProducesExpectedBoundingBox)
{
  const auto input = DynamicObjectBuilder()
                       .withPose(makePose(5.0, 3.0, 0.0))
                       .withShape(makePolygon(kRectangleCCW))
                       .build();

  types::DynamicObject output;
  ASSERT_TRUE(shapes::convertConvexHullToBoundingBox(input, output));

  EXPECT_EQ(output.shape.type, ShapeMsg::BOUNDING_BOX);
  EXPECT_NEAR(output.shape.dimensions.x, 4.0, 1e-6);
  EXPECT_NEAR(output.shape.dimensions.y, 2.0, 1e-6);
  EXPECT_NEAR(output.pose.position.x, 5.0, 1e-6);
  EXPECT_NEAR(output.pose.position.y, 3.0, 1e-6);
  EXPECT_NEAR(normalizeAngle(tf2::getYaw(output.pose.orientation)), 0.0, 1e-6);
}

TEST(ConvertConvexHullToBoundingBox, RotatedRectangleProducesExpectedYawAndDimensions)
{
  constexpr double rotation = M_PI / 6.0;  // 30 deg, applied to the footprint in local frame
  const auto rotated_points = rotatePoints(kRectangleCCW, rotation);
  const auto input = DynamicObjectBuilder()
                       .withPose(makePose(1.0, 1.0, 0.0))
                       .withShape(makePolygon(rotated_points))
                       .build();

  types::DynamicObject output;
  ASSERT_TRUE(shapes::convertConvexHullToBoundingBox(input, output));

  EXPECT_NEAR(output.shape.dimensions.x, 4.0, 1e-3);
  EXPECT_NEAR(output.shape.dimensions.y, 2.0, 1e-3);
  EXPECT_NEAR(output.pose.position.x, 1.0, 1e-3);
  EXPECT_NEAR(output.pose.position.y, 1.0, 1e-3);
  EXPECT_NEAR(normalizeAngle(tf2::getYaw(output.pose.orientation) - rotation), 0.0, 1e-3);
}

TEST(ConvertConvexHullToBoundingBox, EgoPositionPrefersEgoFacingEdgeOverGlobalAreaMinimum)
{
  // 4x2 axis-aligned rectangle: all four edges produce the same bbox area (8), so without an ego
  // position the FIRST edge (the long bottom edge, edge 0) wins and dims.x=4 > dims.y=2.
  // Placing the ego to the RIGHT of the object makes the short right-side edge (edge 1) ego-facing,
  // which gets priority and wins even though its bbox area equals the global minimum. The result
  // swaps length and width: dims.x=2, dims.y=4.
  const std::vector<std::pair<double, double>> rect = {{0.0, 0.0}, {4.0, 0.0}, {4.0, 2.0},
                                                       {0.0, 2.0}};
  const auto input = DynamicObjectBuilder()
                       .withPose(makePose(0.0, 0.0, 0.0))
                       .withShape(makePolygon(rect))
                       .build();

  types::DynamicObject without_ego;
  ASSERT_TRUE(shapes::convertConvexHullToBoundingBox(input, without_ego, std::nullopt));
  EXPECT_NEAR(without_ego.shape.dimensions.x, 4.0, 1e-6);
  EXPECT_NEAR(without_ego.shape.dimensions.y, 2.0, 1e-6);

  geometry_msgs::msg::Point ego_pos;
  ego_pos.x = 10.0;
  ego_pos.y = 1.0;
  types::DynamicObject with_ego;
  ASSERT_TRUE(shapes::convertConvexHullToBoundingBox(input, with_ego, ego_pos));
  EXPECT_NEAR(with_ego.shape.dimensions.x, 2.0, 1e-6);
  EXPECT_NEAR(with_ego.shape.dimensions.y, 4.0, 1e-6);

  // The two strategies must disagree -- this is the behavioral signature of the ego-facing-edge
  // preference changing which of the tied minimum-area edges gets selected.
  EXPECT_NE(without_ego.shape.dimensions.x, with_ego.shape.dimensions.x);
}

TEST(ConvertConvexHullToBoundingBox, WindingOrderIsInvariantForSymmetricRectangle)
{
  // For a symmetric rectangle all four edge orientations produce equal bbox areas, so ties resolve
  // to the first edge in iteration order. Reversing the point list (CW) makes the first CW edge go
  // in the same direction as the first CCW edge (both run along the long axis in the +x direction),
  // so the algorithm is winding-order-invariant here: both orderings produce identical output.
  std::vector<std::pair<double, double>> ccw_points = kRectangleCCW;
  std::vector<std::pair<double, double>> cw_points = kRectangleCCW;
  std::reverse(cw_points.begin(), cw_points.end());

  const auto ccw_input =
    DynamicObjectBuilder().withPose(makePose(0.0, 0.0, 0.0)).withShape(makePolygon(ccw_points)).build();
  const auto cw_input =
    DynamicObjectBuilder().withPose(makePose(0.0, 0.0, 0.0)).withShape(makePolygon(cw_points)).build();

  types::DynamicObject ccw_output;
  types::DynamicObject cw_output;
  ASSERT_TRUE(shapes::convertConvexHullToBoundingBox(ccw_input, ccw_output));
  ASSERT_TRUE(shapes::convertConvexHullToBoundingBox(cw_input, cw_output));

  EXPECT_NEAR(ccw_output.shape.dimensions.x, 4.0, 1e-6);
  EXPECT_NEAR(ccw_output.shape.dimensions.y, 2.0, 1e-6);
  EXPECT_NEAR(normalizeAngle(tf2::getYaw(ccw_output.pose.orientation)), 0.0, 1e-6);

  // CW reversal of kRectangleCCW produces {(-2,1),(2,1),(2,-1),(-2,-1)}: its first edge
  // (-2,1)→(2,1) goes in the same +x direction as the CCW first edge, so the output is identical.
  EXPECT_NEAR(cw_output.shape.dimensions.x, 4.0, 1e-6);
  EXPECT_NEAR(cw_output.shape.dimensions.y, 2.0, 1e-6);
  EXPECT_NEAR(normalizeAngle(tf2::getYaw(cw_output.pose.orientation)), 0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// alignClusterToOrientation
// ---------------------------------------------------------------------------

TEST(AlignClusterToOrientation, NonPolygonShapeReturnsNullopt)
{
  const auto cluster =
    DynamicObjectBuilder().withShape(makeBoundingBox(4.0, 2.0)).build();
  EXPECT_EQ(shapes::alignClusterToOrientation(cluster, 0.0), std::nullopt);
}

TEST(AlignClusterToOrientation, EmptyFootprintReturnsNullopt)
{
  const auto cluster = DynamicObjectBuilder().withShape(makePolygon({})).build();
  EXPECT_EQ(shapes::alignClusterToOrientation(cluster, 0.0), std::nullopt);
}

TEST(AlignClusterToOrientation, SameYawKeepsPoseAndReportsLocalExtentAsDimensions)
{
  // With target_yaw == cluster_yaw (phi = 0), the oriented extent is computed directly in the
  // cluster's local frame: the trapezoid (0,0)-(4,0)-(4,2)-(1,2) spans x in [0,4], y in [0,2], so
  // its centroid-of-extent (2,1) is added (unrotated) to the cluster pose.
  const std::vector<std::pair<double, double>> trapezoid = {{0.0, 0.0}, {4.0, 0.0}, {4.0, 2.0},
                                                             {1.0, 2.0}};
  const auto cluster = DynamicObjectBuilder()
                         .withPose(makePose(10.0, 5.0, 0.0))
                         .withShape(makePolygon(trapezoid))
                         .build();

  const auto aligned = shapes::alignClusterToOrientation(cluster, 0.0);
  ASSERT_TRUE(aligned.has_value());
  EXPECT_NEAR(aligned->pose.position.x, 12.0, 1e-6);
  EXPECT_NEAR(aligned->pose.position.y, 6.0, 1e-6);
  EXPECT_NEAR(aligned->shape.dimensions.x, 4.0, 1e-6);
  EXPECT_NEAR(aligned->shape.dimensions.y, 2.0, 1e-6);
  EXPECT_NEAR(normalizeAngle(tf2::getYaw(aligned->pose.orientation)), 0.0, 1e-6);
  // `alignClusterToOrientation` keeps the shape as a polygon; it only adjusts pose & dimensions.
  EXPECT_EQ(aligned->shape.type, ShapeMsg::POLYGON);
}

TEST(AlignClusterToOrientation, RotatesExtentIntoTargetFrame)
{
  // Cluster yaw 0, target yaw +90 deg (phi = +90 deg): the oriented-extent axis is rotated 90 deg
  // relative to the footprint's local x/y, so the rectangle's long(4)/short(2) axes are swapped in
  // the reported dimensions.
  const auto cluster =
    DynamicObjectBuilder().withPose(makePose(0.0, 0.0, 0.0)).withShape(makePolygon(kRectangleCCW)).build();

  const auto aligned = shapes::alignClusterToOrientation(cluster, M_PI_2);
  ASSERT_TRUE(aligned.has_value());
  EXPECT_NEAR(aligned->shape.dimensions.x, 2.0, 1e-6);
  EXPECT_NEAR(aligned->shape.dimensions.y, 4.0, 1e-6);
  EXPECT_NEAR(normalizeAngle(tf2::getYaw(aligned->pose.orientation)), M_PI_2, 1e-6);
}

// ---------------------------------------------------------------------------
// getObjectZRange / get3dGeneralizedIoU
// ---------------------------------------------------------------------------

TEST(GetObjectZRange, ReturnsCenterPlusMinusHalfHeight)
{
  const auto object =
    DynamicObjectBuilder().withPose(makePose(0.0, 0.0, 0.0, 2.0)).withShape(makeBoundingBox(4.0, 2.0, 1.5)).build();
  const auto [min_z, max_z] = shapes::getObjectZRange(object);
  EXPECT_NEAR(min_z, 1.25, 1e-9);
  EXPECT_NEAR(max_z, 2.75, 1e-9);
}

TEST(Get3dGeneralizedIoU, NoZOverlapReturnsInvalidScore)
{
  const auto source = DynamicObjectBuilder()
                        .withPose(makePose(0.0, 0.0, 0.0, 0.5))
                        .withShape(makeBoundingBox(4.0, 2.0, 1.0))
                        .build();
  const auto target = DynamicObjectBuilder()
                        .withPose(makePose(0.0, 0.0, 0.0, 5.0))
                        .withShape(makeBoundingBox(4.0, 2.0, 1.0))
                        .build();
  EXPECT_DOUBLE_EQ(shapes::get3dGeneralizedIoU(source, target), -1.0);
}

TEST(Get3dGeneralizedIoU, DegeneratePolygonReturnsInvalidScore)
{
  const auto degenerate = DynamicObjectBuilder()
                            .withPose(0.0, 0.0)
                            .withShape(makePolygon({{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}}))
                            .build();
  const auto normal =
    DynamicObjectBuilder().withPose(0.0, 0.0).withShape(makeBoundingBox(4.0, 2.0)).build();
  EXPECT_DOUBLE_EQ(shapes::get3dGeneralizedIoU(degenerate, normal), -1.0);
}

TEST(Get3dGeneralizedIoU, IdenticalThreeDimensionalBoxesReturnOne)
{
  const auto a = DynamicObjectBuilder()
                   .withPose(makePose(0.0, 0.0, 0.0, 1.0))
                   .withShape(makeBoundingBox(4.0, 2.0, 1.5))
                   .build();
  const auto b = DynamicObjectBuilder()
                   .withPose(makePose(0.0, 0.0, 0.0, 1.0))
                   .withShape(makeBoundingBox(4.0, 2.0, 1.5))
                   .build();
  EXPECT_NEAR(shapes::get3dGeneralizedIoU(a, b), 1.0, 1e-9);
}
