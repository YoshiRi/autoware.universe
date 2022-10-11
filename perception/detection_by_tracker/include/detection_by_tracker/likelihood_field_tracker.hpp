// Copyright 2021 Tier IV, Inc.
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

#ifndef DETECTION_BY_TRACKER__LIKELIHOOD_FIELD_TRACKER_CORE_HPP_
#define DETECTION_BY_TRACKER__LIKELIHOOD_FIELD_TRACKER_CORE_HPP_

#include "detection_by_tracker/debugger.hpp"

#include <euclidean_cluster/euclidean_cluster.hpp>
#include <euclidean_cluster/utils.hpp>
#include <euclidean_cluster/voxel_grid_based_euclidean_cluster.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_estimation/shape_estimator.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <autoware_auto_perception_msgs/msg/detected_objects.hpp>
#include <autoware_auto_perception_msgs/msg/tracked_objects.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>


#include <tf2/LinearMath/Transform.h>
#include <tf2/convert.h>
#include <tf2/transform_datatypes.h>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <deque>
#include <memory>
#include <vector>
#include <array>
#include <random>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "detection_by_tracker/detection_by_tracker_core.hpp"
//using autoware::common::types::float64_t; // convert double to float64_t later


/**
 * @brief rectangle zone calc function
 * 
 */
class RectangleZone
{
//private:
public:
  double xmin_, xmax_, ymin_, ymax_;
  
public:
  RectangleZone();/// default constructor
  RectangleZone(const double xmin, const double xmax, const double ymin, const double ymax);
  void setZone(const double xmin, const double xmax, const double ymin, const double ymax);
  bool contains(const Eigen::Vector2d point); /// if input coordinates is inside the rectangle zone, geometry_msgs::msg::Point
};

/**
 * @brief Express Car Likelihood Field used in coarse fitting
 * 
 * @details likelihood is calc as following equations:
 *       exp(-c/2/sigma/sigma)
 *       ,while c is cost parameter and sigma is measurement noise covariance
 */
class CoarseCarLikelihoodField
{
  private:
    double measurement_covariance_ = 0.1;
    /// cost value of {contour, outside}
    std::vector<double> costs_ = {-0.02, 0};
    std::array<RectangleZone,2> contour_zones_;
    const std::array<std::array<std::uint8_t, 2>, 4> indexes_ = {{{3,0},{0,1},{1,2},{2,3}}};

  
  public:
    explicit CoarseCarLikelihoodField(const double width, const double length,
                                    const double outside_margin, const double inside_margin);
    void setContourZones(const double width, const double length, const double outside_margin, const double inside_margin);
    void setMeasurementCovariance(const double cov);
    void setCostParameters(const std::vector<double> & costs);
    double calcLikelihoods(std::vector<Eigen::Vector2d> localized_measurements, std::uint8_t index_num);
};


/**
 * @brief Express Car Likelihood Field used in fine fitting
 * 
 */
class FineCarLikelihoodField
{
  private:

    double measurement_covariance_ = 0.1;

  public:
    RectangleZone car_contour_; /// Rectangle Area
    /// cost value represent {penalty, contour, inside, outside}
    std::vector<double> costs_ = {0.02,-0.04, -0.01, 0}; 
    const int indexes_[4][2] = {{3,0},{0,1},{1,2},{2,3}};
    std::array<RectangleZone,4> penalty_zones_;
    std::array<RectangleZone,4> contour_zones_;

    explicit FineCarLikelihoodField(const double width, const double length,
                                    const double outside_margin, const double inside_margin);
    void setContourZones(const double width, const double length, const double outside_margin, const double inside_margin);
    void setPenaltyZones(const double width, const double length, const double outside_margin, const double inside_margin);
    void setMeasurementCovariance(const double cov);
    void setCostParameters(const std::vector<double> & costs);
    double calcLikelihoods(std::vector<Eigen::Vector2d> localized_measurements, std::uint8_t index_num);
};


/**
 * @brief Manage Each Vehicle Particle
 * 
 */
class VehicleParticle
{
private:
  const double inside_margin_ = 0.25;
  const double outside_margin_ = 1.0;
  double measurement_noise_ = 0.1;


    
public:
  FineCarLikelihoodField fine_likelihood_; // maybe it's ok to use std::optional instead
  CoarseCarLikelihoodField coarse_likelihood_;
  std::array<Eigen::Vector2d,4> corner_points_;

  Eigen::Vector2d center_;
  double orientation_;
  std::uint8_t corner_index_;
  explicit VehicleParticle(const Eigen::Vector2d center, const double width, const double length, const double orientation);
  void setCornerPoints(const Eigen::Vector2d center, const double width, const double length, const double orientation);
  std::uint8_t getNearestCornerIndex(const Eigen::Vector2d & origin = Eigen::Vector2d(0.0,0.0)); /// ego origin is set 0,0 by default
  void toLocalCoordinate(const std::vector<Eigen::Vector2d> & measurements, const Eigen::Vector2d & center, double orientation, std::vector<Eigen::Vector2d>& local_measurements);
  double calcCoarseLikelihood(const std::vector<Eigen::Vector2d> & measurements);
  double calcFineLikelihood(const std::vector<Eigen::Vector2d> & measurements);
};


class SingleLFTracker
{
  private:
    std::vector<VehicleParticle> vehicle_particle_;
    VehicleParticle default_vehicle_;
    std::uint32_t particle_num_;
    Eigen::Vector2d position_;
    double orientation_;
    double width_;
    double length_;
    double default_likelihood_;
    Eigen::Matrix3d covariance_;

  public:
    SingleLFTracker(const autoware_auto_perception_msgs::msg::TrackedObject & object);
    void createRandomVehiclePositionParticle(const std::uint32_t particle_num);
    void createGridVehiclePositionParticle();
    //void createVehicleShapeParticle();
    std::tuple<Eigen::Vector3d, Eigen::Matrix3d> calcMeanAndCovFromParticles(std::vector<double> & likelihoods, std::vector<Eigen::Vector3d> vectors);
    std::tuple<Eigen::Vector3d, Eigen::Matrix3d> calcBestParticles(std::vector<double> & likelihoods, std::vector<Eigen::Vector3d> vectors);
    void estimateState(const std::vector<Eigen::Vector2d> & scan);
    autoware_auto_perception_msgs::msg::TrackedObject toTrackedObject(autoware_auto_perception_msgs::msg::TrackedObject &object);
};



class LikelihoodFieldTracker : public rclcpp::Node
{
public:
  explicit LikelihoodFieldTracker(const rclcpp::NodeOptions & node_options);

private:
  rclcpp::Publisher<autoware_auto_perception_msgs::msg::DetectedObjects>::SharedPtr objects_pub_;
  rclcpp::Subscription<autoware_auto_perception_msgs::msg::TrackedObjects>::SharedPtr trackers_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scans_sub_;
  rclcpp::Subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr
    initial_objects_sub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  TrackerHandler tracker_handler_;
  std::shared_ptr<ShapeEstimator> shape_estimator_;
  std::shared_ptr<euclidean_cluster::EuclideanClusterInterface> cluster_;
  std::shared_ptr<Debugger> debugger_;

  bool ignore_unknown_tracker_;

  void onObjects(
    const sensor_msgs::msg::LaserScan::ConstSharedPtr input_msg);

  // void divideUnderSegmentedObjects(
  //   const autoware_auto_perception_msgs::msg::DetectedObjects & tracked_objects,
  //   const tier4_perception_msgs::msg::DetectedObjectsWithFeature & in_objects,
  //   autoware_auto_perception_msgs::msg::DetectedObjects & out_no_found_tracked_objects,
  //   tier4_perception_msgs::msg::DetectedObjectsWithFeature & out_objects);

  // float optimizeUnderSegmentedObject(
  //   const autoware_auto_perception_msgs::msg::DetectedObject & target_object,
  //   const sensor_msgs::msg::PointCloud2 & under_segmented_cluster,
  //   tier4_perception_msgs::msg::DetectedObjectWithFeature & output);

  // void mergeOverSegmentedObjects(
  //   const autoware_auto_perception_msgs::msg::DetectedObjects & tracked_objects,
  //   const tier4_perception_msgs::msg::DetectedObjectsWithFeature & in_objects,
  //   autoware_auto_perception_msgs::msg::DetectedObjects & out_no_found_tracked_objects,
  //   tier4_perception_msgs::msg::DetectedObjectsWithFeature & out_objects);
};

#endif  // DETECTION_BY_TRACKER__LIKELIHOOD_FIELD_TRACKER_CORE_HPP_
