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

#include "detection_by_tracker/likelihood_field_tracker.hpp"
#include "detection_by_tracker/nlohmann_json.hpp"

#include "perception_utils/perception_utils.hpp"

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#define EIGEN_MPL2_ONLY
#include <Eigen/Core>
#include <Eigen/Geometry>

using Label = autoware_auto_perception_msgs::msg::ObjectClassification;
static std::random_device seed_gen;

namespace
{

// void setClusterInObjectWithFeature(
//   const std_msgs::msg::Header & header, const pcl::PointCloud<pcl::PointXYZ> & cluster,
//   tier4_perception_msgs::msg::DetectedObjectWithFeature & feature_object)
// {
//   sensor_msgs::msg::PointCloud2 ros_pointcloud;
//   pcl::toROSMsg(cluster, ros_pointcloud);
//   ros_pointcloud.header = header;
//   feature_object.feature.cluster = ros_pointcloud;
// }
// autoware_auto_perception_msgs::msg::Shape extendShape(
//   const autoware_auto_perception_msgs::msg::Shape & shape, const float scale)
// {
//   autoware_auto_perception_msgs::msg::Shape output = shape;
//   output.dimensions.x *= scale;
//   output.dimensions.y *= scale;
//   output.dimensions.z *= scale;
//   for (auto & point : output.footprint.points) {
//     point.x *= scale;
//     point.y *= scale;
//     point.z *= scale;
//   }
//   return output;
// }

// boost::optional<ReferenceYawInfo> getReferenceYawInfo(const uint8_t label, const float yaw)
// {
//   const bool is_vehicle =
//     Label::CAR == label || Label::TRUCK == label || Label::BUS == label || Label::TRAILER == label;
//   if (is_vehicle) {
//     return ReferenceYawInfo{yaw, tier4_autoware_utils::deg2rad(30)};
//   } else {
//     return boost::none;
//   }
// }


Eigen::Matrix3d extractXYYawCovariance(const std::array<double, 36> covariance)
{
  Eigen::Matrix3d cov(3,3);
  cov(0,0) = covariance[0];   cov(0,1) = covariance[1];  cov(0,2) = covariance[5];
  cov(1,0) = covariance[6];   cov(1,1) = covariance[7];  cov(1,2) = covariance[11];
  cov(2,0) = covariance[30];   cov(2,1) = covariance[31];  cov(2,2) = covariance[35];
  return cov;
}

void eigenMatrixToROS2Covariance(const Eigen::Matrix3d  covariance, std::array<double, 36> & cov)
{
  cov[0] = covariance(0,0);   cov[1] = covariance(0,1);  cov[5] = covariance(0,2);
  cov[6] = covariance(1,0);   cov[7] = covariance(1,1);  cov[11] = covariance(1,2);
  cov[30] = covariance(2,0);   cov[31] = covariance(2,1);  cov[35] = covariance(2,2);
}

/// sum of cost to calc likelihood 
double sumOfCostToLikelihood(double cost_sum, double sigma, double scaling_factor=1.0)
{
  (void)scaling_factor;
  return std::exp(-cost_sum/2.0/sigma/sigma);
}

/// find max indices
template <class T>
std::vector<std::size_t> findMaxIndices(std::vector<T> v){
    std::vector<std::size_t> pos;

    auto it = std::max_element(std::begin(v), std::end(v));
    while (it != std::end(v))
    {
        pos.push_back(std::distance(std::begin(v), it));
        it = std::find(std::next(it), std::end(v), *it);
    }
    return pos;
}

/// convert Pose without Stamped Message to Pose
template <class T>
void transformObjectsFromMapToBaselink
(T & object_map, T & object_baselink, 
  std_msgs::msg::Header header, tf2_ros::Buffer & buffer)
{
  geometry_msgs::msg::PoseWithCovarianceStamped before, after;
  before.pose = object_map.kinematics.pose_with_covariance;
  before.header = header;
  before.header.frame_id = "map";
  after = buffer.transform(before, "base_link");
  object_baselink = object_map;
  object_baselink.kinematics.pose_with_covariance = after.pose;
  object_baselink.kinematics.pose_with_covariance.pose.position.z = before.pose.pose.position.z;
}


template <class T>
void transformObjectsFromBaselinkToMap
(T & object_baselink, T & object_map, 
  std_msgs::msg::Header header, tf2_ros::Buffer & buffer)
{
  geometry_msgs::msg::PoseWithCovarianceStamped before, after;
  before.pose = object_baselink.kinematics.pose_with_covariance;
  before.header = header;
  before.header.frame_id = "base_link";
  after = buffer.transform(before, "map");
  object_map = object_baselink;
  object_map.kinematics.pose_with_covariance = after.pose;
  object_map.kinematics.pose_with_covariance.pose.position.z = before.pose.pose.position.z;
}

/// @brief get quadrant number of the input point belongging
/// @param point Eigen::vector
/// @return in which quadrant the point belong 1,2,3,4 starts from x 
int getQuadrantBelonging(Eigen::Vector2d point){
  if(point.x() > 0 && point.y() >= 0){
    return 1;
  }else if(point.x() <= 0 && point.y() > 0){
    return 2;
  }else if(point.x() < 0 && point.y() <= 0){
    return 3;
  }else{
    return 4;
  }
}

}  // namespace


RectangleZone::RectangleZone()
{
  xmin_ = -1;  xmax_ = 1;
  ymin_ = -1;  ymax_ = 1;
}

RectangleZone::RectangleZone(const double xmin, const double xmax, const double ymin, const double ymax)
{
  xmin_ = xmin;  xmax_ = xmax;
  ymin_ = ymin;  ymax_ = ymax;
}

void RectangleZone::setZone(const double xmin, const double xmax, const double ymin, const double ymax)
{
  xmin_ = xmin;  xmax_ = xmax;
  ymin_ = ymin;  ymax_ = ymax;
}

bool RectangleZone::contains(const Eigen::Vector2d p)
{
  if(p.x() >= xmin_ && p.x() <= xmax_ && p.y() >= ymin_ && p.y() <= ymax_){
    return true;
  }else{
    return false;
  }
}


CoarseCarLikelihoodField::CoarseCarLikelihoodField(
    const double width, const double length, const double outside_margin, const double inside_margin)
{
  setContourZones(width, length, outside_margin, inside_margin);
}

void CoarseCarLikelihoodField::setContourZones(const double width, const double length, const double outside_margin, const double inside_margin)
{
  auto W2 = width/2.0; auto L2 = length/2.0;
  contour_zones_[0].setZone(-L2-outside_margin, L2+outside_margin, -W2-outside_margin, -W2 + inside_margin);
  contour_zones_[1].setZone(-L2-outside_margin, -L2+inside_margin, -W2-outside_margin, W2 + outside_margin);
  contour_zones_[2].setZone(-L2-outside_margin, L2+outside_margin, W2-inside_margin, W2 + outside_margin);
  contour_zones_[3].setZone(L2-inside_margin, L2+outside_margin, -W2-outside_margin, W2 + outside_margin);
}

void CoarseCarLikelihoodField::setMeasurementCovariance(const double cov)
{
  measurement_covariance_ = cov;
}

void CoarseCarLikelihoodField::setCostParameters(const std::vector<double> & costs)
{
  costs_ = costs; // {inside, outside}
}

/// calc likelihood of localized measurements
double CoarseCarLikelihoodField::calcLikelihoods(const std::vector<Eigen::Vector2d> localized_measurements,std::uint8_t index_num)
{
  double cost_sum = 0;
  // for each 2d measurements
  for(std::size_t i = 0; i < localized_measurements.size(); ++i) { 
    auto part = indexes_[index_num]; // Zone indexes
    if(contour_zones_[part[0]].contains(localized_measurements[i]) || contour_zones_[part[1]].contains(localized_measurements[i])){
      // inside contour zones
      cost_sum += costs_[0];
    }else{
      // outside contour zones
      cost_sum += costs_[1];
    }
  }

  double likelihood = sumOfCostToLikelihood(cost_sum, measurement_covariance_,(double)localized_measurements.size());
  return likelihood;
}




FineCarLikelihoodField::FineCarLikelihoodField(
    const double width, const double length, const double outside_margin, const double inside_margin)
{
  // set likelihood field zones
  setContourZones(width, length, outside_margin, inside_margin);
  setPenaltyZones(width, length, outside_margin, inside_margin);
  car_contour_.setZone(-length/2.0, length/2.0, width/2.0, width/2.0);
}



void FineCarLikelihoodField::setContourZones(const double width, const double length, const double outside_margin, const double inside_margin)
{
  auto W2 = width/2.0; auto L2 = length/2.0;
  auto contour_margin = inside_margin/2.0;
  contour_zones_[0].setZone(-L2, L2, -W2-contour_margin, -W2+contour_margin);
  contour_zones_[1].setZone(-L2-contour_margin, -L2+contour_margin, -W2, W2);
  contour_zones_[2].setZone(-L2, L2, W2-contour_margin, W2+contour_margin);
  contour_zones_[3].setZone(L2-contour_margin, L2+contour_margin, -W2, W2);
  (void)outside_margin; // currently unused
}

void FineCarLikelihoodField::setPenaltyZones(const double width, const double length, const double outside_margin, const double inside_margin)
{
  auto W2 = width/2.0; auto L2 = length/2.0;
  auto contour_margin = inside_margin/2.0;
  penalty_zones_[0].setZone(-L2, L2, -W2-outside_margin, -W2-contour_margin);
  penalty_zones_[1].setZone(-L2-outside_margin, -L2-contour_margin, -W2, W2);
  penalty_zones_[2].setZone(-L2, L2, W2+contour_margin, W2+outside_margin);
  penalty_zones_[3].setZone(L2+contour_margin , L2+outside_margin, -W2, W2); 
  //(void)inside_margin; // currently unused
}

void FineCarLikelihoodField::setMeasurementCovariance(const double cov)
{
  measurement_covariance_ = cov;
}

void FineCarLikelihoodField::setCostParameters(const std::vector<double> & costs)
{
  costs_ = costs; // {penalty, contour, inside, outside}
}

/// calc likelihood of localized measurements
double FineCarLikelihoodField::calcLikelihoods(const std::vector<Eigen::Vector2d> localized_measurements,std::uint8_t index_num)
{
  double cost_sum = 0;
  // for each 2d measurements
  auto part = indexes_[index_num]; // Zone indexes
  for(std::size_t i = 0; i < localized_measurements.size(); ++i) { 
    
    if(penalty_zones_[part[0]].contains(localized_measurements[i]) || penalty_zones_[part[1]].contains(localized_measurements[i])){
      // inside penalty zones
      cost_sum += costs_[0];
    }else if(contour_zones_[part[0]].contains(localized_measurements[i]) || contour_zones_[part[1]].contains(localized_measurements[i])){
      // inside contour zones
      cost_sum += costs_[1];
    }else if(car_contour_.contains(localized_measurements[i])){
      // inside car
      cost_sum += costs_[2];
    }else{
      // outside the car
      auto quadrant = getQuadrantBelonging(localized_measurements[i]);
      // occlueded or occupied
      if(quadrant == (int)index_num){
        // vehicle is occluded
        cost_sum += costs_[3];
      }else{
        // vehicle should not be in here
        // give penalty
        cost_sum += costs_[0];
      }
      
    }
  }

  //double likelihood = sumOfCostToLikelihood(cost_sum, measurement_covariance_,(double)localized_measurements.size());
  return -cost_sum; // use raw log likelihood
}


VehicleParticle::VehicleParticle(
  const Eigen::Vector2d center, const double width, const double length, const double orientation)
  : fine_likelihood_(width, length, outside_margin_, inside_margin_), coarse_likelihood_(width, length, outside_margin_, inside_margin_)
{
  // init likelihood
  //fine_likelihood_ = FineCarLikelihoodField(width, length, outside_margin_, inside_margin_);
  //coarse_likelihood_ =  CoarseCarLikelihoodField(width, length, outside_margin_, inside_margin_);
  // calc corner points
  setCornerPoints(center, width, length, orientation);
  // set geometry
  center_ = center;
  orientation_ = orientation;
}

void VehicleParticle::setCornerPoints(const Eigen::Vector2d center, const double width, const double length, const double orientation)
{
  Eigen::Rotation2Dd rotate(orientation); /// rotation
  auto R = rotate.toRotationMatrix(); /// Rotation matrix R^T
  // corners in local coordinate
  auto p0 = Eigen::Vector2d(length/2.0,-width/2.0);   auto p1 = Eigen::Vector2d(-length/2.0,-width/2.0);
  auto p2 = Eigen::Vector2d(-length/2.0,width/2.0);  auto p3 = Eigen::Vector2d(length/2.0,width/2.0);
  // set corner points coordinates
  corner_points_[0] = R*p0 + center;
  corner_points_[1] = R*p1 + center;
  corner_points_[2] = R*p2 + center;
  corner_points_[3] = R*p3 + center;
}

std::uint8_t VehicleParticle::getNearestCornerIndex(const Eigen::Vector2d & center)
{
  std::uint8_t closest_index = 0;
  double min_dist = 100; // distance[m]
  double buff = 0;
  for(std::uint8_t i=0; i<4; i++){
    buff = (corner_points_[i] - center).norm();
    if(buff<min_dist){
      min_dist = buff;
      closest_index = i;
    }
  }
  return closest_index;
}

void VehicleParticle::toLocalCoordinate(const std::vector<Eigen::Vector2d> & measurements, const Eigen::Vector2d & center, double orientation, std::vector<Eigen::Vector2d> & localized)
{
  auto point_num = measurements.size();
  //std::vector<Eigen::Vector2d> localized(point_num);

  Eigen::Rotation2Dd rotate(-orientation); /// rotation
  auto Rt = rotate.toRotationMatrix(); /// Rotation matrix R^T

  for(std::size_t i=0;i<point_num;i++){
    localized.push_back(Rt*(measurements[i]-center));
  }
  //return localized;
}

/**
 * @brief 
 * 
 * @param measurements : 2d measurement Points
 * @return double likelihood  
 */
double VehicleParticle::calcCoarseLikelihood(const std::vector<Eigen::Vector2d> & measurements)
{
  //auto corner_index = getNearestCornerIndex();
  std::vector<Eigen::Vector2d> local_measurements;
  toLocalCoordinate(measurements, center_, orientation_, local_measurements);
  auto likelihood = coarse_likelihood_.calcLikelihoods(local_measurements, corner_index_);
  return likelihood;
}

/**
 * @brief 
 * 
 * @param measurements : 2d measurement Points
 * @return double likelihood  
 */
double VehicleParticle::calcFineLikelihood(const std::vector<Eigen::Vector2d> & measurements)
{
  // get min and max angles for this rectangle
  auto corner_index = getNearestCornerIndex(); // use default nearest corner index
  int corner_index1 = corner_index + 1;
  int corner_index2 = corner_index - 1;
  if(corner_index1 == 4){ corner_index1 = 0;}
  if(corner_index2 == -1){ corner_index2 = 3;}
  double max_angle = std::atan2(corner_points_[corner_index1].y(), corner_points_[corner_index1].x());
  double min_angle = std::atan2(corner_points_[corner_index2].y(), corner_points_[corner_index2].x());
  if(max_angle < min_angle){
    max_angle += 2.0 * M_PI;
  }

  // convert scan
  std::vector<Eigen::Vector2d>  xy_measurements;
  for(auto scan: measurements){
    // skip invalid scan
    if(!std::isfinite(scan.y())){
      continue;
    }
    // calc scan
    if( scan.x() < max_angle && scan.x() > min_angle){
      Eigen::Vector2d xy{scan.y()*cos(scan.x()), scan.y()*sin(scan.x())};
      xy_measurements.push_back(xy);
    }else if( scan.x()+ 2.0*M_PI < max_angle && scan.x()+2.0*M_PI > min_angle ){
      Eigen::Vector2d xy{scan.y()*cos(scan.x()), scan.y()*sin(scan.x())};
      xy_measurements.push_back(xy);
    }
  }
  xy_measurements.push_back(Eigen::Vector2d{0,0}); // for zerodivision escape 

  // to local coordinates
  std::vector<Eigen::Vector2d> local_measurements;
  toLocalCoordinate(xy_measurements, center_, orientation_, local_measurements);
  auto likelihood = fine_likelihood_.calcLikelihoods(local_measurements, corner_index_);
  return likelihood / (double)xy_measurements.size();// mean likelihood

}


SingleLFTracker::SingleLFTracker(const autoware_auto_perception_msgs::msg::TrackedObject & object)
:default_vehicle_(Eigen::Vector2d(object.kinematics.pose_with_covariance.pose.position.x, object.kinematics.pose_with_covariance.pose.position.y), object.shape.dimensions.y, object.shape.dimensions.x, tf2::getYaw(object.kinematics.pose_with_covariance.pose.orientation))
{
  // set private variable
  position_ = Eigen::Vector2d(object.kinematics.pose_with_covariance.pose.position.x, object.kinematics.pose_with_covariance.pose.position.y);
  orientation_ = tf2::getYaw(object.kinematics.pose_with_covariance.pose.orientation);
  length_ = object.shape.dimensions.x;
  width_ = object.shape.dimensions.y;
  //covariance_ = 
  covariance_ = extractXYYawCovariance(object.kinematics.pose_with_covariance.covariance);
  
  // tmp 
  // covariance_(0,0) = 2; // 0.3m error
  // covariance_(1,1) = 2;
  // covariance_(2,2) = DEG2RAD(10.)*DEG2RAD(10.); // 10deg 
  
  // control parameter
  particle_num_ = 100;
}


void SingleLFTracker::createRandomVehiclePositionParticle(const std::uint32_t particle_num)
{ 
  // Strictly, we should use multivariate normal distribution
  // Relaxed Results
  std::default_random_engine engine(seed_gen());
  std::uniform_real_distribution<> x_distribution(-covariance_(0, 0), covariance_(0, 0)); //std::normal_distribution<> x_distribution(0.0, std::sqrt(covariance_(0, 0)));
  std::uniform_real_distribution<> y_distribution(-covariance_(1, 1), covariance_(1, 1)); //std::normal_distribution<> y_distribution(0.0, std::sqrt(covariance_(1, 1)));
  //std::normal_distribution<> yaw_distribution(0.0, std::sqrt(covariance_(2, 2)));
  // generate vp
  VehicleParticle vp(position_ , width_, length_, orientation_);
  std::uint8_t index = vp.getNearestCornerIndex();

  // generate random vehicle particles
  for(std::uint32_t i =0; i<particle_num; i++){

    Eigen::Vector2d variation{x_distribution(engine),y_distribution(engine)};
    double orientation = orientation_; // + yaw_distribution(engine);
    VehicleParticle vp(position_ + variation, width_, length_, orientation);
    vp.corner_index_ = index;
    vehicle_particle_.push_back(vp);
  }

}


double get_interpV(int i, int len, double wid){
  return (  2.0*(double)i/(len-1.0) - 1.0 ) * wid;
}

void SingleLFTracker::createGridVehiclePositionParticle()
{ 
  // Strictly, we should use multivariate normal distribution
  // Relaxed Results
  // double x_wid, y_wid
  double yaw_wid, longitude, latitude;
  //double max_wid = 2; // 2m is max 
  int yaw_num = 1;
  int len_num = 25;
  int wid_num = 5;

  // x_wid = std::min(std::sqrt(covariance_(0,0)), max_wid);
  // y_wid = std::min(std::sqrt(covariance_(1,1)), max_wid);
  yaw_wid = DEG2RAD(15.0); 

  longitude = 3; //1.5m x2
  latitude = 0.4; // 0.2 m x 2
  // generate vp
  //VehicleParticle vp(position_ , width_, length_, orientation_);
  std::uint8_t index = default_vehicle_.getNearestCornerIndex();

  for(int i=0;i<len_num;i++){
    Eigen::Vector2d variationX{get_interpV(i,len_num,longitude)*cos(orientation_),get_interpV(i,len_num,longitude)*sin(orientation_)};
    for(int j=0;j<wid_num;j++){
      Eigen::Vector2d variationY{-get_interpV(j,wid_num,latitude)*sin(orientation_),get_interpV(j,wid_num,latitude)*cos(orientation_)};
      for(int k=0;k<yaw_num;k++){
        double orientation = orientation_ + get_interpV(k, yaw_num, yaw_wid);
        VehicleParticle vp(position_ + variationX + variationY, width_, length_, orientation);
        vp.corner_index_ = index;
        vehicle_particle_.push_back(vp);
      }
    }
  }
  particle_num_ = yaw_num*len_num*wid_num;

}


/**
 * @brief Extract Measurement Value from Particles
 * 
 * @note cov = 1/(1-w2) * (err*weights) @ err.T
 * @param likelihoods 
 * @param vectors 
 * @return Eigen::VectorXd 
 * 
 */
std::tuple<Eigen::Vector3d, Eigen::Matrix3d> SingleLFTracker::calcMeanAndCovFromParticles(std::vector<double> & likelihoods, std::vector<Eigen::Vector3d> vectors)
{

  auto loop = likelihoods.size();
  Eigen::Vector3d mean;
  Eigen::Matrix3d cov;
  double w2 = 0;
  double sum_likelihoods = 0;

  for(std::size_t i=0; i < loop; i++){
    sum_likelihoods +=  likelihoods[i]; 
  }


  for(std::size_t i=0; i < loop; i++){
    mean += vectors[i] * likelihoods[i] / sum_likelihoods; 
    w2 += likelihoods[i]*likelihoods[i] / sum_likelihoods  / sum_likelihoods;
  }

  for(std::size_t i=0; i < loop; i++){
    cov += (mean-vectors[i]) * (mean-vectors[i]).transpose() * likelihoods[i]  / sum_likelihoods /(1-w2);
  }

  return std::make_tuple(mean,cov); 
}


std::tuple<Eigen::Vector3d, Eigen::Matrix3d> SingleLFTracker::calcBestParticles(std::vector<double> & likelihoods, std::vector<Eigen::Vector3d> vectors)
{

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  Eigen::Matrix3d cov= Eigen::Matrix3d::Zero();
  double w2 = 0;
  double sum_likelihoods = 0;

  std::vector<std::size_t> max_indexes = findMaxIndices(likelihoods);

  //ROS_DEBUG("log:%i", count);
  //std::cout << "Max num " <<max_indexes.size() << ", " <<  default_likelihood_ << std::endl;
  //std::cout << "vp position " << default_vehicle_.center_.x() << ", "<< default_vehicle_.center_.y() << ", " << default_vehicle_.orientation_ <<std::endl;
  //std::cout << "costsum" << default_vehicle_.fine_likelihood_.costs_[0] << ", indexes " << default_vehicle_.corner_index_ << std::endl;

  for(std::size_t i: max_indexes){
    sum_likelihoods +=  likelihoods[i];
    //std::cout << "likelihoods sample " << likelihoods[i] << std::endl;
    // Use default Tracker value if not changed
    if(likelihoods[i] <= default_likelihood_){
      Eigen::Vector3d state(default_vehicle_.center_.x(), default_vehicle_.center_.y(), default_vehicle_.orientation_); 
      return std::make_tuple(state,cov);
    }
  }


  for(std::size_t i: max_indexes){
    mean += vectors[i] ; 
    w2 += likelihoods[i]*likelihoods[i] / sum_likelihoods  / sum_likelihoods;
  }
  mean = mean/(double) max_indexes.size();

  for(std::size_t i: max_indexes){
    cov += (mean-vectors[i]) * (mean-vectors[i]).transpose() * likelihoods[i]  / sum_likelihoods /(1-w2);
  }
  cov(0,0) = 0.4;
  cov(1,1) = 0.4;

  return std::make_tuple(mean,cov); 
}


// input: scan (theta, range) to calc likelihood
void SingleLFTracker::estimateState(const std::vector<Eigen::Vector2d> & scan)
{
  // temporary
  //createRandomVehiclePositionParticle(particle_num_);
  createGridVehiclePositionParticle();


  std::vector<double> likelihoods;
  std::vector<Eigen::Vector3d> states;

  // for(std::uint32_t i=0; i < particle_num_; i++){
  //   double likelihood = vehicle_particle_[i].calcCoarseLikelihood(scan);
  //   likelihoods.push_back(likelihood);
  //   Eigen::Vector3d state(vehicle_particle_[i].center_.x(), vehicle_particle_[i].center_.y(), vehicle_particle_[i].orientation_);
  //   states.push_back(state);
  // }

  // //auto mean_cov  = calcMeanAndCovFromParticles(likelihoods, states);
  // auto mean_cov  = calcBestParticles(likelihoods, states);
  
  // auto mstate = std::get<0>(mean_cov);
  // //covariance_ = std::get<1>(mean_cov);
  // position_ = Eigen::Vector2d(mstate.x(),mstate.y());
  // orientation_ = mstate.z();

  // // fine tracking
  // createRandomVehiclePositionParticle(particle_num_);
  // // tmp
  // likelihoods.clear();
  // states.clear();
  //std::vector<double> likelihoods2;
  //std::vector<Eigen::Vector3d> states2;

  for(std::uint32_t i=0; i < particle_num_; i++){
    double likelihood = vehicle_particle_[i].calcFineLikelihood(scan);
    likelihoods.push_back(likelihood);
    Eigen::Vector3d state(vehicle_particle_[i].center_.x(), vehicle_particle_[i].center_.y(), vehicle_particle_[i].orientation_);
    states.push_back(state);
  }

  default_vehicle_.corner_index_ = default_vehicle_.getNearestCornerIndex();
  default_likelihood_ = default_vehicle_.calcFineLikelihood(scan);

  

  int corner_index1 = default_vehicle_.corner_index_ + 1;
  int corner_index2 = default_vehicle_.corner_index_ - 1;
  if(corner_index1 == 4){ corner_index1 = 0;}
  if(corner_index2 == -1){ corner_index2 = 3;}
  double max_angle = std::atan2(default_vehicle_.corner_points_[corner_index1].y(), default_vehicle_.corner_points_[corner_index1].x());
  double min_angle = std::atan2(default_vehicle_.corner_points_[corner_index2].y(), default_vehicle_.corner_points_[corner_index2].x());
  if(max_angle < min_angle){
    max_angle += 2.0 * M_PI;
  }

  //output scan data
  std::vector<Eigen::Vector2d>  xy_measurements;
  for(auto sc_: scan){
    // skip invalid scan
    if(!std::isfinite(sc_.y())){
      continue;
    }
    if( sc_.x() < max_angle && sc_.x() > min_angle){
      Eigen::Vector2d xy{sc_.y()*cos(sc_.x()), sc_.y()*sin(sc_.x())};
      xy_measurements.push_back(xy);
    }else if( sc_.x()+ 2.0*M_PI < max_angle && sc_.x()+2.0*M_PI > min_angle ){
      Eigen::Vector2d xy{sc_.y()*cos(sc_.x()), sc_.y()*sin(sc_.x())};
      xy_measurements.push_back(xy);
    }
  }
  xy_measurements.push_back(Eigen::Vector2d{0,0}); // for zerodivision escape 

  std::vector<Eigen::Vector2d> local_scan;
  default_vehicle_.toLocalCoordinate(xy_measurements, default_vehicle_.center_, default_vehicle_.orientation_, local_scan);
//  std::cout << local_scan[0].x() << ", " << local_scan[0].y() << std::endl;
//  auto mean_cov_  = calcMeanAndCovFromParticles(likelihoods, states);
auto mean_cov_  = calcBestParticles(likelihoods, states);


  auto mstate_ = std::get<0>(mean_cov_);
  covariance_ = std::get<1>(mean_cov_);


  // Debugging
  // check warping objects
  if(( default_vehicle_.center_ - mstate_.head<2>()).norm()>1.0 || std::abs(orientation_ - mstate_.z() > DEG2RAD(9))){
    //std::cout << "max indexes: " << max_indexes.size() << std::endl;
    //std::cout << default_likelihood_ << " " << likelihoods[max_indexes[0]] << std::endl;
    std::cout << "vp position " << default_vehicle_.center_.x() << ", "<< default_vehicle_.center_.y() << ", " << default_vehicle_.orientation_ <<std::endl;
    //std::cout << "mean position " << mean.x() << ", "<< mean.y() << ", " << mean.z() <<std::endl;
    // output to file
    std::ofstream writing_file;
    std::string filename = "likelihoods.txt";
    writing_file.open(filename, std::ios::app);
    nlohmann::json vehicles;
    for(std::uint32_t i=0; i<particle_num_; i++){
      nlohmann::json vp;
      vp["likelihood"] = likelihoods[i];
      vp["x"] = vehicle_particle_[i].center_.x();
      vp["y"] = vehicle_particle_[i].center_.y();
      vp["yaw"] = vehicle_particle_[i].orientation_;
      vehicles.push_back(vp);
    }
    {
      nlohmann::json vp;
      vp["likelihood"] = default_likelihood_;
      vp["x"] = default_vehicle_.center_.x();
      vp["y"] = default_vehicle_.center_.y();
      vp["yaw"] = default_vehicle_.orientation_;
      vp["min_angle"] = min_angle;
      vp["max_angle"] = max_angle;
      vp["corner_index"] = default_vehicle_.corner_index_;
      vp["w"] = width_;
      vp["l"] = length_;
      vehicles.push_back(vp);
    }
    writing_file << vehicles << "\n" <<std::endl;
    writing_file.close();

    filename = "scan_points.txt";
    nlohmann::json json_scan;
    for(auto each_scan: local_scan){
      nlohmann::json sc;
      sc["x"] = each_scan.x();
      sc["y"] = each_scan.y();
      json_scan.push_back(sc);
    }
    writing_file.open(filename, std::ios::app);
    writing_file << json_scan << "\n" <<std::endl;
    writing_file.close();
    
  }

  position_ = Eigen::Vector2d(mstate_.x(),mstate_.y());
  orientation_ = mstate_.z();

}


autoware_auto_perception_msgs::msg::TrackedObject SingleLFTracker::toTrackedObject(autoware_auto_perception_msgs::msg::TrackedObject &object)
{
  autoware_auto_perception_msgs::msg::TrackedObject estimated_object;
  estimated_object = object;
  // estimated_object.object_id = object.object_id;
  // estimated_object.existence_probability = object.existence_probability;
  // estimated_object.classification = object.classification;
  // estimated_object.shape = object.shape;
  // estimated_object.kinematics.pose_with_covariance = object.kinematics.pose_with_covariance;
  estimated_object.kinematics.pose_with_covariance.pose.position.x = position_.x();
  estimated_object.kinematics.pose_with_covariance.pose.position.y = position_.y();
  const float yaw_hat = tier4_autoware_utils::normalizeRadian(orientation_);
  estimated_object.kinematics.pose_with_covariance.pose.orientation = tier4_autoware_utils::createQuaternionFromYaw(yaw_hat);
  eigenMatrixToROS2Covariance(covariance_, estimated_object.kinematics.pose_with_covariance.covariance);
  return estimated_object;
}

/// PointCloud2 to eigen vector
// std::vector<Eigen::Vector2d> PoinCloud2ToScan(sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud)
// {
//   cloud->data.
// }



LikelihoodFieldTracker::LikelihoodFieldTracker(const rclcpp::NodeOptions & node_options)
: rclcpp::Node("likelihood_field_tracker", node_options),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_)
{
  // Create publishers and subscribers
  trackers_sub_ = create_subscription<autoware_auto_perception_msgs::msg::TrackedObjects>(
    "~/input/tracked_objects", rclcpp::QoS{1},
    std::bind(&TrackerHandler::onTrackedObjects, &tracker_handler_, std::placeholders::_1));

  scans_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
    "/perception/occupancy_grid_map/virtual_scan/laserscan", rclcpp::QoS{1}.best_effort(),
    std::bind(&LikelihoodFieldTracker::onObjects, this, std::placeholders::_1) );
  
  // initial_objects_sub_ =
  //   create_subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
  //     "~/input/initial_objects", rclcpp::QoS{1},
  //     std::bind(&LikelihoodFieldTracker::onObjects, this, std::placeholders::_1));
  objects_pub_ = create_publisher<autoware_auto_perception_msgs::msg::DetectedObjects>(
    "/perception/lftracker", rclcpp::QoS{1});

  //ignore_unknown_tracker_ = declare_parameter<bool>("ignore_unknown_tracker", true);

  /// ???
  // shape_estimator_ = std::make_shared<ShapeEstimator>(true, true);
  // cluster_ = std::make_shared<euclidean_cluster::VoxelGridBasedEuclideanCluster>(
  //   false, 10, 10000, 0.7, 0.3, 0);
  // debugger_ = std::make_shared<Debugger>(this);
}

void LikelihoodFieldTracker::onObjects(
  const sensor_msgs::msg::LaserScan::ConstSharedPtr input_msg)
{
  autoware_auto_perception_msgs::msg::DetectedObjects detected_objects;
  detected_objects.header = input_msg->header;

  // get objects from tracking module
  autoware_auto_perception_msgs::msg::DetectedObjects tracked_objects;
  {
    autoware_auto_perception_msgs::msg::TrackedObjects objects, estimated_objects;
    estimated_objects.header = input_msg->header;
    estimated_objects.header.frame_id = "map";
    // Do prediction
    const bool available_trackers =
      tracker_handler_.estimateTrackedObjects(input_msg->header.stamp, objects);
    
    // Return if none object is tracked
    if (!available_trackers) {
      objects_pub_->publish(detected_objects);
      return;
    } 

    // LaserScan to std::vector
    std::vector<Eigen::Vector2d> scan_vec;
    for(long unsigned int i=0; i < input_msg->ranges.size();i++){
  
    //std::cout<<i<< std::endl;
      double th = input_msg->angle_min+ (double)i*input_msg->angle_increment;
      double range = input_msg->ranges[i];
      //if( range > input_msg->range_max || range < input_msg->range_min){
      //  continue;
      //}
      //Eigen::Vector2d xy{range*cos(th), range*sin(th)};
      Eigen::Vector2d d_theta{th, range};
      scan_vec.push_back(d_theta);
    }
    //std::cout << scan_vec[0].x() << ", " <<scan_vec[0].y();

    RCLCPP_WARN(
      rclcpp::get_logger("LFTracker"),
      "SubsCribed Scan");

    // map -> base_link (may be unneccesary)
    //geometry_msgs::msg::TransformStamped  map_to_baselink = tf_buffer_.lookupTransform(input_msg->header.frame_id, "base_link", input_msg->header.stamp);
    
    // tmp
    // lf fitting for each objects
    for(long unsigned int i=0; i < objects.objects.size();i++){
      // apply only for vehicle
      const auto label = objects.objects[i].classification.front().label;
      //std::cout << label <<std::endl;
      const bool is_vehicle =   Label::CAR == label || Label::TRUCK == label || Label::BUS == label || Label::TRAILER == label;
      if(!is_vehicle){
        continue;
      }

      // Create Tracker
      autoware_auto_perception_msgs::msg::TrackedObject local_object, estimated_local_object, estimated_object;

      // init each vehicle particle
      transformObjectsFromMapToBaselink(objects.objects[i], local_object, input_msg->header, tf_buffer_);
      SingleLFTracker vehicle(local_object);
      vehicle.estimateState(scan_vec);
      estimated_local_object = vehicle.toTrackedObject(objects.objects[i]);

      transformObjectsFromBaselinkToMap(estimated_local_object, estimated_object, input_msg->header, tf_buffer_);
      estimated_objects.objects.push_back(estimated_object);
    }
    

    // to simplify post processes, convert tracked_objects to DetectedObjects message.
    detected_objects = perception_utils::toDetectedObjects(estimated_objects);

    // clear objects
    //tracker_handler_
  }

  // // merge over segmented objects
  // tier4_perception_msgs::msg::DetectedObjectsWithFeature merged_objects;
  // autoware_auto_perception_msgs::msg::DetectedObjects no_found_tracked_objects ;
  // mergeOverSegmentedObjects(tracked_objects, *input_msg, no_found_tracked_objects, merged_objects);
  // debugger_->publishMergedObjects(merged_objects);

  // // divide under segmented objects
  // tier4_perception_msgs::msg::DetectedObjectsWithFeature divided_objects;
  // autoware_auto_perception_msgs::msg::DetectedObjects temp_no_found_tracked_objects;
  // divideUnderSegmentedObjects(
  //   no_found_tracked_objects, *input_msg, temp_no_found_tracked_objects, divided_objects);
  // debugger_->publishDividedObjects(divided_objects);

  // // merge under/over segmented objects to build output objects
  // for (const auto & merged_object : merged_objects.feature_objects) {
  //   detected_objects.objects.push_back(merged_object.object);
  // }
  // for (const auto & divided_object : divided_objects.feature_objects) {
  //   detected_objects.objects.push_back(divided_object.object);
  // }

  objects_pub_->publish(detected_objects);
}

// void LikelihoodFieldTracker::divideUnderSegmentedObjects(
//   const autoware_auto_perception_msgs::msg::DetectedObjects & tracked_objects,
//   const tier4_perception_msgs::msg::DetectedObjectsWithFeature & in_cluster_objects,
//   autoware_auto_perception_msgs::msg::DetectedObjects & out_no_found_tracked_objects,
//   tier4_perception_msgs::msg::DetectedObjectsWithFeature & out_objects)
// {
//   constexpr float recall_min_threshold = 0.4;
//   constexpr float precision_max_threshold = 0.5;
//   constexpr float max_search_range = 6.0;
//   constexpr float min_score_threshold = 0.4;

//   out_objects.header = in_cluster_objects.header;
//   out_no_found_tracked_objects.header = tracked_objects.header;

//   for (const auto & tracked_object : tracked_objects.objects) {
//     const auto & label = tracked_object.classification.front().label;
//     if (ignore_unknown_tracker_ && (label == Label::UNKNOWN)) continue;

//     std::optional<tier4_perception_msgs::msg::DetectedObjectWithFeature>
//       highest_score_divided_object = std::nullopt;
//     float highest_score = 0.0;

//     for (const auto & initial_object : in_cluster_objects.feature_objects) {
//       // search near object
//       const float distance = tier4_autoware_utils::calcDistance2d(
//         tracked_object.kinematics.pose_with_covariance.pose,
//         initial_object.object.kinematics.pose_with_covariance.pose);
//       if (max_search_range < distance) {
//         continue;
//       }
//       // detect under segmented cluster
//       const float recall = perception_utils::get2dRecall(initial_object.object, tracked_object);
//       const float precision =
//         perception_utils::get2dPrecision(initial_object.object, tracked_object);
//       const bool is_under_segmented =
//         (recall_min_threshold < recall && precision < precision_max_threshold);
//       if (!is_under_segmented) {
//         continue;
//       }
//       // optimize clustering
//       tier4_perception_msgs::msg::DetectedObjectWithFeature divided_object;
//       float score = optimizeUnderSegmentedObject(
//         tracked_object, initial_object.feature.cluster, divided_object);
//       if (score < min_score_threshold) {
//         continue;
//       }

//       if (highest_score < score) {
//         highest_score = score;
//         highest_score_divided_object = divided_object;
//       }
//     }
//     if (highest_score_divided_object) {  // found
//       out_objects.feature_objects.push_back(highest_score_divided_object.value());
//     } else {  // not found
//       out_no_found_tracked_objects.objects.push_back(tracked_object);
//     }
//   }
// }

// float LikelihoodFieldTracker::optimizeUnderSegmentedObject(
//   const autoware_auto_perception_msgs::msg::DetectedObject & target_object,
//   const sensor_msgs::msg::PointCloud2 & under_segmented_cluster,
//   tier4_perception_msgs::msg::DetectedObjectWithFeature & output)
// {
//   constexpr float iter_rate = 0.8;
//   constexpr int iter_max_count = 5;
//   constexpr float initial_cluster_range = 0.7;
//   float cluster_range = initial_cluster_range;
//   constexpr float initial_voxel_size = initial_cluster_range / 2.0f;
//   float voxel_size = initial_voxel_size;

//   const auto & label = target_object.classification.front().label;

//   // initialize clustering parameters
//   euclidean_cluster::VoxelGridBasedEuclideanCluster cluster(
//     false, 4, 10000, initial_cluster_range, initial_voxel_size, 0);

//   // convert to pcl
//   pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cluster(new pcl::PointCloud<pcl::PointXYZ>);
//   pcl::fromROSMsg(under_segmented_cluster, *pcl_cluster);

//   // iterate to find best fit divided object
//   float highest_iou = 0.0;
//   tier4_perception_msgs::msg::DetectedObjectWithFeature highest_iou_object;
//   for (int iter_count = 0; iter_count < iter_max_count;
//        ++iter_count, cluster_range *= iter_rate, voxel_size *= iter_rate) {
//     // divide under segmented cluster
//     std::vector<pcl::PointCloud<pcl::PointXYZ>> divided_clusters;
//     cluster.setTolerance(cluster_range);
//     cluster.setVoxelLeafSize(voxel_size);
//     cluster.cluster(pcl_cluster, divided_clusters);

//     // find highest iou object in divided clusters
//     float highest_iou_in_current_iter = 0.0f;
//     tier4_perception_msgs::msg::DetectedObjectWithFeature highest_iou_object_in_current_iter;
//     highest_iou_object_in_current_iter.object.classification = target_object.classification;
//     for (const auto & divided_cluster : divided_clusters) {
//       bool is_shape_estimated = shape_estimator_->estimateShapeAndPose(
//         label, divided_cluster,
//         getReferenceYawInfo(
//           label, tf2::getYaw(target_object.kinematics.pose_with_covariance.pose.orientation)),
//         highest_iou_object_in_current_iter.object.shape,
//         highest_iou_object_in_current_iter.object.kinematics.pose_with_covariance.pose);
//       if (!is_shape_estimated) {
//         continue;
//       }
//       const float iou =
//         perception_utils::get2dIoU(highest_iou_object_in_current_iter.object, target_object);
//       if (highest_iou_in_current_iter < iou) {
//         highest_iou_in_current_iter = iou;
//         setClusterInObjectWithFeature(
//           under_segmented_cluster.header, divided_cluster, highest_iou_object_in_current_iter);
//       }
//     }

//     // finish iteration when current score is under previous score
//     if (highest_iou_in_current_iter < highest_iou) {
//       break;
//     }

//     // copy for next iteration
//     highest_iou = highest_iou_in_current_iter;
//     highest_iou_object = highest_iou_object_in_current_iter;
//   }

//   // build output
//   highest_iou_object.object.classification = target_object.classification;
//   highest_iou_object.object.existence_probability =
//     perception_utils::get2dIoU(target_object, highest_iou_object.object);

//   output = highest_iou_object;
//   return highest_iou;
// }

// void LikelihoodFieldTracker::mergeOverSegmentedObjects(
//   const autoware_auto_perception_msgs::msg::DetectedObjects & tracked_objects,
//   const tier4_perception_msgs::msg::DetectedObjectsWithFeature & in_cluster_objects,
//   autoware_auto_perception_msgs::msg::DetectedObjects & out_no_found_tracked_objects,
//   tier4_perception_msgs::msg::DetectedObjectsWithFeature & out_objects)
// {
//   constexpr float precision_threshold = 0.5;
//   constexpr float max_search_range = 5.0;
//   out_objects.header = in_cluster_objects.header;
//   out_no_found_tracked_objects.header = tracked_objects.header;

//   for (const auto & tracked_object : tracked_objects.objects) {
//     const auto & label = tracked_object.classification.front().label;
//     if (ignore_unknown_tracker_ && (label == Label::UNKNOWN)) continue;

//     // extend shape
//     autoware_auto_perception_msgs::msg::DetectedObject extended_tracked_object = tracked_object;
//     extended_tracked_object.shape = extendShape(tracked_object.shape, /*scale*/ 1.1);

//     pcl::PointCloud<pcl::PointXYZ> pcl_merged_cluster;
//     for (const auto & initial_object : in_cluster_objects.feature_objects) {
//       const float distance = tier4_autoware_utils::calcDistance2d(
//         tracked_object.kinematics.pose_with_covariance.pose,
//         initial_object.object.kinematics.pose_with_covariance.pose);

//       if (max_search_range < distance) {
//         continue;
//       }

//       // If there is an initial object in the tracker, it will be merged.
//       const float precision =
//         perception_utils::get2dPrecision(initial_object.object, extended_tracked_object);
//       if (precision < precision_threshold) {
//         continue;
//       }
//       pcl::PointCloud<pcl::PointXYZ> pcl_cluster;
//       pcl::fromROSMsg(initial_object.feature.cluster, pcl_cluster);
//       pcl_merged_cluster += pcl_cluster;
//     }

//     if (pcl_merged_cluster.points.empty()) {  // if clusters aren't found
//       out_no_found_tracked_objects.objects.push_back(tracked_object);
//       continue;
//     }

//     // build output clusters
//     tier4_perception_msgs::msg::DetectedObjectWithFeature feature_object;
//     feature_object.object.classification = tracked_object.classification;

//     bool is_shape_estimated = shape_estimator_->estimateShapeAndPose(
//       label, pcl_merged_cluster,
//       getReferenceYawInfo(
//         label, tf2::getYaw(tracked_object.kinematics.pose_with_covariance.pose.orientation)),
//       feature_object.object.shape, feature_object.object.kinematics.pose_with_covariance.pose);
//     if (!is_shape_estimated) {
//       out_no_found_tracked_objects.objects.push_back(tracked_object);
//       continue;
//     }

//     feature_object.object.existence_probability =
//       perception_utils::get2dIoU(tracked_object, feature_object.object);
//     setClusterInObjectWithFeature(in_cluster_objects.header, pcl_merged_cluster, feature_object);
//     out_objects.feature_objects.push_back(feature_object);
//   }
// }

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(LikelihoodFieldTracker)
