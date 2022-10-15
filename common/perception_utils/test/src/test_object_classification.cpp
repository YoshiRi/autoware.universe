// Copyright 2022 TIER IV, Inc.
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

#include "perception_utils/object_classification.hpp"

#include <gtest/gtest.h>

namespace
{
autoware_auto_perception_msgs::msg::ObjectClassification createObjectClassification(
  const std::uint8_t label, const double probability)
{
  autoware_auto_perception_msgs::msg::ObjectClassification classification;
  classification.label = label;
  classification.probability = probability;

  return classification;
}


TEST(object_classification, test_getHighestProbLabel)
{
  using autoware_auto_perception_msgs::msg::ObjectClassification;
  using perception_utils::getHighestProbLabel;

  {  // empty
    std::vector<autoware_auto_perception_msgs::msg::ObjectClassification> classification;
    std::uint8_t label = getHighestProbLabel(classification);
    EXPECT_EQ(label, ObjectClassification::UNKNOWN);
  }

  {  // normal case
    std::vector<autoware_auto_perception_msgs::msg::ObjectClassification> classification;
    classification.push_back(createObjectClassification(ObjectClassification::CAR, 0.5));
    classification.push_back(createObjectClassification(ObjectClassification::TRUCK, 0.8));
    classification.push_back(createObjectClassification(ObjectClassification::BUS, 0.7));

    std::uint8_t label = getHighestProbLabel(classification);
    EXPECT_EQ(label, ObjectClassification::TRUCK);
  }

  {  // labels with the same probability
    std::vector<autoware_auto_perception_msgs::msg::ObjectClassification> classification;
    classification.push_back(createObjectClassification(ObjectClassification::CAR, 0.8));
    classification.push_back(createObjectClassification(ObjectClassification::TRUCK, 0.8));
    classification.push_back(createObjectClassification(ObjectClassification::BUS, 0.7));

    std::uint8_t label = getHighestProbLabel(classification);
    EXPECT_EQ(label, ObjectClassification::CAR);
  }
}

// Test isVehicle
TEST(object_classification, test_isVehicle)
{

  using autoware_auto_perception_msgs::msg::ObjectClassification;
  using perception_utils::isVehicle;

  {// True Case with uint8_t
  std::vector<std::uint8_t> vehicle_labels;
  vehicle_labels.push_back(ObjectClassification::BICYCLE);
  vehicle_labels.push_back(ObjectClassification::BUS);
  vehicle_labels.push_back(ObjectClassification::CAR);
  vehicle_labels.push_back(ObjectClassification::MOTORCYCLE);
  vehicle_labels.push_back(ObjectClassification::TRAILER);
  vehicle_labels.push_back(ObjectClassification::TRUCK);
  
  for(auto label: vehicle_labels){
    EXPECT_TRUE(isVehicle(label));
  }
  }

  // False Case with uint8_t
  {
  std::vector<std::uint8_t> non_vehicle_labels;
  non_vehicle_labels.push_back(ObjectClassification::UNKNOWN);
  non_vehicle_labels.push_back(ObjectClassification::PEDESTRIAN);
  for(auto label: non_vehicle_labels){
    EXPECT_FALSE(isVehicle(label));
  }
  }

  // True Case with object_classifications
  {  // normal case
    std::vector<autoware_auto_perception_msgs::msg::ObjectClassification> classification;
    classification.push_back(createObjectClassification(ObjectClassification::CAR, 0.5));
    classification.push_back(createObjectClassification(ObjectClassification::TRUCK, 0.8));
    classification.push_back(createObjectClassification(ObjectClassification::BUS, 0.7));
    EXPECT_TRUE(isVehicle(classification));
  }

  // False Case with object_classifications
  {  // false case
    std::vector<autoware_auto_perception_msgs::msg::ObjectClassification> classification;
    classification.push_back(createObjectClassification(ObjectClassification::PEDESTRIAN, 0.8));
    classification.push_back(createObjectClassification(ObjectClassification::BICYCLE, 0.7));
    EXPECT_FALSE(isVehicle(classification));
  }

} // TEST isVehicle

}  // namespace
