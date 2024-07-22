// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).

#pragma once

#include <utility>  // std::pair
#include <iostream>

#include <boost/shared_ptr.hpp>

#include <svo/global.h>


namespace svo
{
// forward declarations
class FrameHandlerBase;

/// Publish visualisation messages to ROS.
class Visualizer
{
public:
  size_t img_pub_level_;
  bool viz_caption_str_;
  std::string img_caption_;

  Visualizer() = default;
  ~Visualizer() = default;

  cv::Mat createImagesWithFeatures(const FrameBundlePtr& frame_bundle,
                                   const int64_t timestamp,
                                   const bool draw_boundary);
  
  void writeCaptionStr(cv::Mat img);
};

}  // end namespace svo
