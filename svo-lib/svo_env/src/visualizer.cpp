// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).

#include <svo_env/visualizer.h>

#include <deque>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>

#include <svo/common/frame.h>
#include <svo/direct/feature_detection_utils.h>

namespace svo
{
cv::Mat Visualizer::createImagesWithFeatures(const FrameBundlePtr& frame_bundle,
                                           const int64_t timestamp,
                                           const bool draw_boundary)
{
  cv::Mat img_rgb;
  for (size_t i = 0; i < frame_bundle->size(); ++i)
  {
    FramePtr frame = frame_bundle->at(i);
    
    feature_detection_utils::drawFeatures(*frame, img_pub_level_, true,
                                          &img_rgb);
    if (draw_boundary)
    {
      cv::rectangle(img_rgb, cv::Point2f(0.0, 0.0),
                    cv::Point2f(img_rgb.cols, img_rgb.rows),
                    cv::Scalar(0, 255, 0), 6);
    }
    writeCaptionStr(img_rgb);
    
  }

  return img_rgb;
}

void Visualizer::writeCaptionStr(cv::Mat img_rgb)
{
  if (viz_caption_str_)
  {
    cv::putText(img_rgb, img_caption_, cv::Point(20, 20),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,250),
                1, cv::LINE_AA);
  }
}

}  // end namespace svo
