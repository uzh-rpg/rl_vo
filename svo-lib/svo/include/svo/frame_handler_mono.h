// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <svo/frame_handler_base.h>

namespace svo {

/// Monocular Semi-Direct Visual Odometry Pipeline
///
/// References:
/// [1] Christian Forster, Matia Pizzoli, Davide Scaramuzza, "SVO: Semi-Direct
/// Monocular Visual Odometry for Micro Aerial Vehicles", Proc. IEEE International
/// Conference on Robotics and Automation, Honkong 2015.
///
/// This is the main interface class of the VO. It derives from FrameHandlerBase
/// which maintains the state machine (start, stop, reset).
class FrameHandlerMono : public FrameHandlerBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<FrameHandlerMono> Ptr;

  /// Default constructor
  FrameHandlerMono(
      const BaseOptions& base_options,
      const DepthFilterOptions& depth_filter_options,
      const DetectorOptions& feature_detector_options,
      const InitializationOptions& init_options,
      const ReprojectorOptions& reprojector_options,
      const FeatureTrackerOptions& tracker_options,
      const CameraBundle::Ptr& cam);

  virtual ~FrameHandlerMono() = default;

  /// @name Main Interface
  ///
  /// These are the main functions that you will need to interface
  /// with the visual odometry pipeline. Other important functions like
  /// start(), reset(), stage() are provided in the FrameHandlerBase class which
  /// maintains the state-machine.
  ///
  /// @{

  /// After adding an image to SVO with addImage(), the image is saved in a
  /// Frame class, the frame is processed and it's pose computed. Most likely you
  /// want to know the pose of the cam. Therefore, use this function to access
  /// the pose of the last provided image (e.g., vo->lastFrame()->T_world_cam()).
  /// \return FramePtr !!!IMPORTANT!!! can be nullptr if something did not work
  /// well or if you call this function when pipeline is not running, i.e.,
  /// vo->stage()==STAGE_PAUSED.
  FramePtr lastFrame() const;

  /// @}

  inline CameraPtr cam() const { return cams_->getCameraShared(0); }

protected:

  // helpers
  const FramePtr& newFrame() const;

  // unsafe because last_frame might be nullptr. use this function only
  // when you are sure that it is set!
  const FramePtr& lastFrameUnsafe() const;

  bool haveLastFrame() const;

  /// Pipeline implementation. Called by base class.
  virtual UpdateResult processFrameBundle() override;

  /// Processes the first frame and sets it as a keyframe.
  virtual UpdateResult processFirstFrame();

  /// Processes all frames after the first two keyframes.
  virtual UpdateResult processFrame();

  /// Try relocalizing the frame at relative position to provided keyframe.
  virtual UpdateResult relocalizeFrame(
      const Transformation& T_cur_ref,
      const FramePtr& ref_keyframe);

  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll() override;
};

} // namespace svo

