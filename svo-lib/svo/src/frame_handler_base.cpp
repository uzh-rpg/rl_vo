// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include "svo/frame_handler_base.h"

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <opencv2/imgproc/imgproc.hpp>

#include <svo/common/conversions.h>
#include <svo/common/point.h>
#include <svo/direct/depth_filter.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>
#include <svo/direct/matcher.h>
#include <svo/tracker/feature_tracker.h>

# include <svo/img_align/sparse_img_align.h>

#include "svo/initialization.h"
#include "svo/map.h"
#include "svo/reprojector.h"
#include "svo/pose_optimizer.h"


namespace svo
{

FrameHandlerBase::FrameHandlerBase(const BaseOptions& base_options, const ReprojectorOptions& reprojector_options,
                                   const DepthFilterOptions& depthfilter_options,
                                   const DetectorOptions& detector_options, const InitializationOptions& init_options,
                                   const FeatureTrackerOptions& tracker_options, const CameraBundle::Ptr& cameras) :
    options_(base_options), cams_(cameras), stage_(Stage::kPaused), set_reset_(false), set_start_(false), map_(new Map), acc_frame_timings_(
        10), acc_num_obs_(10), num_obs_last_(0), tracking_quality_(TrackingQuality::kInsufficient), relocalization_n_trials_(
        0)
{
  // sanity checks
  CHECK_EQ(reprojector_options.cell_size, detector_options.cell_size);

  // RL addition
  initial_cell_size_ = reprojector_options.cell_size;

  need_new_kf_ = std::bind(&FrameHandlerBase::needNewKf, this, std::placeholders::_1);

  // init modules
  reprojectors_.reserve(cams_->getNumCameras());
  for (size_t camera_idx = 0; camera_idx < cams_->getNumCameras(); ++camera_idx)
  {
    reprojectors_.emplace_back(new Reprojector(reprojector_options, camera_idx));
  }
  SparseImgAlignOptions img_align_options;
  img_align_options.max_level = options_.img_align_max_level;
  img_align_options.min_level = options_.img_align_min_level;
  img_align_options.robustification = options_.img_align_robustification;
  img_align_options.use_distortion_jacobian = options_.img_align_use_distortion_jacobian;
  img_align_options.estimate_illumination_gain = options_.img_align_est_illumination_gain;
  img_align_options.estimate_illumination_offset = options_.img_align_est_illumination_offset;

  sparse_img_align_.reset(new SparseImgAlign(SparseImgAlign::getDefaultSolverOptions(), img_align_options));
  pose_optimizer_.reset(new PoseOptimizer(PoseOptimizer::getDefaultSolverOptions()));
  if (options_.poseoptim_using_unit_sphere)
    pose_optimizer_->setErrorType(PoseOptimizer::ErrorType::kBearingVectorDiff);

  // DEBUG ***
  //pose_optimizer_->initTracing(options_.trace_dir);
  DetectorOptions detector_options2 = detector_options;
  //detector_options2.detector_type = DetectorType::kGridGrad;

  depth_filter_.reset(new DepthFilter(depthfilter_options, detector_options2, cams_));
  initializer_ = initialization_utils::makeInitializer(init_options, tracker_options, detector_options, cams_);
  overlap_kfs_.resize(cams_->getNumCameras());

  VLOG(1) << "SVO initialized";
}

FrameHandlerBase::~FrameHandlerBase()
{
  VLOG(1) << "SVO destructor invoked";
}

//------------------------------------------------------------------------------
bool FrameHandlerBase::addImageBundle(const std::vector<cv::Mat>& imgs, 
                                      const uint64_t timestamp, 
                                      Ref<Vector<>> actions, 
                                      const bool use_RL_actions,
                                      Ref<Vector<>> observations,
                                      double &dones,
                                      double &stage,
                                      double &runtime,
                                      const bool use_gt_init_pose,
                                      Ref<Vector<>> gt_init_pose)
{
  /// @implements set actions ===================================
  // Keyframe action
  use_RL_actions_ = use_RL_actions;
  action_newkeyframe_ = actions(0) == 1;
  use_gt_init_pose_ = use_gt_init_pose;

  // Grid Size action
  float new_cell_size;
  if (!use_RL_actions || prev_state_reset_) {
    new_cell_size = initial_cell_size_;
  } else {
    new_cell_size = actions(1);
  }

  if (use_RL_actions || prev_state_reset_) {
      int new_n_cols = std::ceil(static_cast<double>(cams_->getCameraShared(0)->imageWidth())/new_cell_size);
      int new_n_rows = std::ceil(static_cast<double>(cams_->getCameraShared(0)->imageHeight())/new_cell_size);

      depth_filter_->feature_detector_->grid_.change_cell_size(new_cell_size, new_n_cols, new_n_rows);
      depth_filter_->feature_detector_->closeness_check_grid_.change_cell_size(new_cell_size, new_n_cols, new_n_rows);
      depth_filter_->feature_detector_->options_.cell_size = new_cell_size;
      reprojectors_.at(0)->grid_.reset(new OccupandyGrid2D(new_cell_size, new_n_cols, new_n_rows));
      reprojectors_.at(0)->options_.cell_size = new_cell_size;
  }
  prev_state_reset_ = false;

  auto start_time = std::chrono::high_resolution_clock::now();
  /// @implements end ===========================================

  if (last_frames_)
  {
    // check if the timestamp is valid
    if (last_frames_->getMinTimestampNanoseconds() >= static_cast<int64_t>(timestamp))
    {
      VLOG(4) << "Dropping frame: timestamp older than last frame of id " << last_frames_->getBundleId();
      SVO_WARN_STREAM("Dropping frame: timestamp older than last frame.");
      return false;
    }
  }

//  if (options_.trace_statistics)
//  {
//    SVO_START_TIMER("pyramid_creation");
//  }
  CHECK_EQ(imgs.size(), cams_->getNumCameras());
  std::vector<FramePtr> frames;
  for (size_t i = 0; i < imgs.size(); ++i)
  {
    frames.push_back(
        std::make_shared < Frame
            > (cams_->getCameraShared(i), imgs[i].clone(), timestamp, options_.img_align_max_level + 1, added_frames_));
    /// @todo remove
    // frames.back()->set_T_cam_imu(cams_->get_T_C_B(i));
    frames.back()->setNFrameIndex(i);

    if (use_gt_init_pose_) {
        Quaternion gt_init_quat = Quaternion(gt_init_pose(6), gt_init_pose(3), gt_init_pose(4), gt_init_pose(5));
        frames.back()->initial_gt_T_f_w_ = Transformation(gt_init_quat,
                                                          gt_init_pose.segment(0, 3));
    }
    added_frames_++;
  }
  FrameBundlePtr frame_bundle(new FrameBundle(frames));
//  if (options_.trace_statistics)
//  {
//    SVO_STOP_TIMER("pyramid_creation");
//  }

  // Process frame bundle.
  bool success_frame = addFrameBundle(frame_bundle);


  /// @implements Observations ===============================================================
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  runtime = duration.count();

  if (last_frames_==nullptr) {
    return success_frame;
  }
  svo::FramePtr last_frame = last_frames_->at(0);

  // Observations
  // Number of features
  observations(0) = last_frame->numTrackedLandmarks();

  // Observation last keyframe
  svo::FramePtr last_keyframe = map_->getLastKeyframe();
  if (last_keyframe == nullptr) {
    observations(1) = 0;
    observations.segment(2, 7).setConstant(-1.0);
  } else {
    // Timesteps since last keyframe
    observations(1) = last_frame->id() - last_keyframe->id();

    // Relative pose to last keyframe
    kindr::minimal::QuatTransformation T_WC_last_key = last_keyframe->T_world_cam();
    kindr::minimal::QuatTransformation T_WC_last_frame = last_frame->T_world_cam();
    kindr::minimal::QuatTransformation T_relative = T_WC_last_key.inverse() * T_WC_last_frame;
    observations.segment(2, 3) = T_relative.getPosition();
    Eigen::Quaterniond quaternion_data = T_relative.getEigenQuaternion();
    Eigen::Vector4d quaternion_vector;
    quaternion_vector << quaternion_data.x(), quaternion_data.y(), quaternion_data.z(), quaternion_data.w();
    observations.segment(5, 4) = quaternion_vector;
  }

  // Observation first visible keyframe
  if (!(map_->keyframes_.size() < 2)) {
    FramePtr first_keyframe;
    double min_id = last_frame->id();
    for(const auto& kf : map_->keyframes_)
    {
      double id = kf.second->id();
      if(id < min_id) {
        min_id = id;
        first_keyframe = kf.second;
       }
    }

    // Timesteps since first keyframe
    observations(9) = last_frame->id() - first_keyframe->id();

    // Relative pose to first keyframe
    kindr::minimal::QuatTransformation T_WC_first_key = first_keyframe->T_world_cam();
    kindr::minimal::QuatTransformation T_WC_last_frame = last_frame->T_world_cam();
    kindr::minimal::QuatTransformation T_relative = T_WC_first_key.inverse() * T_WC_last_frame;
    observations.segment(10, 3) = T_relative.getPosition();
    Eigen::Quaterniond quaternion_data = T_relative.getEigenQuaternion();
    Eigen::Vector4d quaternion_vector;
    quaternion_vector << quaternion_data.x(), quaternion_data.y(), quaternion_data.z(), quaternion_data.w();
    observations.segment(13, 4) = quaternion_vector;
  } else {
    observations(9) = 0;
    observations.segment(10, 7).setConstant(-1.0);
  }

  // Observations relative pose to last position prediction
  if (T_WC_last_pose_set_) {
    kindr::minimal::QuatTransformation T_WC_this_frame = last_frame->T_world_cam();
    kindr::minimal::QuatTransformation T_relative = T_WC_last_pose_.inverse() * T_WC_this_frame;
    observations.segment(17, 3) = T_relative.getPosition();
    Eigen::Quaterniond quaternion_data = T_relative.getEigenQuaternion();
    Eigen::Vector4d quaternion_vector;
    quaternion_vector << quaternion_data.x(), quaternion_data.y(), quaternion_data.z(), quaternion_data.w();
    observations.segment(20, 4) = quaternion_vector;
  }
  T_WC_last_pose_ = last_frame->T_world_cam();
  T_WC_last_pose_set_ = true;

  // Features
  Eigen::Matrix<FloatType, 2, Eigen::Dynamic, Eigen::ColMajor> keypoints = last_frame->px_vec_;
  for(size_t i = 0; i < last_frame->num_features_; ++i) {
    if (i >=180) { break; }
    if (isSeed(last_frame->type_vec_[i])) {
      observations(24+i*3) = std::min(last_frame->invmu_sigma2_a_b_vec_.col(i)(0), 20.);
    }
//    observations.segment(17+i*3 + 1, 2) = last_frame->px_vec_.col(i);
    observations(24+i*3 + 1) = last_frame->px_vec_.col(i)(0) / cams_->getCameraShared(0)->imageWidth();
    observations(24+i*3 + 2) = last_frame->px_vec_.col(i)(1) / cams_->getCameraShared(0)->imageHeight();
  }

  // Done
  dones = 0.0;

  // Stage
  stage = int(stage_);

  /// @implements end ===============================================================


  return success_frame;
}

//------------------------------------------------------------------------------
bool FrameHandlerBase::addFrameBundle(const FrameBundlePtr& frame_bundle)
{
  VLOG(40) << "New Frame Bundle received: " << frame_bundle->getBundleId();  
  CHECK_EQ(frame_bundle->size(), cams_->numCameras());

  // ---------------------------------------------------------------------------
  // Prepare processing.

  if (set_start_)
  {
    // Temporary copy rotation prior. TODO(cfo): fix this.
    Quaternion R_imu_world = R_imu_world_;
    bool have_rotation_prior = have_rotation_prior_;
    resetAll();
    R_imu_world_ = R_imu_world;
    have_rotation_prior_ = have_rotation_prior;
    setInitialPose(frame_bundle);
    stage_ = Stage::kInitializing;
  }

  if (stage_ == Stage::kPaused)
  {
    return false;
  }

//  if (options_.trace_statistics)
//  {
//    SVO_LOG("timestamp", frame_bundle->at(0)->getTimestampNSec());
//    SVO_START_TIMER("frontend_time");
//  }
  timer_.start();

  // ---------------------------------------------------------------------------
  // Add to pipeline.
  new_frames_ = frame_bundle;
  ++frame_counter_;

  // for scale check
  double svo_dist_first_two_kfs = -1;
  double opt_dist_first_two_kfs = -1;

  // handle motion prior
  if (have_motion_prior_)
  {
    have_rotation_prior_ = true;
    R_imu_world_ = new_frames_->get_T_W_B().inverse().getRotation();
    if (last_frames_)
    {
      T_newimu_lastimu_prior_ = new_frames_->get_T_W_B().inverse() * last_frames_->get_T_W_B();
      have_motion_prior_ = true;
    }
  }
  else
  {
    // Predict pose of new frame using motion prior.
    // TODO(cfo): remove same from processFrame in mono.
    if (last_frames_)
    {
      VLOG(40) << "Predict pose of new image using motion prior.";
      getMotionPrior(false);

      // set initial pose estimate
      for (size_t i = 0; i < new_frames_->size(); ++i)
      {
        new_frames_->at(i)->T_f_w_ = new_frames_->at(i)->T_cam_imu() * T_newimu_lastimu_prior_
            * last_frames_->at(i)->T_imu_world();
      }
    }
  }

  // Perform tracking.
  update_res_ = processFrameBundle();

  if (update_res_ == UpdateResult::kKeyframe)
  {
    // Set flag in bundle. Before we only set each frame individually.
    new_frames_->setKeyframe();
    last_kf_time_sec_ = new_frames_->at(0)->getTimestampSec();
  }

  if (last_frames_)
  {
    // Set translation motion prior for next frame.
    t_lastimu_newimu_ = new_frames_->at(0)->T_imu_world().getRotation().rotate(
        new_frames_->at(0)->imuPos() - last_frames_->at(0)->imuPos());
  }

  // Statistics.
  acc_frame_timings_.push_back(timer_.stop());
  num_obs_last_ = new_frames_->numTrackedFeatures() +
      new_frames_->numFixedLandmarks();
  if (stage_ == Stage::kTracking)
  {
    if (isInRecovery())
    {
      CHECK_GT(new_frames_->getMinTimestampSeconds(),
               last_good_tracking_time_sec_);
    }
    else
    {
      last_good_tracking_time_sec_ = new_frames_->getMinTimestampSeconds();
    }
    acc_num_obs_.push_back(num_obs_last_);
  }

  // Try relocalizing if tracking failed.
  if(update_res_ == UpdateResult::kFailure)
  {
    VLOG(2) << "Tracking failed: RELOCALIZE.";
    CHECK(stage_ == Stage::kTracking || stage_ == Stage::kInitializing || stage_ == Stage::kRelocalization);

    // Let's try to relocalize with respect to the last keyframe:
    reloc_keyframe_ = map_->getLastKeyframe();
    CHECK_NOTNULL(reloc_keyframe_.get());

    // Reset pose to previous frame to avoid crazy jumps.
    if (stage_ == Stage::kTracking && last_frames_)
    {
      for (size_t i = 0; i < last_frames_->size(); ++i)
        new_frames_->at(i)->T_f_w_ = last_frames_->at(i)->T_f_w_;
    }

    // Reset if we tried many times unsuccessfully to relocalize.
    if (stage_ == Stage::kRelocalization &&
        relocalization_n_trials_ >= options_.relocalization_max_trials)
    {
      VLOG(2) << "Relocalization failed "
              << options_.relocalization_max_trials << " times: RESET.";
      set_reset_ = true;
    }

    // Set stage.
    stage_ = Stage::kRelocalization;
    tracking_quality_ = TrackingQuality::kInsufficient;
  }

  // Set last frame.
  last_frames_ = new_frames_;
  new_frames_.reset();

  // Reset if we should.
  if (set_reset_)
  {
    resetVisionFrontendCommon();
  }

  // Reset rotation prior.
  have_rotation_prior_ = false;
  R_imulast_world_ = R_imu_world_;

  // Reset motion prior
  have_motion_prior_ = false;
  T_newimu_lastimu_prior_.setIdentity();

  // tracing
//  if (options_.trace_statistics)
//  {
//    SVO_LOG("dropout", static_cast<int>(update_res_));
//    SVO_STOP_TIMER("frontend_time");
////    g_permon->writeToFile();
//  }
  // Call callbacks.
  VLOG(40) << "Triggering addFrameBundle() callbacks...";
  triggerCallbacks(last_frames_);
  return true;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
void FrameHandlerBase::setRotationPrior(const Quaternion& R_imu_world)
{
  VLOG(40) << "Set rotation prior.";
  R_imu_world_ = R_imu_world;
  have_rotation_prior_ = true;
}

void FrameHandlerBase::setRotationIncrementPrior(const Quaternion& R_lastimu_newimu)
{
  VLOG(40) << "Set rotation increment prior.";
  R_imu_world_ = R_lastimu_newimu.inverse() * R_imulast_world_;
  have_rotation_prior_ = true;
}

//------------------------------------------------------------------------------
void FrameHandlerBase::setInitialPose(const FrameBundlePtr& frame_bundle) const
{
  if (have_rotation_prior_)
  {
    VLOG(40) << "Set initial pose: With rotation prior";
    for (size_t i = 0; i < frame_bundle->size(); ++i)
    {
      frame_bundle->at(i)->T_f_w_ = cams_->get_T_C_B(i) * Transformation(R_imu_world_, Vector3d::Zero());
    }
  }
  else
  {
    VLOG(40) << "Set initial pose: set such that T_imu_world is identity.";
    for (size_t i = 0; i < frame_bundle->size(); ++i)
    {
      frame_bundle->at(i)->T_f_w_ = cams_->get_T_C_B(i) * T_world_imuinit.inverse();
    }
  }
}

//------------------------------------------------------------------------------
size_t FrameHandlerBase::sparseImageAlignment()
{
  // optimize the pose of the new frame such that it matches the pose of the previous frame best
  // this will improve the relative transformation between the previous and the new frame
  // the result is the number of feature points which could be tracked
  // this is a hierarchical KLT solver
  VLOG(40) << "Sparse image alignment.";
//  if (options_.trace_statistics)
//  {
//    SVO_START_TIMER("sparse_img_align");
//  }
  sparse_img_align_->reset();
  if (have_motion_prior_)
  {
    SVO_DEBUG_STREAM("Apply IMU Prior to Image align");
    double prior_trans = options_.img_align_prior_lambda_trans;
    if (map_->size() < 5)
      prior_trans = 0; // during the first few frames we don't want a prior (TODO)

    sparse_img_align_->setWeightedPrior(T_newimu_lastimu_prior_, 0.0, 0.0,
                                        options_.img_align_prior_lambda_rot,
                                        prior_trans, 0.0, 0.0);
  }
  sparse_img_align_->setMaxNumFeaturesToAlign(options_.img_align_max_num_features);
  size_t img_align_n_tracked = sparse_img_align_->run(last_frames_, new_frames_);

//  if (options_.trace_statistics)
//  {
//    SVO_STOP_TIMER("sparse_img_align");
//    SVO_LOG("img_align_n_tracked", img_align_n_tracked);
//  }
  VLOG(40) << "Sparse image alignment tracked " << img_align_n_tracked << " features.";
  return img_align_n_tracked;
}

//------------------------------------------------------------------------------
size_t FrameHandlerBase::projectMapInFrame()
{
  VLOG(40) << "Project map in frame.";
//  if (options_.trace_statistics)
//  {
//    SVO_START_TIMER("reproject");
//  }
  // compute overlap keyframes
  for (size_t camera_idx = 0; camera_idx < cams_->numCameras(); ++camera_idx)
  {
    ReprojectorPtr& cur_reprojector = reprojectors_.at(camera_idx);
    overlap_kfs_.at(camera_idx).clear();
    map_->getClosestNKeyframesWithOverlap(
          new_frames_->at(camera_idx),
          cur_reprojector->options_.max_n_kfs,
          &overlap_kfs_.at(camera_idx));
  }

  std::vector<std::vector<PointPtr>> trash_points;
  trash_points.resize(cams_->numCameras());
  if (options_.use_async_reprojectors && cams_->numCameras() > 1)
  {
    // start reprojection workers
    std::vector<std::future<void>> reprojector_workers;
    for (size_t camera_idx = 0; camera_idx < cams_->numCameras(); ++camera_idx)
    {
      auto func = std::bind(&Reprojector::reprojectFrames, reprojectors_.at(camera_idx).get(),
                            new_frames_->at(camera_idx), overlap_kfs_.at(camera_idx), trash_points.at(camera_idx));
      reprojector_workers.push_back(std::async(std::launch::async, func));
    }

    // make sure all of them are finished
    for (size_t i = 0; i < reprojector_workers.size(); ++i)
      reprojector_workers[i].get();
  }
  else
  {
    for (size_t camera_idx = 0; camera_idx < cams_->numCameras(); ++camera_idx)
    {
      reprojectors_.at(camera_idx)->reprojectFrames(
            new_frames_->at(camera_idx), overlap_kfs_.at(camera_idx),
            trash_points.at(camera_idx));
    }
  }

  // Effectively clear the points that were discarded by the reprojectors
  for (auto point_vec : trash_points)
    for (auto point : point_vec)
      map_->safeDeletePoint(point);

  // Count the total number of trials and matches for all reprojectors
  Reprojector::Statistics cumul_stats_;
  Reprojector::Statistics cumul_stats_global_map;
  for (const ReprojectorPtr& reprojector : reprojectors_)
  {
    cumul_stats_.n_matches += reprojector->stats_.n_matches;
    cumul_stats_.n_trials += reprojector->stats_.n_trials;
    cumul_stats_global_map.n_matches += reprojector->fixed_lm_stats_.n_matches;
    cumul_stats_global_map.n_trials += reprojector->fixed_lm_stats_.n_trials;
  }

//  if (options_.trace_statistics)
//  {
//    SVO_STOP_TIMER("reproject");
//    SVO_LOG("repr_n_matches_local_map", cumul_stats_.n_matches);
//    SVO_LOG("repr_n_trials_local_map", cumul_stats_.n_trials);
//    SVO_LOG("repr_n_matches_global_map", cumul_stats_global_map.n_matches);
//    SVO_LOG("repr_n_trials_global_map", cumul_stats_global_map.n_trials);
//  }
  VLOG(40) << "Reprojection:" << "\t nPoints = " << cumul_stats_.n_trials << "\t\t nMatches = "
      << cumul_stats_.n_matches;

  size_t n_total_ftrs = cumul_stats_.n_matches +
      (cumul_stats_global_map.n_matches <= 10? 0 : cumul_stats_global_map.n_matches);

  if (n_total_ftrs < options_.quality_min_fts)
  {
    SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features: " +
                             std::to_string(n_total_ftrs));
  }

  return n_total_ftrs;
}

//------------------------------------------------------------------------------
size_t FrameHandlerBase::optimizePose()
{
  // pose optimization
  // optimize the pose of the frame in such a way, that the projection of all feature world coordinates
  // is not far off the position of the feature points within the frame. The optimization is done for all points
  // in the same time, hence optimizing frame pose.
//  if (options_.trace_statistics)
//  {
//    SVO_START_TIMER("pose_optimizer");
//  }

  pose_optimizer_->reset();
  if (have_motion_prior_)
  {
    VLOG(40) << "Apply prior to pose optimization";
    pose_optimizer_->setRotationPrior(new_frames_->get_T_W_B().getRotation().inverse(),
                                      options_.poseoptim_prior_lambda);
  }
  size_t sfba_n_edges_final = pose_optimizer_->run(new_frames_, options_.poseoptim_thresh);

//  if (options_.trace_statistics)
//  {
//    SVO_LOG("sfba_error_init", pose_optimizer_->stats_.reproj_error_before);
//    SVO_LOG("sfba_error_final", pose_optimizer_->stats_.reproj_error_after);
//    SVO_LOG("sfba_n_edges_final", sfba_n_edges_final);
//    SVO_STOP_TIMER("pose_optimizer");
//  }
  SVO_DEBUG_STREAM(
      "PoseOptimizer:" << "\t ErrInit = " << pose_optimizer_->stats_.reproj_error_before << "\t ErrFin = " << pose_optimizer_->stats_.reproj_error_after << "\t nObs = " << sfba_n_edges_final);
  return sfba_n_edges_final;
}

//------------------------------------------------------------------------------
void FrameHandlerBase::optimizeStructure(const FrameBundle::Ptr& frames, int max_n_pts, int max_iter)
{
  VLOG(40) << "Optimize structure.";
  // some feature points will be optimized w.r.t keyframes they were observed
  // in the way that their projection error into all other keyframes is minimzed

  if (max_n_pts == 0)
    return; // don't return if max_n_pts == -1, this means we optimize ALL points

//  if (options_.trace_statistics)
//  {
//    SVO_START_TIMER("point_optimizer");
//  }
  for (const FramePtr& frame : frames->frames_)
  {
    bool optimize_on_sphere = false;
    if (frame->cam()->getType() == Camera::Type::kOmni)
      optimize_on_sphere = true;
    std::deque<PointPtr> pts;
    for (size_t i = 0; i < frame->num_features_; ++i)
    {
      if (frame->landmark_vec_[i] == nullptr || isEdgelet(frame->type_vec_[i]))
        continue;
      pts.push_back((frame->landmark_vec_[i]));
    }
    auto it_end = pts.end();
    if (max_n_pts > 0)
    {
      max_n_pts = std::min(static_cast<size_t>(max_n_pts), pts.size());
      std::nth_element(pts.begin(), pts.begin() + max_n_pts, pts.end(), [](const PointPtr& lhs, const PointPtr& rhs)
      {
        // we favour points that have not been optimized in a while
                       return (lhs->last_structure_optim_ < rhs->last_structure_optim_);
                     });
      it_end = pts.begin() + max_n_pts;
    }
    for (const PointPtr& point : pts)
    {
      point->optimize(max_iter, optimize_on_sphere);
      point->last_structure_optim_ = frame->id_;
    }
  }
//  if (options_.trace_statistics)
//  {
//    SVO_STOP_TIMER("point_optimizer");
//  }
}

//------------------------------------------------------------------------------
void FrameHandlerBase::upgradeSeedsToFeatures(const FramePtr& frame)
{
  VLOG(40) << "Upgrade seeds to features";
  size_t update_count = 0;
  size_t unconverged_cnt = 0;
  for (size_t i = 0; i < frame->num_features_; ++i)
  {
    if (frame->landmark_vec_[i])
    {
      const FeatureType& type = frame->type_vec_[i];
      if (type == FeatureType::kCorner || type == FeatureType::kEdgelet ||
          type == FeatureType::kMapPoint)
      {
        frame->landmark_vec_[i]->addObservation(frame, i);
      }
      else
      {
        CHECK(isFixedLandmark(type));
        frame->landmark_vec_[i]->addObservation(frame, i);
      }
    }
    else if (frame->seed_ref_vec_[i].keyframe)
    {
      if (isUnconvergedSeed(frame->type_vec_[i]))
      {
        unconverged_cnt++;
      }
      SeedRef& ref = frame->seed_ref_vec_[i];

      // In multi-camera case, it might be that we already created a 3d-point
      // for this seed previously when processing another frame from the bundle.
      PointPtr point = ref.keyframe->landmark_vec_[ref.seed_id];
      if (point == nullptr)
      {
        // That's not the case. Therefore, create a new 3d point.
        Position xyz_world =
            ref.keyframe->T_world_cam() *
            ref.keyframe->getSeedPosInFrame(ref.seed_id);
        point = std::make_shared < Point > (xyz_world);
        ref.keyframe->landmark_vec_[ref.seed_id] = point;
        ref.keyframe->track_id_vec_[ref.seed_id] = point->id();
        point->addObservation(ref.keyframe, ref.seed_id);
      }

      // add reference to current frame.
      frame->landmark_vec_[i] = point;
      frame->track_id_vec_[i] = point->id();
      point->addObservation(frame, i);
      if (isCorner(ref.keyframe->type_vec_[ref.seed_id]))
      {
        ref.keyframe->type_vec_[ref.seed_id] = FeatureType::kCorner;
        frame->type_vec_[i] = FeatureType::kCorner;
      }
      else if (isMapPoint(ref.keyframe->type_vec_[ref.seed_id]))
      {
        ref.keyframe->type_vec_[ref.seed_id] = FeatureType::kMapPoint;
        frame->type_vec_[i] = FeatureType::kMapPoint;
      }
      else if (isEdgelet(ref.keyframe->type_vec_[ref.seed_id]))
      {
        ref.keyframe->type_vec_[ref.seed_id] = FeatureType::kEdgelet;
        frame->type_vec_[i] = FeatureType::kEdgelet;

        // Update the edgelet direction.
        double angle = feature_detection_utils::getAngleAtPixelUsingHistogram(
            frame->img_pyr_[frame->level_vec_[i]],
            (frame->px_vec_.col(i) / (1 << frame->level_vec_[i])).cast<int>(),
            4u);
        frame->grad_vec_.col(i) = GradientVector(std::cos(angle),
                                                 std::sin(angle));
      }
      else
      {
        CHECK(false) << "Seed-Type not known";
      }
      ++update_count;
    }

    // when using the feature-wrapper, we might copy some old references?
    frame->seed_ref_vec_[i].keyframe.reset();
    frame->seed_ref_vec_[i].seed_id = -1;
  }
  VLOG(5) << "NEW KEYFRAME: Updated "
          << update_count << " seeds to features in reference frame, "
          << "including " << unconverged_cnt << " unconverged points.\n";
  const double ratio = (1.0 * unconverged_cnt) / update_count;
  if (ratio > 0.2)
  {
    LOG(WARNING) << ratio * 100 << "% updated seeds are unconverged.";
  }
}

//------------------------------------------------------------------------------
void FrameHandlerBase::resetVisionFrontendCommon()
{
  stage_ = Stage::kPaused;
  tracking_quality_ = TrackingQuality::kInsufficient;
  set_reset_ = false;
  set_start_ = false;
  num_obs_last_ = 0;
  reloc_keyframe_.reset();
  relocalization_n_trials_ = 0;
  t_lastimu_newimu_ = Vector3d::Zero();
  have_motion_prior_ = false;
  T_newimu_lastimu_prior_.setIdentity();
  have_rotation_prior_ = false;
  for (auto& frame_vec : overlap_kfs_)
  {
    frame_vec.clear();
  }

  new_frames_.reset();
  last_frames_.reset();
  map_->reset();

  sparse_img_align_->reset();
  depth_filter_->reset();
  initializer_->reset();

  VLOG(1) << "SVO RESET ALL";
}

void FrameHandlerBase::setRecovery(const bool recovery)
{
  loss_without_correction_ = recovery;
}

//------------------------------------------------------------------------------
void FrameHandlerBase::setTrackingQuality(const size_t num_observations)
{
  tracking_quality_ = TrackingQuality::kGood;
  if (num_observations < options_.quality_min_fts)
  {
    SVO_WARN_STREAM_THROTTLE(0.5, "Tracking less than "
                             << options_.quality_min_fts <<" features!");
    tracking_quality_ = TrackingQuality::kInsufficient;
  }
  const int feature_drop = static_cast<int>(num_obs_last_) - num_observations;
  // seeds are extracted at keyframe,
  // so the number is not indicative of tracking quality
  if (!last_frames_->isKeyframe() &&
      feature_drop > options_.quality_max_fts_drop)
  {
    SVO_WARN_STREAM("Lost "<< feature_drop <<" features!");
    tracking_quality_ = TrackingQuality::kInsufficient;
  }
}

//------------------------------------------------------------------------------
bool FrameHandlerBase::needNewKf(const Transformation&)
{
  const std::vector<FramePtr>& visible_kfs = overlap_kfs_.at(0);
  if (options_.kfselect_criterion == KeyframeCriterion::DOWNLOOKING)
  {
    for (const auto& frame : visible_kfs)
    {
      Vector3d relpos = new_frames_->at(0)->T_cam_world() * frame->pos();
      if (fabs(relpos.x()) / depth_median_ < options_.kfselect_min_dist
          && fabs(relpos.y()) / depth_median_ < options_.kfselect_min_dist * 0.8
          && fabs(relpos.z()) / depth_median_ < options_.kfselect_min_dist * 1.3)
        return false;
    }
    VLOG(40) << "KF Select: NEW KEYFRAME";
    return true;
  }

  size_t n_tracked_fts = new_frames_->numTrackedLandmarks();

  if (n_tracked_fts > options_.kfselect_numkfs_upper_thresh)
  {
    VLOG(40) << "KF Select: NO NEW KEYFRAME Above upper bound";
    return false;
  }

  // TODO: this only works for mono!
  if (last_frames_->at(0)->id() - map_->last_added_kf_id_ <
      options_.kfselect_min_num_frames_between_kfs)
  {
    VLOG(40) << "KF Select: NO NEW KEYFRAME We just had a KF";
    return false;
  }

  if (n_tracked_fts < options_.kfselect_numkfs_lower_thresh)
  {
    VLOG(40) << "KF Select: NEW KEYFRAME Below lower bound";
    return true;
  }

  // check that we have at least X disparity w.r.t to last keyframe
  if (options_.kfselect_min_disparity > 0)
  {
    int kf_id = map_->getLastKeyframe()->id();
    std::vector<double> disparities;
    const FramePtr& frame = new_frames_->at(0);
    disparities.reserve(frame->num_features_);
    for (size_t i = 0; i < frame->num_features_; ++i)
    {
      if (frame->landmark_vec_[i])
      {
        const Point::KeypointIdentifierList& observations =
            frame->landmark_vec_[i]->obs_;
        for (auto it = observations.rbegin(); it != observations.rend(); ++it)
        {
          if (it->frame_id == kf_id)
          {
            if (FramePtr kf = it->frame.lock())
            {
              disparities.push_back(
                    (frame->px_vec_.col(i) -
                     kf->px_vec_.col(it->keypoint_index_)).norm());
            }
            break;
          }
        }
      }
      // TODO(cfo): loop also over seed references!
    }

    if (!disparities.empty())
    {
      double disparity = vk::getMedian(disparities);
      VLOG(40) << "KF Select: disparity = " << disparity;
      if (disparity < options_.kfselect_min_disparity)
      {
        VLOG(40) << "KF Select: NO NEW KEYFRAME disparity not large enough";
        return false;
      }
    }
  }

  for (const auto& kf : visible_kfs)
  {
    // TODO: doesn't generalize to rig!
    const double a =
        Quaternion::log(new_frames_->at(0)->T_f_w_.getRotation() *
                        kf->T_f_w_.getRotation().inverse()).norm()
            * 180/M_PI;
    const double d = (new_frames_->at(0)->pos() - kf->pos()).norm();
    if (a < options_.kfselect_min_angle
        && d < options_.kfselect_min_dist_metric)
    {
      VLOG(40) << "KF Select: NO NEW KEYFRAME Min angle = " << a
               << ", min dist = " << d;
      return false;
    }
  }
  VLOG(40) << "KF Select: NEW KEYFRAME";
  return true;
}

void FrameHandlerBase::getMotionPrior(const bool /*use_velocity_in_frame*/)
{
  if (have_rotation_prior_)
  {
    VLOG(40) << "Get motion prior from provided rotation prior.";
    T_newimu_lastimu_prior_ = Transformation(R_imulast_world_ * R_imu_world_.inverse(), t_lastimu_newimu_).inverse();
    have_motion_prior_ = true;
  }
  else if (options_.poseoptim_prior_lambda > 0
           || options_.img_align_prior_lambda_rot > 0
           || options_.img_align_prior_lambda_trans > 0)
  {
    VLOG(40) << "Get motion prior by assuming constant velocity.";
    T_newimu_lastimu_prior_ = Transformation(Quaternion(), t_lastimu_newimu_).inverse();
    have_motion_prior_ = true;
  }
  return;
}

//------------------------------------------------------------------------------
void FrameHandlerBase::setDetectorOccupiedCells(
    const size_t reprojector_grid_idx, const DetectorPtr& feature_detector)
{
  const Reprojector& rep = *reprojectors_.at(reprojector_grid_idx);
  CHECK_EQ(feature_detector->grid_.size(), rep.grid_->size());
  feature_detector->grid_.occupancy_ = rep.grid_->occupancy_;
  if (rep.fixed_landmark_grid_)
  {
    for (size_t idx = 0; idx < rep.fixed_landmark_grid_->size(); idx++)
    {
      if (rep.fixed_landmark_grid_->isOccupied(idx))
      {
        feature_detector->grid_.occupancy_[idx] = true;
      }
    }
  }
}

void FrameHandlerBase::setFirstFrames(const std::vector<FramePtr>& first_frames)
{
  resetAll();
  last_frames_.reset(new FrameBundle(first_frames));
  for (auto f : last_frames_->frames_)
  {
    f->setKeyframe();
    map_->addKeyframe(f, false);
  }
  stage_ = Stage::kTracking;
}

std::vector<FramePtr> FrameHandlerBase::closeKeyframes() const
{
  std::vector<FramePtr> close_kfs;
  for (const auto& kfs : overlap_kfs_)
  {
    close_kfs.insert(close_kfs.begin(), kfs.begin(), kfs.end());
  }
  return close_kfs;
}

void FrameHandlerBase::setCompensation(const bool do_compensation)
{
  sparse_img_align_->setCompensation(do_compensation);
  for (const ReprojectorPtr& rep : reprojectors_)
  {
    rep->options_.affine_est_gain = do_compensation;
  }
  depth_filter_->getMatcher().options_.affine_est_gain_ = do_compensation;
}

} // namespace svo
