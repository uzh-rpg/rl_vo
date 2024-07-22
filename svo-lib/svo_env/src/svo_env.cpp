#include <svo_env/svo_env.h>

#include <svo_env/svo_factory.h>
#include <svo_env/visualizer.h>
#include <svo/common/frame.h>
#include <svo/map.h>
#include <svo/common/camera.h>
#include <svo/common/conversions.h>
#include <svo/frame_handler_mono.h>
#include <svo/initialization.h>
#include <svo/direct/depth_filter.h>

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>

#include <cstdio>
#include <thread>
#include <chrono>
#include <boost/filesystem.hpp>
#include <yaml-cpp/yaml.h>


namespace svo {

SvoEnv::SvoEnv(
    const std::string config_filepath,
    const std::string calib_filepath,
    const int env_id)
{
  YAML::Node config = YAML::LoadFile(config_filepath);
  automatic_reinitialization_ = config["automatic_reinitialization"] ? config["automatic_reinitialization"].as<bool>() : false;

  svo_ = factory::makeMono(config_filepath, calib_filepath);
  ncam_ = svo_->getNCamera();

  visualizer_.reset(new Visualizer());

  svo_->start();
}

SvoEnv::~SvoEnv()
{
  VLOG(1) << "Destructed SVO.";
}

void SvoEnv::processImageBundle(
    const std::vector<cv::Mat>& images,
    const int64_t timestamp_nanoseconds,
    Ref<Vector<>> actions,
    const bool use_RL_actions,
    Ref<Vector<>> observations,
    double &dones,
    double &stage,
    double &runtime,
    const bool use_gt_init_pose,
    Ref<Vector<>> gt_init_pose)
{
  svo_->addImageBundle(images, timestamp_nanoseconds, actions, use_RL_actions, observations, dones, stage, runtime,
                       use_gt_init_pose, gt_init_pose);
}

void SvoEnv::getImageWithFeatures(
    const int64_t timestamp_nanoseconds,
    cv::Mat& img_rgb)
{
  CHECK_NOTNULL(svo_.get());
  CHECK_NOTNULL(visualizer_.get());

  visualizer_->img_caption_.clear();

  switch (svo_->stage())
  {
    case Stage::kTracking: {
      bool draw_boundary = false;
      cv::Mat img_features = visualizer_->createImagesWithFeatures(svo_->getLastFrames(),
                                                                   timestamp_nanoseconds, draw_boundary);
      img_features.copyTo(img_rgb);
      break;
    }
    case Stage::kInitializing: {
      break;
    }
    case Stage::kPaused:
    case Stage::kRelocalization:
      break;
    default:
      LOG(FATAL) << "Unknown stage";
      break;
  }
}

void SvoEnv::reset() {
  svo_->reset_svo();
  svo_->start();
}

void SvoEnv::step(cv::Mat &image, 
                  long time_nsec, 
                  Ref<Vector<>> actions, 
                  const bool use_RL_actions, 
                  Ref<Vector<>> out_pose,
                  Ref<Vector<>> observations,
                  double &dones,
                  double &stage,
                  double &runtime,
                  const bool use_gt_init_pose,
                  Ref<Vector<>> gt_init_pose)
{
  std::vector<cv::Mat> images;
  images.push_back(image);
  processImageBundle(images, time_nsec, actions, use_RL_actions, observations, dones, stage, runtime, use_gt_init_pose,
                     gt_init_pose);

  // Extract output
  if (svo_->getLastFrames()==nullptr) {
    dones = 1;
    return;
  }

  svo::FramePtr frame = svo_->getLastFrames()->at(0);
  Eigen::Matrix<double, 4, 4> pose_matrix = frame->T_world_cam().getTransformationMatrix();
  Eigen::Map<Eigen::Matrix<double, 16, 1>> pose_vector(pose_matrix.data(), pose_matrix.size());
  out_pose = pose_vector;

  if(svo_->stage() == Stage::kPaused && automatic_reinitialization_)
    svo_->start();
}

} // namespace svo
