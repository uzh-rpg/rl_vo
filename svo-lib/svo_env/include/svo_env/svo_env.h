#pragma once

#include <thread>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <svo_env/types.h>
#include <svo/common/types.h>
#include <svo/common/camera_fwd.h>
#include <svo/common/transformation.h>

namespace svo {

// forward declarations
class FrameHandlerBase;
class Visualizer;

enum class PipelineType {
  kMono
};

/// SVO Interface
class SvoEnv
{
public:
  // SVO modules.
  std::shared_ptr<FrameHandlerBase> svo_;
  std::shared_ptr<Visualizer> visualizer_;

  CameraBundlePtr ncam_;

  // System state.
  bool quit_ = false;
  int counter_ = 0;
  bool automatic_reinitialization_ = false;

  SvoEnv(const std::string config_filepath,
               const std::string calib_filepath,
               const int env_id);

  virtual ~SvoEnv();

  // Processing
  void processImageBundle(
      const std::vector<cv::Mat>& images,
      int64_t timestamp_nanoseconds,
      Ref<Vector<>> actions,
      const bool use_RL_actions,
      Ref<Vector<>> observations,
      double &dones,
      double &stage,
      double &runtime,
      const bool use_gt_init_pose,
      Ref<Vector<>> gt_init_pose);

  void getImageWithFeatures(
    const int64_t timestamp_nanoseconds,
    cv::Mat& img_rgb);

  void monoCallback(std::vector<cv::Mat>& images, 
                    long time_nsec,
                    cv::Mat& img_rgb);

  void reset();

  void step(cv::Mat &image, 
            long time_nsec, 
            Ref<Vector<>> actions, 
            const bool use_RL_actions, 
            Ref<Vector<>> out_pose, 
            Ref<Vector<>> observations,
            double &dones,
            double &stage,
            double &runtime,
            const bool use_gt_init_pose,
            Ref<Vector<>> gt_init_pose);

  void setSeed(const int seed) { std::srand(seed); };

};

} // namespace svo
