#include <svo_env/svo_vec_env.h>
#include <opencv2/opencv.hpp>

namespace svo {

SvoVecEnv::SvoVecEnv(
    const std::string config_filepath,
    const std::string calib_filepath,
    const int num_envs,
    const bool initialize_glog) {
  // initialization
//  num_threads_ = 64;
  num_threads_ = 8;
  num_envs_ = num_envs;
  initialize_glog_ = initialize_glog;

  if (initialize_glog_) {
    google::InitGoogleLogging("svo_env");
    google::SetStderrLogging(google::GLOG_FATAL);
//    FLAGS_v = 100;
//    google::SetStderrLogging(google::GLOG_INFO);
  }

  // set threads
  omp_set_num_threads(num_threads_);

  // create & setup environments
  for (int env_id = 0; env_id < num_envs_; env_id++) {
    envs_.push_back(std::make_unique<SvoEnv>(config_filepath, calib_filepath, env_id));
  }
}

SvoVecEnv::~SvoVecEnv() {
  if (initialize_glog_) {
    google::ShutdownGoogleLogging();
  }
}

void SvoVecEnv::setSeed(const int seed) {
  int seed_inc = seed;
  for (int i = 0; i < num_envs_; i++) envs_[i]->setSeed(seed_inc++);
}

void SvoVecEnv::reset(Ref<Vector<>> indices) {
  int num_indices = indices.rows();
  for (int i = 0; i < num_indices; i++) {
    this->envs_[indices[i]]->reset();
  }
}

void SvoVecEnv::step(pybind11::array_t<uint8_t> &input_images,
                     Ref<Vector<>> times_nsec, 
                     Ref<MatrixRowMajor<>> actions,
                     Ref<Vector<>> use_RL_actions,
                     Ref<MatrixRowMajor<>> out_pose,
                     Ref<MatrixRowMajor<>> observations,
                     Ref<Vector<>> dones,
                     Ref<Vector<>> stages,
                     Ref<Vector<>> runtime,
                     Ref<Vector<>> use_gt_init_pose,
                     Ref<MatrixRowMajor<>> gt_init_pose
                     ) {
  pybind11::buffer_info buffer_info = input_images.request();
  uint8_t *data = static_cast<uint8_t *>(buffer_info.ptr);
  std::vector<ssize_t> shape = buffer_info.shape;
  Eigen::TensorMap<Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>> images_tensor(data, shape[0], shape[1], shape[2], shape[3]);

//  omp_set_num_threads(num_threads_);

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < this->num_envs_; i++) {
    perAgentStep(i, i, images_tensor, times_nsec, actions, use_RL_actions, out_pose, observations, dones, stages,
                 runtime, use_gt_init_pose, gt_init_pose);
  }
}

void SvoVecEnv::perAgentStep(const int agent_id,
                             const int data_idx,
                             Eigen::TensorMap<Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>> &images_tensor,
                             Ref<Vector<>> times_nsec, 
                             Ref<MatrixRowMajor<>> actions,
                             Ref<Vector<>> use_RL_actions,
                             Ref<MatrixRowMajor<>> out_pose,
                             Ref<MatrixRowMajor<>> observations,
                             Ref<Vector<>> dones,
                             Ref<Vector<>> stages,
                             Ref<Vector<>> runtime,
                             Ref<Vector<>> use_gt_init_pose,
                             Ref<MatrixRowMajor<>> gt_init_pose) {
  Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> selected_slice = images_tensor.chip(data_idx, 0);
  const auto& dimensions = selected_slice.dimensions();
  cv::Mat input_image(dimensions[0], dimensions[1], CV_8UC(dimensions[2]), (unsigned char*)selected_slice.data());

  this->envs_[agent_id]->step(input_image, 
                              times_nsec(data_idx),
                              actions.row(data_idx),
                              use_RL_actions(data_idx) == 1,
                              out_pose.row(data_idx),
                              observations.row(data_idx),
                              dones(data_idx),
                              stages(data_idx),
                              runtime(data_idx),
                              use_gt_init_pose(data_idx) == 1,
                              gt_init_pose.row(data_idx));
}

void SvoVecEnv::env_step(Ref<Vector<>> indices,
                         pybind11::array_t<uint8_t> input_images,
                         Ref<Vector<>> times_nsec,
                         Ref<MatrixRowMajor<>> actions,
                         Ref<Vector<>> use_RL_actions,
                         Ref<MatrixRowMajor<>> out_pose,
                         Ref<MatrixRowMajor<>> observations,
                         Ref<Vector<>> dones,
                         Ref<Vector<>> stages,
                         Ref<Vector<>> runtime,
                         Ref<Vector<>> use_gt_init_pose,
                         Ref<MatrixRowMajor<>> gt_init_pose) {
  pybind11::buffer_info buffer_info = input_images.request();
  uint8_t *data = static_cast<uint8_t *>(buffer_info.ptr);
  std::vector<ssize_t> shape = buffer_info.shape;
  Eigen::TensorMap<Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>> images_tensor(data, shape[0], shape[1], shape[2], shape[3]);

  int num_indices = indices.rows();

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < num_indices; i++) {
    perAgentStep(indices[i], i, images_tensor, times_nsec, actions, use_RL_actions, out_pose, observations, dones,
                 stages, runtime, use_gt_init_pose, gt_init_pose);
  }
}

void SvoVecEnv::env_visualize_features(const int env_idx,
                                       pybind11::array_t<uint8_t>& image,
                                       const int64_t timestamp_nanoseconds) {
  pybind11::buffer_info buf_info = image.request();
  int height = buf_info.shape[0];
  int width = buf_info.shape[1];
  int channels = buf_info.shape[2];
  uint8_t* data = static_cast<uint8_t*>(buf_info.ptr);
  cv::Mat cv_image(height, width, CV_8UC(channels), data, cv::Mat::AUTO_STEP);

  this->envs_[env_idx]->getImageWithFeatures(timestamp_nanoseconds,
                                             cv_image);
}

} // namespace svo