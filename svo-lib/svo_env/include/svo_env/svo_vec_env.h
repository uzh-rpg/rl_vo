#pragma once

#include <thread>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

#include <svo_env/types.h>
#include <svo_env/svo_env.h>


namespace svo {

/// Vectorized SVO Environments
class SvoVecEnv
{
public:
  SvoVecEnv(const std::string config_filepath, const std::string calib_filepath, const int num_envs,
            const bool define_glog);
  ~SvoVecEnv();

  void reset(Ref<Vector<>> indices);

  void step(pybind11::array_t<uint8_t> &input_images,
            Ref<Vector<>> time_nsec, 
            Ref<MatrixRowMajor<>> actions,
            Ref<Vector<>> use_RL_actions,
            Ref<MatrixRowMajor<>> out_pose,
            Ref<MatrixRowMajor<>> observations,
            Ref<Vector<>> dones,
            Ref<Vector<>> stages,
            Ref<Vector<>> runtime,
            Ref<Vector<>> use_gt_init_pose,
            Ref<MatrixRowMajor<>> gt_init_pose);

  void perAgentStep(const int agent_id,
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
                    Ref<MatrixRowMajor<>> gt_init_pose);

  void setSeed(const int seed);

  void env_step(Ref<Vector<>> indices,
                pybind11::array_t<uint8_t> input_images,
                Ref<Vector<>> time_nsec,
                Ref<MatrixRowMajor<>> actions,
                Ref<Vector<>> use_RL_actions,
                Ref<MatrixRowMajor<>> out_pose,
                Ref<MatrixRowMajor<>> observations,
                Ref<Vector<>> dones,
                Ref<Vector<>> stages,
                Ref<Vector<>> runtime,
                Ref<Vector<>> use_gt_init_pose,
                Ref<MatrixRowMajor<>> gt_init_pose);

  void env_visualize_features(const int env_idx,
                              pybind11::array_t<uint8_t>& image,
                              const int64_t timestamp_nanoseconds);

protected:
  std::vector<std::unique_ptr<SvoEnv>> envs_;
  int num_envs_, num_threads_;
  bool initialize_glog_;
};

} // namespace svo
