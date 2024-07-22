#include <svo/svo.h>
#include <svo_env/svo_factory.h>
#include <svo/frame_handler_mono.h>
#include <yaml-cpp/yaml.h>

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <yaml-cpp/yaml.h>
#pragma diagnostic pop

namespace svo {
namespace factory {

// BaseOptions loadBaseOptions(const ros::NodeHandle& pnh, bool forward_default)
BaseOptions loadBaseOptions(const YAML::Node config, bool forward_default)
{
  BaseOptions o;
  // o.max_n_kfs = vk::param<int>(pnh, "max_n_kfs", 5);
  o.max_n_kfs = config["max_n_kfs"] ? config["max_n_kfs"].as<int>() : 5;
  // o.trace_dir = vk::param<std::string>(pnh, "trace_dir", ros::package::getPath("svo")+"/trace");
  o.trace_dir = config["trace_dir"] ? config["trace_dir"].as<std::string>() : "<path>";
  // o.quality_min_fts = vk::param<int>(pnh, "quality_min_fts", 50);
  o.quality_min_fts = config["quality_min_fts"] ? config["quality_min_fts"].as<int>() : 50;
  // o.quality_max_fts_drop = vk::param<int>(pnh, "quality_max_drop_fts", 40);
  o.quality_max_fts_drop = config["quality_max_drop_fts"] ? config["quality_max_drop_fts"].as<int>() : 40;
  // o.relocalization_max_trials = vk::param<int>(pnh, "relocalization_max_trials", 50);
  o.relocalization_max_trials = config["relocalization_max_trials"] ? config["relocalization_max_trials"].as<int>() : 50;
  // o.poseoptim_prior_lambda = vk::param<double>(pnh, "poseoptim_prior_lambda", 0.0);
  o.poseoptim_prior_lambda = config["poseoptim_prior_lambda"] ? config["poseoptim_prior_lambda"].as<double>() : 0.0;
  // o.poseoptim_using_unit_sphere = vk::param<bool>(pnh, "poseoptim_using_unit_sphere", false);
  o.poseoptim_using_unit_sphere = config["poseoptim_using_unit_sphere"] ? config["poseoptim_using_unit_sphere"].as<bool>() : false;
  // o.img_align_prior_lambda_rot = vk::param<double>(pnh, "img_align_prior_lambda_rot", 0.0);
  o.img_align_prior_lambda_rot = config["img_align_prior_lambda_rot"] ? config["img_align_prior_lambda_rot"].as<double>() : 0.0;
  // o.img_align_prior_lambda_trans = vk::param<double>(pnh, "img_align_prior_lambda_trans", 0.0);
  o.img_align_prior_lambda_trans = config["img_align_prior_lambda_trans"] ? config["img_align_prior_lambda_trans"].as<double>() : 0.0;
  // o.structure_optimization_max_pts = vk::param<int>(pnh, "structure_optimization_max_pts", 20);
  o.structure_optimization_max_pts = config["structure_optimization_max_pts"] ? config["structure_optimization_max_pts"].as<int>() : 20;
  // o.init_map_scale = vk::param<double>(pnh, "map_scale", 1.0);
  o.init_map_scale = config["map_scale"] ? config["map_scale"].as<double>() : 1.0;
  std::string default_kf_criterion = forward_default ? "FORWARD" : "DOWNLOOKING";
  // if(vk::param<std::string>(pnh, "kfselect_criterion", default_kf_criterion) == "FORWARD")
  if((config["kfselect_criterion"] ? config["kfselect_criterion"].as<std::string>() : default_kf_criterion) == "FORWARD")
    o.kfselect_criterion = KeyframeCriterion::FORWARD;
  else
    o.kfselect_criterion = KeyframeCriterion::DOWNLOOKING;
  // o.kfselect_min_dist = vk::param<double>(pnh, "kfselect_min_dist", 0.12);
  o.kfselect_min_dist = config["kfselect_min_dist"] ? config["kfselect_min_dist"].as<double>() : 0.12;
  // o.kfselect_numkfs_upper_thresh = vk::param<int>(pnh, "kfselect_numkfs_upper_thresh", 120);
  o.kfselect_numkfs_upper_thresh = config["kfselect_numkfs_upper_thresh"] ? config["kfselect_numkfs_upper_thresh"].as<int>() : 120;
  // o.kfselect_numkfs_lower_thresh = vk::param<double>(pnh, "kfselect_numkfs_lower_thresh", 70);
  o.kfselect_numkfs_lower_thresh = config["kfselect_numkfs_lower_thresh"] ? config["kfselect_numkfs_lower_thresh"].as<double>() : 70;
  // o.kfselect_min_dist_metric = vk::param<double>(pnh, "kfselect_min_dist_metric", 0.01);
  o.kfselect_min_dist_metric =config["kfselect_min_dist_metric"] ? config["kfselect_min_dist_metric"].as<double>() : 0.01;
  // o.kfselect_min_angle = vk::param<double>(pnh, "kfselect_min_angle", 20);
  o.kfselect_min_angle = config["kfselect_min_angle"] ? config["kfselect_min_angle"].as<double>() : 20;
  // o.kfselect_min_disparity = vk::param<double>(pnh, "kfselect_min_disparity", 40);
  o.kfselect_min_disparity = config["kfselect_min_disparity"] ? config["kfselect_min_disparity"].as<double>() : 40;
  // o.kfselect_min_num_frames_between_kfs = vk::param<int>(pnh, "kfselect_min_num_frames_between_kfs", 2);
  o.kfselect_min_num_frames_between_kfs = config["kfselect_min_num_frames_between_kfs"] ? config["kfselect_min_num_frames_between_kfs"].as<int>() : 2;
  // o.kfselect_backend_max_time_sec = vk::param<double>(pnh, "kfselect_backend_max_time_sec", 3.0);
  o.kfselect_backend_max_time_sec = config["kfselect_backend_max_time_sec"] ? config["kfselect_backend_max_time_sec"].as<double>() : 3.0;
  // o.img_align_max_level = vk::param<int>(pnh, "img_align_max_level", 4);
  o.img_align_max_level = config["img_align_max_level"] ? config["img_align_max_level"].as<int>() : 4;
  // o.img_align_min_level = vk::param<int>(pnh, "img_align_min_level", 2);
  o.img_align_min_level = config["img_align_min_level"] ? config["img_align_min_level"].as<int>() : 2;
  // o.img_align_robustification = vk::param<bool>(pnh, "img_align_robustification", false);
  o.img_align_robustification = config["img_align_robustification"] ? config["img_align_robustification"].as<bool>() : false;
  // o.img_align_use_distortion_jacobian = vk::param<bool>(pnh, "img_align_use_distortion_jacobian", false);
  o.img_align_use_distortion_jacobian = config["img_align_use_distortion_jacobian"] ? config["img_align_use_distortion_jacobian"].as<bool>() : false;
  // o.img_align_est_illumination_gain = vk::param<bool>(pnh, "img_align_est_illumination_gain", false);
  o.img_align_est_illumination_gain = config["img_align_est_illumination_gain"] ? config["img_align_est_illumination_gain"].as<bool>() : false;
  // o.img_align_est_illumination_offset = vk::param<bool>(pnh, "img_align_est_illumination_offset", false);
  o.img_align_est_illumination_offset = config["img_align_est_illumination_offset"] ? config["img_align_est_illumination_offset"].as<bool>() : false;
  // o.poseoptim_thresh = vk::param<double>(pnh, "poseoptim_thresh", 2.0);
  o.poseoptim_thresh = config["poseoptim_thresh"] ? config["poseoptim_thresh"].as<double>() : 2.0;
  // o.update_seeds_with_old_keyframes = vk::param<bool>(pnh, "update_seeds_with_old_keyframes", true);
  o.update_seeds_with_old_keyframes = config["update_seeds_with_old_keyframes"] ? config["update_seeds_with_old_keyframes"].as<bool>() : true;
  // o.use_async_reprojectors = vk::param<bool>(pnh, "use_async_reprojectors", false);
  o.use_async_reprojectors = config["use_async_reprojectors"] ? config["use_async_reprojectors"].as<bool>() : false;
  // o.trace_statistics = vk::param<bool>(pnh, "trace_statistics", false);
  o.trace_statistics = config["trace_statistics"] ? config["trace_statistics"].as<bool>() : false;
  return o;
}

// DetectorOptions loadDetectorOptions(const ros::NodeHandle& pnh)
DetectorOptions loadDetectorOptions(const YAML::Node config)
{
  DetectorOptions o;
  // o.cell_size = vk::param<int>(pnh, "grid_size", 35);
  o.cell_size = config["grid_size"] ? config["grid_size"].as<int>() : 35;
  // o.max_level = vk::param<int>(pnh, "n_pyr_levels", 3) - 1;
  o.max_level = config["n_pyr_levels"] ? config["n_pyr_levels"].as<int>() - 1: 2;
  // o.threshold_primary = vk::param<int>(pnh, "detector_threshold_primary", 10);
  o.threshold_primary = config["detector_threshold_primary"] ? config["detector_threshold_primary"].as<int>() : 10;
  // o.threshold_secondary = vk::param<int>(pnh, "detector_threshold_secondary", 200);
  o.threshold_secondary = config["detector_threshold_secondary"] ? config["detector_threshold_secondary"].as<int>() : 200;
  // o.threshold_shitomasi = vk::param<int>(pnh, "detector_threshold_shitomasi", 100);
  o.threshold_shitomasi = config["detector_threshold_shitomasi"] ? config["detector_threshold_shitomasi"].as<int>() : 100;
  // if(vk::param<bool>(pnh, "use_edgelets", true))
  if(config["use_edgelets"] ? config["use_edgelets"].as<bool>() : true)
    o.detector_type = DetectorType::kFastGrad;
  else
    o.detector_type = DetectorType::kFast;
  return o;
}

// DepthFilterOptions loadDepthFilterOptions(const ros::NodeHandle& pnh)
DepthFilterOptions loadDepthFilterOptions(const YAML::Node config)
{
  DepthFilterOptions o;
  // o.max_search_level = vk::param<int>(pnh, "n_pyr_levels", 3) - 1;
  o.max_search_level = config["n_pyr_levels"] ? config["n_pyr_levels"].as<int>() - 1: 2;
  // o.use_threaded_depthfilter = vk::param<bool>(pnh, "use_threaded_depthfilter", true);
  o.use_threaded_depthfilter = config["use_threaded_depthfilter"] ? config["use_threaded_depthfilter"].as<bool>() : true;
  // o.seed_convergence_sigma2_thresh = vk::param<double>(pnh, "seed_convergence_sigma2_thresh", 200.0);
  o.seed_convergence_sigma2_thresh = config["seed_convergence_sigma2_thresh"] ? config["seed_convergence_sigma2_thresh"].as<double>() : 200.0;
  // o.mappoint_convergence_sigma2_thresh = vk::param<double>(pnh, "mappoint_convergence_sigma2_thresh", 500.0);
  o.mappoint_convergence_sigma2_thresh = config["mappoint_convergence_sigma2_thresh"] ? config["mappoint_convergence_sigma2_thresh"].as<double>() : 500.0;
  // o.scan_epi_unit_sphere = vk::param<bool>(pnh, "scan_epi_unit_sphere", false);
  o.scan_epi_unit_sphere = config["scan_epi_unit_sphere"] ? config["scan_epi_unit_sphere"].as<bool>() : false;
  // o.affine_est_offset= vk::param<bool>(pnh, "depth_filter_affine_est_offset", true);
  o.affine_est_offset= config["depth_filter_affine_est_offset"] ? config["depth_filter_affine_est_offset"].as<bool>() : true;
  // o.affine_est_gain = vk::param<bool>(pnh, "depth_filter_affine_est_gain", false);
  o.affine_est_gain = config["depth_filter_affine_est_gain"] ? config["depth_filter_affine_est_gain"].as<bool>() : false;
  // o.max_n_seeds_per_frame = static_cast<size_t>(static_cast<double>(vk::param<int>(pnh, "max_fts", 120)) * vk::param<double>(pnh, "max_seeds_ratio", 3.0));
  o.max_n_seeds_per_frame = static_cast<size_t>(static_cast<double>(config["max_fts"] ? config["max_fts"].as<double>() : 120) * (config["max_seeds_ratio"] ? config["max_seeds_ratio"].as<double>() : 3.0));
  // o.max_map_seeds_per_frame = static_cast<size_t>(static_cast<double>(vk::param<int>(pnh, "max_map_fts", 120)));
  o.max_map_seeds_per_frame = static_cast<size_t>(static_cast<double>(config["max_map_fts"] ? config["max_map_fts"].as<double>() : 120));
  // o.extra_map_points = vk::param<bool>(pnh, "depth_filter_extra_map_points", false);
  o.extra_map_points = config["depth_filter_extra_map_points"] ? config["depth_filter_extra_map_points"].as<bool>() : false;
  return o;
}

// InitializationOptions loadInitializationOptions(const ros::NodeHandle& pnh)
InitializationOptions loadInitializationOptions(const YAML::Node config)
{
  InitializationOptions o;
  // o.init_min_features = vk::param<int>(pnh, "init_min_features", 100);
  o.init_min_features = config["init_min_features"] ? config["init_min_features"].as<int>() : 100;
  // o.init_min_tracked = vk::param<int>(pnh, "init_min_tracked", 80);
  o.init_min_tracked = config["init_min_tracked"] ? config["init_min_tracked"].as<int>() : 80;
  // o.init_min_inliers = vk::param<int>(pnh, "init_min_inliers", 70);
  o.init_min_inliers = config["init_min_inliers"] ? config["init_min_inliers"].as<int>() : 70;
  // o.init_min_disparity = vk::param<double>(pnh, "init_min_disparity", 40.0);
  o.init_min_disparity = config["init_min_disparity"] ? config["init_min_disparity"].as<double>() : 40.0;
  // o.init_min_features_factor = vk::param<double>(pnh, "init_min_features_factor", 2.0);
  o.init_min_features_factor = config["init_min_features_factor"] ? config["init_min_features_factor"].as<double>() : 2.0;
  // o.reproj_error_thresh = vk::param<double>(pnh, "reproj_err_thresh", 2.0);
  o.reproj_error_thresh = config["reproj_err_thresh"] ? config["reproj_err_thresh"].as<double>() : 2.0;
  // o.init_disparity_pivot_ratio = vk::param<double>(pnh, "init_disparity_pivot_ratio", 0.5);
  o.init_disparity_pivot_ratio = config["init_disparity_pivot_ratio"] ? config["init_disparity_pivot_ratio"].as<double>() : 0.5;
  // std::string init_method = vk::param<std::string>(pnh, "init_method", "FivePoint");
  std::string init_method = config["init_method"] ? config["init_method"].as<std::string>() : "FivePoint";
  if(init_method == "Homography")
    o.init_type = InitializerType::kHomography;
  else if(init_method == "FivePoint")
    o.init_type = InitializerType::kFivePoint;
  else if(init_method == "OneShot")
    o.init_type = InitializerType::kOneShot;
  else
    SVO_ERROR_STREAM("Initialization Method not supported: " << init_method);
  return o;
}

// FeatureTrackerOptions loadTrackerOptions(const ros::NodeHandle& pnh)
FeatureTrackerOptions loadTrackerOptions(const YAML::Node config)
{
  FeatureTrackerOptions o;
  // o.klt_max_level = vk::param<int>(pnh, "klt_max_level", 4);
  o.klt_max_level = config["klt_max_level"] ? config["klt_max_level"].as<int>() : 4;
  // o.klt_min_level = vk::param<int>(pnh, "klt_min_level", 0.001);
  o.klt_min_level = config["klt_min_level"] ? config["klt_min_level"].as<int>() : 0.001;
  return o;
}

// ReprojectorOptions loadReprojectorOptions(const ros::NodeHandle& pnh)
ReprojectorOptions loadReprojectorOptions(const YAML::Node config)
{
  ReprojectorOptions o;
  // o.max_n_kfs = vk::param<int>(pnh, "reprojector_max_n_kfs", 5);
  o.max_n_kfs = config["reprojector_max_n_kfs"] ? config["reprojector_max_n_kfs"].as<int>() : 5;
  // o.max_n_features_per_frame = vk::param<int>(pnh, "max_fts", 160);
  o.max_n_features_per_frame = config["max_fts"] ? config["max_fts"].as<int>() : 160;
  // o.cell_size = vk::param<int>(pnh, "grid_size", 35);
  o.cell_size = config["grid_size"] ? config["grid_size"].as<int>() : 35;
  // o.reproject_unconverged_seeds = vk::param<bool>(pnh, "reproject_unconverged_seeds", true);
  o.reproject_unconverged_seeds = config["reproject_unconverged_seeds"] ? config["reproject_unconverged_seeds"].as<bool>() : true;
  // o.max_unconverged_seeds_ratio = vk::param<double>(pnh, "max_unconverged_seeds_ratio", -1.0);
  o.max_unconverged_seeds_ratio = config["max_unconverged_seeds_ratio"] ? config["max_unconverged_seeds_ratio"].as<double>() : -1.0;
  // o.min_required_features = vk::param<int>(pnh, "quality_min_fts", 50);
  o.min_required_features = config["quality_min_fts"] ? config["quality_min_fts"].as<int>() : 50;
  // o.seed_sigma2_thresh = vk::param<double>(pnh, "seed_convergence_sigma2_thresh", 200.0);
  o.seed_sigma2_thresh = config["seed_convergence_sigma2_thresh"] ? config["seed_convergence_sigma2_thresh"].as<double>() : 200.0;
  // o.affine_est_offset = vk::param<bool>(pnh, "reprojector_affine_est_offset", true);
  o.affine_est_offset = config["reprojector_affine_est_offset"] ? config["reprojector_affine_est_offset"].as<bool>() : true;
  // o.affine_est_gain = vk::param<bool>(pnh, "reprojector_affine_est_gain", false);
  o.affine_est_gain = config["reprojector_affine_est_gain"] ? config["reprojector_affine_est_gain"].as<bool>() : false;
  // o.max_fixed_landmarks = vk::param<int>(pnh, "reprojector_max_fixed_landmarks", 50);
  o.max_fixed_landmarks = config["reprojector_max_fixed_landmarks"] ? config["reprojector_max_fixed_landmarks"].as<int>() : 50;
  // o.max_n_global_kfs = vk::param<int>(pnh, "reprojector_max_n_global_kfs", 20);
  o.max_n_global_kfs = config["reprojector_max_n_global_kfs"] ? config["reprojector_max_n_global_kfs"].as<int>() : 20;
  // o.use_kfs_from_global_map = vk::param<bool>(pnh, "reprojector_use_kfs_from_global_map", false);
  o.use_kfs_from_global_map = config["reprojector_use_kfs_from_global_map"] ? config["reprojector_use_kfs_from_global_map"].as<bool>() : false;
  // o.fixed_lm_grid_size = vk::param<int>(pnh, "reprojector_fixed_lm_grid_size", 50);
  o.fixed_lm_grid_size = config["reprojector_fixed_lm_grid_size"] ? config["reprojector_fixed_lm_grid_size"].as<int>() : 50;

  return o;
}

// CameraBundle::Ptr loadCameraFromYaml(const ros::NodeHandle& pnh)
CameraBundle::Ptr loadCameraFromYaml(std::string calib_file)
{
  // std::string calib_file = vk::param<std::string>(pnh, "calib_file", "~/cam.yaml");
  CameraBundle::Ptr ncam = CameraBundle::loadFromYaml(calib_file);
  std::cout << "loaded " << ncam->numCameras() << " cameras";
  for(const auto& cam : ncam->getCameraVector())
    cam->printParameters(std::cout, "");
  return ncam;
}

// void setInitialPose(const ros::NodeHandle& pnh, FrameHandlerBase& vo)
void setInitialPose(const YAML::Node config, FrameHandlerBase& vo)
{
  Transformation T_world_imuinit(
        // Quaternion(vk::param<double>(pnh, "T_world_imuinit/qw", 1.0),
        //            vk::param<double>(pnh, "T_world_imuinit/qx", 0.0),
        //            vk::param<double>(pnh, "T_world_imuinit/qy", 0.0),
        //            vk::param<double>(pnh, "T_world_imuinit/qz", 0.0)),
        // Vector3d(vk::param<double>(pnh, "T_world_imuinit/tx", 0.0),
        //          vk::param<double>(pnh, "T_world_imuinit/ty", 0.0),
        //          vk::param<double>(pnh, "T_world_imuinit/tz", 0.0)));
        Quaternion(config["T_world_imuinit/qw"] ? config["T_world_imuinit/qw"].as<double>() : 1.0,
                   config["T_world_imuinit/qx"] ? config["T_world_imuinit/qx"].as<double>() : 0.0,
                   config["T_world_imuinit/qy"] ? config["T_world_imuinit/qy"].as<double>() : 0.0,
                   config["T_world_imuinit/qz"] ? config["T_world_imuinit/qz"].as<double>() : 0.0),
        Vector3d(config["T_world_imuinit/tx"] ? config["T_world_imuinit/tx"].as<double>() : 0.0,
                 config["T_world_imuinit/ty"] ? config["T_world_imuinit/ty"].as<double>() : 0.0,
                 config["T_world_imuinit/tz"] ? config["T_world_imuinit/tz"].as<double>() : 0.0));
  vo.setInitialImuPose(T_world_imuinit);
}


FrameHandlerMono::Ptr makeMono(const std::string config_filepath, const std::string calib_filepath, const CameraBundlePtr& cam)
{
  // Loading of yaml parameters
  YAML::Node config = YAML::LoadFile(config_filepath);
  // Create camera
  CameraBundle::Ptr ncam = (cam) ? cam : loadCameraFromYaml(calib_filepath);
  if (ncam->numCameras() > 1)
  {
    LOG(WARNING) << "Load more cameras than needed, will erase from the end.";
    ncam->keepFirstNCams(1);
  }


  // Init VO
  FrameHandlerMono::Ptr vo =
      std::make_shared<FrameHandlerMono>(
        loadBaseOptions(config, false),
        loadDepthFilterOptions(config),
        loadDetectorOptions(config),
        loadInitializationOptions(config),
        loadReprojectorOptions(config),
        loadTrackerOptions(config),
        ncam);

  // Get initial position and orientation of IMU
  setInitialPose(config, *vo);

  return vo;
}

} // namespace factory
} // namespace svo
