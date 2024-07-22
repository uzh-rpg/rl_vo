#include <glog/logging.h>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

#include "svo_env/svo_env.h"
#include <svo/frame_handler_mono.h>
#include <gflags/gflags.h>


void TestResults(std::string file1_path , std::string file2_path);
void ReadImageTimestamp(int img_counter, std::vector<cv::Mat>& images, long& time_nsec);
void StoreResults(svo::FramePtr frame, int counter, cv::Mat img_rgb, std::string pose_path, std::string img_dir_path);


int main(int argc, char **argv)
{
  // google::InitGoogleLogging(argv[0]);
  // gflags::ParseCommandLineFlags(&argc, &argv, true);
  // google::InstallFailureSignalHandler();

  // Loading Params
  std::string config_filepath = "/workspace/svo-rl/svo_ws/src/svo-lib/svo_env/param/pinhole.yaml";
  std::string calib_filepath = "/workspace/svo-rl/svo_ws/src/svo-lib/svo_env/param/calib/svo_test_pinhole.yaml";
  svo::SvoEnv svo_env(config_filepath, calib_filepath);

  // Storing stuff
  std::string pose_path = "/workspace/svo-rl/data/Test/test_poses.txt";
  std::string img_dir_path = "/workspace/svo-rl/data/Test/feature_images";
  std::remove(pose_path.c_str());
  boost::filesystem::remove_all(img_dir_path);
  boost::filesystem::create_directory(img_dir_path);


  // Main Loop
  for (int i = 0; i < 300; ++i) {
    std::vector<cv::Mat> images;
    long time_nsec;
    ReadImageTimestamp(i, images, time_nsec);

    cv::Mat img_rgb;
    svo_env.monoCallback(images, time_nsec, img_rgb);

    svo::FramePtr frame = svo_env.svo_->getLastFrames()->at(0);
    StoreResults(frame, i, img_rgb, pose_path, img_dir_path);
    
    std::this_thread::sleep_for(std::chrono::duration<double>(0.05));
    std::cout<<"Processing Iteration: "<<i<<std::endl;
  }

  // Check if output is the same
  std::string file1_path = "/workspace/svo-rl/data/Test/gt_poses.txt";
  TestResults(file1_path, pose_path);

  std::cout<<"Finished"<<std::endl;

}


void TestResults(std::string file1_path , std::string file2_path) {
  std::ifstream file1(file1_path);
  std::ifstream file2(file2_path);
  std::string line1, line2;
  int line_number = 1; 
  double threshold = 0.05;
  while (std::getline(file1, line1) && std::getline(file2, line2)) {
    std::vector<double> numbers1, numbers2;
    std::istringstream ss1(line1), ss2(line2);
    double num1, num2;

    for (int i=0; i<12; ++i) {
        ss1 >> num1;
        ss2 >> num2;
        numbers1.push_back(num1);
        numbers2.push_back(num2);
    }

    bool all_within_threshold = true;
    for (size_t i = 0; i < numbers1.size(); ++i) {
        if (std::abs(numbers1[i] - numbers2[i]) > threshold) {
            std::cout << "Line " << line_number << " exceeds the threshold:" << std::endl;
            std::cout << "File 1: " << line1 << std::endl;
            std::cout << "File 2: " << line2 << std::endl;
        }
    }

    ++line_number;
  }
}


void ReadImageTimestamp(int img_counter, std::vector<cv::Mat>& images, long& time_nsec)  {
  // Read image and timestamp
  cv::Mat image;
  std::stringstream ss;
  ss << "/workspace/svo-rl/data/images/" << "frame" << std::setw(6) << std::setfill('0') << img_counter << ".png";
  std::string img_filename = ss.str();
  image = cv::imread(img_filename);
  images.push_back(image.clone());

  int line_number = img_counter;
  std::string timestamp_path = "/workspace/svo-rl/data/images/timestamps.txt";
  std::ifstream file(timestamp_path);
  std::string line_content;
  for (int current_line = 0; current_line <= line_number; ++current_line) {
    if (!std::getline(file, line_content)) {
      std::cerr << "Line " << line_number << " not found in the file." << std::endl;
      break;
      }};
  file.close();
  time_nsec = std::stol(line_content);
}


void StoreResults(svo::FramePtr frame, int counter, cv::Mat img_rgb, std::string pose_path, std::string img_dir_path)  {
  Eigen::Map<Eigen::VectorXd> pose_matrix_vector(frame->T_world_cam().getTransformationMatrix().data(), 
                                                 frame->T_world_cam().getTransformationMatrix().size());
  std::ofstream outfile(pose_path, std::ios::app);
  if (outfile.is_open()) {
      // Write the vector elements to the file
      for (int i = 0; i < pose_matrix_vector.size(); ++i) {
          outfile << pose_matrix_vector(i) << " ";
      }
      outfile << "\n"; // Add a newline character to separate lines
      // Close the file
      outfile.close();
  } else {
      std::cerr << "Unable to open the file: " << pose_path << std::endl;
  }

  if (!img_rgb.empty()) {
    std::string img_base_filename = img_dir_path + "/feature_image_";
    std::stringstream ss;
    ss << counter;
    std::string output_filename = img_base_filename + ss.str() + ".jpg";
    bool success = cv::imwrite(output_filename, img_rgb);
  }
}
