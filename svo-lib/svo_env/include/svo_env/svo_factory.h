#pragma once

#include <memory>
#include <svo/common/camera_fwd.h>

namespace svo {

// forward declarations
class FrameHandlerMono;

namespace factory {

/// Factory for Mono-SVO.
std::shared_ptr<FrameHandlerMono> makeMono(
    const std::string config_filepath, 
    const std::string calib_filepath,
    const CameraBundlePtr& cam = nullptr);

} // namespace factory
} // namespace mono
