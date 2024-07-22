#!/bin/sh
xhost local:root


XAUTH=/tmp/.docker.xauth

docker run --privileged --rm -it \
    --volume <path>/vo_rl/:<path>/vo_rl/:rw \
    --volume<path>/TartanAir/:/datasets/TartanAir/:ro \
    --volume <path>/EuRoC/:/datasets/EuRoC/:ro \
    --volume <path>/TUM-RGBD/:/datasets/TUM-RGBD/:ro \
    --volume <path>/log_voRL/:/logs/log_voRL/:rw \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --net=host \
    --ipc=host \
    --privileged \
    --user $(id -u):$(id -g) \
    --gpus=all \
    vo_rl
    bash
