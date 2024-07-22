#!/bin/sh
xhost local:root


XAUTH=/tmp/.docker.xauth

docker run --privileged --rm -it \
    --volume <path>/svo-rl/:/workspace/svo-rl/:rw \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --net=host \
    --privileged \
    svo_lib
    bash