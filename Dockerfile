FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

RUN apt-get update && apt-get -y install sudo

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV USERNAME <your_username>
ENV HOME /home/$USERNAME

RUN useradd -m $USERNAME && \
        echo "$USERNAME:$USERNAME" | chpasswd && \
        usermod --shell /bin/bash $USERNAME && \
        usermod -aG sudo $USERNAME && \
        mkdir -p /etc/sudoers.d && \
        echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
        chmod 0440 /etc/sudoers.d/$USERNAME && \
        # Replace 1003 with your user/group id
        usermod  --uid 1003 $USERNAME && \
  groupmod --gid 1001 $USERNAME

# svo-lib dependecies
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-importlib-metadata python3-more-itertools python3-zipp python3-tk tmux
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libglew-dev libopencv-dev libyaml-cpp-dev cmake libboost-all-dev nvidia-container-toolkit
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip python3.8-venv libeigen3-dev python3-pybind11
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libsuitesparse-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y valgrind
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libssl-dev

# Create Python Environmnent
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip3 install networkx==3.1
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt

USER <your_username>
WORKDIR <path>/vo_rl