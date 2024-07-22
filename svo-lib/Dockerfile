FROM ubuntu:20.04

# Own dependecies
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-importlib-metadata python3-more-itertools python3-zipp
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libglew-dev libopencv-dev libyaml-cpp-dev cmake libboost-all-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip python3.8-venv libeigen3-dev python3-pybind11

# Create Python Environmnent
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt


# Set work directory to repo
WORKDIR /workspace/svo-rl/svo-lib
