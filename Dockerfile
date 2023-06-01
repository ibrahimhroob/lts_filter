FROM nvidia/cudagl:11.4.1-base-ubuntu20.04

# Config
ENV ROS_DISTRO noetic
ENV USER UoL

# Minimal setup
RUN apt-get update --fix-missing && apt-get install -y locales lsb-release
ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales

# Setup torch, cuda for the model and other dependencies 
RUN apt install -y python3-pip git &&\  
    pip install numpy==1.20.1 \
    torch==1.7.1+cu110 \
    torchvision==0.8.2+cu110 \
    torchaudio===0.7.2 -f https://download.pytorch.org/whl/cu110/torch_stable.html 

# Install ROS
# [ROS] a. Setup your sources.list
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# [ROS] b. Set up the keys
RUN apt install -y curl wget && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# [ROS] c. Installation
RUN apt update && \
    apt install -y ros-${ROS_DISTRO}-ros-base && \
    echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

# Clone the ndt localization package, filter and Download the model
RUN cd  /home && \
    mkdir -p ${USER}/c_ws/src && cd ${USER}/c_ws/src && \
    git clone https://github.com/koide3/ndt_omp && \
    git clone https://github.com/SMRT-AIST/fast_gicp --recursive && \
    git clone https://github.com/ibrahimhroob/hdl_localization.git && \
    git clone https://github.com/koide3/hdl_global_localization && \
    git clone https://github.com/ibrahimhroob/inference_model.git 

# Download base map for localization 
RUN cd /home/${USER}/c_ws/src/hdl_localization/data  && \ 
    wget https://lcas.lincoln.ac.uk/nextcloud/index.php/s/9FM2HmQAG5PzDdF/download -O map.pcd

# Download the model 
RUN cd /home/${USER}/c_ws/src/inference_model/lts_filter/model && \
    wget https://lcas.lincoln.ac.uk/nextcloud/index.php/s/KTS4XYWxGxbYtXs/download -O best_model.pth

# Install catkin tools and other packages 
RUN apt update && \
    apt install -y ros-${ROS_DISTRO}-catkin python3-catkin-tools ros-${ROS_DISTRO}-ros-numpy

# install ros packages
RUN apt-get update && apt-get install --no-install-recommends -y nano build-essential \
    libgtest-dev libomp-dev libboost-all-dev libopencv-dev ros-${ROS_DISTRO}-pcl-ros ros-${ROS_DISTRO}-rviz \
    ros-${ROS_DISTRO}-tf2 ros-${ROS_DISTRO}-tf2-ros ros-${ROS_DISTRO}-tf2-geometry-msgs ros-${ROS_DISTRO}-image-transport ros-${ROS_DISTRO}-image-transport-plugins \
    ros-${ROS_DISTRO}-eigen-conversions ros-${ROS_DISTRO}-tf-conversions python3 python3-pip python3-venv \
    ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-message-filters wget \
    libatlas-base-dev libgoogle-glog-dev libsuitesparse-dev libglew-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install ceres-solver
WORKDIR /root/thirdParty
RUN wget https://github.com/ceres-solver/ceres-solver/archive/refs/tags/1.14.0.tar.gz
RUN tar zxf 1.14.0.tar.gz
RUN cd ceres-solver-1.14.0
RUN mkdir build && cd build
RUN cmake -DCMAKE_BUILD_TYPE=Release ./ceres-solver-1.14.0 && make -j2 && make install

WORKDIR /home/${USER}/c_ws
RUN . /opt/ros/${ROS_DISTRO}/setup.sh; catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m

