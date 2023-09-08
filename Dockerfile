FROM nvidia/cudagl:11.4.1-base-ubuntu20.04

# For faster build, use more jobs.
ENV MAX_JOBS=6

ENV ROS_DISTRO noetic

ENV PROJECT=/lts

ENV DATA=$PROJECT/data

##############################################
# Minimal ubuntu setup
RUN apt update --fix-missing && apt install -y locales lsb-release && apt clean

ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales

##############################################
# Setup torch, cuda for the model and other dependencies 
RUN apt install -y python3-pip git &&\  
    pip install numpy==1.20.1 install scipy\
    torch==1.7.1+cu110 \
    torchvision==0.8.2+cu110 \
    torchaudio===0.7.2 -f https://download.pytorch.org/whl/cu110/torch_stable.html 

##############################################
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

# Install catkin tools and other packages 
RUN apt update && \
    apt install -y ros-${ROS_DISTRO}-catkin python3-catkin-tools ros-${ROS_DISTRO}-ros-numpy

# install ros packages
# Install catkin tools and other packages 
RUN apt update && apt install -y --no-install-recommends nano build-essential \
    ros-${ROS_DISTRO}-catkin python3-catkin-tools ros-${ROS_DISTRO}-ros-numpy \
    libomp-dev libboost-all-dev ros-${ROS_DISTRO}-pcl-ros \
    ros-${ROS_DISTRO}-tf2 ros-${ROS_DISTRO}-tf2-ros ros-${ROS_DISTRO}-tf2-geometry-msgs  \
    ros-${ROS_DISTRO}-eigen-conversions ros-${ROS_DISTRO}-tf-conversions python3 \
    && apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install scikit-learn
##############################################
# Install project related dependencies
RUN mkdir -p $PROJECT/logs && mkdir -p $DATA

WORKDIR $PROJECT
COPY . $PROJECT
RUN python3 setup.py develop 

RUN rm -rf $PROJECT

##############################################
# Add user to share files between container and host system
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid 1000 user \
    && adduser --disabled-password --gecos '' --uid 1000 --gid 1000 user \
    && chown -R user:user /lts
