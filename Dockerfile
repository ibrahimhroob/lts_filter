FROM nvidia/cudagl:11.4.1-base-ubuntu20.04

# Config
ENV ROS_DISTRO noetic

# Minimal setup
RUN apt-get update && apt-get install -y locales lsb-release
ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales

# Install ROS
# a. Setup your sources.list
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# b. Set up the keys
RUN apt install -y curl wget && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# c. Installation
RUN apt update && \
    apt install -y ros-${ROS_DISTRO}-desktop-full && \
    echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

# Setup torch, cuda for the model and other dependencies 
RUN apt install -y python3-pip ros-noetic-ros-numpy git &&\
    pip install torch==1.7.1+cu110 \
    torchvision==0.8.2+cu110 \
    torchaudio===0.7.2 -f https://download.pytorch.org/whl/cu110/torch_stable.html 


# Clone the filter 
RUN cd  /home && \
    git clone https://github.com/ibrahimhroob/inference_model.git

# Download the model 
RUN cd /home/inference_model/ktima && \
    wget https://lcas.lincoln.ac.uk/nextcloud/index.php/s/KTS4XYWxGxbYtXs/download -O best_model.pth

# Run main.py when the container launches
# CMD ["python3", "/home/inference_model/stability_filter.py"]
