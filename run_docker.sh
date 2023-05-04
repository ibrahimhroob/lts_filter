#!/bin/bash

# ENV
ROS_MASTER_URI=http://Mars:11311/

image_name=lts_filter


echo "Starting ${image_name} docker container..."
docker run --privileged                            \
           --network host                          \
           --gpus all                              \
           --env="NVIDIA_DRIVER_CAPABILITIES=all"  \
           --rm         \
           -e ROS_MASTER_URI=${ROS_MASTER_URI}     \
           -w /home               \
           -it ${image_name} 
     