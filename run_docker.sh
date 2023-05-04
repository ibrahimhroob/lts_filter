#!/bin/bash

######### BASH TERMINAL COLORS ################################################
# Black        0;30     Dark Gray     1;30
# Red          0;31     Light Red     1;31
# Green        0;32     Light Green   1;32
# Brown/Orange 0;33     Yellow        1;33
# Blue         0;34     Light Blue    1;34
# Purple       0;35     Light Purple  1;35
# Cyan         0;36     Light Cyan    1;36
# Light Gray   0;37     White         1;37

RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

printf "\n${PURPLE}***Script ID: run_docker***${NC}\n"


######### ENVIRONMENTAL VARIABLES  ############################################
ROS_MASTER_URI=http://Mars:11311/
RAW_CLOUD=/os_cloud_node/points
FILTERED_CLOUD=/cloud_filtered
EPSILON_0=0.04
EPSILON_1=0.85

# const variables
image_name=lts_filter
ROS_DISTRO=noetic

###############################################################################
#parse bash args of they exists to override the default params 
while getopts U:R:F:G:D: flag
do
    case "${flag}" in
        U) ROS_MASTER_URI=${OPTARG};;
        R) RAW_CLOUD=${OPTARG};;
        F) FILTERED_CLOUD=${OPTARG};;
        G) EPSILON_0=${OPTARG};;
        D) EPSILON_1=${OPTARG};;
    esac
done

printf "Args:\n"
printf "ROS_MASTER_(U)RI:     ${CYAN}${ROS_MASTER_URI}${NC}\n"
printf "(R)AW_CLOUD:          ${CYAN}${RAW_CLOUD}${NC}\n"
printf "(F)ILTERED_CLOUD:     ${CYAN}${FILTERED_CLOUD}${NC}\n"
printf "EPSILON_0 (G)round:   ${CYAN}${EPSILON_0}${NC}\n"
printf "EPSILON_1 (D)ynamic:  ${CYAN}${EPSILON_1}${NC}\n"
printf "\n"

###############################################################################
#Main code is here

printf "${GREEN}Starting ${image_name} docker node...${NC}\n"
docker run --privileged                            \
           --network host                          \
           --gpus all                              \
           --env="NVIDIA_DRIVER_CAPABILITIES=all"  \
           --rm                                    \
           -e ROS_MASTER_URI=${ROS_MASTER_URI}     \
           -it ${image_name}                       \
           /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
                  source /home/c_ws/devel/setup.bash && \
                  roslaunch lts_filter filter.launch raw_cloud:=${RAW_CLOUD} \
                                                     filtered_cloud:=${FILTERED_CLOUD} \
                                                     epsilon_0:=${EPSILON_0} \
                                                     epsilon_1:=${EPSILON_1}"

