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

printf "\n${PURPLE}***Script ID: run_localizer***${NC}\n"


######### ENVIRONMENTAL VARIABLES  ############################################
CLOUD_TOPIC=/os_cloud_node/points

# const variables
image_name=local
ROS_DISTRO=noetic

###############################################################################
#parse bash args of they exists to override the default params 
while getopts C: flag
do
    case "${flag}" in
        C) CLOUD_TOPIC=${OPTARG};;
    esac
done

printf "Args:\n"
printf "(C)LOUD_TOPIC:     ${CYAN}${CLOUD_TOPIC}${NC}\n"
printf "\n"

###############################################################################
#Main code is here

printf "${GREEN}Starting ${image_name} localizer node...${NC}\n"
source /home/zotac/uol_localization_ws/devel/setup.bash
roslaunch hdl_localization hdl_localization.launch \
                  points_topic:=${CLOUD_TOPIC} 2> >(grep -v TF_REPEATED_DATA buffer_core)
