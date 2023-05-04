# LTS-Filter for Docker

![LTS-Filter](img/lts_filter.png)

This is a Docker configuration for the long-term stability (LTS) points filter. 

To use the Dockerfile, run the following command to build the filter image:

```bash
./build_docker.sh
```

To run the Docker node, set the following parameters:
* ROS_MASTER_URI: the ROS master URI (default: http://Mars:11311/)
* RAW_CLOUD: raw point cloud topic (default: /os_cloud_node/points)
* FILTERED_CLOUD: filtered point cloud topic (default: /cloud_filtered)
* EPSILON_0: bottom threshold used to filter the ground points (default: 0.04)
* EPSILON_1: upper threshold used to filter the dynamic points (default: 0.85)

These parameters can be set in the `run_docker.sh` file or passed as arguments as follows:

```bash
./run_docker.sh -U <ROS_MASTER_URI> -R <RAW_CLOUD> -G <EPSILON_0> -D <EPSILON_1>
```

You can find [rosbags](https://drive.google.com/drive/folders/1QXDFUI_gCjb6L3F0F-lc-5blk1GA3lAT?usp=sharing) that can be used to test the filter.

---
The Docker image is set up to run as a ROS node that subscribes to the __RAW_CLOUD__ topic and publishes the __FILTERED_CLOUD__ topic. Here are the steps to test it:
* Open a terminal and run `roscore` to get the ROS master URI.
* In a second terminal, run the Docker node and pass the correct master URI as explained above.
* To visualize the topics, open another terminal and launch `rviz`, then add the cloud topics. If you are using the supplied bags for testing, you may need to change the fixed frame ID in `rviz` to `os_sensor`.
