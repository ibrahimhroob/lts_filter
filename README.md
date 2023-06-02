# LTS-Filter for Docker 

![LTS-Filter](img/lts_filter.png)

This Docker configuration is designed for the map-based hdl-localization package and the Long-Term Stability (LTS) points filter.

To use the Docker container, run the following command to build the package:

```bash
./build_docker.sh
```

Before launching the container, please make sure to set the correct `ROS_MASTER_URI` in `docker-compose.yml` file. 

To launch the map based localization node, we need first to login into the container then launch the node:
```basha
docker-compose up -d
```

To launch the filter: 
```bash
./run_filter.sh
```
If you want to reconfigure the default point cloud topics for the filter, you can modify them in the run_filter.sh script or pass them as arguments to the script. Please note that the performance of the filter depends on the GPU. Real-time performance may not be feasible with less powerful GPUs.


To run the localizer, execute the following command:
```bash
./run_localizer.sh -C <CLOUD_TOPIC>
```
Replace <CLOUD_TOPIC> with the topic that will be used for localization, either the raw or the filtered topic.

The robot pose within the map is exposed in the `/odom` topic. 

For setting the initial robot pose within the map, it is recommended to use RVIZ's 2D Pose Estimate functionality. RVIZ visualisation configurations can be found in the config folder:

```bash
rviz -d config/hdl_localization.rviz
```

You can find [rosbags](https://drive.google.com/drive/folders/1QXDFUI_gCjb6L3F0F-lc-5blk1GA3lAT?usp=sharing) that can be used to test the filter.

After you finish working with the container, please shut it down using the following command:

```bash
docker-compose down
```


