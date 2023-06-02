# LTS-Filter for Docker 

![LTS-Filter](img/lts_filter.png)

This is a Docker configuration for the map based hdl-localization package and the long-term stability (LTS) points filter. 

To use the Dockerfile, run the following command to build the package:

```bash
./build_docker.sh
```

Before launching the container, please make sure to set the correct `ROS_MASTER_URI` in `docker-compose.yml` file. 

To launch the map based localizatoin node, we need first to login into the container then launch the node:
```basha
docker-compose up -d
```

To launch the filter: 
```bash
./run_filter.sh
```
You may need to reconfigure the point cloud topics of the filter based on the point cloud topics of the system, please modify them in the `run_filter.sh` or you can pass them as arguments to the script. 

To run the localizer, please run: 
```bash
./run_localizer.sh -C <CLOUT_TOPIC>
```
Please set the <CLOUT_TOPIC> to the topic that will be used for localization, you could use either the raw or the filtered topic. 

The robot pose within the map is exposed in the `/odom` topic. 

To set the initial robot pose within the map, we recommend to use RVIZ `2D Pose Estimate`, RVIZ visualizatoin configuratoins are in the config folder:
```bash
rviz -d config/hdl_localization.rviz
```

You can find [rosbags](https://drive.google.com/drive/folders/1QXDFUI_gCjb6L3F0F-lc-5blk1GA3lAT?usp=sharing) that can be used to test the filter.

After finishing with the container, please shut it down using:
```bash
docker-compose down
```
