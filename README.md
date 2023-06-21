# LTS-Filter for Docker 

![LTS-Filter](img/lts_filter.png)

## 1. Getting the Docker Image

This Docker configuration is designed for the map-based hdl-localization package and the Long-Term Stability (LTS) points filter.

### 1.1 Pulling a precompiled image

We precompiled a Docker image and hosted on our LCAS Docker Hub. In order to download it locally, you can login first in the server and then download it by using the following commands:

```bash
docker login lcas.lincoln.ac.uk -u lcas -p lincoln
docker pull lcas.lincoln.ac.uk/lts_filter
```

This step is recommended because the most time-efficient.

### 1.2 (Expert users/developers) Locally compiling the image

To use the Docker container, run the following command to build the package:

```bash
./build_docker.sh
```

## 2. Launching the Docker environment

Before launching the container, please make sure to set the correct `ROS_MASTER_URI` in the `docker-compose.yml` file. 

To launch the map-based localization node, we need first to login into the container and then launch the node:
```bash
docker-compose up -d
```

### 2.1 Launching the filter

In order to launch the pre-trained filter, please run in terminal the following: 

```bash
./run_filter.sh
```

If you want to reconfigure the default point cloud topics for the filter, you can modify them in the `run_filter.sh` script or pass them as arguments to the script. 
`NB:` Please note that the performance of the filter depends on the GPU. Real-time performance may not be feasible with less powerful GPUs.

### 2.2 Launching the NDT-localizer

To run the localizer, execute the following command:

```bash
./run_localizer.sh -C <CLOUD_TOPIC>
```

Replace <CLOUD_TOPIC> with the topic that will be used for localization, either the raw or the filtered topic.

The robot's pose within the map is exposed in the `/ndt/odom` topic, while a transformation `map -> /ndt_odom` is published on the `tf_tree` to not interfere with an alternative localization method (e.g., RTK-GPS). 

For setting the initial robot pose within the map, it is recommended to use RVIZ's 2D Pose Estimate functionality. RVIZ visualisation configurations can be found in the config folder:

```bash
rviz -d config/hdl_localization.rviz
```

## 3. Testing with a ROS bag
You can find [rosbags](https://drive.google.com/drive/folders/1QXDFUI_gCjb6L3F0F-lc-5blk1GA3lAT?usp=sharing) that can be used to test the filter.


## 4. Cleaning the environment

After you finish working with the container, please shut it down using the following command:

```bash
docker-compose down
```


