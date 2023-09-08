# LTS-Filter for Docker 

### Pre setup
In your catkin_ws, please clone and build the following packages:
```bash
cd </path/to/catkin_ws>/src
git clone https://github.com/koide3/ndt_omp
git clone https://github.com/SMRT-AIST/fast_gicp --recursive 
git clone https://github.com/koide3/hdl_global_localization 
git clone --branch SPS https://github.com/ibrahimhroob/hdl_localization.git
```

Then build the packages:
```bash
cd </path/to/catkin_ws>
catkin build
```

### Building the Docker image
We provide a ```Dockerfile``` and a ```docker-compose.yaml``` to run all docker commands. 

**IMPORTANT** To have GPU access during the build stage, make ```nvidia``` the default runtime in ```/etc/docker/daemon.json```:

    ```yaml
    {
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
            } 
        },
        "default-runtime": "nvidia" 
    }
    ```
    Save the file and run ```sudo systemctl restart docker``` to restart docker.


To build the image, simply type the following in the terminal:
```bash
bash build_docker.sh
```

To run the container:
```bash
docker compose up -d
```

### Dataset
Bacchus dataset:
```bash
wget https://lcas.lincoln.ac.uk/nextcloud/index.php/s/ssibg4rtrC4XFNJ/download -O Bacchus.zip && unzip Bacchus.zip && rm Bacchus.zip
```


### Running
To run, export the path to the data

```bash
export DATA=path/to/dataset
```

### Training
```bash
docker exec -it lts_filter-project-1 bash
python scripts/train.py
```

## Cleaning the environment

After you finish working with the container, please shut it down using the following command:

```bash
docker compose down
```

## License
This project is free software made available under the MIT License. For details see the LICENSE file.
