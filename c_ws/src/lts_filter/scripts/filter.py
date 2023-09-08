#!/usr/bin/env python3

import time
import torch
import struct
import numpy as np

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField

import lts.models.pointnet as pn
import lts.models.transformer as pct

from lts.util.slices_loader import ScansDataLoader

class Stability():
    def __init__(self):
        rospy.init_node('pointcloud_stability_inference')

        raw_cloud_topic      = rospy.get_param('~raw_cloud', "/os_cloud_node/points")
        filtered_cloud_topic = rospy.get_param('~filtered_cloud', "/cloud_filtered_kiss")


        epsilon_0 = rospy.get_param('~epsilon_0', 0.05)
        epsilon_1 = rospy.get_param('~epsilon_1', 0.85)

        self.model_type = rospy.get_param('~model', 'pct')
        weights_pth = rospy.get_param('~model_weights_pth', "/lts/log/point_reg/train_0/checkpoints/best_model.pth")

        rospy.Subscriber(raw_cloud_topic, PointCloud2, self.callback)

        # Initialize the publisher
        self.pub = rospy.Publisher(filtered_cloud_topic, PointCloud2, queue_size=10)

        rospy.loginfo('raw_cloud: %s', raw_cloud_topic)
        rospy.loginfo('filtered_cloud: %s', filtered_cloud_topic)
        rospy.loginfo('Bottom threshold: %f', epsilon_0)
        rospy.loginfo('Upper threshold: %f', epsilon_1)

        self.threshold_ground = epsilon_0
        self.threshold_dynamic = epsilon_1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(self.device, weights_pth)

        rospy.spin()

    def callback(self, pointcloud_msg):
        pc = ros_numpy.numpify(pointcloud_msg)
        height = pc.shape[0]
        try:
            width = pc.shape[1]
        except:
            width = 1
        data = np.zeros((height * width, 4), dtype=np.float32)
        data[:, 0] = np.resize(pc['x'], height * width)
        data[:, 1] = np.resize(pc['y'], height * width)
        data[:, 2] = np.resize(pc['z'], height * width)
        data[:, 3] = np.resize(pc['intensity'], height * width)

        # Infere the stability labels
        data = self.infer(data)

        filtered_cloud = self.to_rosmsg(data, pointcloud_msg.header)

        self.pub.publish(filtered_cloud)


    def to_rosmsg(self, data, header):
        filtered_cloud = PointCloud2()
        filtered_cloud.header = header

        # Define the fields for the filtered point cloud
        filtered_fields = [PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        PointField('intensity', 12, PointField.FLOAT32, 1)]

        filtered_cloud.fields = filtered_fields
        filtered_cloud.is_bigendian = False
        filtered_cloud.point_step = 16
        filtered_cloud.row_step = filtered_cloud.point_step * len(data)
        filtered_cloud.is_bigendian = False
        filtered_cloud.is_dense = True
        filtered_cloud.width = len(data)
        filtered_cloud.height = 1


        # Filter the point cloud based on intensity
        for point in data:
            filtered_cloud.data += struct.pack('ffff', point[0], point[1], point[2], point[3])

        return filtered_cloud


    def load_model(self, device, weights_pth):

        # 2. Define model
        model = pn
        if self.model_type == 'pct':
            model = pct

        model = model.get_model()
        model.to(device)

        checkpoint = torch.load( weights_pth )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.eval()   

        rospy.loginfo("Model loaded successfully!")

        return model

    def infer(self, pointcloud):
        
        start_time = time.time()
        FRAME_DATASET = ScansDataLoader(pointcloud)

        points, _ = FRAME_DATASET[0]

        points = torch.from_numpy(points)
        points = points.unsqueeze(0)

        for i in range(1, len(FRAME_DATASET)):
            p, _ = FRAME_DATASET[i]
            p = torch.from_numpy(p)
            p = p.unsqueeze(0)
            points = torch.vstack((points, p))

        points = points.float().to(self.device)
        points = points.transpose(2, 1)
        labels = self.model(points)

        points = points.permute(0,2,1).cpu().data.numpy().reshape((-1, 3))
        labels = labels.permute(0,2,1).cpu().data.numpy().reshape((-1, ))

        data = np.column_stack((points, labels))

        data = data[(data[:,3] < self.threshold_dynamic) & (data[:,3] >= self.threshold_ground)]

        end_time = time.time()
        elapsed_time = end_time - start_time
        rospy.loginfo("Frame inference and filter processing time: {:.4f} seconds [{:.2f} Hz]".format(elapsed_time, 1/elapsed_time))

        return data


if __name__ == '__main__':
    stability_node = Stability()
    