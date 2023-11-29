#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ImuSubscriber:
    def __init__(self):
        rospy.init_node('imu_subscriber', anonymous=True)
        self.imu_data_list = []
        self.data_count = 0

        # Subscribe to the /imu/data_raw topic
        rospy.Subscriber('/imu/data_raw', Imu, self.imu_callback)

    def imu_callback(self, data):
        # Update imu_data_list with the latest data
        self.imu_data_list.append(data)

        # Increment the data_count
        self.data_count += 1

        # If 50 data points have been received, plot and reset the count
        if self.data_count == 50:
            self.plot_data()
            self.data_count = 0

    def plot_data(self):
        # Plot the accelerometer data in a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Extract accelerometer data from imu_data_list
        accelerations = [data.linear_acceleration for data in self.imu_data_list]

        # Plot X, Y, and Z components
        ax.plot([acc.x for acc in accelerations], [acc.y for acc in accelerations], [acc.z for acc in accelerations], label='Accelerometer Data')

        # Customize the plot
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('Accelerometer Data')

        # Show the legend
        ax.legend()

        # Display the plot
        plt.show()

if __name__ == '__main__':
    imu_subscriber = ImuSubscriber()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
