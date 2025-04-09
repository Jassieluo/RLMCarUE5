import time

import matplotlib.pyplot as plt
from utils_read import SharedUETrainMessageReadTools
import numpy as np
import math


class UE5MessagePlot:
    def __init__(self):
        self.UE5MessageReadTools = SharedUETrainMessageReadTools()

    def create_vectors(self):
        self.UE5MessageReadTools.create_vector_recv_map()

    def meshgrid_z(self, z_list):
        z = np.zeros((len(z_list), len(z_list)))
        for i in range(len(z_list)):
            z[i, i] = z_list[i]
        return z

    def plot_vector_reward_map(self):
        x_list = []
        y_list = []
        z_list = []
        delta_e = 1e-5
        figure = plt.figure(num=0)
        ax = figure.add_subplot(111, projection='3d')
        plt.ion()
        count = 0

        while count <= 100000000:
            vector_list = self.UE5MessageReadTools.get_shared_vector_and_collision_list()
            # x_list.append(vector_list[0])
            # y_list.append(vector_list[1])
            # z_list.append(1/(math.sqrt(vector_list[4]**2+vector_list[5]**2)+delta_e))
            x_list.append(vector_list[0])
            y_list.append(vector_list[1])
            z = 1/(math.sqrt(vector_list[4]**2+vector_list[5]**2)+delta_e)
            count += 1

            surf = ax.scatter3D(vector_list[0], vector_list[1], z, cmap='viridis')
            plt.show()
            plt.pause(0.005)
            # figure.clear()


if __name__ == '__main__':
    U = UE5MessagePlot()
    U.create_vectors()
    U.plot_vector_reward_map()