import numpy as np
import taichi as ti
import taichi.math as tm
import time

ti.init(arch=ti.cpu)


class testClass():
    def __init__(self):
        pass

    def method1(self):
        print("method1")

@ti.func
def create_T_Matrix(x: ti.f64, y: ti.f64, z: ti.f64) -> tm.mat4:
    return tm.mat4([[1, 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]])

@ti.func
def create_P_Matrix(p: tm.vec2, yaw:ti.f64) -> tm.mat4:
    return tm.mat4([[ti.cos(yaw), -ti.sin(yaw), 0, p[0]],
                    [ti.sin(yaw), ti.cos(yaw), 0, p[1]],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

@ti.kernel
def test_kernel():
    for i in range(1000):
        mm = create_T_Matrix(i+0.1, i+0.2, i+0.3)
        nn = create_P_Matrix(tm.vec2(i+0.1, i+0.2), i+0.3)
        aa = mm @ nn

@ti.kernel
def calcoffsetpose_ti(pose:tm.vec2, yaw:ti.f64, offset:tm.vec3) -> tm.vec2:
    T = create_T_Matrix(offset[0], offset[1], offset[2])
    P = create_P_Matrix(pose, yaw)
    offset_pose = P @ T

    return tm.vec2(offset_pose[0, 3], offset_pose[1, 3])


def calcoffsetpose(position, yaw, x, y, z):
    '''Only use numpy to calculate the offset pose.
    '''
    c_yaw = np.cos(yaw)
    s_yaw = np.sin(yaw)
    transform_q_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=np.float64)
    tr = position
    pose_matrix = np.array([[c_yaw, -s_yaw, 0, tr[0]], [s_yaw, c_yaw, 0, tr[1]], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    offset_position = np.dot(pose_matrix, transform_q_matrix)

    return offset_position[:2, 3]


def test_calcoffsetpose():
    pose = tm.vec2(3849.0256916445646, 73705.4632446311)
    yaw = -1.06
    offset = tm.vec3(3.041403437917443, 0.0, 0.0)

    position = np.array([3849.0256916445646, 73705.4632446311], dtype=np.float64)
    x = 3.041403437917443
    y = 0.0
    z = 0.0

    s1 = time.time()
    for i in range(1000):
        offset_pose = calcoffsetpose(position, yaw, x, y, z)
    e1 = time.time()
    print("np time: ", (e1-s1)/1000)

    s2 = time.time()
    for i in range(1000):
        offset_pose = calcoffsetpose_ti(pose, yaw, offset)
    e2 = time.time()
    print("ti time: ", (e2-s2)/1000)




def main():
    #s1 = time.time()

    #test_kernel()
    test_calcoffsetpose()

    #e1 = time.time()
    #print("time: ", (e1-s1)/1000)


if __name__ == "__main__":
    main()