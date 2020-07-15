import pybullet as p
import pybullet_data
import numpy as np
import math
import time

def sim(goal, epsilon):
    physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version, p.DIRECT is faster
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")
    cubeStartPos = [0,0,0]
    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    robotId = p.loadURDF("custom.urdf",cubeStartPos, cubeStartOrientation)

    # inverse kinematics
    finalJoint = p.getNumJoints(robotId)
    angles = p.calculateInverseKinematics(robotId, finalJoint - 1, goal)
    for i in range(len(angles)):
        p.resetJointState(robotId, 3*i + 2, 0.1)

    #p.setRealTimeSimulation(1) # don't need if doing p.DIRECT for server, but do need for p.GUI

    # forward kinematics
    dist = 0
    endEffPos = 0
    for i in range(20):
        angles = p.calculateInverseKinematics(robotId, finalJoint - 1, goal)
        for j in range(len(angles)):
            p.resetJointState(robotId, 3 * j + 2, angles[j])
        endEffPos = p.getLinkState(robotId, finalJoint - 1)[0]
        dist = np.linalg.norm(np.array([a_i - b_i for a_i, b_i in zip(endEffPos, goal)]))
        if dist < epsilon:
            break

    # determine dist from tip of EE to goal
    # delete arm from simulation
    p.disconnect()
    #print('dist: ', dist)
    return dist, endEffPos