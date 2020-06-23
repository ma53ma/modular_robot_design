import pybullet as p
import pybullet_data
import numpy as np
import math
import time

def sim(goal):
    physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version, p.DIRECT is faster
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    cubeStartPos = [0,0,0]
    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    robotId = p.loadURDF("custom.urdf",cubeStartPos, cubeStartOrientation)

    finalJoint = p.getNumJoints(robotId)

    # inverse kinematics
    angles = p.calculateInverseKinematics(robotId, finalJoint - 1, goal)

    #p.setRealTimeSimulation(1) # don't need if doing p.DIRECT for server, but do need for p.GUI

    # forward kinematics
    for i in range(len(angles)):
        p.resetJointState(robotId, 3 * i + 2, angles[i])
    #for joint in range(finalJoint):
    endEffPos = p.getLinkState(robotId, finalJoint - 1)[0]

    # determine dist from tip of EE to goal
    dist = np.linalg.norm(np.array([a_i - b_i for a_i, b_i in zip(endEffPos, goal)]))
    # delete arm from simulation
    p.disconnect()
    #print('dist: ', dist)
    return dist, endEffPos