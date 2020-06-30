import pybullet as p
import pybullet_data
import numpy as np
import math
import time

def sim(goal, epsilon):
    physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version, p.DIRECT is faster
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    cubeStartPos = [0,0,0]
    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    robotId = p.loadURDF("custom.urdf",cubeStartPos, cubeStartOrientation)

    finalJoint = p.getNumJoints(robotId)

    # inverse kinematics

    #p.setRealTimeSimulation(1) # don't need if doing p.DIRECT for server, but do need for p.GUI
    #endEffPos = 0
    #dist = 0
    # forward kinematics
    for i in range(20):
        angles = p.calculateInverseKinematics(robotId, finalJoint - 1, goal)#)
        for i in range(len(angles)):
            p.resetJointState(robotId, 3 * i + 2, angles[i])
    endEffPos = p.getLinkState(robotId, finalJoint - 1)[0]
    dist = np.linalg.norm(np.array([a_i - b_i for a_i, b_i in zip(endEffPos, goal)]))
        #print('iteration ', j)
        #print('distance ', dist)

    #for joint in range(finalJoint):

    # determine dist from tip of EE to goal
    # delete arm from simulation
    p.disconnect()
    #print('dist: ', dist)
    return dist, endEffPos