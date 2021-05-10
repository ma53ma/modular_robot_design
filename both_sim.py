import pybullet as p
import pybullet_data
import numpy as np
import math
import time

# pybullet simulation for inverse kinematics
def sim(goal, pos_epsilon, orient_epsilon):
    physicsClient = p.connect(p.DIRECT)  # P.GUI for image or p.DIRECT for non-graphical version, p.DIRECT is faster
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # need to keep in
    #p.setRealTimeSimulation(1) # don't need if doing p.DIRECT for server, but do need for p.GUI
    # loading in plane
    planeId = p.loadURDF("plane.urdf")
    # loading in arrangement for episode
    robotStartPos = [0,0,0]
    robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
    robotId = p.loadURDF("dqn_sac.urdf", robotStartPos, robotStartOrientation)

    # get number of joints in robot
    finalJoint = p.getNumJoints(robotId)
    # obtaining first iteration of IK to act as initial condition
    angles = p.calculateInverseKinematics(robotId, finalJoint - 1, goal[0:3],goal[3:])
    # move arm into first iteration angles
    for i in range(len(angles)):
        p.resetJointState(robotId, 3*i + 2, angles[i])

    if len(goal) > 3:
        orientation = True
    else:
        orientation = False

    # initializing position and orientation distances and end-effector positions and orientations
    pos_dist = 0
    orient_dist = 0
    endEffPos = 0
    endEffOrient = 0

    # running IK for 10 iterations
    for i in range(10):
        angles = p.calculateInverseKinematics(robotId, finalJoint - 1, goal[0:3],goal[3:])
        for j in range(len(angles)):
            p.resetJointState(robotId, 3 * j + 2, angles[j])
        #p.resetJointStatesMultiDof(robotId, np.arange(2, finalJoint, 3), angles)
        endEffPos = p.getLinkState(robotId, finalJoint - 1)[0]
        endEffOrient = p.getLinkState(robotId, finalJoint - 1)[1]
        # calculating euclidean distance between end effector and goal
        pos_dist = np.linalg.norm(np.array([a_i - b_i for a_i, b_i in zip(endEffPos, goal[0:3])]))
        #print('end eff orient: ', endEffOrient)
        #print('goal: ', goal)
        #print('goal[0:3], ', goal[0:3])
        #print('goal[3:-1]: ', goal[3:])
        if orientation:
            endEffOrientMatrix = np.reshape(p.getMatrixFromQuaternion(endEffOrient), (3,3))
            goalMatrix = np.reshape(p.getMatrixFromQuaternion(goal[3:]), (3,3))
            # calculating 2-norm between orientations
            orient_dist = np.linalg.norm(np.eye(3) - np.matmul(goalMatrix, np.transpose(endEffOrientMatrix)), ord=2)
        else:
            orient_dist = 0
        #print('transposed endEffOrientMatrix: ', np.transpose(endEffOrientMatrix))
        #print('goalMatrix: ', goalMatrix)
        #print('matrix product: ', np.matmul(goalMatrix, np.transpose(endEffOrientMatrix)))
        if orientation:
            if pos_dist < pos_epsilon and orient_dist < orient_epsilon:
                break
        else:
            if pos_dist < pos_epsilon:
                break
    if orientation:
        endEffPosAndOrient = np.concatenate((endEffPos, endEffOrient), axis=0)
    else:
        endEffPosAndOrient = endEffPos
    # determine dist from tip of EE to goal
    # delete arm from simulation
    p.disconnect()
    #print('dist: ', dist)
    return pos_dist,orient_dist, endEffPosAndOrient