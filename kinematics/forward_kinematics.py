'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    1. the local_trans has to consider different joint axes and link parameters for different joints
    2. Please use radians and meters as unit.
'''

# add PYTHONPATH
import os
import sys
import numpy as np
import math
from numpy import sin, cos, pi, matrix, random
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))

from numpy.matlib import matrix, identity

from recognize_posture import PostureRecognitionAgent


class ForwardKinematicsAgent(PostureRecognitionAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {'Head': ['HeadYaw', 'HeadPitch'],
                       'LArm': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll'],
                       'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll'],
                       'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'],
                       # YOUR CODE HERE
                       }

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        '''calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''
        T = identity(4)
        # YOUR CODE HERE
        #joint_angle = np.pi / 2
        dicti_trans ={'HeadYaw': [0,0,126.5], 'HeadPitch': [0,0,0], 'LShoulderPitch': [0,98,100], 'LShoulderRoll':[0,0,0],
                      'LElbowYaw': [105,15,0], 'LElbowRoll': [0,0,0],
                      'LHipYawPitch': [0,50,-85], 'LHipRoll': [0,0,0], 'LHipPitch':[0,0,0], 'LKneePitch':[0,0,-100], 'LAnklePitch':[0,0,-102.9], 'LAnkleRoll':[0,0,0],
                      'RHipYawPitch': [0,-50,-85], 'RHipRoll':[0,0,0], 'RHipPitch':[0,0,0], 'RKneePitch':[0,0,-100], 'RAnklePitch':[0,0,-102.9],
                      'RAnkleRoll':[0,0,0], 'RShoulderPitch':[0,-98,100], 'RShoulderRoll': [0,0,0], 'RElbowYaw':[105,-15,0], 'RElbowRoll':[0,0,0]}
        '''dicti_rot = {'HeadYaw': 3, 'HeadPitch':2, 'LShoulderPitch':2, 'LShoulderRoll':1, # 1 roll, 2 pitch, 3 yaw rotation
                      'LElbowYaw':3, 'LElbowRoll':1, 'LWristYaw':3, 'LHipYawPitch':2,
                      'LHipRoll':1, 'LHipPitch':2, 'LKneePitch':2, 'LAnklePitch':2, 'LAnkleRoll':1, 'RHipYawPitch':2,
                      'RHipRoll':1, 'RHipPitch':2, 'RKneePitch':2, 'RAnklePitch':2,
                      'RAnkleRoll':1, 'RShoulderPitch':2, 'RShoulderRoll':1, 'RElbowYaw':3, 'RElbowRoll':1, 'RWristYaw':3}'''
        c = cos(joint_angle)
        s = sin(joint_angle)

        R_x = matrix([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        R_y = matrix([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]])
        R_z = matrix([[c, s, 0],
                      [-s, c, 0],
                      [0, 0, 1]])
        if joint_name == 'HipYawPitch':  #spezialfall weil joint um spezielle achse dreht
            #hip_offset_z = 85/1000 # unit vector (x,y,z)=(l,m,n), l = 0
            #hip_offset_y = 50/1000
            term = 1-c
            m = math.sqrt(2) #rot achse ist 45° also (0,sqrt(2), sqrt(2)), l = 0
            n = math.sqrt(2)
            T = matrix([[l*l*term+c, -n*s, m*s],
                        [n*s, m*m*term+c, n*m*term],
                        [-m*s, m*n*term, n*n*term+c]])
            row = dicti_trans[joint_name]
            newrow = [x / 1000 for x in row]
            T = np.vstack([T, newrow])
            newcolumn = [[0], [0], [0], [1]]
            T = np.hstack([T, newcolumn])
        if 'Pitch' in joint_name:
            T = R_y
            row = dicti_trans[joint_name]
            newrow = [x / 1000 for x in row]
            T = np.vstack([T, newrow])
            newcolumn = [[0], [0], [0], [1]]
            T = np.hstack([T, newcolumn])
            #print('pitch')

        if 'Roll' in joint_name:
            T = R_x
            row = dicti_trans[joint_name]
            newrow = [x / 1000 for x in row]
            T = np.vstack([T, newrow])
            newcolumn = [[0], [0], [0], [1]]
            T = np.hstack([T, newcolumn])

        if 'Yaw' in joint_name:
            T = R_z
            row = dicti_trans[joint_name]
            newrow = [x / 1000 for x in row]
            T = np.vstack([T, newrow])
            newcolumn = [[0], [0], [0], [1]]
            T = np.hstack([T, newcolumn])
            #print('yaw')
        #print(T)

        return T

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        for chain_joints in self.chains.values():  # gehe die values der chains durch (listen von joints)
            T = identity(4)
            for joint in chain_joints:  #gehe z.b. in head chain alle joints durch,
                angle = joints[joint]
                Tl = self.local_trans(joint, angle)
                # YOUR CODE HERE
                T = T * Tl #mulitpl. alle matrizen für letzten joint am arm z.b. zusammen
                self.transforms[joint] = T #kriegen für jeden joint eine matrix relativ zu torso abgespeichert

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.run()
