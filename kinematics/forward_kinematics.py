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
                       'RArm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll'],
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
                T = Tl @ T #mulitpl. alle matrizen für letzten joint am arm z.b. zusammen
                self.transforms[joint] = T #kriegen für jeden joint eine matrix relativ zu torso abgespeichert


# Visualization to verify each joints rotation direction is correct
# Shown are:
# - All chains in random (but deterministic) colors
# - All local transformation directions (x red, y green, z blue) of each joint
# - All local rotation axes in black of each joint
def draw(i, ax, agent, transforms=None, anim_angles=False):
    global anim_data

    import itertools

    if i == 0:
        ax.clear()
        ax.set(xlim=(-0.225, 0.225), ylim=(-0.225, 0.225), zlim=(-0.30, 0.15))
        anim_data = {}

    if anim_angles:
        # Choose which joints to animate and how fast/far.
        d = {
            'HeadYaw': 0.0,
            'HeadPitch': 0.0,
            'LShoulderPitch': 0.0,
            'LShoulderRoll': 0.0,
            'LElbowYaw': 0.0,
            'LElbowRoll': 0.0,
            'LWristYaw': 0.0,
            'LHipYawPitch': 0.0,
            'LHipRoll': 0.0,
            'LHipPitch': 0.5,
            'LKneePitch': 0.0,
            'LAnklePitch': 0.0,
            'LAnkleRoll': 0.0,
        }
        for k in list(d.keys()):
            if k[0] == 'L':
                d[f"R{k[1:]}"] = d[k]
        # Linearly go from -pi/2 (-90°) to pi/2 (90°) in 32 steps.
        for k in d.keys():
            d[k] *= -np.pi / 2 + (i % 33) * np.pi / 32

        transforms = agent.forward_kinematics(d)
    if transforms is None or len(transforms) == 0:
        transforms = agent.transforms

    new_ts = {}
    for k, v in transforms.items():
        new_ts[k] = np.array(v, np.float32).T
    transforms = new_ts

    p_data = {}
    t_ax_len = 0.05
    j_line_width = 5
    t_line_width = 2
    for chain_joints in agent.chains.values():
        old_P = np.array([0, 0, 0], np.float32)
        for joint in chain_joints:
            P = transforms[joint] @ np.array([[0], [0], [0], [1]], np.float32)
            Px = t_ax_len * (transforms[joint] @ np.array([[1], [0], [0], [0]], np.float32))
            Py = t_ax_len * (transforms[joint] @ np.array([[0], [1], [0], [0]], np.float32))
            Pz = t_ax_len * (transforms[joint] @ np.array([[0], [0], [1], [0]], np.float32))

            if i == 0:
                p_data[joint] = [zip(old_P, P[:3, 0])] + \
                                [zip(P[:3, 0], (P + p)[:3, 0]) for p in [Px, Py, Pz]]
            else:
                anim_data[joint][0].set_data_3d(*zip(old_P, P[:3, 0]))
                for i, p in enumerate((Px, Py, Pz), 1):
                    anim_data[joint][i].set_data_3d(*zip(P[:3, 0], (P + p)[:3, 0]))
            old_P = P[:3, 0]
    if i != 0:
        return
    # Draw all links first
    line_widths = [j_line_width] + [t_line_width] * 3
    for joint in itertools.chain.from_iterable(agent.chains.values()):
        anim_data[joint] = [ax.plot(*p_data[joint][0], linewidth=j_line_width)[0]]
    # And only then the transformations, so they are on top.
    for joint in itertools.chain.from_iterable(agent.chains.values()):
        for i, color in enumerate(('r-', 'g-', 'b-'), 1):
            anim_data[joint] += [ax.plot(*p_data[joint][i], color, linewidth=line_widths[i])[0]]

def play_animation():
    global ax, agent

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ani = FuncAnimation(fig, draw, frames=128, interval=50, repeat=True,
                        fargs=(ax, agent, {}, True))
    plt.show(block=True)

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    play_animation()
    agent.run()
