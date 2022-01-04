'''In this exercise you need to implement the PID controller for joints of robot.

* Task:
    1. complete the control function in PIDController with prediction
    2. adjust PID parameters for NAO in simulation

* Hints:
    1. the motor in simulation can simple modelled by angle(t) = angle(t-1) + speed * dt
    2. use self.y to buffer model prediction
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'software_installation'))

import numpy as np
from collections import deque
from spark_agent import SparkAgent, JOINT_CMD_NAMES


class PIDController(object):
    '''a discretized PID controller, it controls an array of servos,
       e.g. input is an array and output is also an array
    '''
    def __init__(self, dt, size, sensor_limits):
        '''
        @param dt: step time
        @param size: number of control values
        @param delay: delay in number of steps
        '''
        self.dt = dt
        self.u = np.zeros(size)
        self.e1 = np.zeros(size)
        self.e2 = np.zeros(size)
        self.e0 = np.zeros(size)
        # ADJUST PARAMETERS BELOW
        delay = 0
        self.Kp = 24
        self.Ki = 0
        self.Kd = 0.2
        self.y = deque(np.zeros(size), maxlen=delay + 1)
        self.enabled = True
        self.sensor_limits = [sensor_limits[name] for name in JOINT_CMD_NAMES]
        self.speed_limit = 100.0

    def set_delay(self, delay):
        '''
        @param delay: delay in number of steps
        '''
        self.y = deque(self.y, delay + 1)

    def set_enabled(self, enabled):
        if not self.enabled and enabled:
            self.u *= 0.0
            self.e1 *= 0.0
            self.e2 *= 0.0
        self.enabled = enabled

    def control(self, target, sensor):
        '''apply PID control
        @param target: reference values
        @param sensor: current values from sensor
        @return control signal
        '''
        # YOUR CODE HERE
        #print(target)

        '''if not self.enabled:
            return self.u
            # Clamp the targets to a known safe range.
        for i in range(len(target)):
            target[i] = max(self.sensor_limits[i][0], min(target[i], self.sensor_limits[i][1]))'''

        A0 = self.Kp + (self.Ki * self.dt) + (self.Kd / self.dt)
        A1 = - self.Kp - 2 * (self.Kd / self.dt)
        A2 = self.Kd / self.dt

        #error = np.zeros(3) # error[2] = e(t-2), [1]=e(t-1), [0]=e(t) jeder error muss ein np array sein
        #output = self.u
        #loop wird implizit außerhalb dieser Funktion definiert


        #self.e0 = target - sensor #da ich nicht wusste wie ich e1 aus dem alten_e0 brechnen sollte
        # habe ich das erts so versucht

        error0 = target - sensor

        self.u = self.u + A0 * error0 +A1 * self.e1 +A2 * self.e2

        self.e2 = self.e1
        self.e1 = error0
        #self.e1 = self.e0

        return self.u



class PIDAgent(SparkAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PIDAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.joint_names = JOINT_CMD_NAMES.keys()
        number_of_joints = len(self.joint_names)
        #self.joint_controller = PIDController(dt=0.01, size=number_of_joints)
        self.target_joints = {k: 0 for k in self.joint_names}
        self.sensor_joints = {k: 0 for k in self.joint_names}
        self.sensor_limits = {
            'HeadYaw': (-2.08, 2.08),
            'HeadPitch': (-0.51, 0.67),
            'LShoulderPitch': (-2.08, 2.08),
            'RShoulderPitch': (-2.08, 2.08),
            'LShoulderRoll': (-0.31, 1.31),
            'RShoulderRoll': (-1.31, 0.31),
            'LElbowYaw': (-2.08, 2.08),
            'RElbowYaw': (-2.08, 2.08),
            'LElbowRoll': (-1.54, -0.04),
            'RElbowRoll': (0.04, 1.54),
            'LHipYawPitch': (-1.14, 0.74),
            'RHipYawPitch': (-1.14, 0.74),
            'LHipRoll': (-0.37, 0.78),
            'RHipRoll': (-0.78, 0.37),
            'LHipPitch': (-1.53, 0.48),
            'RHipPitch': (-1.53, 0.48),
            'LKneePitch': (-0.09, 2.12),
            'RKneePitch': (-0.09, 2.12),
            'LAnklePitch': (-1.18, 0.92),
            'RAnklePitch': (-1.18, 0.92),
            'LAnkleRoll': (-0.76, 0.39),
            'RAnkleRoll': (-0.39, 0.76),
        }
        self.joint_controller = PIDController(0.01, number_of_joints, self.sensor_limits)

    def think(self, perception):
        action = super(PIDAgent, self).think(perception)
        '''calculate control vector (speeds) from
        perception.joint:   current joints' positions (dict: joint_id -> position (current))
        self.target_joints: target positions (dict: joint_id -> position (target)) '''
        joint_angles = np.asarray(
            [perception.joint[joint_id]  for joint_id in JOINT_CMD_NAMES])
        target_angles = np.asarray([self.target_joints.get(joint_id, 
            perception.joint[joint_id]) for joint_id in JOINT_CMD_NAMES])
        u = self.joint_controller.control(target_angles, joint_angles)
        action.speed = dict(zip(JOINT_CMD_NAMES.keys(), u))  # dict: joint_id -> speed
        self.sensor_joints = perception.joint
        return action


if __name__ == '__main__':
    agent = PIDAgent()
    agent.target_joints['HeadYaw'] = 1.0
    agent.run()
