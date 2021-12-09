'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing
    # the handle offsets relative to the angle and time of the point.
    # The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''

#import keyframes
from pid import PIDAgent
from keyframes import hello
import numpy as np
from scipy import interpolate
from scipy.interpolate import CubicSpline

#musste den ganzen keyframe ordner als source root deklarieren damit leftBackToStand.py importiert
# werden konnte, also wenn man eigene Module importieren möchte

class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        #self.counting = 0
        #self.keyframestarttime = 0
        #self.start_time = self.perception.time
        #self.st = -1
        self.time = 0 #variable for saving start time
        self.set_time = False #variable to check if starttime was set

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        target_joints = {} #soll dict werden für jeden joint ein value (target)
        # YOUR CODE HERE



        names, times, keys = keyframes  # get names of joints, list of times and angles belonging to each time

        if not self.set_time:  # check if animation started
            if len(times) == 0:
                return target_joints
            self.set_time = True
            self.time = perception.time  # initialize start time if not started

        current_time = perception.time - self.time #- 1  # calculate current time from start time

        if (current_time < 0):  # skip the first timesteps to avoid executing angle interpolation in mid air
            return target_joints

        num_joints = len(names)  # calculate number of joints to move
        for joint in range(num_joints):

            joint_times = times[joint]  # get time list belonging to current joint

            len_time_seq = len(joint_times)
            if (current_time <= joint_times[-1]):  # if the animation is running
                joint_angles = []
                for i in keys[joint]:
                    joint_angles.append(i[0])  # get joint angles from keyframes array
                if (len_time_seq <= 3):  # avoid interpolation with splrep with less than 4 values
                    joint_name = names[joint]
                    target_joints[joint_name] = np.interp(current_time, joint_times, joint_angles)  # interpolate with numpy
                else:  # enough values to make cubic spline interpolation
                    spl = interpolate.splrep(joint_times, joint_angles, w=None, xb=0, xe=None, k=3)  # calculate interpolation function for time and angle values
                    # f = CubicSpline(joint_times, joint_angles, bc_type='natural',extrapolate=True)
                    joint_name = names[joint]
                    target_joints[joint_name] = interpolate.splev(current_time, spl)  # save angle calculated from spline and current time in target_joints
                    # target_joints[joint_name] = f(current_time)

        if current_time > joint_times[-1]:
            self.set_time = False
            self.time = perception.time
            current_time = perception.time - self.time
            self.keyframes = ([], [], [])

        if "LHipYawPitch" in target_joints:
            target_joints["RHipYawPitch"] = target_joints["LHipYawPitch"]
        return target_joints


if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run() #Zeitpunkt zu dem der Roboter gespawnt wird, Simulation kann auch so laufen
