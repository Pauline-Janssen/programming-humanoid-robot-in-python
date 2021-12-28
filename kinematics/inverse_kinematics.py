'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity
import numpy as np
from numpy import random, linalg
from scipy.optimize import fmin

class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix, target matrix
        :return: list of joint angles
        '''
        joint_angles = []
        # YOUR CODE HERE

        def error_func(angles, target, effect_name):
            d = {j: 0.0 for j in self.joint_names} #erstelle dict mit joints und alle winkel auf 0.0
            for i, j in enumerate(self.chains[effect_name]): #gehe alle joints in chain d. effector durch
                d[j] = angles[i] #setze dict angles auf angles aus angles

            #forw_kin_m = self.forward_kinematics(d) #aktueller wert durch forward mit akt. theta
            self.forward_kinematics(d)
            forw_kin_m = self.transforms[self.chains[effect_name][-1]]
            e = target - forw_kin_m  #fehler zw. target und aktuellen werten, matrizen
            return linalg.norm(e)

        #func = lambda t: error_func(t, transform, effector_name)
        joint_angles = fmin(        # fmin(function, x0, args...)
            error_func,
            np.random.rand(len(self.chains[effector_name])),
            (transform, effector_name),
            xtol=10**-9, ftol=10**-9)
        return joint_angles

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE (erstelle keyframe um inverse bewegung auszuführen)
        angles = self.inverse_kinematics(effector_name, transform)
        names = []
        times = []
        keys = []
        for j in self.joint_names:
            names.append(j)
            times.append([0.0, 1.0]) #aktuelle zeit, endzeit
            key = [[self.target_joints[j], [3, -0.1, 0.0], [3, 0.1, 0.0]]]

            #keys.append([self.target_joints[j], [3, -0.1, 0.0], [3, 0.1, 0.0]]) #initialer keyframe, start
            if j in self.chains[effector_name]:
                key += [[angles[self.chains[effector_name].index(j)], [3, -0.1, 0.0], [3, 0.1, 0.0]]]
                keys.append(key)
                #keys.append([angles[self.chains[effector_name].index(j)], [3, -0.1, 0.0], [3, 0.1, 0.0]]) #füge winkel aus inverse kinematics hinzu (ziel)
            else:
                key += [[self.target_joints[j], [3, -0.1, 0.0], [3, 0.1, 0.0]]]
                keys.append(key)
                #keys.append([self.target_joints[j], [3, 0.1, 0.0], [3, 0.1, 0.0]])
        self.keyframes = (names, times, keys)


        #self.keyframes = ([], [], [])  # the result joint angles have to fill in

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    #T[-1, 0] =
    T[-1, 1] = 0.05
    T[-1, 2] = -0.2879
        #-0.26

    agent.set_transforms('LLeg', T)
    agent.run()
