'''In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

'''


from angle_interpolation import AngleInterpolationAgent
from keyframes import hello
from keyframes import leftBackToStand
import pickle

class PostureRecognitionAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PostureRecognitionAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.posture = 'unknown'
        self.posture_classifier = pickle.load(open('robot_pose.pkl' , 'rb'))  # LOAD YOUR CLASSIFIER

    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        posture = 'unknown'
        # YOUR CODE HERE
        jointangles = perception.joint #get the joint agles out of perception class object, dictionary
        #print('1',jointangles)
        keys = ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch']
        poses = ['Crouch', 'Belly', 'Knee', 'Frog', 'HeadBack', 'Back', 'Left', 'StandInit', 'Stand', 'Right', 'Sit']
        jointang = [jointangles.get(key) for key in keys] # myDictionary.get(key) for key in keys]
        #print('2',jointang)
        bodyangles = perception.imu  #print(bodyangles) #[-0.0,-0.0]

        all_angles = jointang + bodyangles
        #print('3',all_angles)
        posture_array = self.posture_classifier.predict([all_angles]) #eckige klammern wieder weil sonst nur 1D array brauchen liste von elementen
        posture_int=posture_array[0] #schlecht in int umgewandelt naja wayne
        posture = poses[posture_int] #posture_array hat die Form [8]
        #print('4', posture)
        return posture

if __name__ == '__main__':
    agent = PostureRecognitionAgent()
    agent.keyframes = leftBackToStand()
        #hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
