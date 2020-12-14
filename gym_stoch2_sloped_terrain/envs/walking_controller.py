# ### Walking controller
# Written by Shishir Kolathaya shishirk@iisc.ac.in
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for realizing walking controllers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataclasses import dataclass
from collections import namedtuple
from utils.ik_class import Stoch2Kinematics
from utils.ik_class import LaikagoKinematics
from utils.ik_class import HyqKinematics
from utils.ik_class import Stoch3Kinematics
import numpy as np

PI = np.pi
no_of_points = 100

@dataclass
class leg_data:
    name: str
    motor_hip: float = 0.0
    motor_knee: float = 0.0
    motor_abduction: float = 0.0
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    phi: float = 0.0
    b: float = 1.0
    step_length: float = 0.0
    x_shift = 0.0
    y_shift = 0.0
    z_shift = 0.0

@dataclass
class robot_data:
    front_right: leg_data = leg_data('fr')
    front_left: leg_data = leg_data('fl')
    back_right: leg_data = leg_data('br')
    back_left: leg_data = leg_data('bl')

class WalkingController():
    def __init__(self,
                 gait_type='trot',
                 phase=[0, 0, 0, 0],
                 ):
        self._phase = robot_data(front_right=phase[0], front_left=phase[1], back_right=phase[2], back_left=phase[3])
        self.front_left = leg_data('fl')
        self.front_right = leg_data('fr')
        self.back_left = leg_data('bl')
        self.back_right = leg_data('br')
        self.gait_type = gait_type

        self.MOTOROFFSETS_Stoch = [2.3562, 1.2217]                                                        #?????
        self.MOTOROFFSETS_Laikago = [0.87, 0.7]  # [np.pi*0.9, 0]#
        self.MOTOROFFSETS_HYQ = [1.57, 0]
        self.MOTOROFFSETS_Stoch3 = [0,np.pi]

        self.swing_points = np.array([[-0.14, -0.45], [-0.3, -0.26], [0.2, -0.1], [0.2, -0.1], [0.3, -0.26], [0.14, -0.45]])
        self.stance_points = np.array([[0.14, -0.45], [0, -0.45], [-0.14, -0.45]])


        self.leg_name_to_sol_branch_HyQ = {'fl': 0, 'fr': 0, 'bl': 1, 'br': 1}                            #????????
        self.leg_name_to_dir_Laikago = {'fl': 1, 'fr': -1, 'bl': 1, 'br': -1}
        self.leg_name_to_sol_branch_Laikago = {'fl': 0, 'fr': 0, 'bl': 0, 'br': 0}

        self.body_width = 0.24
        self.body_length = 0.37
        self.Stoch2_Kin = Stoch2Kinematics()
        self.Laikago_Kin = LaikagoKinematics()
        self.Stoch3_Kin = Stoch3Kinematics()
        self.Hyq_Kin = HyqKinematics()

    def update_leg_theta(self, theta):
        """ Depending on the gait, the theta for every leg is calculated"""

        def constrain_theta(theta):
            theta = np.fmod(theta, 2 * no_of_points)
            if (theta < 0):
                theta = theta + 2 * no_of_points
            return theta

        self.front_right.theta = constrain_theta(theta + self._phase.front_right)
        self.front_left.theta = constrain_theta(theta + self._phase.front_left)
        self.back_right.theta = constrain_theta(theta + self._phase.back_right)
        self.back_left.theta = constrain_theta(theta + self._phase.back_left)

    def get_swing_stance_weights(self, action):
        # if action[1] < 0.1:
        #     action[1] = 0.1
        # print(action.tolist())
        swing_weights = np.array([action[0], action[1], action[2], action[3], action[4], action[5]])
        stance_weights = np.array([action[5], 1, action[0]])
        return swing_weights, stance_weights

    def initialize_leg_state(self, theta, action):
        '''
        Initialize all the parameters of the leg trajectories
        Args:
            theta  : trajectory cycle parameter theta
            action : trajectory modulation parameters predicted by the policy
        Ret:
            legs   : namedtuple('legs', 'front_right front_left back_right back_left')
        '''
        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right=self.front_right, front_left=self.front_left, back_right=self.back_right,
                    back_left=self.back_left)

        self.update_leg_theta(theta)
        leg_phi = action[12:16]  # fr fl br bl
        self._update_leg_phi_val(leg_phi)
        return legs

    def run_bezier_trajectory(self, theta, action):
        legs = self.initialize_leg_state(theta, action)

        actionf = action[:6]
        actionb = action[6:12]

        swing_weightsf, stance_weightsf = self.get_swing_stance_weights(actionf)
        swing_weightsb, stance_weightsb = self.get_swing_stance_weights(actionb)

        for leg in legs:

            tau = leg.theta / no_of_points

            if leg.name == "fr" or "fl":
                x, y = self.drawfullBezier(self.swing_points, swing_weightsf, self.stance_points, stance_weightsf, tau)
            else:
                x, y = self.drawfullBezier(self.swing_points, swing_weightsb, self.stance_points, stance_weightsb, tau)

            leg.x, leg.y, leg.z = np.array(
                [[np.cos(leg.phi), 0, np.sin(leg.phi)], [0, 1, 0], [-np.sin(leg.phi), 0, np.cos(leg.phi)]]) @ np.array(
                [x, y, 0])

            # print(leg.x, leg.y)
            leg.motor_knee, leg.motor_hip, leg.motor_abduction = self.Stoch3_Kin.inverseKinematics(leg.x, leg.y, leg.z,
                                                                                                   self.leg_name_to_sol_branch_Laikago[
                                                                                                       leg.name])
            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS_Stoch3[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS_Stoch3[1]

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_left.motor_abduction,
                            legs.back_right.motor_hip, legs.back_right.motor_knee, legs.back_right.motor_abduction,
                            legs.front_right.motor_hip, legs.front_right.motor_knee, legs.front_right.motor_abduction,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_left.motor_abduction]
        return leg_motor_angles


    def drawfullBezier(self, swing_points, swing_weights, stance_points, stance_weights, t):

        def drawCurve(points, weights, t):
            if (points.shape[0] == 1):
                return [points[0, 0] / weights[0], points[0, 1] / weights[0]]
            else:
                newpoints = np.zeros([points.shape[0] - 1, points.shape[1]])
                newweights = np.zeros(weights.size)
                for i in np.arange(newpoints.shape[0]):
                    x = (1 - t) * points[i, 0] + t * points[i + 1, 0]
                    y = (1 - t) * points[i, 1] + t * points[i + 1, 1]
                    w = (1 - t) * weights[i] + t * weights[i + 1]
                    newpoints[i, 0] = x
                    newpoints[i, 1] = y
                    newweights[i] = w

                return drawCurve(newpoints, newweights, t)

        swing_newpoints = np.zeros(swing_points.shape)
        stance_newpoints = np.zeros(stance_points.shape)

        for i in np.arange(swing_points.shape[0]):
            swing_newpoints[i] = swing_points[i] * swing_weights[i]

        for i in np.arange(stance_points.shape[0]):
            stance_newpoints[i] = stance_points[i] * stance_weights[i]

        if (t < 1):
            return drawCurve(swing_newpoints, swing_weights, t)
        if (t >= 1):
            # return [stance_points[0,0]+ (t-1)*(stance_points[-1,0] - stance_points[0,0]), -0.21]
            return drawCurve(stance_newpoints, stance_weights, t - 1)

    def _update_leg_phi_val(self, leg_phi):
        '''
        Args:
             leg_phi : steering angles for each leg trajectories
        '''
        self.front_right.phi = leg_phi[0]
        self.front_left.phi = leg_phi[1]
        self.back_right.phi = leg_phi[2]
        self.back_left.phi = leg_phi[3]

    def _update_leg_step_length_val(self, step_length):
        '''
        Args:
            step_length : step length of each leg trajectories
        '''
        self.front_right.step_length = step_length[0]
        self.front_left.step_length = step_length[1]
        self.back_right.step_length = step_length[2]
        self.back_left.step_length = step_length[3]


def constrain_abduction(angle):
    '''
    constrain abduction command with respect to the kinematic limits of the abduction joint
    '''
    if (angle < 0):
        angle = 0
    elif (angle > 0.35):
        angle = 0.35
    return angle


if (__name__ == "__main__"):
    walkcon = WalkingController(phase=[PI, 0, 0, PI])
    walkcon._update_leg_step_length(0.068 * 2, 0.4)
    walkcon._update_leg_phi(0.4)

