import numpy as np
import gym
from gym import spaces
import gym_stoch2_sloped_terrain.envs.walking_controller as walking_controller
import math
import random
from collections import deque
import pybullet
import gym_stoch2_sloped_terrain.envs.bullet_client as bullet_client
import pybullet_data
import gym_stoch2_sloped_terrain.envs.planeEstimation.get_terrain_normal as normal_estimator

LEG_POSITION = ["fl_", "bl_", "fr_", "br_"]
KNEE_CONSTRAINT_POINT_RIGHT = [0.014, 0, 0.076]  # hip
KNEE_CONSTRAINT_POINT_LEFT = [0.0, 0.0, -0.077]  # knee
RENDER_HEIGHT = 720                              #??
RENDER_WIDTH = 960                                #??
PI = np.pi
no_of_points = 100

def constrain_theta(theta):
    theta = np.fmod(theta, 2 * no_of_points)                                           #its use
    if (theta < 0):
        theta = theta + 2 * no_of_points
    return theta


class Stoch3Env(gym.Env):

    def __init__(self,
				 render=False,
				 on_rack=False,
				 gait='trot',
				 phase=[0, no_of_points, no_of_points, 0],  # [FR, FL, BR, BL]          #its use
				 action_dim=16,
				 end_steps=1000,
				 stairs=False,
				 downhill=False,
				 seed_value=100,                                                        #its use
				 wedge=False,
				 IMU_Noise=False,
				 deg=5):

        self._is_stairs = stairs
        self._is_wedge = wedge
        self._is_render = render
        self._on_rack = on_rack
        self.rh_along_normal = 0.24

        self.seed_value = seed_value
        random.seed(self.seed_value)

        if self._is_render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient()

        self._theta = 0

        self._frequency = 4
        self.termination_steps = end_steps
        self.downhill = downhill

        # PD gains
        self._kp = 400
        self._kd = 28.9

        self.dt = 0.003
        self._frame_skip = 40
        self._n_steps = 0
        self._action_dim = action_dim

        self._obs_dim = 9


        self.action = np.zeros(self._action_dim)

        self._last_base_position = [0, 0, 0]
        self.last_yaw = 0
        self._distance_limit = float("inf")

        self.current_com_height = 0.7

        # wedge_parameters
        self.wedge_start = 0.5
        self.wedge_halflength = 2

        if gait is 'trot':
            phase = [0, no_of_points, no_of_points, 0]
        elif gait is 'walk':
            phase = [0, no_of_points, 3* no_of_points / 2, no_of_points / 2]
        self._walkcon = walking_controller.WalkingController(gait_type=gait, phase=phase)
        self.inverse = False
        self._cam_dist = 1.0
        self._cam_yaw = 0.0
        self._cam_pitch = 0.0

        self.avg_vel_per_step = 0
        self.avg_omega_per_step = 0

        self.linearV = 0
        self.angV = 0
        self.prev_vel = [0, 0, 0]
        self.prev_feet_points = np.ndarray((5,3))
        self.x_f = 0
        self.y_f = 0

        self.clips = 100

        self.friction = 0.8
        self.ori_history_length = 3
        self.ori_history_queue = deque([0] * 3 * self.ori_history_length,
                                       maxlen=3 * self.ori_history_length)  # observation queue

        self.step_disp = deque([0] * 100, maxlen=100)
        self.stride = 5

        self.incline_deg = deg
        self.incline_ori = 0

        self.prev_incline_vec = (0, 0, 1)

        self.terrain_pitch = []
        self.add_IMU_noise = IMU_Noise

        self.INIT_POSITION = [0, 0, 0.6]
        self.INIT_ORIENTATION = [0, 0, 0, 1]

        self.support_plane_estimated_pitch = 0
        self.support_plane_estimated_roll = 0

        self.pertub_steps = 0
        self.x_f = 0
        self.y_f = 0

        ## Gym env related mandatory variables
        #self._obs_dim = 3 * self.ori_history_length   # [r,p,y]x previous time steps, suport plane roll and pitch
        self._obs_dim = 15 #3 * self.ori_history_length   # [r,p,y]x previous time steps, suport plane roll and pitch

        observation_high = np.array([np.pi / 2] * self._obs_dim)
        observation_low = -observation_high
        self.observation_space = spaces.Box(observation_low, observation_high)

        action_high = np.array([1] * self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)

        self.hard_reset()

        self.randomize_only_inclines(default=True)

        if (self._is_stairs):
            boxHalfLength = 0.1
            boxHalfWidth = 1
            boxHalfHeight = 0.015
            sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
                                                                   halfExtents=[boxHalfLength, boxHalfWidth,
                                                                                boxHalfHeight])
            boxOrigin = 0.3
            n_steps = 15
            self.stairs = []
            for i in range(n_steps):
                step = self._pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                             basePosition=[boxOrigin + i * 2 * boxHalfLength, 0,
                                                                           boxHalfHeight + i * 2 * boxHalfHeight],
                                                             baseOrientation=[0.0, 0.0, 0.0, 1])
                self.stairs.append(step)
                self._pybullet_client.changeDynamics(step, -1, lateralFriction=0.8)

    def hard_reset(self):
        '''
		Function to
		1) Set simulation parameters which remains constant throughout the experiments
		2) load urdf of plane, wedge and robot in initial conditions
		'''
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(300))
        self._pybullet_client.setTimeStep(self.dt / self._frame_skip)

        self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
        self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
        self._pybullet_client.setGravity(0, 0, -9.8)

        if self._is_wedge:

            wedge_halfheight_offset = 0.01

            self.wedge_halfheight = wedge_halfheight_offset + 1.5 * math.tan(math.radians(self.incline_deg)) / 2.0
            self.wedgePos = [0, 0, self.wedge_halfheight]
            self.wedgeOrientation = self._pybullet_client.getQuaternionFromEuler([0, 0, self.incline_ori])

            if not (self.downhill):
                wedge_model_path = "gym_stoch2_sloped_terrain/envs/Wedges/uphill/urdf/wedge_" + str(
                    self.incline_deg) + ".urdf"

                self.INIT_ORIENTATION = self._pybullet_client.getQuaternionFromEuler(
                    [math.radians(self.incline_deg) * math.sin(self.incline_ori),
                     -math.radians(self.incline_deg) * math.cos(self.incline_ori), 0])

                self.robot_landing_height = wedge_halfheight_offset + 0.65 + math.tan(
                    math.radians(self.incline_deg)) * abs(self.wedge_start)

                self.INIT_POSITION = [self.INIT_POSITION[0], self.INIT_POSITION[1], self.robot_landing_height]

            else:
                wedge_model_path = "gym_stoch2_sloped_terrain/envs/Wedges/downhill/urdf/wedge_" + str(
                    self.incline_deg) + ".urdf"

                self.robot_landing_height = wedge_halfheight_offset + 0.5 + math.tan(
                    math.radians(self.incline_deg)) * 1.5

                self.INIT_POSITION = [0, 0, self.robot_landing_height]  # [0.5, 0.7, 0.3] #[-0.5,-0.5,0.3]

                self.INIT_ORIENTATION = [0, 0, 0, 1]

            self.wedge = self._pybullet_client.loadURDF(wedge_model_path, self.wedgePos, self.wedgeOrientation)

            self.SetWedgeFriction(0.7)

        model_path = 'gym_stoch2_sloped_terrain/envs/robots/stoch3/urdf/stoch3.urdf'
        self.Stoch3 = self._pybullet_client.loadURDF(model_path, self.INIT_POSITION, self.INIT_ORIENTATION)

        self._joint_name_to_id, self._motor_id_list = self.BuildMotorIdList()

        self.ResetLeg()
        self.ResetPoseForAbd()
        #self.ActuateSpring()

        if self._on_rack:
            self._pybullet_client.createConstraint(
                self.Stoch3, -1, -1, -1, self._pybullet_client.JOINT_FIXED,
                [0, 0, 0], [0, 0, 0], [0, 0, self.INIT_POSITION[2]])

        self._pybullet_client.resetBasePositionAndOrientation(self.Stoch3, self.INIT_POSITION, self.INIT_ORIENTATION)
        self._pybullet_client.resetBaseVelocity(self.Stoch3, [0, 0, 0], [0, 0, 0])

        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        self.SetFootFriction(self.friction)

    def reset_standing_position(self):

        self.ResetLeg(standing_torque=200)
        self.ResetPoseForAbd()

        # Conditions for standstill
        for i in range(2000):
            self._pybullet_client.stepSimulation()

        self.ResetLeg(standing_torque=0)

    def reset(self):
        '''
		This function resets the environment
		Note : Set_Randomization() is called before reset() to either randomize or set environment in default conditions.
		'''
        self._theta = 0
        self._last_base_position = [0, 0, 0]
        self.last_yaw = 0
        self.inverse = False

        if self._is_wedge:
            self._pybullet_client.removeBody(self.wedge)

            wedge_halfheight_offset = 0.01

            self.wedge_halfheight = wedge_halfheight_offset + 1.5 * math.tan(math.radians(self.incline_deg)) / 2.0
            self.wedgePos = [0, 0, self.wedge_halfheight]
            self.wedgeOrientation = self._pybullet_client.getQuaternionFromEuler([0, 0, self.incline_ori])

            if not (self.downhill):
                wedge_model_path = "gym_stoch2_sloped_terrain/envs/Wedges/uphill/urdf/wedge_" + str(
                    self.incline_deg) + ".urdf"

                self.INIT_ORIENTATION = self._pybullet_client.getQuaternionFromEuler(
                    [math.radians(self.incline_deg) * math.sin(self.incline_ori),
                     -math.radians(self.incline_deg) * math.cos(self.incline_ori), 0])

                self.robot_landing_height = wedge_halfheight_offset + 0.65 + math.tan(
                    math.radians(self.incline_deg)) * abs(self.wedge_start)

                self.INIT_POSITION = [self.INIT_POSITION[0], self.INIT_POSITION[1], self.robot_landing_height]

            else:
                wedge_model_path = "gym_stoch2_sloped_terrain/envs/Wedges/downhill/urdf/wedge_" + str(
                    self.incline_deg) + ".urdf"

                self.robot_landing_height = wedge_halfheight_offset + 0.65 + math.tan(
                    math.radians(self.incline_deg)) * 1.5

                self.INIT_POSITION = [0.3, 0, self.robot_landing_height]

                self.INIT_ORIENTATION = [0, 0, 0, 1]

            self.wedge = self._pybullet_client.loadURDF(wedge_model_path, self.wedgePos, self.wedgeOrientation)
            self.SetWedgeFriction(0.7)

        self._pybullet_client.resetBasePositionAndOrientation(self.Stoch3, self.INIT_POSITION, self.INIT_ORIENTATION)
        self._pybullet_client.resetBaseVelocity(self.Stoch3, [0, 0, 0], [0, 0, 0])
        self.reset_standing_position()


        LINK_ID = [0,3,7,11,15]
        i=0
        for  link_id in LINK_ID:
            if(link_id!=0):
                self.prev_feet_points[i] = np.array(self._pybullet_client.getLinkState(self.Stoch3,link_id)[0])
            else:
                self.prev_feet_points[i] = np.array(self.GetBasePosAndOrientation()[0])
            i+=1


        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        self._n_steps = 0
        return self.GetObservation()

    def apply_Ext_Force(self, x_f, y_f, link_index=1, visulaize=False, life_time=0.01):
        '''
		function to apply external force on the robot
		Args:
			x_f  :  external force in x direction
			y_f  : 	external force in y direction
			link_index : link index of the robot where the force need to be applied
			visulaize  :  bool, whether to visulaize external force by arrow symbols
			life_time  :  life time of the visualization
 		'''
        force_applied = [x_f, y_f, 0]
        self._pybullet_client.applyExternalForce(self.Stoch3, link_index, forceObj=[x_f, y_f, 0], posObj=[0, 0, 0],
                                                 flags=self._pybullet_client.LINK_FRAME)
        f_mag = np.linalg.norm(np.array(force_applied))

        if (visulaize and f_mag != 0.0):
            point_of_force = self._pybullet_client.getLinkState(self.Stoch3, link_index)[0]

            lam = 1 / (2 * f_mag)
            dummy_pt = [point_of_force[0] - lam * force_applied[0],
                        point_of_force[1] - lam * force_applied[1],
                        point_of_force[2] - lam * force_applied[2]]
            self._pybullet_client.addUserDebugText(str(round(f_mag, 2)) + " N", dummy_pt, [0.13, 0.54, 0.13],
                                                   textSize=2, lifeTime=life_time)
            self._pybullet_client.addUserDebugLine(point_of_force, dummy_pt, [0, 0, 1], 3, lifeTime=life_time)

    def SetLinkMass(self, link_idx, mass=0):
        '''
		Function to add extra mass to front and back link of the robot

		Args:
			link_idx : link index of the robot whose weight to need be modified
			mass     : value of extra mass to be added

		Ret:
			new_mass : mass of the link after addition
		Note : Presently, this function supports addition of masses in the front and back link only (0, 11)
		'''
        link_mass = self._pybullet_client.getDynamicsInfo(self.Stoch3, link_idx)[0]
        if (link_idx == 0):
            link_mass = mass + 1.1
            self._pybullet_client.changeDynamics(self.Stoch3, 0, mass=link_mass)
        elif (link_idx == 11):
            link_mass = mass + 1.1
            self._pybullet_client.changeDynamics(self.Stoch3, 11, mass=link_mass)

        return link_mass

    def getlinkmass(self, link_idx):
        '''
		function to retrieve mass of any link
		Args:
			link_idx : link index of the robot
		Ret:
			m[0] : mass of the link
		'''
        m = self._pybullet_client.getDynamicsInfo(self.Stoch3, link_idx)
        return m[0]

    def Set_Randomization(self, default=False, idx1=0, idx2=0, idx3=1, idx0=0, idx11=0, idxc=2, idxp=0, deg=5, ori=0):
        '''
		This function helps in randomizing the physical and dynamics parameters of the environment to robustify the policy.
		These parameters include wedge incline, wedge orientation, friction, mass of links, motor strength and external perturbation force.
		Note : If default argument is True, this function set above mentioned parameters in user defined manner
		'''
        if default:
            frc = [0.55, 0.6, 0.8]
            extra_link_mass = [0, 0.05, 0.1, 0.15]
            cli = [5.2, 6, 7, 8]
            pertub_range = [0, -60, 60, -100, 100]
            self.pertub_steps = 150
            self.x_f = 0
            self.y_f = pertub_range[idxp]
            self.incline_deg = deg + 2 * idx1
            self.incline_ori = ori + PI / 6 * idx2
            self.new_fric_val = frc[idx3]
            self.friction = self.SetFootFriction(self.new_fric_val)
            self.FrontMass = self.SetLinkMass(0, extra_link_mass[idx0])
            self.BackMass = self.SetLinkMass(11, extra_link_mass[idx11])
            self.clips = cli[idxc]

        else:
            avail_deg = [5, 7, 9, 11]
            extra_link_mass = [0, .05, 0.1, 0.15]
            pertub_range = [0, -60, 60, -100, 100]
            cli = [5, 6, 7, 8]
            self.pertub_steps = 150  # random.randint(90,200) #Keeping fixed for now
            self.x_f = 0
            self.y_f = pertub_range[random.randint(0, 4)]
            self.incline_deg = avail_deg[random.randint(0, 3)]
            self.incline_ori = (PI / 12) * random.randint(0, 6)  # resolution of 15 degree
            self.new_fric_val = np.round(np.clip(np.random.normal(0.6, 0.08), 0.55, 0.8), 2)
            self.friction = self.SetFootFriction(self.new_fric_val)
            i = random.randint(0, 3)
            self.FrontMass = self.SetLinkMass(0, extra_link_mass[i])
            i = random.randint(0, 3)
            self.BackMass = self.SetLinkMass(11, extra_link_mass[i])
            self.clips = np.round(np.clip(np.random.normal(6.5, 0.4), 5, 8), 2)

    def randomize_only_inclines(self, default=False, idx1=0, idx2=0, deg=7, ori=0):
        '''
        This function only randomizes the wedge incline and orientation and is called during training without Domain Randomization
        '''
        if default:
            self.incline_deg = deg + 2 * idx1
            self.incline_ori = ori + PI / 12 * idx2

        else:
            avail_deg = [7, 9, 11, 13, 15]
            self.incline_deg = avail_deg[random.randint(0, 4)]
            self.incline_ori = (PI / 12) * random.randint(0, 3)  # resolution of 15 degree

    def boundYshift(self, x, y):
        '''
		This function bounds Y shift with respect to current X shift
		Args:
			 x : absolute X-shift
			 y : Y-Shift
		Ret :
			 y : bounded Y-shift
		'''
        if x > 0.5619:
            if y > 1 / (0.5619 - 1) * (x - 1):
                y = 1 / (0.5619 - 1) * (x - 1)
        return y

    def getYXshift(self, yx):
        '''
		This function bounds X and Y shifts in a trapezoidal workspace
		'''
        y = yx[:4]
        x = yx[4:]
        for i in range(0, 4):
            y[i] = self.boundYshift(abs(x[i]), y[i])
            y[i] = y[i] * 0.038
            x[i] = x[i] * 0.0418
        yx = np.concatenate([y, x])
        return yx

    def transform_action(self, action):
        '''
		Transform normalized actions to scaled offsets
		Args:
			action : 20 dimensional 1D array of predicted action values from policy in following order :
					 [(step lengths of FR, FL, BR, BL), (steer angles of FR, FL, BR, BL),
					  (Y-shifts of FR, FL, BR, BL), (X-shifts of FR, FL, BR, BL),
					  (Z-shifts of FR, FL, BR, BL)]
		Ret :
			action : scaled action parameters

		Note : The convention of Cartesian axes for leg frame in the codebase follow this order, Y points up, X forward and Z right.
		       While in research paper we follow this order, Z points up, X forward and Y right.
		'''

        action = np.clip(action, -1, 1)

        action[:12] =(action[:12] +1)/2
        for i in range(12):  # Zero all the negative power terms
            if action[i] == 0.0:
                action[i] = 0.01
        action[12:16] = np.pi/3 * action[12:16]
        return action


    def get_foot_contacts(self):
        '''
		Retrieve foot contact information with the supporting ground and any special structure (wedge/stairs).
		Ret:
			foot_contact_info : 8 dimensional binary array, first four values denote contact information of feet [FR, FL, BR, BL] with the ground
			while next four with the special structure.
		'''
        foot_ids = [3, 7, 11, 15]
        foot_contact_info = np.zeros(8)

        for leg in range(4):
            contact_points_with_ground = self._pybullet_client.getContactPoints(self.plane, self.Stoch3, -1,
                                                                                foot_ids[leg])
            if len(contact_points_with_ground) > 0:
                foot_contact_info[leg] = 1

            if self._is_wedge:
                contact_points_with_wedge = self._pybullet_client.getContactPoints(self.wedge, self.Stoch3, -1,
                                                                                   foot_ids[leg])
                if len(contact_points_with_wedge) > 0:
                    foot_contact_info[leg + 4] = 1

            if self._is_stairs:
                for steps in self.stairs:
                    contact_points_with_stairs = self._pybullet_client.getContactPoints(steps, self.Stoch3, -1,
                                                                                        foot_ids[leg])
                    if len(contact_points_with_stairs) > 0:
                        foot_contact_info[leg + 4] = 1

        return foot_contact_info

    def step(self, action):
        '''
		function to perform one step in the environment
		Args:
			action : array of action values
		Ret:
			ob 	   : observation after taking step
			reward     : reward received after taking step
			done       : whether the step terminates the env
			{}	   : any information of the env (will be added later)
		'''
        action = self.transform_action(action)


        energy_spent = self.do_simulation(action, n_frames=self._frame_skip)

        ob = self.GetObservation()
        reward, done = self._get_reward(energy_spent)
        return ob, reward, done, {}

    def CurrentVelocities(self):
        '''
		Returns robot's linear and angular velocities
		Ret:
			radial_v  : linear velocity
			current_w : angular velocity
		'''
        current_w = self.GetBaseAngularVelocity()[2]
        current_v = self.GetBaseLinearVelocity()
        radial_v = math.sqrt(current_v[0] ** 2 + current_v[1] ** 2)
        return radial_v, current_w

    def do_simulation(self, action, n_frames):
        '''
		Converts action parameters to corresponding motor commands with the help of a elliptical trajectory controller
		'''

        omega = 2 * no_of_points * self._frequency
        self.action = action
        energy_spent_per_step = 0
        current_theta = self._theta

        while (np.abs(self._theta - current_theta) <= no_of_points * 0.5):

            leg_m_angle_cmd = self._walkcon.run_bezier_trajectory(self._theta, action)
            #print(leg_m_angle_cmd)
            #if (self._theta % 5 ==0):
            # self.vis_foot_traj()
            self._theta = constrain_theta(omega * self.dt + self._theta)

            m_angle_cmd_ext = np.array(leg_m_angle_cmd)

            m_vel_cmd_ext = np.zeros(12)

            force_visualizing_counter = 0

            for _ in range(n_frames):
               # self.ActuateSpring()
                applied_motor_torque = self._apply_pd_control(m_angle_cmd_ext, m_vel_cmd_ext)
                self._pybullet_client.stepSimulation()


                joint_power = np.multiply(applied_motor_torque,self.GetMotorVelocities())  # Power output of individual actuators

                for i in range (12):   # Zero all the negative power terms
                   if joint_power[i] < 0.0 :
                       joint_power[i]=0

                energy_spent = np.sum(joint_power) * self.dt / n_frames
                energy_spent_per_step += energy_spent

        self._n_steps += 1
        return energy_spent_per_step



    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, _ = self.GetBasePosAndOrientation()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px).reshape(RENDER_WIDTH, RENDER_HEIGHT, 4)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self, pos, orientation):
        '''
		Check termination conditions of the environment
		Args:
			pos 		: current position of the robot's base in world frame
			orientation : current orientation of robot's base (Quaternions) in world frame
		Ret:
			done 		: return True if termination conditions satisfied
		'''
        done = False
        RPY = self._pybullet_client.getEulerFromQuaternion(orientation)

        if self._n_steps >= self.termination_steps:
            done = True
        else:
            if abs(RPY[0]) > math.radians(30):
                print('Oops, Robot about to fall sideways! Terminated')
                done = True

            if abs(RPY[1]) > math.radians(35):
                print('Oops, Robot doing wheely! Terminated')
                done = True

            if pos[2] > 0.9:
                print('Robot was too high! Terminated')
                done = True

        return done

    def _get_reward(self, energy_spent_per_step):
        '''
        Calculates reward achieved by the robot for RPY stability, torso height criterion and forward distance moved on the slope:
        Ret:
            reward : reward achieved
            done   : return True if environment terminates

        '''

        pos, ori = self.GetBasePosAndOrientation()

        RPY_orig = self._pybullet_client.getEulerFromQuaternion(ori)
        RPY = np.round(RPY_orig, 4)

        roll_reward = np.exp(-45 * ((RPY[0] - 0) ** 2))
        pitch_reward = np.exp(-45 * ((RPY[1] - 0) ** 2))
        yaw_reward = np.exp(-40 * (RPY[2] ** 2))

        x = pos[0]
        y = pos[1]
        x_l = self._last_base_position[0]
        y_l = self._last_base_position[1]
        self._last_base_position = pos

        step_distance_x = (x - x_l)
        step_distance_y = abs(y - y_l)

        done = self._termination(pos, ori)
        if done:
            reward = 0
        else:
            reward = round(yaw_reward, 4) + round(pitch_reward, 4) + round(roll_reward, 4) +  10*round(step_distance_x, 4) - energy_spent_per_step/15
        #print(10* round(step_distance_x, 4), energy_spent_per_step/15)

        return reward, done

    def vis_foot_traj(self,line_thickness = 5,life_time = 15):
        LINK_ID = [0,3,7,11,15]
        i=0
        for  link_id in LINK_ID:
            if(link_id!=0):
                current_point = self._pybullet_client.getLinkState(self.Stoch3,link_id)[0]
                self._pybullet_client.addUserDebugLine(current_point,self.prev_feet_points[i],[1,0,0],line_thickness,lifeTime=life_time)
            else:
                current_point = self.GetBasePosAndOrientation()[0]
                #self._pybullet_client.addUserDebugLine(current_point,self.prev_feet_points[i],[0,0,1],line_thickness,lifeTime=100)
            self.prev_feet_points[i] = current_point
            i+=1


    def _apply_pd_control(self, motor_commands, motor_vel_commands):
        '''
        Apply PD control to reach desired motor position commands
        Ret:
			applied_motor_torque : array of applied motor torque values in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]
		'''

        qpos_act = self.GetMotorAngles()
        qvel_act = self.GetMotorVelocities()
        applied_motor_torque = self._kp * (motor_commands - qpos_act[: 12]) + self._kd * (motor_vel_commands - qvel_act[: 12])
        motor_strength = 150
        applied_motor_torque = np.clip(np.array(applied_motor_torque), -motor_strength, motor_strength)
        applied_motor_torque = applied_motor_torque.tolist()
        #print(applied_motor_torque)
        for motor_id, motor_torque in zip(self._motor_id_list[: 12], applied_motor_torque):
            self.SetMotorTorqueById(motor_id, motor_torque)
        return applied_motor_torque


    def add_noise(self, sensor_value, SD=0.04):
        '''
		Adds sensor noise of user defined standard deviation in current sensor_value
		'''
        noise = np.random.normal(0, SD, 1)
        sensor_value = sensor_value + noise[0]
        return sensor_value

    def GetObservation(self):
        '''
		This function returns the current observation of the environment for the interested task
		Ret:
			obs : [R(t-2), P(t-2), Y(t-2), R(t-1), P(t-1), Y(t-1), R(t), P(t), Y(t), estimated support plane (roll, pitch) ]
		'''
        pos, ori = self.GetBasePosAndOrientation()
        motor_angles = self.GetMotorAngles()
        RPY = self._pybullet_client.getEulerFromQuaternion(ori)
        RPY = np.round(RPY, 5)

        for val in RPY:
            if (self.add_IMU_noise):
                val = self.add_noise(val)
            self.ori_history_queue.append(val)

        #obs = self.ori_history_queue
        obs = np.concatenate((motor_angles, RPY)).ravel()

        return obs

    def GetMotorAngles(self):
        '''
		This function returns the current joint angles in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA ]
		'''
        motor_ang = [self._pybullet_client.getJointState(self.Stoch3, motor_id)[0] for motor_id in self._motor_id_list]
        return motor_ang[:12]

    def GetMotorVelocities(self):
        '''
		This function returns the current joint velocities in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA ]
		'''
        motor_vel = [self._pybullet_client.getJointState(self.Stoch3, motor_id)[1] for motor_id in self._motor_id_list]
        return motor_vel[:12]

    def GetBasePosAndOrientation(self):
        '''
		This function returns the robot torso position(X,Y,Z) and orientation(Quaternions) in world frame
		'''
        position, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.Stoch3))
        return position, orientation

    def GetBaseAngularVelocity(self):
        '''
		This function returns the robot base angular velocity in world frame
		Ret: list of 3 floats
		'''
        basevelocity = self._pybullet_client.getBaseVelocity(self.Stoch3)
        return basevelocity[1]

    def GetBaseLinearVelocity(self):
        '''
		This function returns the robot base linear velocity in world frame
		Ret: list of 3 floats
		'''
        basevelocity = self._pybullet_client.getBaseVelocity(self.Stoch3)
        return basevelocity[0]

    def SetFootFriction(self, foot_friction):
        '''
		This function modify coefficient of friction of the robot feet
		Args :
		foot_friction :  desired friction coefficient of feet
		Ret  :
		foot_friction :  current coefficient of friction
		'''
        FOOT_LINK_ID = [3, 7, 11, 15]
        for link_id in FOOT_LINK_ID:
            self._pybullet_client.changeDynamics(
                self.Stoch3, link_id, lateralFriction=foot_friction)
        return foot_friction

    def SetWedgeFriction(self, friction):
        '''
		This function modify friction coefficient of the wedge
		Args :
		foot_friction :  desired friction coefficient of the wedge
		'''
        self._pybullet_client.changeDynamics(
            self.wedge, -1, lateralFriction=friction)

    def SetMotorTorqueById(self, motor_id, torque):
        '''
		function to set motor torque for respective motor_id
		'''
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.Stoch3,
            jointIndex=motor_id,
            controlMode=self._pybullet_client.TORQUE_CONTROL,
            force=torque)

    def BuildMotorIdList(self):
        '''
		function to map joint_names with respective motor_ids as well as create a list of motor_ids
		Ret:
		joint_name_to_id : Dictionary of joint_name to motor_id
		motor_id_list	 : List of joint_ids for respective motors in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA ]
		'''
        num_joints = self._pybullet_client.getNumJoints(self.Stoch3)
        joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.Stoch3, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

            # adding abduction
            MOTOR_NAMES = ["fl_hip_joint",
                           "fl_knee_joint",
                           "fl_abd_joint",

                           "br_hip_joint",
                           "br_knee_joint",
                           "br_abd_joint",

                           "fr_hip_joint",
                           "fr_knee_joint",
                           "fr_abd_joint",

                           "bl_hip_joint",
                           "bl_knee_joint",
                           "bl_abd_joint"

                           ]

        motor_id_list = [joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]

        return joint_name_to_id, motor_id_list

    def ResetLeg(self, standing_torque=0):
        '''
		function to reset hip and knee joints' state
		Args:
			 leg_id 		  : denotes leg index
			 add_constraint   : bool to create constraints in lower joints of five bar leg mechanisim
			 standstilltorque : value of initial torque to set in hip and knee motors for standing condition
		'''
        self._pybullet_client.resetJointState(
            self.Stoch3,
            self._joint_name_to_id["fl_hip_joint"],  # motor
            targetValue=np.pi/3, targetVelocity=0)

        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.Stoch3,
            jointIndex=(self._joint_name_to_id["fl_hip_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=standing_torque,
            targetVelocity=0
        )

        self._pybullet_client.resetJointState(
            self.Stoch3,
            self._joint_name_to_id["fl_knee_joint"],
            targetValue=2*np.pi/3, targetVelocity=0)

        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.Stoch3,
            jointIndex=(self._joint_name_to_id["fl_knee_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=standing_torque,
            targetVelocity=0
        )

        self._pybullet_client.resetJointState(
            self.Stoch3,
            self._joint_name_to_id["br_hip_joint"],  # motor
            targetValue=np.pi/3, targetVelocity=0)

        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.Stoch3,
            jointIndex=(self._joint_name_to_id["br_hip_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=standing_torque,
            targetVelocity=0
        )

        self._pybullet_client.resetJointState(
            self.Stoch3,
            self._joint_name_to_id["br_knee_joint"],
            targetValue=2*np.pi/3, targetVelocity=0)

        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.Stoch3,
            jointIndex=(self._joint_name_to_id["br_knee_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=standing_torque,
            targetVelocity=0
        )

        self._pybullet_client.resetJointState(
            self.Stoch3,
            self._joint_name_to_id["fr_hip_joint"],  # motor
            targetValue=np.pi/3, targetVelocity=0)

        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.Stoch3,
            jointIndex=(self._joint_name_to_id["fr_hip_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=standing_torque,
            targetVelocity=0
        )

        self._pybullet_client.resetJointState(
            self.Stoch3,
            self._joint_name_to_id["fr_knee_joint"],
            targetValue=2*np.pi/3, targetVelocity=0)

        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.Stoch3,
            jointIndex=(self._joint_name_to_id["fr_knee_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=standing_torque,
            targetVelocity=0
        )

        self._pybullet_client.resetJointState(
            self.Stoch3,
            self._joint_name_to_id["bl_hip_joint"],  # motor
            targetValue=np.pi/3, targetVelocity=0)

        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.Stoch3,
            jointIndex=(self._joint_name_to_id["bl_hip_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=standing_torque,
            targetVelocity=0
        )

        self._pybullet_client.resetJointState(
            self.Stoch3,
            self._joint_name_to_id["bl_knee_joint"],
            targetValue=2*np.pi/3, targetVelocity=0)

        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.Stoch3,
            jointIndex=(self._joint_name_to_id["bl_knee_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=standing_torque,
            targetVelocity=0
        )

    def ResetPoseForAbd(self):
        '''
		Reset initial conditions of abduction joints
		'''
        self._pybullet_client.resetJointState(
            self.Stoch3,
            self._joint_name_to_id["fl_abd_joint"],
            targetValue=0, targetVelocity=0)
        self._pybullet_client.resetJointState(
            self.Stoch3,
            self._joint_name_to_id["br_abd_joint"],
            targetValue=0, targetVelocity=0)
        self._pybullet_client.resetJointState(
            self.Stoch3,
            self._joint_name_to_id["fr_abd_joint"],
            targetValue=0, targetVelocity=0)
        self._pybullet_client.resetJointState(
            self.Stoch3,
            self._joint_name_to_id["bl_abd_joint"],
            targetValue=0, targetVelocity=0)

        # Set control mode for each motor and initial conditions
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.Stoch3,
            jointIndex=(self._joint_name_to_id["fl_abd_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.Stoch3,
            jointIndex=(self._joint_name_to_id["br_abd_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.Stoch3,
            jointIndex=(self._joint_name_to_id["fr_abd_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.Stoch3,
            jointIndex=(self._joint_name_to_id["bl_abd_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )

