import sys, os
#sys.path.append(os.path.realpath('../')) False, True
import gym_stoch2_sloped_terrain.envs.Stoch3_pybullet_env as e
import argparse
from fabulous.color import blue,green,red,bold
import numpy as np
import math
import datetime
from matplotlib import pyplot as plt
PI = np.pi


#policy to be tested 
policy = np.load("experiments/15Dec1/iterations/best_policy.npy")

rpy_accurate = []
rpy_noisy = []
if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument('--FrontMass', help='mass to be added in the first', type=float, default=0)
	parser.add_argument('--BackMass', help='mass to be added in the back', type=float, default=0)
	parser.add_argument('--FrictionCoeff', help='foot friction value to be set', type=float, default=0.6)
	parser.add_argument('--WedgeIncline', help='wedge incline degree of the wedge', type=int, default=11)
	parser.add_argument('--WedgeOrientation', help='wedge orientation degree of the wedge', type=float, default=0)
	parser.add_argument('--MotorStrength', help='maximum motor Strength to be applied', type=float, default=7.0)
	parser.add_argument('--RandomTest', help='flag to sample test values randomly ', type=bool, default=False)
	parser.add_argument('--seed', help='seed for the random sampling', type=float, default=100)
	parser.add_argument('--EpisodeLength', help='number of gait steps of a episode', type=int, default=40)

	args = parser.parse_args()
	WedgePresent = False
	if(args.WedgeIncline == 0):
		WedgePresent = False
	
	env = e.Stoch3Env(render=True, wedge=WedgePresent, downhill=True, stairs = False,seed_value=args.seed,
				      on_rack=False, gait = 'trot')
	steps = 0
	t_r = 0
	if(args.RandomTest):
		env.Set_Randomization(default=False)
	else:
		env.incline_deg = args.WedgeIncline

		env.incline_ori = math.radians(args.WedgeOrientation)


	state = env.reset()


	print (
	bold(blue("\nTest Parameters:\n")),
	green('\nWedge Inclination:'),red(env.incline_deg),
	green('\nWedge Orientation:'),red(math.degrees(args.WedgeOrientation)),
	green('\nCoeff. of friction:'),red(env.friction),
	green('\nMotor saturation torque:'),red(env.clips))


	ROT=[]
	tot_energy=[]

	for i_step in range(args.EpisodeLength):

		#env._pybullet_client.stepSimulation()
		#print('Roll:',math.degrees(env.support_plane_estimated_roll),
		      #'Pitch:',math.degrees(env.support_plane_estimated_pitch))
		#action = policy.dot(state)
		action = np.array([1,.5,.5,.5,.5,1,1,.5,.5,.5,.5,1,0,0,0,0])
		state, r, _, angle,energy = env.step(action)

		RPY=state[-3:].tolist()
		ROT.append([datetime.datetime.now(),*RPY]) #*un zip

		ene=energy.tolist()
		tot_energy.append(ene)
		# print(tot_energy)
		final_energy=np.sum(tot_energy)
		print("total energy", final_energy)

		t_r += r

		# if(i_step % 1 ==0):
		# 	    env.vis_foot_traj()

	print("Total_reward "+ str(t_r))


	plt.style.use('classic')
	RPY_DATA = list(zip(*ROT))
	fig, ax = plt.subplots()
	ax.plot(RPY_DATA[0], RPY_DATA[1], color='b', linestyle='--', label='Roll')
	ax.plot(RPY_DATA[0], RPY_DATA[2], color='r', linestyle=':',label='Pitch')
	#ax.plot(RPY_DATA[0], RPY_DATA[3], color='g', linestyle='--',label='Yaw')
	leg = ax.legend();
	# Add gridlines
	ax.grid(linestyle='-', linewidth='0.5')
	plt.show()
