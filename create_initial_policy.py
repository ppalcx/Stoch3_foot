import numpy as np
#import gym_stoch2_sloped_terrain.envs.stoch2_pybullet_env as e
#import gym_stoch2_sloped_terrain.envs.HyQ_pybullet_env as e

import gym_stoch2_sloped_terrain.envs.Stoch3_pybullet_env as e

import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

if(__name__ == "__main__"):


	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument('--policyName', help='file name of the initial policy', type=str, default='initial_policy_Stoch3')
	args = parser.parse_args()	
	env = e.Stoch3Env(render=True, wedge = False, stairs = False,on_rack=False)

	tuned_actions= np.array(
							[1,.5,.5,.5,.5,1,1,.5,.5,.5,.5,1,0,0,0,0]
					  		)

	# NUmber of steps per episode
	num_of_steps = 40

	# list that tracks the states and actions
	states = []
	actions = []
	do_supervised_learning = True

	experiment_counter = 0

	cstate = env.reset()
	roll = 0
	pitch = 0
	t_r =0
	for ii in np.arange(0,num_of_steps):
		cstate, r, _, info = env.step(tuned_actions)
		t_r +=r
		states.append(cstate)
		actions.append(tuned_actions)
	experiment_counter = experiment_counter +1
	print("Returns of the experiment:",t_r)

	if(do_supervised_learning):
		model = LinearRegression(fit_intercept = False)
		states = np.array(states)
		actions = np.array(actions)

		#train
		print("Shape_X_Labels:",states.shape,"Shape_Y_Labels:",actions.shape)
		model.fit(states,actions)
		action_pred= model.predict(states)
		
		#test
		print('Mean squared error:', mean_squared_error(actions, action_pred))
		res = np.array(model.coef_)
		np.save("./initial_policies/"+args.policyName+".npy", res)