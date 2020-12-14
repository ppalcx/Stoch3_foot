import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches



fig2 = plt.figure()
ax2 = fig2.add_subplot(111, aspect='equal')

points = [[-0.3968, -0.45], [0.3968, -0.45], [0.2, -0.1], [-0.2, -0.1], [-0.3968, -0.45]] #the points to trace the edges.
polygon= plt.Polygon(points,  fill=None, edgecolor='r')
ax2.add_patch(polygon)
#fig2.savefig('reg-polygon.png', dpi=90, bbox_inches='tight') 

def drawBezier(swing_points, swing_weights, stance_points, stance_weights,  t):

    def drawCurve(points, weights, t):
        if(points.shape[0]==1):
            return [points[0,0]/weights[0], points[0,1]/weights[0]]
        else:
            newpoints=np.zeros([points.shape[0]-1, points.shape[1]])
            newweights=np.zeros(weights.size)
            for i in np.arange(newpoints.shape[0]):
                x = (1-t) * points[i,0] + t * points[i+1,0]
                y = (1-t) * points[i,1] + t * points[i+1,1]
                w = (1-t) * weights[i] + t*weights[i+1]
                newpoints[i,0] = x
                newpoints[i,1] = y
                newweights[i] = w

            return drawCurve(newpoints, newweights, t)

    swing_newpoints = np.zeros(swing_points.shape)
    stance_newpoints = np.zeros(stance_points.shape)

    for i in np.arange(swing_points.shape[0]):
        swing_newpoints[i]=swing_points[i]*swing_weights[i]

    for i in np.arange(stance_points.shape[0]):
        stance_newpoints[i]=stance_points[i]*stance_weights[i]

    if(t<1):
        return drawCurve(swing_newpoints, swing_weights, t)
    if(t>=1):
        return drawCurve(stance_newpoints, stance_weights, t-1)
        #return [stance_points[0,0]+ (t-1)*(stance_points[-1,0] - stance_points[0,0]), -0.21]




x= np.zeros(80)
y =np.zeros(80)


swing_points = np.array([[-0.14,-0.45],[-0.3,-0.26],[0.2,-0.1],[0.2,-0.1],[0.3,-0.26],[0.14,-0.45]])
stance_points = np.array([[0.14,-0.45],[0, -0.45],[-0.14,-0.45]])

#pts_r = points.copy()
#pts_l = points.copy()
action = [1.0, 1.0, 0.7444449361252171, 1.0, 0.1, 1,1.0]


# def get_swing_stance_weights(action):
#     swing_weights = np.array([action[0], action[1], action[1], action[2]])
#     stance_weights = np.array([action[2],action[3], action[4], action[5],action[0]])
#     return swing_weights, stance_weights

def get_swing_stance_weights(action):
    swing_weights = np.array([action[0], action[1], action[2],action[3],action[4], action[5]])
    stance_weights = np.array([action[5],action[6],action[0]])
    return  swing_weights, stance_weights

# swing_weights = np.ones(4)
# stance_weights = np.ones(5)
swing_weights, stance_weights = get_swing_stance_weights(action)
print(swing_weights)
#weightsl = np.ones(4)

count = 0
for t in np.arange(0,2, 0.025):
    x[count], y[count] = drawBezier(swing_points, swing_weights, stance_points, stance_weights, t)
    count = count+1
# count =0
# for t in np.arange(0,2, 0.001):
#     x1[count], y1[count] = drawBezier(pts_l,weightsl, t)
#     count = count+1

plt.plot(x,y,".", label = 'robot trajectoryr')
#plt.plot(x1,y1,'r', label = 'robot trajectoryl')
plt.legend()
plt.show()
