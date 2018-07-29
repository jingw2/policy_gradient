# /usr/bin/env python 3.6
# -*-coding:utf-8-*-

'''
main program
'''
import gym

env = gym.make("CartPole-v0").unwrapped

maxEps = 400
numFeatures = 4
numActions = 2
lr = 0.01
gamma = 0.95
pg = Cart_Pytorch(numFeatures, numActions, lr, gamma)

rewardList_pg = []
for t in range(maxEps):
	state = env.reset()
	done = False

	rewardSum = 0
	while not done:
		action = pg._selectAction(state)

		newState, reward, done, _ = env.step(action)
		pg.storeTrasition(state, action, reward)
		state = newState

		rewardSum += reward

	discountedReward = pg.learn()
	rewardList.append(rewardSum)

	if t % 100 == 0:
		print("Finish epsilon {}".format(t))

plt.plot(list(range(maxEps)), rewardList, 'r-', linewidth = 2)
plt.xlabel("episode")
plt.ylabel("rewards")
plt.show()

state = env.reset()
done = False
while not done:
	action = pg.greedy(state)
	state, reward, done, _ = env.step(action)
	env.render()

env.close()