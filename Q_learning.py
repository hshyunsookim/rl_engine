# Tabular Q-learning // SARSA

import taxi_domain as dm
import random as r
import collections


# Initialize game state
s = dm.gameState()
(xt,yt,xp,yp,pickup), (xg,yg) = [[0,0,0,4,0],[4,4]]
s.setStates(xt,yt,xp,yp,pickup, xg,yg)

# Set parameters and variables
alpha = 0.5    # Learning rate
gamma = 0.9    # Discount factor
epsilon = 0.1 # Exploration factor (exploration vs. exploitation)
lamda = 0.5    # Eligibility trace decay rate 
			   # (didn't want to overwrite python built-in name "lambda" here)


numMoves = 0  # Count number of moves in an episode

# Q-value Greedy Policy 
def evaluatePolicy(state):
	for s in state.Q.keys():
		Q_values = state.Q.get(s)

		# for each state, choose argmax_a Q(s,a) 
		maxVal = max(Q_values)						# max_a Q(s,a)
		maxIndices = findMaxInd(maxVal, Q_values)	# index of maximum Q (0~5)
		maxIdx = r.randint(0,len(maxIndices)-1)		# random tie-break 
		greedy_action = maxIndices[maxIdx]+1		# choose greedy aciton (index+1, 1~6) 
		state.pi[(s)] = greedy_action


def findMaxInd(val, list):
	idx = -1
	indices = []
	while True:
		try:
			idx = list.index(val, idx+1)
			indices.append(idx)
		except ValueError:
			break
	return indices

# Functions for debugging
def printEtrace():
	for v in s.e.values():
		for i in  range(len(v)):
			if v[i] != 0.0:
				print v[i]

def printUniqueQ():
	for v in s.Q.values():
		for i in range(len(v)):
			if v[i] != 0:
				print v[i]
	raw_input(" ")	

def printOrderedQ():
	orderedQ = collections.OrderedDict(sorted(s.Q.items()))
	for state, value in orderedQ.iteritems():
		print state, value
	raw_input(" ")

def printOrderedQnE():
	orderedQ = collections.OrderedDict(sorted(s.Q.items()))
	for state, value in orderedQ.iteritems():
		print state, value, s.e.get((state))
	raw_input(" ")



for i in range(51):
	(xt,yt,xp,yp,pickup), (xg,yg) = [[0,0,0,4,0],[4,4]]
	# (xt,yt,xp,yp,pickup), (xg,yg) = [[4,4,4,4,1],[4,4]]
	s.setStates(xt,yt,xp,yp,pickup, xg,yg)
	s.clearTraces()	
	# retain Q-function from previous episode	
	# printOrderedQ()
	### printOrderedQnE()
	### print numMoves

	numMoves = 0


	## Time = t0
	# Store current state (state at time t0)
	s_t0 = s.getCurrState()

	# Get best action for current state
	if r.random() > epsilon:
		a_t0 = s.getAction()
	else:
		a_t0 = r.randint(1,6)

	while not s.checkEoE():
		# Table look-up for Q-value at (s,a) 
		Qval_curr = s.getValue(a_t0)

		# Eligibility Trace 
		s.updateEtrace(a_t0, lamda)
		# printEtrace()

		# Change current state using chosen action, and receive a reward
		reward = s.changeState(a_t0)

		## Time = t1
		# Store new state (state at time t1)
		s_t1 = s.getCurrState()

		# Get best action for next state
		if r.random() > epsilon:
			a_t1 = s.getAction()
		else:
			a_t1 = r.randint(1,6)

		# Table look-up for Q-value at (s',a')
		if s.EoE:
			Qval_next = 0 
		else:
			Qval_next = s.getValue(a_t1)

		## Update Q-value
		# TD-error
		delta = reward + gamma*Qval_next - Qval_curr

		#### print reward, Qval_next, Qval_curr

		# Update Q
		for key in s.Q.keys():
			values = s.e.get((key))
			for actions in range(len(values)):
				newVal = s.Q.get((key))[actions-1] + values[actions-1]*alpha*delta
				s.setValue(key, actions, newVal)
		

		# print s_t0, a_t0, Qval_curr, s_t1, a_t1, Qval_next, s.Q.get((s_t0))[a_t0-1]

		## Take a time-step
		s_t0 = s_t1
		a_t0 = a_t1

		# Update the policy for the new Q-function 
		evaluatePolicy(s)

		# Print progress
		numMoves += 1
		if numMoves%1000 == 0:
			print "."
		if numMoves > 10000:
			print "quit"
			break
		#print "Eps #", i, "  Move #", numMoves, "  State: ", s_t0, "  Action: ", a_t0

	print "End of Episode: ", i, " in ", numMoves, " moves"