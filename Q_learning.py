# Tabular Q-learning

import taxi_domain as dm
import random as r


# Initialize game state
s = dm.gameState()
(xt,yt,xp,yp,pickup), (xg,yg) = [[0,0,0,4,0],[4,4]]
s.setStates(xt,yt,xp,yp,pickup, xg,yg)

# Set parameters and variables
alpha = 0.1   # Learning rate
gamma = 0.9   # Discount factor
epsilon = 0.1 # Exploration factor (exploration vs. exploitation)
lamda = 0.1   # Eligibility trace decay rate 
			  # (didn't want to overwrite python built-in name "lambda" here)

numMoves = 0  # Count number of moves in an episode

## Time = t0
# Store current state (state at time t0)
s_t0 = s.getCurrState()

# Get best action for current state
if r.random() < epsilon:
	a_t0 = s.getAction()
else:
	a_t0 = r.randint(1,6)

# Table look-up for Q-value at (s,a) 
Qval_curr = s.getValue(a_t0)

while not s.checkEoE():
	# Change current state using chosen action, and receive a reward
	reward = s.changeState(a_t0)

	## Time = t1
	# Store new state (state at time t1)
	s_t1 = s.getCurrState()

	# Get best action for next state
	if r.random() < epsilon:
		a_t1 = s.getAction()
	else:
		a_t1 = r.randint(1,6)

	# Table look-up for Q-value at (s',a')
	Qval_next = s.getValue(a_t1)

	## Update Q-value
	# TD-error
	delta = reward + gamma*Qval_next - Qval_curr

	# Eligibility Trace 
	''' Need to Implement Here '''
	e = 1

	# Update Q
	s.setValue(s_t0, a_t0, alpha*delta)

	## Take a time-step
	s_t0 = s_t1
	a_t0 = a_t1
	Qval_curr = Qval_next

	# Print progress
	numMoves += 1
	print "Move #", numMoves, "\n"

print "End of Episode"


# To Do:
# implement e-traces
# update policy by max-ing Q(s,a) after each Q-value update