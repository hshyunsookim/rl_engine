# Taxi-Game Domain

import numpy as np
import sys
from random import randint

# Create map 
# Each number indicates valid actions 
# (1:UP, 2:RIGHT, 3:DOWN, 4:LEFT)
# (5: PICK-UP PASSENGER, 6: DROP-OFF PASSENGER) are executable from every position
taxiMap = [[(2,3,5,6),   (3,4,5,6),     (2,3,5,6),     (2,3,4,5,6),   (3,4,5,6)], 
           [(1,2,3,5,6), (1,3,4,5,6),   (1,2,3,5,6),   (1,2,3,4,5,6), (1,3,4,5,6)], 
           [(1,2,3,5,6), (1,2,3,4,5,6), (1,2,3,4,5,6), (1,2,3,4,5,6), (1,3,4,5,6)], 
           [(1,3,5,6),   (1,2,3,5,6),   (1,3,4,5,6),   (1,2,3,5,6),   (1,3,4,5,6)], 
           [(1,5,6),     (1,2,5,6),     (1,4,5,6),     (1,2,5,6),     (1,4,5,6) ]]

# 1) Initialize tabular Q-function, set values for all 625 states to zero
# Structured as a map, where key = (xt,yt,xp,yp,pickup) and value = 6-element tuple, 
# holding one value for each action.  
# Ex.) Q.get( (xt,yt,xp,yp,pickup) )[action]
# Ex.) Q.get( (1,1,5,5,0) )[3] gives the value of executing action 3 (move DOWN) 
#      when the taxi is at position (1,1), passenger is at (5,5) and the passenger 
# 	   is not picked up (pick-up = 0) 

# 2) Initialize tabular Policy (Pi)
# Before any learning, the policy is initialized as random action for every state
# because the value of each state-action pair is all zero. (random tie-breaks)

# 3) Initialize eligibility traces (to be optionally used)
# e-trace keeps track of a list of past states, and a weight that decays with time
# Most recently visited state receives a weight of 1 (full update), whereas
# weight of past states are scaled by a factor (lambda) at each time step
Q, pi, e = {}, {}, {}
for xt in range(5):
	for yt in range(5):
		for xp in range(5):
			for yp in range(5):
				Q[(xt,yt,xp,yp,0)] = (0,0,0,0,0,0)
				pi[(xt,yt,xp,yp,0)] = randint(1,6)
				e[(xt,yt,xp,yp,0)] = (0,0,0,0,0,0)
		# When passenger is picked-up, (x,y) positions of taxi and passenger are the same
		Q[(xt,yt,xt,yt,1)] = (0,0,0,0,0,0)
		pi[(xt,yt,xt,yt,1)] = randint(1,6)
		e[(xt,yt,xt,yt,1)] = (0,0,0,0,0,0)

# GameState Class:
# start state: (x,y) positions of taxi, passenger, and pick-up state
# goal state: (x,y) position of goal location 
# goal state implicitly states xt==xp, yt==yp, pickup = 0 (passenger dropped off)
class gameState:
	def __init__(self):
		# Current State
		self.xt = 0
		self.yt = 0
		self.xp = 0
		self.yp = 4
		self.pickup = 0
		
		# Goal State
		self.xg = 4
		self.yg = 4

		# Q-values
		self.Q = Q

		# Policy
		self.pi = pi

		# Eligibility traces
		self.e = e

		# Domain map
		self.map = taxiMap
		
 		# indicates End-of-Episode
		self.EoE = 0

	def setStates(self, xt, yt, xp, yp, pickup, xg, yg):
		self.xt = xt
		self.yt = yt
		self.xp = xp
		self.yp = yp
		self.pickup = pickup 
		self.xg = xg
		self.yg = yg

	def printState(self):
		print "current state : ", (self.xt, self.yt, self.xp, self.yp, self.pickup)
		print "goal state : ", (self.xg, self.yg)

	def getCurrState(self):
		return (self.xt, self.yt, self.xp, self.yp, self.pickup)

	def getGoalState(self):
		return (self.xg, self.yg)

	def getValue(self, action):
		state = (self.xt, self.yt, self.xp, self.yp, self.pickup)
		# print "State: ", state, " Action: ", action
		return self.Q.get((state))[action-1] # actions: 1~6, index: 0~5

	def setValue(self, state, action, newValue):
		curr = self.Q.get((state))
		values = list(self.Q.get((state)))
		values[action-1] = newValue
		self.Q[(state)] = tuple(values)
		# print "s:", state, "curr: ", curr, " newVal: ", newValue, "  new: ", self.Q.get((state))

	def getAction(self):
		state = (self.xt, self.yt, self.xp, self.yp, self.pickup)
		return self.pi.get((state))

	# Transition Function for the Taxi-Domain
	# Each action from 1 to 4 moves the taxi in one of four directions
	# If the passenger is picked-up, passenger moves with the taxi
	# Pick-up and Drop-off can be attempted at any position 
	# Episode ends (EoE <-- True) when the taxi arrives at the goal position
	#  with the passenger, and executes the drop-off action
	def changeState(self, action):
		# All moves receive -1 reward (special cases handled below)
		r = -1

		# If the action is valid at the current position, follow the transition rules
		if self.validAction(action):

			# Move UP/RIGHT/DOWN/LEFT
			if action == 1:
				self.yt += 1
			elif action == 2:
				self.xt += 1
			elif action == 3:
				self.yt -= 1
			elif action == 4:
				self.xt -= 1

			# if passenger is already in the taxi, they move together
			if self.pickup == 1:
				self.xp = self.xt
				self.yp = self.yt

			# Pick up 
			if action == 5:
				if self.checkSamePosition():
					self.pickup = 1
					
				else:
					# attempted pick-up, failed because taxi and passenger are
					# not at the same location. Penalize with -10 reward
					r = -10

			# Drop off
			elif action == 6:
				# attempting to drop-off when passenger is not on the taxi
				# is penalized with -10 reward
				if not self.pickup:
					r = -10 

				self.pickup = 0

				# End-of-Episode gets +20 reward
				if self.checkEoE():
					r = 20
					self.EoE = 1

		return r


	def validAction(self, action):
		return action in self.map[4-self.yt][self.xt] 

	def checkSamePosition(self):
		return (self.xt==self.xp) & (self.yt==self.yp)

	def updateEtrace(self, action, lamda):
		for key in e.keys():
			val = list(self.e[(key)])
			for i in range(len(val)):
				val[i] *= lamda
			e[(key)] = tuple(val)

		state = (self.xt, self.yt, self.xp, self.yp, self.pickup)
		val = list(self.e.get((state)))
		val[action-1] = 1
		self.e[(state)] = tuple(val)

	def clearTraces(self):
		for key in self.e.keys():
			self.e[(key)] = (0,0,0,0,0,0);

	def checkEoE(self):
		return self.checkSamePosition() & (self.xg==self.xp) & \
			    (self.yg==self.yp) & (self.pickup == 0)