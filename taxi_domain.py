# Taxi-Game Domain

import numpy as np
import sys
import pdb

# Create map 
# Each number indicates valid actions 
# (1:UP, 2:RIGHT, 3:DOWN, 4:LEFT)
# (5: PICK-UP PASSENGER, 6: DROP-OFF PASSENGER) are executable from every position
taxiMap = [[(2,3,5,6),   (3,4,5,6),     (2,3,5,6),     (2,3,4,5,6),   (3,4,5,6)], 
           [(1,2,3,5,6), (1,3,4,5,6),   (1,2,3,5,6),   (1,2,3,4,5,6), (1,3,4,5,6)], 
           [(1,2,3,5,6), (1,2,3,4,5,6), (1,2,3,4,5,6), (1,2,3,4,5,6), (1,3,4,5,6)], 
           [(1,3,5,6),   (1,2,3,5,6),   (1,3,4,5,6),   (1,2,3,5,6),   (1,3,4,5,6)], 
           [(1,5,6),     (1,2,5,6),     (1,4,5,6),     (1,2,5,6),     (1,4,5,6) ]]

# Initialize tabular value function, set values for all 625 states to zero
# Structured as {state:value} = {(taxi_Xpos, taxi_Ypos, passenger_Xpos, passenger_Ypos):value}
V = {}
for xt in range(5):
	for yt in range(5):
		for xp in range(5):
			for yp in range(5):
				V[(xt,yt,xp,yp)] = 0


# GameState Class:
# start state: (x,y) positions of taxi, passenger, and pick-up state
# goal state: (x,y) position of goal location 
# goal state implicitly states xt==xp, yt==yp, pickup = 0 (passenger dropped off)
class gameState:
	def __init__(self):
		self.xt = 0
		self.yt = 0
		self.xp = 0
		self.yp = 4
		self.pickup = 0
		self.xg = 4
		self.yg = 4

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


'''
s = gameState()
s.setStates(1,1,1,1,1,1,1)
s.printState()
'''

pdb.set_trace()  # breakpoint 9ae05d3c //
				