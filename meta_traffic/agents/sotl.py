"""Self Organizing Agent class."""
import random, os, sys
from itertools import cycle
from collections import deque


class SOTLAgent:
    """Self Organizing Agent class."""

    def __init__(self, env, ts_id, min_green_vehicle=3, max_red_vehicle=6):
        """Initialize Self Organizing agent."""
        self.env = env
        self.ts_id = ts_id
        self.action = 0
        self.min_green_vehicle = min_green_vehicle
        self.max_red_vehicle = max_red_vehicle
        self.traffic_signal = self.env.traffic_signals[self.ts_id]
    
    def act(self):
        
        vehicle_counts = dict()
        for lane in self.env.engine.agent_manager.conn.lane.getIDList():
            vehicle_counts[lane] = len(self.env.engine.agent_manager.conn.lane.getLastStepVehicleIDs(lane))

        num_green_vehicles = sum([vehicle_counts[lane] for lane in self.traffic_signal.phase_available_startlanes[self.action]])
        num_red_vehicles = sum([vehicle_counts[lane] for lane in self.traffic_signal.startlanes])
        num_red_vehicles -= num_green_vehicles
        
        if (num_green_vehicles <= self.min_green_vehicle and num_red_vehicles > self.max_red_vehicle) or ((num_green_vehicles == 0 and num_red_vehicles > 0)):
            action = (self.action+1) % len(self.traffic_signal.green_phases)
            self.action = action
            print(action)
        return self.action
    
    

