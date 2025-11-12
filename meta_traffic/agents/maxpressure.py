"""MaxPressure Agent class."""

class MaxPressureAgent:
    """MaxPressure Agent class."""

    def __init__(self, env, ts_id):
        """Initialize MaxPressure agent."""
        self.env = env
        self.ts_id = ts_id
        self.action = None
        self.traffic_signal = self.env.traffic_signals[self.ts_id]

    def act(self):

    
        vehicle_counts = dict()
        for lane in self.env.sumo.lane.getIDList():
            vehicle_counts[lane] = len(self.env.sumo.lane.getLastStepVehicleIDs(lane))
        max_pressure = None
        action = -1
        for phase_id in range(self.traffic_signal.action_space.n):
            pressure = sum([vehicle_counts[start] - vehicle_counts[end] for start, end in self.traffic_signal.phase_available_lanelinks[phase_id]])
            if max_pressure is None or pressure > max_pressure:
                action = phase_id
                max_pressure = pressure
        self.action = action
        return action
    
    

