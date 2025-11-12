import gymnasium as gym
from metadrive.policy.base_policy import BasePolicy


class SUMOTrafficSignalControlPolicy(BasePolicy):
    DEBUG_MARK_COLOR = (252, 119, 3, 255)

    def __init__(self, obj, seed):
        # Since control object may change
        super(SUMOTrafficSignalControlPolicy, self).__init__(control_object=obj, random_seed=seed)

    def act(self, agent_id):
        action = self.engine.external_actions[agent_id]
        if self.engine.global_config["action_check"]:
            # Do action check for external input in EnvInputPolicy
            assert self.get_input_space().contains(action), "Input {} is not compatible with action space {}!".format(
                action, self.get_input_space()
            )
        self.action_info["action"] = action
        return action

    @classmethod
    def get_input_space(cls):
        """
        The Input space is a class attribute
        """
        raise ValueError("The input space of SUMO Traffic Light control policy is unknown "
                         "until sumo is created in the agent_manager")
