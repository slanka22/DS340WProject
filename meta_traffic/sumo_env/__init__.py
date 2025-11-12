"""MetaTraffic Environment for Traffic Signal Control."""

from gymnasium.envs.registration import register

register(
    id="meta-traffic-v0",
    entry_point="meta_traffic.environment.metadrive_sumo_gym:MetadriveSumoGym",
    kwargs={"single_agent": True},
)
