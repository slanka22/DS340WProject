"""
Script to integrate METADRIVE and SUMO simulations
"""

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import argparse
import logging
import time
from panda3d.core import Vec3
# ==================================================================================================
# -- find metadrive module -----------------------------------------------------------------------------
# ==================================================================================================
import glob
import os
import sys
# ==================================================================================================
# -- find traci module -----------------------------------------------------------------------------
# ==================================================================================================

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# ==================================================================================================
# -- sumo integration imports ----------------------------------------------------------------------
# ==================================================================================================

from .sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from .sumo_integration.metadrive_simulation import MetaDriveSimulation  # pylint: disable=wrong-import-position
from .sumo_integration.constants import INVALID_ACTOR_ID, PANDA_WORLD_SIZE  # pylint: disable=wrong-import-position
from .sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position

# ==================================================================================================
# -- synchronization_loop --------------------------------------------------------------------------
# ==================================================================================================


class SimulationSynchronization(object):
    """
    SimulationSynchronization class is responsible for the synchronization of sumo and meta
    simulations.
    """
    def __init__(
        self,
        sumo_simulation,
        metadrive_simulation,
        tls_manager='sumo',
        sync_vehicle_color=False,
        sync_vehicle_lights=False
    ):

        self.sumo = sumo_simulation
        self.metadrive = metadrive_simulation

        self.metadrive.world.reset()

        self.metadrive.world.vehicle.set_pitch(-90.0)
        self.metadrive.world.vehicle.set_roll(-90.0)
        self.metadrive.world.vehicle.origin.hide()
        self.metadrive.world.vehicle.disable_gravity()
        for node in self.sumo.net.getNodes():
            if node.getType() == 'traffic_light':
                center_p = node._coord
                self.metadrive.world.vehicle.set_position((center_p[0], center_p[1]), height=100)
                print(node.getType())
                
        o, r, tm, tc, info = self.metadrive.world.step([0, 0])



        # self.metadrive.world.engine.sensors['rgb_camera'].track(self.metadrive.world.vehicle)
        # self.world.vehicle.position
        # self.world.vehicle.set_position((11800,13330), height=5)
        self.metadrive.world.engine.force_fps.disable()


        self.tls_manager = tls_manager
        self.sync_vehicle_color = sync_vehicle_color
        self.sync_vehicle_lights = sync_vehicle_lights

        if tls_manager == 'metadrive':
            self.sumo.switch_off_traffic_lights()
        elif tls_manager == 'sumo':
            self.metadrive.switch_off_traffic_lights()

        # Mapped actor ids.
        self.sumo2metadrive_ids = {}  # Contains only actors controlled by sumo.
        self.metadrive2sumo_ids = {}  # Contains only actors controlled by metadrive.

        self.sumo2metadrive_ped_ids = {}  # Contains only ped actors controlled by sumo.
        self.metadrive2sumo_ped_ids = {}  # Contains only ped actors controlled by metadrive.

        # for landmark_id in self.sumo.traffic_light_ids:
        #     print(landmark_id)

        # TODO
        BridgeHelper.blueprint_library = []
        BridgeHelper.offset = self.sumo.offset
        # BridgeHelper.offset = self.sumo.get_net_offset()

    def tick(self):
        # """
        # Tick to simulation synchronization
        # """
        # # -----------------
        # # sumo-->meta sync
        # # -----------------
        self.sumo.tick()

        # Spawning new sumo actors in meta (i.e, not controlled by metadrive).
        sumo_spawned_actors = self.sumo.spawned_actors - set(self.metadrive2sumo_ids.values())
        for sumo_actor_id in sumo_spawned_actors:
            self.sumo.subscribe(sumo_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            metadrive_blueprint = BridgeHelper.get_metadrive_blueprint(sumo_actor, self.sync_vehicle_color)
            if metadrive_blueprint is not None:
                metadrive_transform = BridgeHelper.get_metadrive_transform(
                    sumo_actor.transform, sumo_actor.extent, sumo_actor
                )

                metadrive_actor_id = self.metadrive.spawn_actor(metadrive_blueprint, metadrive_transform)
                if metadrive_actor_id != INVALID_ACTOR_ID:
                    self.sumo2metadrive_ids[sumo_actor_id] = metadrive_actor_id
            else:
                self.sumo.unsubscribe(sumo_actor_id)

        # Destroying sumo arrived actors in meta.
        for sumo_actor_id in self.sumo.destroyed_actors:
            if sumo_actor_id in self.sumo2metadrive_ids:
                id = self.sumo2metadrive_ids.pop(sumo_actor_id)
                self.metadrive.destroy_actor(id)
                self.metadrive.world.engine._clean_color(id)

        # Updating sumo actors in metadrive.
        for sumo_actor_id in self.sumo2metadrive_ids:
            metadrive_actor_id = self.sumo2metadrive_ids[sumo_actor_id]

            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            # metadrive_actor = self.metadrive.get_actor(metadrive_actor_id)

            metadrive_transform = BridgeHelper.get_metadrive_transform(
                sumo_actor.transform, sumo_actor.extent, sumo_actor
            )
            # if self.sync_vehicle_lights:
            #     metadrive_lights = BridgeHelper.get_metadrive_lights_state(metadrive_actor.get_light_state(),
            #                                                        sumo_actor.signals)
            # else:
            metadrive_lights = None

            self.metadrive.synchronize_vehicle(metadrive_actor_id, metadrive_transform, sumo_actor, metadrive_lights)

        # Spawning new sumo actors in meta (i.e, not controlled by metadrive).
        sumo_spawned_ped_actors = self.sumo.spawned_ped_actors - set(self.metadrive2sumo_ped_ids.values())
        for sumo_actor_id in sumo_spawned_ped_actors:
            self.sumo.subscribe_pedestrian(sumo_actor_id)
            sumo_actor = self.sumo.get_actor_pedestrian(sumo_actor_id)

            metadrive_blueprint = BridgeHelper.get_metadrive_blueprint(sumo_actor, self.sync_vehicle_color)
            if metadrive_blueprint is not None:
                metadrive_transform = BridgeHelper.get_metadrive_transform(
                    sumo_actor.transform, sumo_actor.extent, sumo_actor
                )

                metadrive_actor_id = self.metadrive.spawn_actor(metadrive_blueprint, metadrive_transform)
                if metadrive_actor_id != INVALID_ACTOR_ID:
                    self.sumo2metadrive_ped_ids[sumo_actor_id] = metadrive_actor_id
            else:
                self.sumo.unsubscribe_pedestrian(sumo_actor_id)

        # Destroying sumo arrived actors in meta.
        for sumo_actor_id in self.sumo.destroyed_ped_actors:
            if sumo_actor_id in self.sumo2metadrive_ped_ids:
                self.metadrive.destroy_actor(self.sumo2metadrive_ped_ids.pop(sumo_actor_id))

        # Updating sumo actors in metadrive.
        for sumo_actor_id in self.sumo2metadrive_ped_ids:
            metadrive_actor_id = self.sumo2metadrive_ped_ids[sumo_actor_id]

            sumo_actor = self.sumo.get_actor_pedestrian(sumo_actor_id)
            # metadrive_actor = self.metadrive.get_actor(metadrive_actor_id)

            metadrive_transform = BridgeHelper.get_metadrive_transform(
                sumo_actor.transform, sumo_actor.extent, sumo_actor
            )
            metadrive_lights = None

            self.metadrive.synchronize_vehicle(metadrive_actor_id, metadrive_transform, sumo_actor, metadrive_lights)

        # # Updates and creates traffic lights in metadrive based on sumo information.
        # if self.tls_manager == 'sumo':
        #     for landmark_id in self.sumo.traffic_light_ids:
        #         print(landmark_id)
        #         sumo_tl_state = self.sumo.get_traffic_light_state(landmark_id)
        #         metadrive_tl_state = BridgeHelper.get_metadrive_traffic_light_state(sumo_tl_state)

        #         self.metadrive.synchronize_traffic_light(landmark_id, metadrive_tl_state)
        # self.metadrive.world.engine.taskMgr.step()
        self.metadrive.tick()
        # self.metadrive.render()

    def close(self):
        """
        Cleans synchronization.
        """

        # Destroying synchronized actors.
        for metadrive_actor_id in self.sumo2metadrive_ids.values():
            self.metadrive.destroy_actor(metadrive_actor_id)

        for sumo_actor_id in self.metadrive2sumo_ids.values():
            self.sumo.destroy_actor(sumo_actor_id)

        # Closing sumo and meta client.
        self.sumo.close()
        self.metadrive.close()

        self.sumo2metadrive_ids = {}
        self.metadrive2sumo_ids = {}

        self.sumo2metadrive_ped_ids = {}
        self.metadrive2sumo_ped_ids = {}

def synchronization_loop(args):
    """
    Entry point for sumo-meta co-simulation.
    """
    sumo_simulation = SumoSimulation(args.sumo_cfg_file, args.step_length, args.sumo_host,
                                     args.sumo_port, args.sumo_gui, args.client_order)

    # metadrive makes repeat steps with each step length of 0.02s
    metadrive_simulation = MetaDriveSimulation(int(args.step_length/PANDA_WORLD_SIZE))

    synchronization = SimulationSynchronization(sumo_simulation, metadrive_simulation, args.tls_manager,
                                                args.sync_vehicle_color, args.sync_vehicle_lights)

    try:
        while True:
            # start = time.time()
            synchronization.tick()
            # end = time.time()
            # elapsed = end - start
            # if elapsed < args.step_length:
            #     time.sleep(args.step_length - elapsed)

    except KeyboardInterrupt:
        logging.info('Cancelled by user.')

    finally:
        logging.info('Cleaning synchronization')

        synchronization.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('sumo_cfg_file', type=str, help='sumo configuration file')
    argparser.add_argument('--sumo-host',
                           metavar='H',
                           default=None,
                           help='IP of the sumo host server (default: 127.0.0.1)')
    argparser.add_argument('--sumo-port',
                           metavar='P',
                           default=None,
                           type=int,
                           help='TCP port to listen to (default: 8813)')
    argparser.add_argument('--sumo-gui', action='store_true', help='run the gui version of sumo')
    argparser.add_argument('--step-length',
                           default=1,
                           type=float,
                           help='set fixed delta seconds (default: 0.04s)')
    argparser.add_argument('--client-order',
                           metavar='TRACI_CLIENT_ORDER',
                           default=1,
                           type=int,
                           help='client order number for the co-simulation TraCI connection (default: 1)')
    argparser.add_argument('--sync-vehicle-lights',
                           action='store_true',
                           help='synchronize vehicle lights state (default: False)')
    argparser.add_argument('--sync-vehicle-color',
                           action='store_true',
                           help='synchronize vehicle color (default: False)')
    argparser.add_argument('--sync-vehicle-all',
                           action='store_true',
                           help='synchronize all vehicle properties (default: False)')
    argparser.add_argument('--tls-manager',
                           type=str,
                           choices=['none', 'sumo', 'meta'],
                           help="select traffic light manager (default: none)",
                           default='sumo')
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')
    arguments = argparser.parse_args()

    if arguments.sync_vehicle_all is True:
        arguments.sync_vehicle_lights = True
        arguments.sync_vehicle_color = True

    if arguments.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    synchronization_loop(arguments)
