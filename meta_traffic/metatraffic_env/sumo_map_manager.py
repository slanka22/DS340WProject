import os.path

from metadrive.manager.base_manager import BaseManager
import xml.etree.ElementTree as ET
from metadrive.component.map.scenario_map import ScenarioMap
from meta_traffic.metatraffic_env.sumo_integration.map_utils import StreetMap, extract_map_features


class SUMOMapManager(BaseManager):
    """
    It currently only support load one map into the simulation.
    """
    PRIORITY = 0  # Map update has the most high priority

    def __init__(self):
        super(SUMOMapManager, self).__init__()
        self.current_map = None
        street_map = StreetMap()
        street_map.reset(self.get_net_file_path(self.engine.global_config["sumo_cfg_file"]))
        self.map_feature = extract_map_features(street_map)

    def destroy(self):
        self.current_map.destroy()
        super(SUMOMapManager, self).destroy()
        self.current_map = None

    def before_reset(self):
        if self.current_map:
            self.current_map.detach_from_world()

    def reset(self):
        if not self.current_map:
            self.current_map = ScenarioMap(map_index=0, map_data=self.map_feature)
        self.current_map.attach_to_world()

    @staticmethod
    def get_net_file_path(sumocfg_file):
        tree = ET.parse(sumocfg_file)
        root = tree.getroot()

        # Find the net-file element
        net_file_element = root.find(".//net-file")

        if net_file_element is not None and 'value' in net_file_element.attrib:
            # Extract and return the path
            net_file_path = net_file_element.attrib['value']
            return os.path.join(os.path.dirname(sumocfg_file), net_file_path)
        else:
            raise ValueError("Can not find net-file path from the sumo cfg: {}".format(sumocfg_file))
