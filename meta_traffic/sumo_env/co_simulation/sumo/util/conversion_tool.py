import logging
import os

import numpy as np
from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType


import copy
import logging
import os
import pickle
import shutil

from scenarionet.common_utils import save_summary_anda_mapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from map import StreetMap



def extract_map_features(street_map: StreetMap):

    from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
    
    ret = {}
    # # build map boundary
    polygons = []
    
    for junction_id, junction in street_map.graph.junctions.items():
        if len(junction.shape) <= 2:
            continue
        boundary_polygon = Polygon(junction.shape)
        boundary_polygon = [(x,y) for x, y in boundary_polygon.exterior.coords]
        id = "junction_{}".format(junction.name)
        ret[id] = {
                    SD.TYPE: MetaDriveType.LANE_SURFACE_STREET,
                    SD.POLYLINE: junction.shape,
                    SD.POLYGON: boundary_polygon,
                }

    # build map lanes
    for road_id, road in street_map.graph.roads.items():
        for lane in road.lanes:
            
            id = "lane_{}".format(lane.name)
            
            boundary_polygon = [(x,y) for x, y in lane.shape.shape.exterior.coords]
            if lane.type == 'driving':
                ret[id] = {
                    SD.TYPE: MetaDriveType.LANE_SURFACE_STREET,
                    SD.POLYLINE: lane.sumolib_obj.getShape(),
                    SD.POLYGON: boundary_polygon,
                }
            elif lane.type == 'sidewalk':
                ret[id] = {
                    SD.TYPE: MetaDriveType.BOUNDARY_SIDEWALK,
                    SD.POLYGON: boundary_polygon,
                }
            elif lane.type == 'shoulder':
                ret[id] = {
                    SD.TYPE: MetaDriveType.BOUNDARY_SIDEWALK,
                    SD.POLYGON: boundary_polygon,
                }
            elif lane.type == 'crossing':
                print('hello')
                ret[id] = {
                    SD.TYPE: MetaDriveType.CROSSWALK,
                    SD.POLYGON: boundary_polygon,
                }

    for lane_divider_id, lane_divider in enumerate(street_map.graph.lane_dividers):
        id = "lane_divider_{}".format(lane_divider_id)
        ret[id] = {SD.TYPE: MetaDriveType.LINE_BROKEN_SINGLE_WHITE, SD.POLYLINE: lane_divider}

    for edge_divider_id, edge_divider in enumerate(street_map.graph.edge_dividers):
        id = "edge_divider_{}".format(edge_divider_id)
        ret[id] = {SD.TYPE: MetaDriveType.LINE_SOLID_SINGLE_YELLOW, SD.POLYLINE: edge_divider}
            
    return ret


def create_map_file(street_map: StreetMap, output_path: str):

    result = SD()
    result[SD.ID] = 0
    result[SD.VERSION] = "sumo" 
    # metadata
    result[SD.METADATA] = {}
    result[SD.DYNAMIC_MAP_STATES] = {}
    result[SD.TRACKS] = {}
    result[SD.LENGTH] = 100
    result[SD.METADATA]["dataset"] = street_map.sumo_net_path
    result[SD.METADATA]["map"] = street_map.sumo_net_path
    result[SD.METADATA][SD.METADRIVE_PROCESSED] = False
    result[SD.METADATA]["map_version"] = "sumo-maps-v1.0"
    result[SD.METADATA]["coordinate"] = "right-handed"
    result[SD.METADATA]["scenario_token"] = 0
    result[SD.METADATA]["scenario_id"] = 0
    result[SD.METADATA][SD.ID] = 0
    result[SD.METADATA]["scenario_type"] = "scenario_type"

    all_objs = set()
    all_objs.add("ego")
    tracks = {
        k: dict(
            type=MetaDriveType.UNSET,
            state=dict(
                position=np.concatenate([np.array([0.001, 0.001, 0]).reshape(-1,3)*i for i in range(100)],axis=0),
                heading=np.zeros(shape=(100, )),
                velocity=np.ones(shape=(100, 2))*0.001,
                valid=np.ones(shape=(100, )),
                length=np.zeros(shape=(100, 1)),
                width=np.zeros(shape=(100, 1)),
                height=np.zeros(shape=(100, 1))
            ),
            metadata=dict(track_length=100, type=None, object_id=k, original_id=k)
        )
        for k in list(all_objs)
    }

    result[SD.TRACKS] = tracks
    result[SD.METADATA][SD.SDC_ID] = "ego"

    # # map
    result[SD.MAP_FEATURES] = extract_map_features(street_map)

    sd_scenario = result

    dataset_name = "sumo"
    dataset_version = "v1.0"

    save_path = copy.deepcopy(output_path)
    output_path = output_path + "_tmp_twoway"
    # meta recorder and data summary
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=False)

    summary_file = SD.DATASET.SUMMARY_FILE
    mapping_file = SD.DATASET.MAPPING_FILE

    summary_file_path = os.path.join(output_path, summary_file)
    mapping_file_path = os.path.join(output_path, mapping_file)

    summary = {}
    mapping = {}

    scenario_id = sd_scenario[SD.ID]
    export_file_name = SD.get_export_file_name(dataset_name, dataset_version, scenario_id)


    # add agents summary
    summary_dict = {}
    for track_id, track in sd_scenario[SD.TRACKS].items():
        summary_dict[track_id] = SD.get_object_summary(state_dict=track, id=track_id)
    sd_scenario[SD.METADATA][SD.SUMMARY.OBJECT_SUMMARY] = summary_dict

    # count some objects occurrence
    sd_scenario[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY] = SD.get_number_summary(sd_scenario)

    # update summary/mapping dicy
    summary[export_file_name] = copy.deepcopy(sd_scenario[SD.METADATA])
    mapping[export_file_name] = ""  # in the same dir

    # sanity check
    sd_scenario = sd_scenario.to_dict()

    # dump
    p = os.path.join(output_path, export_file_name)
    with open(p, "wb") as f:
        pickle.dump(sd_scenario, f)

    # store summary file
    save_summary_anda_mapping(summary_file_path, mapping_file_path, summary, mapping)

  
if __name__ == "__main__":

    import argparse
    import os

    desc = "Build database from sumo scenarios"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--sumo_net_path",
        required=True,
        default='../examples/net/Town01.net.xml',
        help="the path of the sumo network file",
    )
    parser.add_argument(
        "--output_path",
        default='./',
        help="the path of the output database file")
    
    args = parser.parse_args()
    sumo_net_path = args.sumo_net_path
    output_path = args.output_path


    street_map = StreetMap()
    street_map.reset(sumo_net_path)
    
    create_map_file(street_map, output_path)

