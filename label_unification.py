#!/usr/bin/env python3
"""
Label Unification Module for PROVE Pipeline
Enables joint training of Mapillary Vistas and Cityscapes datasets

This module provides:
1. Complete class definitions for both datasets
2. Mapping strategies for label unification
3. Label transformation utilities
4. Unified dataset configuration for MMSegmentation
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum
import numpy as np


# =============================================================================
# CITYSCAPES CLASS DEFINITIONS
# =============================================================================

class CityscapesTrainID(IntEnum):
    """Cityscapes 19 evaluation classes with their train IDs"""
    ROAD = 0
    SIDEWALK = 1
    BUILDING = 2
    WALL = 3
    FENCE = 4
    POLE = 5
    TRAFFIC_LIGHT = 6
    TRAFFIC_SIGN = 7
    VEGETATION = 8
    TERRAIN = 9
    SKY = 10
    PERSON = 11
    RIDER = 12
    CAR = 13
    TRUCK = 14
    BUS = 15
    TRAIN = 16
    MOTORCYCLE = 17
    BICYCLE = 18
    IGNORE = 255


@dataclass
class CityscapesClass:
    """Cityscapes class definition"""
    name: str
    id: int
    train_id: int
    category: str
    category_id: int
    has_instances: bool
    ignore_in_eval: bool
    color: Tuple[int, int, int]


CITYSCAPES_CLASSES = [
    # void classes
    CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    # flat classes
    CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    # construction classes
    CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    # object classes
    CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    # nature classes
    CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    # sky classes
    CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    # human classes
    CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    # vehicle classes
    CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]


# =============================================================================
# MAPILLARY VISTAS CLASS DEFINITIONS (v1.2 - 66 classes)
# =============================================================================

class MapillaryID(IntEnum):
    """Mapillary Vistas v1.2 class IDs (66 classes)"""
    BIRD = 0
    GROUND_ANIMAL = 1
    CURB = 2
    FENCE = 3
    GUARD_RAIL = 4
    BARRIER = 5
    WALL = 6
    BIKE_LANE = 7
    CROSSWALK_PLAIN = 8
    CURB_CUT = 9
    PARKING = 10
    PEDESTRIAN_AREA = 11
    RAIL_TRACK = 12
    ROAD = 13
    SERVICE_LANE = 14
    SIDEWALK = 15
    BRIDGE = 16
    BUILDING = 17
    TUNNEL = 18
    PERSON = 19
    BICYCLIST = 20
    MOTORCYCLIST = 21
    OTHER_RIDER = 22
    LANE_MARKING_CROSSWALK = 23
    LANE_MARKING_GENERAL = 24
    MOUNTAIN = 25
    SAND = 26
    SKY = 27
    SNOW = 28
    TERRAIN = 29
    VEGETATION = 30
    WATER = 31
    BANNER = 32
    BENCH = 33
    BIKE_RACK = 34
    BILLBOARD = 35
    CATCH_BASIN = 36
    CCTV_CAMERA = 37
    FIRE_HYDRANT = 38
    JUNCTION_BOX = 39
    MAILBOX = 40
    MANHOLE = 41
    PHONE_BOOTH = 42
    POTHOLE = 43
    STREET_LIGHT = 44
    POLE = 45
    TRAFFIC_SIGN_FRAME = 46
    UTILITY_POLE = 47
    TRAFFIC_LIGHT = 48
    TRAFFIC_SIGN_BACK = 49
    TRAFFIC_SIGN_FRONT = 50
    TRASH_CAN = 51
    BICYCLE = 52
    BOAT = 53
    BUS = 54
    CAR = 55
    CARAVAN = 56
    MOTORCYCLE = 57
    ON_RAILS = 58
    OTHER_VEHICLE = 59
    TRAILER = 60
    TRUCK = 61
    WHEELED_SLOW = 62
    CAR_MOUNT = 63
    EGO_VEHICLE = 64
    UNLABELED = 65


@dataclass
class MapillaryClass:
    """Mapillary Vistas class definition"""
    name: str
    id: int
    category: str
    color: Tuple[int, int, int]


MAPILLARY_CLASSES = [
    MapillaryClass('Bird', 0, 'animal', (165, 42, 42)),
    MapillaryClass('Ground Animal', 1, 'animal', (0, 192, 0)),
    MapillaryClass('Curb', 2, 'construction--barrier', (196, 196, 196)),
    MapillaryClass('Fence', 3, 'construction--barrier', (190, 153, 153)),
    MapillaryClass('Guard Rail', 4, 'construction--barrier', (180, 165, 180)),
    MapillaryClass('Barrier', 5, 'construction--barrier', (90, 120, 150)),
    MapillaryClass('Wall', 6, 'construction--barrier', (102, 102, 156)),
    MapillaryClass('Bike Lane', 7, 'construction--flat', (128, 64, 255)),
    MapillaryClass('Crosswalk - Plain', 8, 'construction--flat', (140, 140, 200)),
    MapillaryClass('Curb Cut', 9, 'construction--flat', (170, 170, 170)),
    MapillaryClass('Parking', 10, 'construction--flat', (250, 170, 160)),
    MapillaryClass('Pedestrian Area', 11, 'construction--flat', (96, 96, 96)),
    MapillaryClass('Rail Track', 12, 'construction--flat', (230, 150, 140)),
    MapillaryClass('Road', 13, 'construction--flat', (128, 64, 128)),
    MapillaryClass('Service Lane', 14, 'construction--flat', (110, 110, 110)),
    MapillaryClass('Sidewalk', 15, 'construction--flat', (244, 35, 232)),
    MapillaryClass('Bridge', 16, 'construction--structure', (150, 100, 100)),
    MapillaryClass('Building', 17, 'construction--structure', (70, 70, 70)),
    MapillaryClass('Tunnel', 18, 'construction--structure', (150, 120, 90)),
    MapillaryClass('Person', 19, 'human', (220, 20, 60)),
    MapillaryClass('Bicyclist', 20, 'human--rider', (255, 0, 0)),
    MapillaryClass('Motorcyclist', 21, 'human--rider', (255, 0, 100)),
    MapillaryClass('Other Rider', 22, 'human--rider', (255, 0, 200)),
    MapillaryClass('Lane Marking - Crosswalk', 23, 'marking', (200, 128, 128)),
    MapillaryClass('Lane Marking - General', 24, 'marking', (255, 255, 255)),
    MapillaryClass('Mountain', 25, 'nature', (64, 170, 64)),
    MapillaryClass('Sand', 26, 'nature', (230, 160, 50)),
    MapillaryClass('Sky', 27, 'nature', (70, 130, 180)),
    MapillaryClass('Snow', 28, 'nature', (190, 255, 255)),
    MapillaryClass('Terrain', 29, 'nature', (152, 251, 152)),
    MapillaryClass('Vegetation', 30, 'nature', (107, 142, 35)),
    MapillaryClass('Water', 31, 'nature', (0, 170, 30)),
    MapillaryClass('Banner', 32, 'object', (255, 255, 128)),
    MapillaryClass('Bench', 33, 'object', (250, 0, 30)),
    MapillaryClass('Bike Rack', 34, 'object', (100, 140, 180)),
    MapillaryClass('Billboard', 35, 'object', (220, 220, 220)),
    MapillaryClass('Catch Basin', 36, 'object', (220, 128, 128)),
    MapillaryClass('CCTV Camera', 37, 'object', (222, 40, 40)),
    MapillaryClass('Fire Hydrant', 38, 'object', (100, 170, 30)),
    MapillaryClass('Junction Box', 39, 'object', (40, 40, 40)),
    MapillaryClass('Mailbox', 40, 'object', (33, 33, 33)),
    MapillaryClass('Manhole', 41, 'object', (100, 128, 160)),
    MapillaryClass('Phone Booth', 42, 'object', (142, 0, 0)),
    MapillaryClass('Pothole', 43, 'object', (70, 100, 150)),
    MapillaryClass('Street Light', 44, 'object', (210, 170, 100)),
    MapillaryClass('Pole', 45, 'object--support', (153, 153, 153)),
    MapillaryClass('Traffic Sign Frame', 46, 'object--support', (128, 128, 128)),
    MapillaryClass('Utility Pole', 47, 'object--support', (0, 0, 80)),
    MapillaryClass('Traffic Light', 48, 'object--traffic-sign', (250, 170, 30)),
    MapillaryClass('Traffic Sign (Back)', 49, 'object--traffic-sign', (192, 192, 192)),
    MapillaryClass('Traffic Sign (Front)', 50, 'object--traffic-sign', (220, 220, 0)),
    MapillaryClass('Trash Can', 51, 'object', (140, 140, 20)),
    MapillaryClass('Bicycle', 52, 'object--vehicle', (119, 11, 32)),
    MapillaryClass('Boat', 53, 'object--vehicle', (150, 0, 255)),
    MapillaryClass('Bus', 54, 'object--vehicle', (0, 60, 100)),
    MapillaryClass('Car', 55, 'object--vehicle', (0, 0, 142)),
    MapillaryClass('Caravan', 56, 'object--vehicle', (0, 0, 90)),
    MapillaryClass('Motorcycle', 57, 'object--vehicle', (0, 0, 230)),
    MapillaryClass('On Rails', 58, 'object--vehicle', (0, 80, 100)),
    MapillaryClass('Other Vehicle', 59, 'object--vehicle', (128, 64, 64)),
    MapillaryClass('Trailer', 60, 'object--vehicle', (0, 0, 110)),
    MapillaryClass('Truck', 61, 'object--vehicle', (0, 0, 70)),
    MapillaryClass('Wheeled Slow', 62, 'object--vehicle', (0, 0, 192)),
    MapillaryClass('Car Mount', 63, 'void', (32, 32, 32)),
    MapillaryClass('Ego Vehicle', 64, 'void', (120, 10, 10)),
    MapillaryClass('Unlabeled', 65, 'void', (0, 0, 0)),
]


# =============================================================================
# UNIFIED LABEL DEFINITIONS
# =============================================================================

class UnifiedTrainID(IntEnum):
    """
    Unified training class IDs for joint Cityscapes + Mapillary training.
    
    This unified schema provides a common label space that encompasses
    all classes from both datasets while maintaining semantic consistency.
    """
    # Flat classes (0-4)
    ROAD = 0
    SIDEWALK = 1
    PARKING = 2
    RAIL_TRACK = 3
    BIKE_LANE = 4
    
    # Construction classes (5-11)
    BUILDING = 5
    WALL = 6
    FENCE = 7
    GUARD_RAIL = 8
    BRIDGE = 9
    TUNNEL = 10
    BARRIER = 11
    
    # Object classes (12-17)
    POLE = 12
    TRAFFIC_LIGHT = 13
    TRAFFIC_SIGN = 14
    STREET_LIGHT = 15
    UTILITY_POLE = 16
    OTHER_OBJECT = 17
    
    # Nature classes (18-23)
    VEGETATION = 18
    TERRAIN = 19
    SKY = 20
    WATER = 21
    SNOW = 22
    MOUNTAIN = 23
    
    # Human classes (24-27)
    PERSON = 24
    BICYCLIST = 25
    MOTORCYCLIST = 26
    OTHER_RIDER = 27
    
    # Vehicle classes (28-39)
    CAR = 28
    TRUCK = 29
    BUS = 30
    TRAIN = 31
    MOTORCYCLE = 32
    BICYCLE = 33
    CARAVAN = 34
    TRAILER = 35
    BOAT = 36
    OTHER_VEHICLE = 37
    WHEELED_SLOW = 38
    ANIMAL = 39
    
    # Marking classes (40-41)
    LANE_MARKING = 40
    CROSSWALK = 41
    
    # Void / Ignore
    IGNORE = 255


@dataclass
class UnifiedClass:
    """Unified class definition for joint training"""
    name: str
    id: int
    category: str
    cityscapes_ids: List[int]  # Cityscapes train IDs that map to this class
    mapillary_ids: List[int]   # Mapillary IDs that map to this class
    color: Tuple[int, int, int]


UNIFIED_CLASSES = [
    # Flat classes
    UnifiedClass('road', 0, 'flat', [0], [7, 13, 14], (128, 64, 128)),
    UnifiedClass('sidewalk', 1, 'flat', [1], [9, 11, 15], (244, 35, 232)),
    UnifiedClass('parking', 2, 'flat', [], [10], (250, 170, 160)),
    UnifiedClass('rail track', 3, 'flat', [], [12], (230, 150, 140)),
    UnifiedClass('bike lane', 4, 'flat', [], [7], (128, 64, 255)),
    
    # Construction classes
    UnifiedClass('building', 5, 'construction', [2], [17], (70, 70, 70)),
    UnifiedClass('wall', 6, 'construction', [3], [6], (102, 102, 156)),
    UnifiedClass('fence', 7, 'construction', [4], [3], (190, 153, 153)),
    UnifiedClass('guard rail', 8, 'construction', [], [4], (180, 165, 180)),
    UnifiedClass('bridge', 9, 'construction', [], [16], (150, 100, 100)),
    UnifiedClass('tunnel', 10, 'construction', [], [18], (150, 120, 90)),
    UnifiedClass('barrier', 11, 'construction', [], [2, 5], (90, 120, 150)),
    
    # Object classes
    UnifiedClass('pole', 12, 'object', [5], [45, 46], (153, 153, 153)),
    UnifiedClass('traffic light', 13, 'object', [6], [48], (250, 170, 30)),
    UnifiedClass('traffic sign', 14, 'object', [7], [49, 50], (220, 220, 0)),
    UnifiedClass('street light', 15, 'object', [], [44], (210, 170, 100)),
    UnifiedClass('utility pole', 16, 'object', [], [47], (0, 0, 80)),
    UnifiedClass('other object', 17, 'object', [], [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 51], (140, 140, 140)),
    
    # Nature classes
    UnifiedClass('vegetation', 18, 'nature', [8], [30], (107, 142, 35)),
    UnifiedClass('terrain', 19, 'nature', [9], [26, 29], (152, 251, 152)),
    UnifiedClass('sky', 20, 'nature', [10], [27], (70, 130, 180)),
    UnifiedClass('water', 21, 'nature', [], [31], (0, 170, 30)),
    UnifiedClass('snow', 22, 'nature', [], [28], (190, 255, 255)),
    UnifiedClass('mountain', 23, 'nature', [], [25], (64, 170, 64)),
    
    # Human classes
    UnifiedClass('person', 24, 'human', [11], [19], (220, 20, 60)),
    UnifiedClass('bicyclist', 25, 'human', [12], [20], (255, 0, 0)),
    UnifiedClass('motorcyclist', 26, 'human', [12], [21], (255, 0, 100)),
    UnifiedClass('other rider', 27, 'human', [12], [22], (255, 0, 200)),
    
    # Vehicle classes
    UnifiedClass('car', 28, 'vehicle', [13], [55], (0, 0, 142)),
    UnifiedClass('truck', 29, 'vehicle', [14], [61], (0, 0, 70)),
    UnifiedClass('bus', 30, 'vehicle', [15], [54], (0, 60, 100)),
    UnifiedClass('train', 31, 'vehicle', [16], [58], (0, 80, 100)),
    UnifiedClass('motorcycle', 32, 'vehicle', [17], [57], (0, 0, 230)),
    UnifiedClass('bicycle', 33, 'vehicle', [18], [52], (119, 11, 32)),
    UnifiedClass('caravan', 34, 'vehicle', [], [56], (0, 0, 90)),
    UnifiedClass('trailer', 35, 'vehicle', [], [60], (0, 0, 110)),
    UnifiedClass('boat', 36, 'vehicle', [], [53], (150, 0, 255)),
    UnifiedClass('other vehicle', 37, 'vehicle', [], [59], (128, 64, 64)),
    UnifiedClass('wheeled slow', 38, 'vehicle', [], [62], (0, 0, 192)),
    UnifiedClass('animal', 39, 'animal', [], [0, 1], (165, 42, 42)),
    
    # Marking classes
    UnifiedClass('lane marking', 40, 'marking', [], [24], (255, 255, 255)),
    UnifiedClass('crosswalk', 41, 'marking', [], [8, 23], (200, 128, 128)),
]


# =============================================================================
# MAPPING STRATEGIES
# =============================================================================

class MappingStrategy:
    """Base class for label mapping strategies"""
    
    def __init__(self, ignore_index: int = 255):
        self.ignore_index = ignore_index
    
    def create_lookup_table(self, source_num_classes: int) -> np.ndarray:
        """Create a lookup table for fast label transformation"""
        raise NotImplementedError
    
    def transform_label(self, label: np.ndarray) -> np.ndarray:
        """Transform a label map using the lookup table"""
        raise NotImplementedError


class MapillarytoCityscapes(MappingStrategy):
    """
    Map Mapillary Vistas labels to Cityscapes 19-class format.
    
    This is useful when you want to use Mapillary data to augment
    Cityscapes training while keeping the original Cityscapes classes.
    """
    
    MAPPING = {
        # Flat: road (0)
        MapillaryID.ROAD: CityscapesTrainID.ROAD,
        MapillaryID.BIKE_LANE: CityscapesTrainID.ROAD,
        MapillaryID.SERVICE_LANE: CityscapesTrainID.ROAD,
        MapillaryID.LANE_MARKING_GENERAL: CityscapesTrainID.ROAD,
        MapillaryID.LANE_MARKING_CROSSWALK: CityscapesTrainID.ROAD,
        
        # Flat: sidewalk (1)
        MapillaryID.SIDEWALK: CityscapesTrainID.SIDEWALK,
        MapillaryID.CURB_CUT: CityscapesTrainID.SIDEWALK,
        MapillaryID.PEDESTRIAN_AREA: CityscapesTrainID.SIDEWALK,
        MapillaryID.CURB: CityscapesTrainID.SIDEWALK,
        MapillaryID.CROSSWALK_PLAIN: CityscapesTrainID.SIDEWALK,
        
        # Construction: building (2)
        MapillaryID.BUILDING: CityscapesTrainID.BUILDING,
        MapillaryID.BRIDGE: CityscapesTrainID.BUILDING,
        MapillaryID.TUNNEL: CityscapesTrainID.BUILDING,
        
        # Construction: wall (3)
        MapillaryID.WALL: CityscapesTrainID.WALL,
        MapillaryID.BARRIER: CityscapesTrainID.WALL,
        
        # Construction: fence (4)
        MapillaryID.FENCE: CityscapesTrainID.FENCE,
        MapillaryID.GUARD_RAIL: CityscapesTrainID.FENCE,
        
        # Object: pole (5)
        MapillaryID.POLE: CityscapesTrainID.POLE,
        MapillaryID.UTILITY_POLE: CityscapesTrainID.POLE,
        MapillaryID.TRAFFIC_SIGN_FRAME: CityscapesTrainID.POLE,
        MapillaryID.STREET_LIGHT: CityscapesTrainID.POLE,
        
        # Object: traffic light (6)
        MapillaryID.TRAFFIC_LIGHT: CityscapesTrainID.TRAFFIC_LIGHT,
        
        # Object: traffic sign (7)
        MapillaryID.TRAFFIC_SIGN_FRONT: CityscapesTrainID.TRAFFIC_SIGN,
        MapillaryID.TRAFFIC_SIGN_BACK: CityscapesTrainID.TRAFFIC_SIGN,
        
        # Nature: vegetation (8)
        MapillaryID.VEGETATION: CityscapesTrainID.VEGETATION,
        
        # Nature: terrain (9)
        MapillaryID.TERRAIN: CityscapesTrainID.TERRAIN,
        MapillaryID.SAND: CityscapesTrainID.TERRAIN,
        MapillaryID.MOUNTAIN: CityscapesTrainID.TERRAIN,
        
        # Nature: sky (10)
        MapillaryID.SKY: CityscapesTrainID.SKY,
        
        # Human: person (11)
        MapillaryID.PERSON: CityscapesTrainID.PERSON,
        
        # Human: rider (12) - combine all rider types
        MapillaryID.BICYCLIST: CityscapesTrainID.RIDER,
        MapillaryID.MOTORCYCLIST: CityscapesTrainID.RIDER,
        MapillaryID.OTHER_RIDER: CityscapesTrainID.RIDER,
        
        # Vehicle: car (13)
        MapillaryID.CAR: CityscapesTrainID.CAR,
        
        # Vehicle: truck (14)
        MapillaryID.TRUCK: CityscapesTrainID.TRUCK,
        
        # Vehicle: bus (15)
        MapillaryID.BUS: CityscapesTrainID.BUS,
        
        # Vehicle: train (16)
        MapillaryID.ON_RAILS: CityscapesTrainID.TRAIN,
        
        # Vehicle: motorcycle (17)
        MapillaryID.MOTORCYCLE: CityscapesTrainID.MOTORCYCLE,
        
        # Vehicle: bicycle (18)
        MapillaryID.BICYCLE: CityscapesTrainID.BICYCLE,
        
        # Ignored classes -> 255
        MapillaryID.BIRD: CityscapesTrainID.IGNORE,
        MapillaryID.GROUND_ANIMAL: CityscapesTrainID.IGNORE,
        MapillaryID.PARKING: CityscapesTrainID.IGNORE,
        MapillaryID.RAIL_TRACK: CityscapesTrainID.IGNORE,
        MapillaryID.SNOW: CityscapesTrainID.IGNORE,
        MapillaryID.WATER: CityscapesTrainID.IGNORE,
        MapillaryID.BANNER: CityscapesTrainID.IGNORE,
        MapillaryID.BENCH: CityscapesTrainID.IGNORE,
        MapillaryID.BIKE_RACK: CityscapesTrainID.IGNORE,
        MapillaryID.BILLBOARD: CityscapesTrainID.IGNORE,
        MapillaryID.CATCH_BASIN: CityscapesTrainID.IGNORE,
        MapillaryID.CCTV_CAMERA: CityscapesTrainID.IGNORE,
        MapillaryID.FIRE_HYDRANT: CityscapesTrainID.IGNORE,
        MapillaryID.JUNCTION_BOX: CityscapesTrainID.IGNORE,
        MapillaryID.MAILBOX: CityscapesTrainID.IGNORE,
        MapillaryID.MANHOLE: CityscapesTrainID.IGNORE,
        MapillaryID.PHONE_BOOTH: CityscapesTrainID.IGNORE,
        MapillaryID.POTHOLE: CityscapesTrainID.IGNORE,
        MapillaryID.TRASH_CAN: CityscapesTrainID.IGNORE,
        MapillaryID.BOAT: CityscapesTrainID.IGNORE,
        MapillaryID.CARAVAN: CityscapesTrainID.IGNORE,
        MapillaryID.OTHER_VEHICLE: CityscapesTrainID.IGNORE,
        MapillaryID.TRAILER: CityscapesTrainID.IGNORE,
        MapillaryID.WHEELED_SLOW: CityscapesTrainID.IGNORE,
        MapillaryID.CAR_MOUNT: CityscapesTrainID.IGNORE,
        MapillaryID.EGO_VEHICLE: CityscapesTrainID.IGNORE,
        MapillaryID.UNLABELED: CityscapesTrainID.IGNORE,
    }
    
    def __init__(self, ignore_index: int = 255):
        super().__init__(ignore_index)
        # Use 256 to cover all possible uint8 values (0-255)
        # Mapillary has 66 official classes but labels can contain higher values
        self.lookup_table = self.create_lookup_table(256)
    
    def create_lookup_table(self, source_num_classes: int = 256) -> np.ndarray:
        """Create lookup table for Mapillary -> Cityscapes mapping
        
        Args:
            source_num_classes: Size of lookup table (default 256 to handle all uint8 values)
        """
        lut = np.full(source_num_classes, self.ignore_index, dtype=np.uint8)
        for mapillary_id, cityscapes_id in self.MAPPING.items():
            lut[mapillary_id] = cityscapes_id
        return lut
    
    def transform_label(self, label: np.ndarray) -> np.ndarray:
        """Transform Mapillary label to Cityscapes format
        
        Note: Handles label values beyond the 66 official Mapillary classes
        by mapping them to ignore_index.
        """
        return self.lookup_table[label].astype(np.uint8)


class CityscapesToUnified(MappingStrategy):
    """
    Map Cityscapes train IDs to the unified label space.
    """
    
    MAPPING = {
        # Flat
        CityscapesTrainID.ROAD: UnifiedTrainID.ROAD,
        CityscapesTrainID.SIDEWALK: UnifiedTrainID.SIDEWALK,
        
        # Construction
        CityscapesTrainID.BUILDING: UnifiedTrainID.BUILDING,
        CityscapesTrainID.WALL: UnifiedTrainID.WALL,
        CityscapesTrainID.FENCE: UnifiedTrainID.FENCE,
        
        # Object
        CityscapesTrainID.POLE: UnifiedTrainID.POLE,
        CityscapesTrainID.TRAFFIC_LIGHT: UnifiedTrainID.TRAFFIC_LIGHT,
        CityscapesTrainID.TRAFFIC_SIGN: UnifiedTrainID.TRAFFIC_SIGN,
        
        # Nature
        CityscapesTrainID.VEGETATION: UnifiedTrainID.VEGETATION,
        CityscapesTrainID.TERRAIN: UnifiedTrainID.TERRAIN,
        CityscapesTrainID.SKY: UnifiedTrainID.SKY,
        
        # Human
        CityscapesTrainID.PERSON: UnifiedTrainID.PERSON,
        CityscapesTrainID.RIDER: UnifiedTrainID.BICYCLIST,  # Map generic rider to bicyclist
        
        # Vehicle
        CityscapesTrainID.CAR: UnifiedTrainID.CAR,
        CityscapesTrainID.TRUCK: UnifiedTrainID.TRUCK,
        CityscapesTrainID.BUS: UnifiedTrainID.BUS,
        CityscapesTrainID.TRAIN: UnifiedTrainID.TRAIN,
        CityscapesTrainID.MOTORCYCLE: UnifiedTrainID.MOTORCYCLE,
        CityscapesTrainID.BICYCLE: UnifiedTrainID.BICYCLE,
    }
    
    def __init__(self, ignore_index: int = 255):
        super().__init__(ignore_index)
        self.lookup_table = self.create_lookup_table(256)
    
    def create_lookup_table(self, source_num_classes: int = 256) -> np.ndarray:
        """Create lookup table for Cityscapes -> Unified mapping"""
        lut = np.full(source_num_classes, self.ignore_index, dtype=np.uint8)
        for cs_id, unified_id in self.MAPPING.items():
            if cs_id != CityscapesTrainID.IGNORE:
                lut[cs_id] = unified_id
        return lut
    
    def transform_label(self, label: np.ndarray) -> np.ndarray:
        """Transform Cityscapes label to unified format"""
        return self.lookup_table[label].astype(np.uint8)


class MapillaryToUnified(MappingStrategy):
    """
    Map Mapillary Vistas IDs to the unified label space.
    """
    
    MAPPING = {
        # Flat
        MapillaryID.ROAD: UnifiedTrainID.ROAD,
        MapillaryID.BIKE_LANE: UnifiedTrainID.BIKE_LANE,
        MapillaryID.SERVICE_LANE: UnifiedTrainID.ROAD,
        MapillaryID.SIDEWALK: UnifiedTrainID.SIDEWALK,
        MapillaryID.CURB_CUT: UnifiedTrainID.SIDEWALK,
        MapillaryID.PEDESTRIAN_AREA: UnifiedTrainID.SIDEWALK,
        MapillaryID.CURB: UnifiedTrainID.BARRIER,
        MapillaryID.PARKING: UnifiedTrainID.PARKING,
        MapillaryID.RAIL_TRACK: UnifiedTrainID.RAIL_TRACK,
        MapillaryID.CROSSWALK_PLAIN: UnifiedTrainID.CROSSWALK,
        
        # Construction
        MapillaryID.BUILDING: UnifiedTrainID.BUILDING,
        MapillaryID.WALL: UnifiedTrainID.WALL,
        MapillaryID.FENCE: UnifiedTrainID.FENCE,
        MapillaryID.GUARD_RAIL: UnifiedTrainID.GUARD_RAIL,
        MapillaryID.BARRIER: UnifiedTrainID.BARRIER,
        MapillaryID.BRIDGE: UnifiedTrainID.BRIDGE,
        MapillaryID.TUNNEL: UnifiedTrainID.TUNNEL,
        
        # Object
        MapillaryID.POLE: UnifiedTrainID.POLE,
        MapillaryID.TRAFFIC_SIGN_FRAME: UnifiedTrainID.POLE,
        MapillaryID.UTILITY_POLE: UnifiedTrainID.UTILITY_POLE,
        MapillaryID.TRAFFIC_LIGHT: UnifiedTrainID.TRAFFIC_LIGHT,
        MapillaryID.TRAFFIC_SIGN_FRONT: UnifiedTrainID.TRAFFIC_SIGN,
        MapillaryID.TRAFFIC_SIGN_BACK: UnifiedTrainID.TRAFFIC_SIGN,
        MapillaryID.STREET_LIGHT: UnifiedTrainID.STREET_LIGHT,
        MapillaryID.BANNER: UnifiedTrainID.OTHER_OBJECT,
        MapillaryID.BENCH: UnifiedTrainID.OTHER_OBJECT,
        MapillaryID.BIKE_RACK: UnifiedTrainID.OTHER_OBJECT,
        MapillaryID.BILLBOARD: UnifiedTrainID.OTHER_OBJECT,
        MapillaryID.CATCH_BASIN: UnifiedTrainID.OTHER_OBJECT,
        MapillaryID.CCTV_CAMERA: UnifiedTrainID.OTHER_OBJECT,
        MapillaryID.FIRE_HYDRANT: UnifiedTrainID.OTHER_OBJECT,
        MapillaryID.JUNCTION_BOX: UnifiedTrainID.OTHER_OBJECT,
        MapillaryID.MAILBOX: UnifiedTrainID.OTHER_OBJECT,
        MapillaryID.MANHOLE: UnifiedTrainID.OTHER_OBJECT,
        MapillaryID.PHONE_BOOTH: UnifiedTrainID.OTHER_OBJECT,
        MapillaryID.POTHOLE: UnifiedTrainID.OTHER_OBJECT,
        MapillaryID.TRASH_CAN: UnifiedTrainID.OTHER_OBJECT,
        
        # Nature
        MapillaryID.VEGETATION: UnifiedTrainID.VEGETATION,
        MapillaryID.TERRAIN: UnifiedTrainID.TERRAIN,
        MapillaryID.SAND: UnifiedTrainID.TERRAIN,
        MapillaryID.SKY: UnifiedTrainID.SKY,
        MapillaryID.WATER: UnifiedTrainID.WATER,
        MapillaryID.SNOW: UnifiedTrainID.SNOW,
        MapillaryID.MOUNTAIN: UnifiedTrainID.MOUNTAIN,
        
        # Human
        MapillaryID.PERSON: UnifiedTrainID.PERSON,
        MapillaryID.BICYCLIST: UnifiedTrainID.BICYCLIST,
        MapillaryID.MOTORCYCLIST: UnifiedTrainID.MOTORCYCLIST,
        MapillaryID.OTHER_RIDER: UnifiedTrainID.OTHER_RIDER,
        
        # Vehicle
        MapillaryID.CAR: UnifiedTrainID.CAR,
        MapillaryID.TRUCK: UnifiedTrainID.TRUCK,
        MapillaryID.BUS: UnifiedTrainID.BUS,
        MapillaryID.ON_RAILS: UnifiedTrainID.TRAIN,
        MapillaryID.MOTORCYCLE: UnifiedTrainID.MOTORCYCLE,
        MapillaryID.BICYCLE: UnifiedTrainID.BICYCLE,
        MapillaryID.CARAVAN: UnifiedTrainID.CARAVAN,
        MapillaryID.TRAILER: UnifiedTrainID.TRAILER,
        MapillaryID.BOAT: UnifiedTrainID.BOAT,
        MapillaryID.OTHER_VEHICLE: UnifiedTrainID.OTHER_VEHICLE,
        MapillaryID.WHEELED_SLOW: UnifiedTrainID.WHEELED_SLOW,
        
        # Animal
        MapillaryID.BIRD: UnifiedTrainID.ANIMAL,
        MapillaryID.GROUND_ANIMAL: UnifiedTrainID.ANIMAL,
        
        # Markings
        MapillaryID.LANE_MARKING_GENERAL: UnifiedTrainID.LANE_MARKING,
        MapillaryID.LANE_MARKING_CROSSWALK: UnifiedTrainID.CROSSWALK,
        
        # Void
        MapillaryID.CAR_MOUNT: UnifiedTrainID.IGNORE,
        MapillaryID.EGO_VEHICLE: UnifiedTrainID.IGNORE,
        MapillaryID.UNLABELED: UnifiedTrainID.IGNORE,
    }
    
    def __init__(self, ignore_index: int = 255):
        super().__init__(ignore_index)
        # Use 256 to cover all possible uint8 values (0-255)
        # Mapillary has 66 official classes but labels can contain higher values
        self.lookup_table = self.create_lookup_table(256)
    
    def create_lookup_table(self, source_num_classes: int = 256) -> np.ndarray:
        """Create lookup table for Mapillary -> Unified mapping
        
        Args:
            source_num_classes: Size of lookup table (default 256 to handle all uint8 values)
        """
        lut = np.full(source_num_classes, self.ignore_index, dtype=np.uint8)
        for mapillary_id, unified_id in self.MAPPING.items():
            lut[mapillary_id] = unified_id
        return lut
    
    def transform_label(self, label: np.ndarray) -> np.ndarray:
        """Transform Mapillary label to unified format
        
        Note: Handles label values beyond the 66 official Mapillary classes
        by mapping them to ignore_index.
        """
        return self.lookup_table[label].astype(np.uint8)


class UnifiedToCityscapes(MappingStrategy):
    """
    Map unified labels back to Cityscapes 19-class format for evaluation.
    
    This is useful when training on the unified label space but evaluating
    on the standard Cityscapes benchmark.
    """
    
    MAPPING = {
        UnifiedTrainID.ROAD: CityscapesTrainID.ROAD,
        UnifiedTrainID.SIDEWALK: CityscapesTrainID.SIDEWALK,
        UnifiedTrainID.PARKING: CityscapesTrainID.IGNORE,  # Not in Cityscapes eval
        UnifiedTrainID.RAIL_TRACK: CityscapesTrainID.IGNORE,
        UnifiedTrainID.BIKE_LANE: CityscapesTrainID.ROAD,
        
        UnifiedTrainID.BUILDING: CityscapesTrainID.BUILDING,
        UnifiedTrainID.WALL: CityscapesTrainID.WALL,
        UnifiedTrainID.FENCE: CityscapesTrainID.FENCE,
        UnifiedTrainID.GUARD_RAIL: CityscapesTrainID.IGNORE,
        UnifiedTrainID.BRIDGE: CityscapesTrainID.IGNORE,
        UnifiedTrainID.TUNNEL: CityscapesTrainID.IGNORE,
        UnifiedTrainID.BARRIER: CityscapesTrainID.WALL,
        
        UnifiedTrainID.POLE: CityscapesTrainID.POLE,
        UnifiedTrainID.TRAFFIC_LIGHT: CityscapesTrainID.TRAFFIC_LIGHT,
        UnifiedTrainID.TRAFFIC_SIGN: CityscapesTrainID.TRAFFIC_SIGN,
        UnifiedTrainID.STREET_LIGHT: CityscapesTrainID.POLE,
        UnifiedTrainID.UTILITY_POLE: CityscapesTrainID.POLE,
        UnifiedTrainID.OTHER_OBJECT: CityscapesTrainID.IGNORE,
        
        UnifiedTrainID.VEGETATION: CityscapesTrainID.VEGETATION,
        UnifiedTrainID.TERRAIN: CityscapesTrainID.TERRAIN,
        UnifiedTrainID.SKY: CityscapesTrainID.SKY,
        UnifiedTrainID.WATER: CityscapesTrainID.IGNORE,
        UnifiedTrainID.SNOW: CityscapesTrainID.IGNORE,
        UnifiedTrainID.MOUNTAIN: CityscapesTrainID.TERRAIN,
        
        UnifiedTrainID.PERSON: CityscapesTrainID.PERSON,
        UnifiedTrainID.BICYCLIST: CityscapesTrainID.RIDER,
        UnifiedTrainID.MOTORCYCLIST: CityscapesTrainID.RIDER,
        UnifiedTrainID.OTHER_RIDER: CityscapesTrainID.RIDER,
        
        UnifiedTrainID.CAR: CityscapesTrainID.CAR,
        UnifiedTrainID.TRUCK: CityscapesTrainID.TRUCK,
        UnifiedTrainID.BUS: CityscapesTrainID.BUS,
        UnifiedTrainID.TRAIN: CityscapesTrainID.TRAIN,
        UnifiedTrainID.MOTORCYCLE: CityscapesTrainID.MOTORCYCLE,
        UnifiedTrainID.BICYCLE: CityscapesTrainID.BICYCLE,
        UnifiedTrainID.CARAVAN: CityscapesTrainID.IGNORE,
        UnifiedTrainID.TRAILER: CityscapesTrainID.IGNORE,
        UnifiedTrainID.BOAT: CityscapesTrainID.IGNORE,
        UnifiedTrainID.OTHER_VEHICLE: CityscapesTrainID.IGNORE,
        UnifiedTrainID.WHEELED_SLOW: CityscapesTrainID.IGNORE,
        UnifiedTrainID.ANIMAL: CityscapesTrainID.IGNORE,
        
        UnifiedTrainID.LANE_MARKING: CityscapesTrainID.ROAD,
        UnifiedTrainID.CROSSWALK: CityscapesTrainID.SIDEWALK,
    }
    
    def __init__(self, ignore_index: int = 255):
        super().__init__(ignore_index)
        self.lookup_table = self.create_lookup_table(256)
    
    def create_lookup_table(self, source_num_classes: int = 256) -> np.ndarray:
        """Create lookup table for Unified -> Cityscapes mapping"""
        lut = np.full(source_num_classes, self.ignore_index, dtype=np.uint8)
        for unified_id, cityscapes_id in self.MAPPING.items():
            if unified_id != UnifiedTrainID.IGNORE:
                lut[unified_id] = cityscapes_id
        return lut
    
    def transform_label(self, label: np.ndarray) -> np.ndarray:
        """Transform unified label to Cityscapes format"""
        return self.lookup_table[label].astype(np.uint8)


# =============================================================================
# LABEL UNIFICATION MANAGER
# =============================================================================

class LabelUnificationManager:
    """
    Manager class for handling label unification across datasets.
    
    Provides utilities for:
    - Converting between different label spaces
    - Generating palette configurations
    - Creating dataset configurations for joint training
    """
    
    # Number of classes for different label spaces
    NUM_CITYSCAPES_CLASSES = 19
    NUM_MAPILLARY_CLASSES = 66
    NUM_UNIFIED_CLASSES = 42
    
    def __init__(self, target_space: str = 'cityscapes'):
        """
        Initialize the label unification manager.
        
        Args:
            target_space: Target label space for unification.
                Options: 'cityscapes', 'unified'
        """
        self.target_space = target_space
        
        # Initialize mappers
        self.mapillary_to_cityscapes = MapillarytoCityscapes()
        self.cityscapes_to_unified = CityscapesToUnified()
        self.mapillary_to_unified = MapillaryToUnified()
        self.unified_to_cityscapes = UnifiedToCityscapes()
    
    def get_mapper(self, source: str, target: str) -> MappingStrategy:
        """
        Get the appropriate mapper for source -> target conversion.
        
        Args:
            source: Source dataset ('cityscapes', 'mapillary', 'unified')
            target: Target label space ('cityscapes', 'unified')
            
        Returns:
            MappingStrategy instance
        """
        mapper_key = f"{source}_to_{target}"
        mappers = {
            'mapillary_to_cityscapes': self.mapillary_to_cityscapes,
            'cityscapes_to_unified': self.cityscapes_to_unified,
            'mapillary_to_unified': self.mapillary_to_unified,
            'unified_to_cityscapes': self.unified_to_cityscapes,
        }
        
        if mapper_key not in mappers:
            raise ValueError(f"No mapper available for {source} -> {target}")
        
        return mappers[mapper_key]
    
    def transform_label(self, label: np.ndarray, source: str, target: str) -> np.ndarray:
        """
        Transform a label map from source to target label space.
        
        Args:
            label: Label map as numpy array
            source: Source dataset
            target: Target label space
            
        Returns:
            Transformed label map
        """
        mapper = self.get_mapper(source, target)
        return mapper.transform_label(label)
    
    @staticmethod
    def get_cityscapes_classes() -> List[str]:
        """Get list of Cityscapes class names in train ID order"""
        classes = [''] * 19
        for cls in CITYSCAPES_CLASSES:
            if 0 <= cls.train_id < 19:
                classes[cls.train_id] = cls.name
        return classes
    
    @staticmethod
    def get_cityscapes_palette() -> List[List[int]]:
        """Get Cityscapes color palette in train ID order"""
        palette = [[0, 0, 0]] * 19
        for cls in CITYSCAPES_CLASSES:
            if 0 <= cls.train_id < 19:
                palette[cls.train_id] = list(cls.color)
        return palette
    
    @staticmethod
    def get_mapillary_classes() -> List[str]:
        """Get list of Mapillary class names"""
        return [cls.name for cls in MAPILLARY_CLASSES]
    
    @staticmethod
    def get_mapillary_palette() -> List[List[int]]:
        """Get Mapillary color palette"""
        return [list(cls.color) for cls in MAPILLARY_CLASSES]
    
    @staticmethod
    def get_unified_classes() -> List[str]:
        """Get list of unified class names"""
        classes = [''] * 42
        for cls in UNIFIED_CLASSES:
            classes[cls.id] = cls.name
        return classes
    
    @staticmethod
    def get_unified_palette() -> List[List[int]]:
        """Get unified color palette"""
        palette = [[0, 0, 0]] * 42
        for cls in UNIFIED_CLASSES:
            palette[cls.id] = list(cls.color)
        return palette
    
    def get_num_classes(self, label_space: str) -> int:
        """Get number of classes for a label space"""
        num_classes = {
            'cityscapes': self.NUM_CITYSCAPES_CLASSES,
            'mapillary': self.NUM_MAPILLARY_CLASSES,
            'unified': self.NUM_UNIFIED_CLASSES,
        }
        return num_classes.get(label_space, 19)
    
    def generate_mmseg_dataset_config(self, 
                                       cityscapes_root: str,
                                       mapillary_root: str,
                                       target_space: str = 'cityscapes') -> Dict:
        """
        Generate MMSegmentation dataset configuration for joint training.
        
        Args:
            cityscapes_root: Path to Cityscapes dataset
            mapillary_root: Path to Mapillary Vistas dataset
            target_space: Target label space ('cityscapes' or 'unified')
            
        Returns:
            Dictionary with dataset configuration
        """
        if target_space == 'cityscapes':
            classes = self.get_cityscapes_classes()
            palette = self.get_cityscapes_palette()
            num_classes = self.NUM_CITYSCAPES_CLASSES
        else:
            classes = self.get_unified_classes()
            palette = self.get_unified_palette()
            num_classes = self.NUM_UNIFIED_CLASSES
        
        config = {
            'dataset_type': 'UnifiedSegDataset',
            'num_classes': num_classes,
            'classes': tuple(classes),
            'palette': palette,
            'reduce_zero_label': False,
            
            'data': {
                'samples_per_gpu': 2,
                'workers_per_gpu': 4,
                
                'train': {
                    'type': 'ConcatDataset',
                    'datasets': [
                        # Cityscapes training data
                        {
                            'type': 'CityscapesDataset' if target_space == 'cityscapes' else 'UnifiedCityscapesDataset',
                            'data_root': cityscapes_root,
                            'img_dir': 'leftImg8bit/train',
                            'ann_dir': 'gtFine/train',
                            'seg_map_suffix': '_gtFine_labelTrainIds.png',
                        },
                        # Mapillary training data
                        {
                            'type': 'MapillaryUnifiedDataset',
                            'data_root': mapillary_root,
                            'img_dir': 'training/images',
                            'ann_dir': 'training/v1.2/labels',
                            'target_space': target_space,
                        }
                    ]
                },
                
                'val': {
                    'type': 'CityscapesDataset',
                    'data_root': cityscapes_root,
                    'img_dir': 'leftImg8bit/val',
                    'ann_dir': 'gtFine/val',
                    'seg_map_suffix': '_gtFine_labelTrainIds.png',
                },
                
                'test': {
                    'type': 'CityscapesDataset',
                    'data_root': cityscapes_root,
                    'img_dir': 'leftImg8bit/val',
                    'ann_dir': 'gtFine/val',
                    'seg_map_suffix': '_gtFine_labelTrainIds.png',
                }
            }
        }
        
        return config


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def convert_mapillary_labels_to_cityscapes(input_dir: str, output_dir: str) -> None:
    """
    Batch convert Mapillary label files to Cityscapes format.
    
    Args:
        input_dir: Directory containing Mapillary label PNG files
        output_dir: Output directory for converted labels
    """
    import os
    from PIL import Image
    
    mapper = MapillarytoCityscapes()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            # Load label
            label_path = os.path.join(input_dir, filename)
            label = np.array(Image.open(label_path))
            
            # Transform
            transformed = mapper.transform_label(label)
            
            # Save
            output_path = os.path.join(output_dir, filename)
            Image.fromarray(transformed).save(output_path)
            
    print(f"Converted {len(os.listdir(input_dir))} labels to {output_dir}")


def create_unified_label_file(cityscapes_label: np.ndarray = None,
                               mapillary_label: np.ndarray = None,
                               source: str = 'cityscapes') -> np.ndarray:
    """
    Convert a label to the unified label space.
    
    Args:
        cityscapes_label: Cityscapes label map (if source is 'cityscapes')
        mapillary_label: Mapillary label map (if source is 'mapillary')
        source: Source dataset type
        
    Returns:
        Label map in unified label space
    """
    manager = LabelUnificationManager()
    
    if source == 'cityscapes' and cityscapes_label is not None:
        return manager.transform_label(cityscapes_label, 'cityscapes', 'unified')
    elif source == 'mapillary' and mapillary_label is not None:
        return manager.transform_label(mapillary_label, 'mapillary', 'unified')
    else:
        raise ValueError("Must provide label for the specified source")


def print_mapping_summary():
    """Print a summary of the label mappings"""
    print("=" * 80)
    print("LABEL UNIFICATION MAPPING SUMMARY")
    print("=" * 80)
    
    print("\n--- Cityscapes Classes (19 evaluation classes) ---")
    for cls in CITYSCAPES_CLASSES:
        if 0 <= cls.train_id < 19:
            print(f"  {cls.train_id:2d}: {cls.name}")
    
    print("\n--- Mapillary Vistas Classes (66 classes) ---")
    for cls in MAPILLARY_CLASSES:
        print(f"  {cls.id:2d}: {cls.name} ({cls.category})")
    
    print("\n--- Unified Classes (42 classes) ---")
    for cls in UNIFIED_CLASSES:
        cs_str = ', '.join(str(x) for x in cls.cityscapes_ids) if cls.cityscapes_ids else '-'
        mp_str = ', '.join(str(x) for x in cls.mapillary_ids) if cls.mapillary_ids else '-'
        print(f"  {cls.id:2d}: {cls.name:20s} | CS: [{cs_str:10s}] | MP: [{mp_str}]")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print_mapping_summary()
    
    # Example usage
    print("\n" + "=" * 80)
    print("EXAMPLE USAGE")
    print("=" * 80)
    
    # Create manager
    manager = LabelUnificationManager()
    
    # Example: Create a fake Mapillary label and convert it
    fake_mapillary_label = np.array([[13, 15, 17], [19, 55, 30]], dtype=np.uint8)
    print(f"\nOriginal Mapillary label:\n{fake_mapillary_label}")
    
    # Convert to Cityscapes
    cityscapes_label = manager.transform_label(fake_mapillary_label, 'mapillary', 'cityscapes')
    print(f"\nConverted to Cityscapes:\n{cityscapes_label}")
    
    # Convert to Unified
    unified_label = manager.transform_label(fake_mapillary_label, 'mapillary', 'unified')
    print(f"\nConverted to Unified:\n{unified_label}")
    
    # Get class names
    print(f"\nCityscapes classes: {manager.get_cityscapes_classes()[:5]}...")
    print(f"Unified classes: {manager.get_unified_classes()[:5]}...")
