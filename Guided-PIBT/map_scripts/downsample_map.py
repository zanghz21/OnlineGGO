import numpy as np
from .maps import TrafficMapfMap, save_mapfile_from_np

map_path = 'guided-pibt/benchmark-lifelong/maps/warehouse_large.map'
traffic_map = TrafficMapfMap(map_path)

downsample_map_np = traffic_map.map_np[::2, ::2]

tgt_map_path = 'guided-pibt/benchmark-lifelong/maps/warehouse_downsample.map'

save_mapfile_from_np(downsample_map_np, tgt_map_path)
