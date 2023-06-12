import numpy as np

from pathlib import Path
import pickle 
import os 
from ..registry import PIPELINES

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def read_file(path, tries=2, num_point_feature=4, virtual=False):
    if virtual:
        # WARNING: hard coded for nuScenes 
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]
        tokens = path.split('/')
        seg_path = os.path.join(*tokens[:-2], tokens[-2]+"_VIRTUAL", tokens[-1]+'.pkl.npy')
        data_dict = np.load(seg_path, allow_pickle=True).item()

        # remove reflectance as other virtual points don't have this value  
        virtual_points1 = data_dict['real_points'][:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] 
        virtual_points2 = data_dict['virtual_points']

        points = np.concatenate([points, np.ones([points.shape[0], 15-num_point_feature])], axis=1)
        virtual_points1 = np.concatenate([virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1)
        virtual_points2 = np.concatenate([virtual_points2, -1 * np.ones([virtual_points2.shape[0], 1])], axis=1)
        points = np.concatenate([points, virtual_points1, virtual_points2], axis=0).astype(np.float32)
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, virtual=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), virtual=virtual).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T

def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)
    
    return points 

def read_single_waymo_sweep(sweep):
    obj = get_obj(sweep['path'])

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    
    return points_sweep.T, curr_times.T


def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), virtual=res["virtual"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, virtual=res["virtual"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])
        
        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points

            if nsweeps > 1: 
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])
        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
        elif res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }
        else:
            pass 

        return res, info
