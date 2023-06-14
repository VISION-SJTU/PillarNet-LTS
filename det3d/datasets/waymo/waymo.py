import pickle
import numpy as np

from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.registry import DATASETS


@DATASETS.register_module
class WaymoDataset(PointCloudDataset):
    NumPointFeatures = 5  # x, y, z, intensity, elongation

    def __init__(
        self,
        info_path,
        root_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        sample=False,
        nsweeps=1,
        load_interval=1,
        use_cbgs=False,
        **kwargs,
    ):
        self.load_interval = load_interval 
        self.sample = sample
        self.nsweeps = nsweeps
        self.use_cbgs = use_cbgs
        print("Using {} sweeps".format(nsweeps))
        super(WaymoDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )
        self._num_point_features = WaymoDataset.NumPointFeatures if nsweeps == 1 else WaymoDataset.NumPointFeatures+1

    def reset(self):
        raise NotImplementedError

    def load_infos(self, info_path):

        with open(self._info_path, "rb") as f:
            _waymo_infos_all = pickle.load(f)
        _waymo_infos_all = _waymo_infos_all[::self.load_interval]
        
        if not self.test_mode and self.use_cbgs:
            _cls_infos = {name: [] for name in self._class_names}
            for info in _waymo_infos_all:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
            _cls_dist = {k: len(v) / max(duplicated_samples, 1) for k, v in _cls_infos.items()}

            self._waymo_infos = []

            frac = 1.0 / len(self._class_names)
            ratios = [frac / v for v in _cls_dist.values()]

            for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
                self._waymo_infos += np.random.choice(
                    cls_infos, int(len(cls_infos) * ratio)
                ).tolist()
        else:
            self._waymo_infos = _waymo_infos_all

        print("Using {} Frames".format(len(self._waymo_infos)))

    def __len__(self):

        if not hasattr(self, "_waymo_infos"):
            self.load_infos(self._info_path)

        return len(self._waymo_infos)

    def get_sensor_data(self, idx):
        info = self._waymo_infos[idx]

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "annotations": None,
                "nsweeps": self.nsweeps, 
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "type": "WaymoDataset",
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def evaluation(self, detections, output_dir=None, testset=False):
        from .waymo_common import _create_pd_detection, reorganize_info

        infos = self._waymo_infos 
        infos = reorganize_info(infos)

        _create_pd_detection(detections, infos, output_dir)

        print("use waymo devkit tool for evaluation")

        return None, None 

