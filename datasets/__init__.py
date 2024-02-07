from .kitti_dataset_1215 import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .ds_dataset import DSDataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    'drivingstereo': DSDataset
}
