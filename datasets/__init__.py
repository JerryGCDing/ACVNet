from .kitti_dataset_1215 import VoxelKITTIDataset, KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .ds_dataset import DSDataset, VoxelDSDatasetCalib

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    'drivingstereo': DSDataset
}
