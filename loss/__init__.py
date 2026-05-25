from mmengine.registry import Registry
OPENOCC_LOSS = Registry('openocc_loss')

from .multi_loss import MultiLoss
from .occupancy_loss import OccupancyLoss
from .occupancy_loss_offline import OccupancyLossOffline
from .bce_loss import BinaryCrossEntropyLoss, PixelDistributionLoss
