from torch.utils.data import DataLoader
import torch
import  logging
from typing import  Dict, Union
from reclib.data.fields import MetadataField
from reclib.data.dataset import Batch
logger = logging.getLogger(__name__)

TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]


def add_epoch_number(batch: Batch, epoch: int) -> Batch:
    """
    Add the epoch number to the batch instances as a MetadataField.
    """
    for instance in batch.instances:
        instance.fields["epoch_num"] = MetadataField(epoch)
    return batch


class DataIterator(DataLoader):
    def __init__(self):
        pass