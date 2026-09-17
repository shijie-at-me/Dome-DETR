"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.utils.data as data


class DetDataset(data.Dataset):
    """
    A detection dataset is ``load_item`` plus a transform pipeline. ``load_item`` returns the raw
    ``(image, target)`` of one index, which is also what Mosaic and MixUp call to fetch the extra
    samples they mix in; ``__getitem__`` runs the transforms on top. The transforms get the
    dataset itself as a third argument so that epoch-based policies can read ``epoch``.
    """

    transforms = None

    def __getitem__(self, index):
        img, target = self.load_item(index)
        if self.transforms is not None:
            img, target, _ = self.transforms(img, target, self)
        return img, target

    def load_item(self, index):
        raise NotImplementedError("Please implement this function to return item before `transforms`.")

    def set_epoch(self, epoch) -> None:
        self._epoch = epoch

    @property
    def _epoch_policy(self) -> bool:
        """Whether the transform pipeline reads the epoch, so that a loader can refuse to freeze it."""
        return getattr(self.transforms, "policy", {}).get("name", "default") == "stop_epoch"

    @property
    def epoch(self):
        return getattr(self, "_epoch", -1)
