# depth_dataset.py
from ultralytics.data.dataset import YOLODataset
from train_ppcm import DepthCache
import torch


class YOLODatasetWithDepth(YOLODataset):
    """YOLODataset + depth. augment off 라 파일명 매칭 유효."""
    def __init__(self, *args, depth_cache_root=None, dataset_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_cache = DepthCache(depth_cache_root, dataset_name) \
            if depth_cache_root else None

    def __getitem__(self, index):
        item = super().__getitem__(index)   # 기존 dict (img, im_file, ...)
        if self.depth_cache is not None:
            H, W = item["img"].shape[-2:]
            item["depth"] = self.depth_cache.get(item["im_file"], target_hw=(H, W))
        return item

    @staticmethod
    def collate_fn(batch):
        """부모 collate 로 나머지 처리 후, depth 만 별도 스택."""
        has_depth = "depth" in batch[0]
        depths = [b.pop("depth") for b in batch] if has_depth else None
        out = YOLODataset.collate_fn(batch)
        if has_depth:
            out["depth"] = torch.stack(depths, 0)
        return out