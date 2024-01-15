# ----------------------------------------------
# Created by Wei-Jie Huang
# ----------------------------------------------
import copy
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset

from datasets.coco import CocoDetection, make_coco_transforms, ConvertCocoPolysToMask
from util.misc import get_local_rank, get_local_size, nested_tensor_from_tensor_list


def get_paths(root):
    root = Path(root)
    return {
        'cityscapes': {
            'train_img': root / 'cityscapes/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'cityscapes/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'foggy_cityscapes': {
            'train_img': root / 'cityscapes_foggy/leftImg8bit/train',
            'train_anno': root / 'cityscapes_foggy/annotations/foggy_cityscapes_train.json',
            'val_img': root / 'cityscapes_foggy/leftImg8bit/val',
            'val_anno': root / 'cityscapes_foggy/annotations/foggy_cityscapes_val.json',
        },
        'interp_cityscapes_to_foggy_cityscapes': {
            'train_img': root / 'city_instruction_cityfoggy/train',  # FIXME: modify path
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'city_instruction_cityfoggy/val',  # FIXME: modify path
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'sim10k': {
            'train_img': root / 'sim10k/VOC2012/JPEGImages',
            'train_anno': root / 'sim10k/annotations/sim10k_caronly.json',
        },
        'interp_sim10k_to_cityscapes_caronly': {
            'train_img': root / 'sim_instruction_cityscapes/VOC2012/JPEGImages',  # FIXME: modify path
            'train_anno': root / 'sim10k/annotations/sim10k_caronly.json',
        },
        'bdd_daytime': {
            'train_img': root / 'bdd_daytime/train',
            'train_anno': root / 'bdd_daytime/annotations/bdd_daytime_train.json',
            'val_img': root / 'bdd_daytime/val',
            'val_anno': root / 'bdd_daytime/annotations/bdd_daytime_val.json',
        },
        'interp_cityscapes_to_bdd_daytime': {
            'train_img': root / 'city_instruction_bdd/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'city_instruction_bdd/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
    }


class DADataset(Dataset):
    def __init__(self, source_img_folder, source_ann_file, target_img_folder, target_ann_file,
                 transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):

        # for source data
        self.source = CocoDetection(
            img_folder=source_img_folder,
            ann_file=source_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size
        )

        # for target data
        self.target = CocoDetection(
            img_folder=target_img_folder,
            ann_file=target_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size
        )

    def __len__(self):
        return max(len(self.source), len(self.target))

    def __getitem__(self, idx):
        source_img, source_target = self.source[idx % len(self.source)]
        target_img, target_target = self.target[idx % len(self.target)]

        return source_img, target_img, source_target, target_target


def collate_fn(batch):
    source_imgs, target_imgs, source_targets, target_targets = list(zip(*batch))
    samples = nested_tensor_from_tensor_list(source_imgs + target_imgs)
    
    targets = source_targets + target_targets

    return samples, targets


class MixupDADataset(Dataset):
    def __init__(
        self,
        src_img_folder, src_ann_file,
        tgt_img_folder, tgt_ann_file,
        int_img_folder, int_domain_label,
        transforms, return_masks,
        cache_mode=False, local_rank=0, local_size=1
    ):
        self.src_img_folder = src_img_folder
        self.src_ann_file = src_ann_file
        self.tgt_img_folder = tgt_img_folder
        self.tgt_ann_file = tgt_ann_file
        self.int_img_folder = int_img_folder
        self.int_domain_label = int_domain_label

        self._transforms = transforms
        self.return_masks = return_masks

        self.src = CocoDetection(
            img_folder=src_img_folder,
            ann_file=src_ann_file,
            transforms=None,  # make sure the outputted image is not transformed
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
            to_tensor_first=True
        )

        self.tgt = CocoDetection(
            img_folder=tgt_img_folder,
            ann_file=tgt_ann_file,
            transforms=None,  # make sure the outputted image is not transformed
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
            to_tensor_first=True
        )

        self.int = CocoDetection(
            img_folder=int_img_folder,
            ann_file=src_ann_file,
            transforms=None,  # make sure the outputted image is not transformed
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
            to_tensor_first=True
        )

        assert len(self.src) == len(self.int), f'#src={len(self.src)}, #int={len(self.int)}'
        assert self.src.ids == self.int.ids

    def __len__(self):
        return max(len(self.src), len(self.tgt))

    def __getitem__(self, idx):
        src_img, src_target = self.src[idx % len(self.src)]
        tgt_img, tgt_target = self.tgt[idx % len(self.tgt)]
        int_img, int_target = self.int[idx % len(self.int)]

        return src_img, int_img, tgt_img, src_target, tgt_target

    def transform(self, img, target):
        img, target = self._transforms(img, target)

        return img, target


def collate_fn_mixup(batch):
    src_imgs, tgt_imgs, int_imgs, src_targets, tgt_targets = list(zip(*batch))
    samples = nested_tensor_from_tensor_list(src_imgs + int_imgs + tgt_imgs)
    targets = src_targets + tgt_targets

    return samples, targets


# wrapper
def build(image_set, cfg):
    paths = get_paths(cfg.DATASET.COCO_PATH)
    source_domain, target_domain = cfg.DATASET.DATASET_FILE.split('_to_')

    if 'bdd' in target_domain:
        assert cfg.DATASET.CATEGORY_IDS == [1, 2, 3, 4, 5, 7, 8], 'BDD has no category `train`'
    
    if image_set in ['val', 'val_target']:
        print(f'Val target: {target_domain}')
        return CocoDetection(
            img_folder=paths[target_domain]['val_img'],
            ann_file=paths[target_domain]['val_anno'],
            transforms=make_coco_transforms(image_set),
            return_masks=cfg.MODEL.MASKS,
            cache_mode=cfg.CACHE_MODE,
            local_rank=get_local_rank(),
            local_size=get_local_size()
        )

    elif image_set == 'val_source':
        # sim10k has only training set, no validation set
        if source_domain == 'sim10k':
            return None

        print(f'Val source: {source_domain}')  # FIXME: delete it
        return CocoDetection(
            img_folder=paths[source_domain]['val_img'],
            ann_file=paths[source_domain]['val_anno'],
            transforms=make_coco_transforms(image_set),
            return_masks=cfg.MODEL.MASKS,
            cache_mode=cfg.CACHE_MODE,
            local_rank=get_local_rank(),
            local_size=get_local_size()
        )

    elif image_set == 'train':
        if cfg.DATASET.DA_MODE == 'source_only':
            return CocoDetection(
                img_folder=paths[source_domain]['train_img'],
                ann_file=paths[source_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        elif cfg.DATASET.DA_MODE == 'oracle' or cfg.DATASET.DA_MODE == 'target_only':
            return CocoDetection(
                img_folder=paths[target_domain]['train_img'],
                ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            
        elif cfg.DATASET.DA_MODE == 'uda':
            return DADataset(
                source_img_folder=paths[source_domain]['train_img'],
                source_ann_file=paths[source_domain]['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

        elif cfg.DATASET.DA_MODE == 'blenda':
            interp_domain = f'interp_{cfg.DATASET.DATASET_FILE}'
            dataset = MixupDADataset(
                src_img_folder=paths[source_domain]['train_img'],
                src_ann_file=paths[source_domain]['train_anno'],
                tgt_img_folder=paths[target_domain]['train_img'],
                tgt_ann_file=paths[target_domain]['train_anno'],
                int_img_folder=paths[interp_domain]['train_img'],
                int_domain_label=1,
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            return dataset

        else:
            raise ValueError(f'Unknown argument cfg.DATASET.DA_MODE {cfg.DATASET.DA_MODE}')
    raise ValueError(f'unknown image set {image_set}')
