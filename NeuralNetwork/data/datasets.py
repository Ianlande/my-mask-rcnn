#-*- coding:utf-8 -*-

import bisect

import torch
import torchvision
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from NeuralNetwork.structures.bounding_box import BoxList
from NeuralNetwork.structures.segmentation_mask import SegmentationMask


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)
            
        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    """

    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_img_info(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_img_info(sample_idx)


class AbstractDataset(torch.utils.data.Dataset):
    """
    Serves as a common interface to reduce boilerplate and help dataset
    customization

    A generic Dataset for the NeuralNetwork must have the following
    non-trivial fields / methods implemented:
        CLASSES - list/tuple:
            A list of strings representing the classes. It must have
            "__background__" as its 0th element for correct id mapping.

        __getitem__ - function(idx):
            This has to return three things: img, target, idx.
            img is the input image, which has to be load as a PIL Image object
            implementing the target requires the most effort, since it must have
            multiple fields: the size, bounding boxes, labels (contiguous), and
            masks (either COCO-style Polygons, RLE or torch BinaryMask).
            Usually the target is a BoxList instance with extra fields.
            Lastly, idx is simply the input argument of the function.

    also the following is required:
        __len__ - function():
            return the size of the dataset
        get_img_info - function(idx):
            return metadata, at least width and height of the input image
    """

    def __init__(self, *args, **kwargs):
        self.name_to_id = None
        self.id_to_name = None


    def __getitem__(self, idx):
        raise NotImplementedError


    def initMaps(self):
        """
        Can be called optionally to initialize the id<->category name mapping


        Initialize default mapping between:
            class <==> index
        class: this is a string that represents the class
        index: positive int, used directly by the ROI heads.


        NOTE:
            make sure that the background is always indexed by 0.
            "__background__" <==> 0

            if initialized by hand, double check that the indexing is correct.
        """
        assert isinstance(self.CLASSES, (list, tuple))
        assert self.CLASSES[0] == "__background__"
        cls = self.CLASSES
        self.name_to_id = dict(zip(cls, range(len(cls))))
        self.id_to_name = dict(zip(range(len(cls)), cls))


    def get_img_info(self, index):
        raise NotImplementedError


    def __len__(self):
        raise NotImplementedError

