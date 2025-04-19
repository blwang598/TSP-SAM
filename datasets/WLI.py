import os
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler, BatchSampler
from collections import OrderedDict

from datasets.tools import ResizeAndPad, soft_transform, collate_fn, collate_fn_soft, collate_fn_train, collate_fn_, decode_mask


class WLIDataset(Dataset):
    def __init__(self, cfg, file_txt, transform=None, if_self_training=False):
        self.cfg = cfg 
        f = open(file_txt, 'r')
        self.name_list = sorted(f.readlines()) 
        self.name = [a.split()[0] for a in self.name_list]
        self.bl_label = [a.split()[1] for a in self.name_list] 
 
        self.transform = transform
        self.images_class = [0, 1] 

        self.if_self_training = if_self_training

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        content = name.split()
        image_path, bingli_label, fenhua_label, shendu_label = content[0], int(content[1]), int(content[2]), int(content[3])
        name = image_path.split("/")[-1] 
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, c = image.shape 
        gt_path = image_path[:-4] + "_mask.png" 
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) 

        masks = []
        bboxes = []
        categories = []
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()
        delta = 100
        for mask in gt_masks:
            masks.append(mask) 
            x, y, w, h = cv2.boundingRect(mask)
            # delta_x = int(random.uniform(0.05, 0.2) * x)
            # delta_y = int(random.uniform(0.05, 0.2) * y)
            delta_x, delta_y = 0, 0

            w_new = w if x+w+delta_x > W else w+delta_x
            h_new = h if y+h+delta_y > H else h+delta_y 
            bboxes.append([x, y, x + w_new, y + h_new]) 
            categories.append("0") 
 

        if self.if_self_training: 
            image_weak, bboxes_weak, masks_weak = soft_transform(image, bboxes, masks, categories)
            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak)) 

            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0) 
 
            return (image_weak, torch.tensor(bboxes_weak),
                    torch.tensor(masks_weak).float(), bingli_label, fenhua_label, shendu_label, W, H, name)

        elif self.cfg.visual: 
            file_name=name
            origin_image = image
            origin_bboxes = bboxes
            origin_masks = masks
            if self.transform:
                padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), True)

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            origin_bboxes = np.stack(origin_bboxes, axis=0)
            origin_masks = np.stack(origin_masks, axis=0)
            return (file_name, padding, origin_image, origin_bboxes, origin_masks,
                    image, torch.tensor(bboxes), torch.tensor(masks).float(),bingli_label, fenhua_label, shendu_label)

        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float(), bingli_label, fenhua_label, shendu_label, W, H, name


    def resize_scale_bbox(self, mask, expand_range=0.8):
        """
        Generate a new bounding box through random offset and scaling to accommodate uncertainty
        args:
        mask -- .png
        expand_range -- random scale e.g. 120%
        """ 
        x, y, w, h = cv2.boundingRect(mask) 
        expand_factor_w = 1 + random.uniform(0, expand_range)
        expand_factor_h = 1 + random.uniform(0, expand_range) 
        new_w = int(w * expand_factor_w)
        new_h = int(h * expand_factor_h) 
        new_x = max(0, x - (new_w - w) // 2)
        new_y = max(0, y - (new_h - h) // 2) 
        new_w = min(new_x+new_w, mask.shape[1]) - new_x
        new_h = min(new_y+new_h, mask.shape[0]) - new_y 
        offset_range_x = (new_w - w) // 2
        offset_range_y = (new_h - h) // 2 
        random_offset_x = random.randint(-offset_range_x // 2, offset_range_x // 2)
        random_offset_y = random.randint(-offset_range_y // 2, offset_range_y // 2) 
        new_x = max(0, new_x + random_offset_x)
        new_y = max(0, new_y + random_offset_y) 
        new_w = min(new_x+new_w, mask.shape[1]) - new_x
        new_h = min(new_y+new_h, mask.shape[0]) - new_y

        return new_x, new_y, new_w, new_h


class WLIDataset_train(Dataset):
    def __init__(self, cfg, file_txt, transform=None, if_self_training=False):
        self.cfg = cfg 
        f = open(file_txt, 'r')
        self.name_list = sorted(f.readlines()) 
        self.name = [a.split()[0] for a in self.name_list]
        self.bl_label = [a.split()[1] for a in self.name_list]
        self.fh_label = [a.split()[2] for a in self.name_list]
        self.sd_label = [a.split()[3] for a in self.name_list] 
        self.composed_label = [self.bl_label[i] + self.fh_label[i] + self.sd_label[i] for i in range(len(self.name_list))] 
 
        self.transform = transform
        self.images_class = [0, 1] 

        self.if_self_training = if_self_training

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        batch = []
        for idx in index:
            name = self.name_list[idx]
            content = name.split()
            image_path, bingli_label, fenhua_label, shendu_label = content[0], int(content[1]), int(content[2]), int(content[3]) 
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            H, W, c = image.shape 
            gt_path = image_path[:-4] + "_mask.png" 
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            masks = []
            bboxes = []
            categories = []
            gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
            assert gt_masks.sum() == (gt_mask > 0).sum()
            for mask in gt_masks:
                masks.append(mask) 
                x, y, w, h = cv2.boundingRect(mask)
                bboxes.append([x, y, x + w, y + h]) 
                categories.append("0")

            if self.if_self_training: 
                image_weak, bboxes_weak, masks_weak = soft_transform(image, bboxes, masks, categories)
                if self.transform:
                    image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak)) 

                bboxes_weak = np.stack(bboxes_weak, axis=0)
                masks_weak = np.stack(masks_weak, axis=0)
                batch.append((image_weak, torch.tensor(bboxes_weak),
                        torch.tensor(masks_weak).float(), bingli_label, fenhua_label, shendu_label))

            elif self.cfg.visual:
                file_name = os.path.splitext(os.path.basename(name))[0]
                origin_image = image
                origin_bboxes = bboxes
                origin_masks = masks
                if self.transform:
                    padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), True)

                bboxes = np.stack(bboxes, axis=0)
                masks = np.stack(masks, axis=0)
                origin_bboxes = np.stack(origin_bboxes, axis=0)
                origin_masks = np.stack(origin_masks, axis=0)
                return (file_name, padding, origin_image, origin_bboxes, origin_masks,
                        image, torch.tensor(bboxes), torch.tensor(masks).float(),bingli_label, fenhua_label, shendu_label)

            else:
                if self.transform:
                    image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

                bboxes = np.stack(bboxes, axis=0)
                masks = np.stack(masks, axis=0)
                return image, torch.tensor(bboxes), torch.tensor(masks).float(), bingli_label, fenhua_label, shendu_label
        return batch



    def resize_scale_bbox(self, mask, expand_range=0.8):
        """
        Generate a new bounding box through random offset and scaling to accommodate uncertainty
        args:
        mask -- .png
        expand_range -- random scale e.g. 120%
        """ 
        x, y, w, h = cv2.boundingRect(mask) 
        expand_factor_w = 1 + random.uniform(0, expand_range)
        expand_factor_h = 1 + random.uniform(0, expand_range) 
        new_w = int(w * expand_factor_w)
        new_h = int(h * expand_factor_h) 
        new_x = max(0, x - (new_w - w) // 2)
        new_y = max(0, y - (new_h - h) // 2) 
        new_w = min(new_x+new_w, mask.shape[1]) - new_x
        new_h = min(new_y+new_h, mask.shape[0]) - new_y 
        offset_range_x = (new_w - w) // 2
        offset_range_y = (new_h - h) // 2 
        random_offset_x = random.randint(-offset_range_x // 2, offset_range_x // 2)
        random_offset_y = random.randint(-offset_range_y // 2, offset_range_y // 2) 
        new_x = max(0, new_x + random_offset_x)
        new_y = max(0, new_y + random_offset_y) 
        new_w = min(new_x+new_w, mask.shape[1]) - new_x
        new_h = min(new_y+new_h, mask.shape[0]) - new_y

        return new_x, new_y, new_w, new_h



def load_datasets_soft(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = WLIDataset(
        cfg,
        file_txt=cfg.datasets.WLI.val_txt, 
        transform=transform,
    )
    val_train = WLIDataset(
        cfg,
        file_txt=cfg.datasets.WLI.train_txt, 
        transform=transform,
    )

    soft_train = WLIDataset_train(
        cfg, 
        file_txt=cfg.datasets.WLI.train_txt,
        transform=transform,
        if_self_training=True,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    hierarchical_nway_kshot_train_dataloader = hierarchical_nway_kshot_dataloader(soft_train, train_steps=cfg.opt.train_steps)
    return hierarchical_nway_kshot_train_dataloader, val_dataloader




class Singleton(type):
    """
    Define an Instance operation that lets clients access its unique
    instance.
    """

    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class TaskSampler(metaclass=Singleton):
    def __init__(self, unique_classes, n_way, k_shot):
        self.unique_classes = sorted(unique_classes) 
 
        self.n_way = n_way
        self.k_shot = k_shot
        self.counter = 0
        self.sampled_classes = None

    def sample_N_classes_as_a_task(self):
        self.sampled_classes = random.sample(self.unique_classes, self.n_way)
        return self.sampled_classes


class N_Way_K_Shot_BatchSampler(Sampler):
    def __init__(self, y, max_iter, task_sampler): 
        self.y = y 
        self.max_iter = max_iter
        self.task_sampler = task_sampler
        self.label_dict = self.build_label_dict()
        self.batch_size = 8 
        self.unique_classes_from_y = sorted(set(self.y)) 

    def build_label_dict(self):
        label_dict = OrderedDict()
        for i, label in enumerate(self.y): 
            if label not in label_dict:
                label_dict[label] = [i]
            else:
                label_dict[label].append(i)
        return label_dict 

    def sample_examples_by_class(self, cls): 
        if cls not in self.label_dict:
            return [] 
        if self.task_sampler.k_shot <= len(self.label_dict[cls]): 
            sampled_examples = random.sample(self.label_dict[cls], 
                                             self.task_sampler.k_shot) 
        else:
            sampled_examples = random.choices(self.label_dict[cls], 
                                              k=self.task_sampler.k_shot) 
        return sampled_examples

    def __iter__(self):
        for _ in range(self.max_iter):
            batch = []
            classes = self.task_sampler.sample_N_classes_as_a_task() 
            if len(batch) == 0:
                for cls in classes: 
                    samples_for_this_class = self.sample_examples_by_class(cls)
                    batch.extend(samples_for_this_class)

            yield batch 

    def __len__(self):
        return self.max_iter


def nway_kshot_dataloader(dataset, n_way, k_shot, train_steps):
    task_sampler = TaskSampler(set(dataset.bl_label), n_way, k_shot) 
    n_way_k_shot_sampler = N_Way_K_Shot_BatchSampler(dataset.bl_label, train_steps, task_sampler) 
 
    meta_loader = torch.utils.data.DataLoader(dataset, shuffle=False, sampler=n_way_k_shot_sampler, collate_fn=collate_fn_train)
    return meta_loader 


class Hierarchical_TaskSampler(metaclass=Singleton):
    def __init__(self, unique_classes, n_way_list, k_shot_dict): 
 
        self.unique_classes = sorted(unique_classes)
        self.n_way_list = n_way_list
        self.k_shot_dict = k_shot_dict
        self.counter = 0
        self.sampled_classes = None

    def hierarchical_sample_N_classes_as_a_task(self):
        self.sampled_classes = self.unique_classes 
        return self.sampled_classes


class Hierarchical_N_Way_K_Shot_BatchSampler(Sampler):
    def __init__(self, y, max_iter, task_sampler): 
        self.y = y 
        self.max_iter = max_iter
        self.task_sampler = task_sampler
        self.label_dict = self.build_label_dict()
        self.batch_size = 9 
        self.unique_classes_from_y = sorted(set(self.y)) 

    def build_label_dict(self):
        label_dict = OrderedDict()
        for i, label in enumerate(self.y): 
            if label not in label_dict:
                label_dict[label] = [i]
            else:
                label_dict[label].append(i)
        return label_dict 

    def sample_examples_by_class(self, cls): 
        if cls not in self.label_dict:
            return [] 
        if self.task_sampler.k_shot_dict[cls] <= len(self.label_dict[cls]): 
            sampled_examples = random.sample(self.label_dict[cls], 
                                             self.task_sampler.k_shot_dict[cls]) 
        else:
            sampled_examples = random.choices(self.label_dict[cls], 
                                              k=self.task_sampler.k_shot_dict[cls]) 
        return sampled_examples

    def __iter__(self):
        for _ in range(self.max_iter):
            batch = []
            classes = self.task_sampler.hierarchical_sample_N_classes_as_a_task() 
            if len(batch) == 0:
                for cls in classes: 
                    samples_for_this_class = self.sample_examples_by_class(cls)
                    batch.extend(samples_for_this_class)

            yield batch 

    def __len__(self):
        return self.max_iter 
 
def  hierarchical_nway_kshot_dataloader(dataset, train_steps):
    n_way_list = 6
    k_shot_dict = {'022': 2, '100': 1, '101': 1, '110': 1, '111': 1, '122': 1}
    hierarchical_task_sampler = Hierarchical_TaskSampler(set(dataset.composed_label), n_way_list, k_shot_dict)
    hierarchical_n_way_k_shot_sampler = Hierarchical_N_Way_K_Shot_BatchSampler(dataset.composed_label, train_steps, hierarchical_task_sampler)
    meta_loader = torch.utils.data.DataLoader(dataset, shuffle=False, sampler=hierarchical_n_way_k_shot_sampler, collate_fn=collate_fn_train)
    return meta_loader 



