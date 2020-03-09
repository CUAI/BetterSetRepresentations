import torchvision
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import os
import random
import cv2
import h5py
import json


def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )
        
class IsoColorCircles(torch.utils.data.Dataset):
    def __init__(self, train=True, root='circles', size=1000, n = None):
        self.train = train
        self.root = root
        self.size = size
        self.n = n
        self.data = self.cache()  

    def cache(self):
        cache_path = os.path.join(self.root, f"iso_color_circles_{self.train}_{self.n}.pth")
        if os.path.exists(cache_path):
            return torch.load(cache_path)

        print("Processing dataset...")
        data = []
        for i in range(self.size):
            if i%10000 == 0:
                print(i)
            img = np.zeros((64, 64,3), dtype = "float") 
            n = int(random.randint(1, 10))
            if self.n is not None:
                n = self.n
            color_count = [0,0]
            circle_features = torch.zeros([10,4]).float()
            # Creating circle 
            j = 0
            while j < n:
                tmp = np.zeros((64, 64,3), dtype = "float") 
                l = range(1,12)
                r = l[int(random.random()*11)]
                center = (int(random.random()*(64-2*r)+r), int(random.random()*(64-2*r)+r))
                c_p = random.randint(0, 1)
                c = [0,0,0]
                c[c_p] = 1
                tmp = cv2.circle(tmp, center, r+1, c, -1)
                if (img + tmp).max() > 1:
                    continue
                elif img.min() >= 1:
                    assert(False)
                else:
                    tmp = np.zeros((64, 64,3), dtype = "float") 
                    tmp = cv2.circle(tmp, center, r, c, -1)
                    color_count[c_p] += 1
                    img+= tmp
                    circle_features[j] = torch.tensor([center[0], center[1],r, c_p+1])
                    j+=1
            

            
            l = range(1,12)
            
            # iso
            
            fail = True
            while fail:
                s = torch.zeros([10]).float()
                fail = False
                iso = np.zeros((64, 64,3), dtype = "float") 
                for idx, f in enumerate(circle_features):
                    if f[3].int() == 0 :
                        break
                    tmp = np.zeros((64, 64,3), dtype = "float") 
                    r = f[2]
                    c = [0,0,0]
                    c[f[3].int() - 1] = 1
                    center = (int(random.random()*(64-2*r)+r), int(random.random()*(64-2*r)+r))

                    tmp = cv2.circle(tmp, center, r+1, c, -1)
                    if (iso + tmp).max() > 1:
                        fail = True
                        break
                    elif iso.min() >= 1:
                        assert(False)
                    else:
                        tmp = np.zeros((64, 64,3), dtype = "float") 
                        tmp = cv2.circle(tmp, center, r, c, -1)
                        s[idx] = (f[0] - center[0])**2 + (f[1] - center[1])**2
                        iso+= tmp
                
            i+=1
            data.append((torch.tensor(img).transpose(0,2).float(), torch.tensor(iso).transpose(0,2).float(), s))
        torch.save(data, cache_path)
        print("Done!")
        return data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.size

class MarkedColorCircles(torch.utils.data.Dataset):
    def __init__(self, train=True, root='circles', size=1000, colors = [[1,0,0],[0,1,0]]):
        self.train = train
        self.root = root
        self.size = size
        self.data = self.cache()  
        self.colors = colors

    def cache(self):
        cache_path = os.path.join(self.root, f"marked_color_circles_{self.train}.pth")
        if os.path.exists(cache_path):
            return torch.load(cache_path)

        print("Processing dataset...")
        data = []
        for i in range(self.size):
            img = np.zeros((64, 64,3), dtype = "float") 
            n = int(random.randint(0, 10))
            color_count = [0,0]
            circle_features = torch.zeros([10,4]).float()
            # Creating circle 
            j = 0
            while j < n:
                tmp = np.zeros((64, 64,3), dtype = "float") 
                l = range(1,12)
                r = l[int(random.random()*11)]
                center = (int(random.random()*(64-2*r)+r), int(random.random()*(64-2*r)+r))
                c_p = random.randint(0, 1)
                c = [0,0,0]
                c[c_p] = 1
                tmp = cv2.circle(tmp, center, r+1, c, -1)
                if (img + tmp).max() > 1:
                    continue
                elif img.min() >= 1:
                    assert(False)
                else:
                    tmp = np.zeros((64, 64,3), dtype = "float") 
                    tmp = cv2.circle(tmp, center, r, c, -1)
                    color_count[c_p] += 1
                    img+= tmp
                    circle_features[j] = torch.tensor([center[0], center[1],r, c_p+1])
                    j+=1
            i+=1
            data.append((torch.tensor(img).transpose(0,2).float(), circle_features))
        torch.save(data, cache_path)
        print("Done!")
        return data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.size

class CLEVR(torch.utils.data.Dataset):
    def __init__(self, base_path, split, box=False, full=False, chamfer=False):
        assert split in {
            "train",
            "val",
            "test",
        }  # note: test isn't very useful since it doesn't have ground-truth scene information
        self.base_path = base_path
        self.split = split
        self.max_objects = 10
        self.box = box  # True if clevr-box version, False if clevr-state version
        self.full = full  # Use full validation set?
        self.chamfer = chamfer  # Use Chamfer data?

        with self.img_db() as db:
            ids = db["image_ids"]
            self.image_id_to_index = {id: i for i, id in enumerate(ids)}
        self.image_db = None

        with open(self.scenes_path) as fd:
            scenes = json.load(fd)["scenes"]
        self.img_ids, self.scenes = self.prepare_scenes(scenes)

    def object_to_fv(self, obj):
        coords = [p / 3 for p in obj["3d_coords"]]
        one_hot = lambda key: [obj[key] == x for x in CLASSES[key]]
        material = one_hot("material")
        color = one_hot("color")
        shape = one_hot("shape")
        size = one_hot("size")
        assert sum(material) == 1
        assert sum(color) == 1
        assert sum(shape) == 1
        assert sum(size) == 1
        # concatenate all the classes
        return coords + material + color + shape + size

    def prepare_scenes(self, scenes_json):
        img_ids = []
        scenes = []
        for scene in scenes_json:
            img_idx = scene["image_index"]
            # different objects depending on bbox version or attribute version of CLEVR sets
            if self.box:
                objects = self.extract_bounding_boxes(scene)
                objects = torch.FloatTensor(objects)
            else:
                objects = [self.object_to_fv(obj) for obj in scene["objects"]]
                objects = torch.FloatTensor(objects).transpose(0, 1)
            num_objects = objects.size(1)
            # pad with 0s
            if num_objects < self.max_objects:
                objects = torch.cat(
                    [
                        objects,
                        torch.zeros(objects.size(0), self.max_objects - num_objects),
                    ],
                    dim=1,
                )
            # fill in masks
            mask = torch.zeros(self.max_objects)
            mask[:num_objects] = 1

            img_ids.append(img_idx)
            scenes.append((objects, mask))
        return img_ids, scenes

    def extract_bounding_boxes(self, scene):
        """
        Code used for 'Object-based Reasoning in VQA' to generate bboxes
        https://arxiv.org/abs/1801.09718
        https://github.com/larchen/clevr-vqa/blob/master/bounding_box.py#L51-L107
        """
        objs = scene["objects"]
        rotation = scene["directions"]["right"]

        num_boxes = len(objs)

        boxes = np.zeros((1, num_boxes, 4))

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []

        for i, obj in enumerate(objs):
            [x, y, z] = obj["pixel_coords"]

            [x1, y1, z1] = obj["3d_coords"]

            cos_theta, sin_theta, _ = rotation

            x1 = x1 * cos_theta + y1 * sin_theta
            y1 = x1 * -sin_theta + y1 * cos_theta

            height_d = 6.9 * z1 * (15 - y1) / 2.0
            height_u = height_d
            width_l = height_d
            width_r = height_d

            if obj["shape"] == "cylinder":
                d = 9.4 + y1
                h = 6.4
                s = z1

                height_u *= (s * (h / d + 1)) / ((s * (h / d + 1)) - (s * (h - s) / d))
                height_d = height_u * (h - s + d) / (h + s + d)

                width_l *= 11 / (10 + y1)
                width_r = width_l

            if obj["shape"] == "cube":
                height_u *= 1.3 * 10 / (10 + y1)
                height_d = height_u
                width_l = height_u
                width_r = height_u

            obj_name = (
                obj["size"]
                + " "
                + obj["color"]
                + " "
                + obj["material"]
                + " "
                + obj["shape"]
            )
            ymin.append((y - height_d) / 320.0)
            ymax.append((y + height_u) / 320.0)
            xmin.append((x - width_l) / 480.0)
            xmax.append((x + width_r) / 480.0)

        return xmin, ymin, xmax, ymax

    @property
    def images_folder(self):
        return os.path.join(self.base_path, "images", self.split)

    @property
    def scenes_path(self):
        if self.split == "test":
            raise ValueError("Scenes are not available for test")
        return os.path.join(
            self.base_path, "scenes", "CLEVR_{}_scenes.json".format(self.split)
        )

    def img_db(self):
        path = os.path.join(self.base_path, "{}-images.h5".format(self.split))
        return h5py.File(path, "r")

    def load_image(self, image_id):
        if self.image_db is None:
            self.image_db = self.img_db()
        index = self.image_id_to_index[image_id]
        image = self.image_db["images"][index]
        return image

    def make_mask(self, objects, size, num_objs):
        num_objs = len(size[size == 1])
        masks = torch.zeros([16,128,128])
        for i in range(num_objs):
            masks[i, objects[1, i]:objects[3, i], objects[0, i]:objects[2, i]] = 1
        return masks

    def __getitem__(self, item):
        image_id = self.img_ids[item]
        image = self.load_image(image_id)
        objects, size = self.scenes[item]
        if self.chamfer:
            objects = (objects * 128).to(dtype=torch.uint8)
            num_objs = len(size[size == 1])
            return image, self.make_mask(objects, size, num_objs)
        return image

    def __len__(self):
        if self.split == "train" or self.full:
            return len(self.scenes)
        else:
            return len(self.scenes) // 10


class CLEVRMasked(torch.utils.data.Dataset):
    def __init__(self, base_path, split, full=False, iou=False):
        assert split in {
            "train",
            "test",
        }  # note: test isn't very useful since it doesn't have ground-truth scene information
        self.base_path = base_path
        self.split = split
        self.full = full  # Use full validation set?
        self.iou = iou

        with self.img_db() as db:
            ids = db["image_ids"]
            self.image_id_to_index = {id: i for i, id in enumerate(ids)}
        self.image_db = None
        self.img_ids = [i for i in range(len(self.image_id_to_index))]

    @property
    def images_folder(self):
        return os.path.join(self.base_path, "images", self.split)

    def img_db(self):
        path = os.path.join(self.base_path, "{}-images-foreground.h5".format(self.split))
        return h5py.File(path, "r")

    def load_image(self, image_id):
        if self.image_db is None:
            self.image_db = self.img_db()
        index = self.image_id_to_index[image_id]
        image = self.image_db["images"][index]
        image_mask = self.image_db["images_mask"][index]
        image_foreground = self.image_db["images_foreground"][index]
        return image, image_mask, image_foreground

    def __getitem__(self, item):
        image_id = self.img_ids[item]
        image, image_mask, image_foreground = self.load_image(image_id)
        if self.iou:
            return image, image_mask, image_foreground
        return image, image_foreground

    def __len__(self):
        if self.split == "train" or self.full:
            return len(self.img_ids)
        else:
            return len(self.img_ids) // 10
