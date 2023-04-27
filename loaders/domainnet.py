import numpy as np
import os
import os.path
from PIL import Image
from loaders.randaugment import RandAugmentMC
from torchvision import transforms
import pickle
import torch.utils.data as TorchData
from torch.utils.data.distributed import DistributedSampler
from loaders.utils import TransformFixMatch, TransformFixMatchTwo
from loaders.randaugment import RandAugmentMC
from loaders.simsiam_aug import SimSiamTransform, SimSiamTransformTwo
import copy


def get_transforms(crop_size):
    data_transforms = {
        'labeled': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            # RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'fixmatch': TransformFixMatch(crop_size=crop_size),
        'fixmatch_two': TransformFixMatchTwo(crop_size=crop_size),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'simsiam': SimSiamTransform(224),
        'simsiamTwo': SimSiamTransformTwo(224),
    }
    return data_transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class Imagelists_Domainnet(TorchData.Dataset):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False, repeat=False):
        super().__init__()
        imgs, labels = make_dataset_fromlist(image_list)
        if len(imgs) == 0:
            print(image_list)
        if repeat:
            imgs = imgs.repeat(10)
            labels = labels.repeat(10)
        self.imgs = imgs
        self.labels = labels
        self.is_gt = [1] * len(self.labels)
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.test = test
        self.loader = pil_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, self.imgs[index], self.is_gt[index]

    def __len__(self):
        return len(self.imgs)


class Imagelists_Domainnet_Repeat(TorchData.Dataset):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False, repeat=False):
        super().__init__()
        imgs, labels = make_dataset_fromlist(image_list)
        if len(imgs) == 0:
            print(image_list)
        # if repeat:
        # imgs = imgs.repeat(10)
        # labels = labels.repeat(10)
        self.old_imgs = copy.deepcopy(imgs)
        self.old_labels = copy.deepcopy(labels)
        self.imgs = imgs
        self.labels = labels
        self.is_gt = [1] * len(self.labels)
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.test = test
        self.loader = pil_loader
        self.num_repeat = 50
        self.shuffle_repeat()
        self.is_gt = self.is_gt * self.num_repeat

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, self.imgs[index], self.is_gt[index]

    def __len__(self):
        # return len(self.imgs)
        return len(self.old_imgs) * self.num_repeat

    def shuffle_repeat(self):
        print('Shuffling')
        num_repeat = self.num_repeat
        ind = np.arange(len(self.old_imgs))
        new_imgs = []
        new_labels = []
        for i in range(num_repeat):
            np.random.shuffle(ind)
            new_imgs.append(self.old_imgs[ind])
            new_labels.append(self.old_labels[ind])
        self.imgs = np.concatenate(new_imgs, axis=0)
        self.labels = np.concatenate(new_labels, axis=0)
        print('Finished')


def build_dataset(args):
    base_path = args.base_path
    root = args.data_root
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')

    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % args.num)
    image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt')
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))

    if args.arch == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    data_transforms = get_transforms(crop_size)

    source_dataset = Imagelists_Domainnet(image_set_file_s, root=root, transform=data_transforms[args.labeled_transform])
    # target labeled
    # target_dataset = Imagelists_Domainnet(image_set_file_t, root=root, transform=data_transforms[args.labeled_transform], repeat=True)
    target_dataset = Imagelists_Domainnet_Repeat(image_set_file_t, root=root, transform=data_transforms[args.labeled_transform])
    target_dataset.shuffle_repeat()
    # target unlabeled
    target_dataset_unl = Imagelists_Domainnet(image_set_file_unl, root=root, transform=data_transforms[args.unl_transform])
    # target validation
    target_dataset_val = Imagelists_Domainnet(image_set_file_t_val, root=root, transform=data_transforms['val'])
    # target test
    target_dataset_test = Imagelists_Domainnet(image_set_file_unl, root=root, transform=data_transforms['test'])

    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.arch == 'alexnet':
        bs = 32
    else:
        bs = 24

    train_sampler = TorchData.RandomSampler if args.local_rank == -1 else DistributedSampler

    drop_last = True
    bs = args.bs
    source_loader = TorchData.DataLoader(source_dataset, sampler=train_sampler(source_dataset),
                                         batch_size=bs, num_workers=args.n_workers, drop_last=drop_last, pin_memory=True)
    # target_loader = TorchData.DataLoader(target_dataset, sampler=train_sampler(target_dataset),
    #                                      batch_size=min(bs, len(target_dataset)),
    #                                      num_workers=args.n_workers, drop_last=drop_last, pin_memory=True)
    target_loader = TorchData.DataLoader(target_dataset, batch_size=min(bs, len(target_dataset)), shuffle=False,
                                         num_workers=args.n_workers, drop_last=drop_last, pin_memory=True)
    target_loader_unl = TorchData.DataLoader(target_dataset_unl, sampler=train_sampler(target_dataset_unl),
                                             batch_size=bs * args.bs_unl_multi, num_workers=args.n_workers, drop_last=drop_last, pin_memory=True)

    target_loader_val = TorchData.DataLoader(target_dataset_val, sampler=TorchData.SequentialSampler(target_dataset_val), batch_size=bs * 2,
                                             num_workers=args.n_workers, drop_last=False, pin_memory=True)

    target_loader_test = TorchData.DataLoader(target_dataset_test, sampler=TorchData.SequentialSampler(target_dataset_test), batch_size=bs * 2,
                                              num_workers=args.n_workers, drop_last=False, pin_memory=True)

    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list
