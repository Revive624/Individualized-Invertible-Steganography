import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
from natsort import natsorted


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if self.mode == "train":
            if c.Dataset_mode == 'PARIS':
                self.TRAIN_PATH = c.TRAIN_PATH_PARIS
                self.format_train = 'JPG'
                print('TRAIN DATASET is PARIS')

            if c.Dataset_mode == 'ImageNet':
                self.TRAIN_PATH = c.TRAIN_PATH_IMAGENET
                self.format_train = 'JPEG'
                print('TRAIN DATASET is ImageNet')

            # train
            self.files = natsorted(sorted(glob.glob(self.TRAIN_PATH + "/*." + self.format_train)))

        if self.mode == "val":
            if c.Dataset_VAL_mode == 'PARIS':
                self.VAL_PATH = c.VAL_PATH_PARIS
                self.format_val = 'png'
                print('VAL DATASET is PARIS')

            if c.Dataset_VAL_mode == 'ImageNet':
                self.VAL_PATH = c.VAL_PATH_IMAGENET
                self.format_val = 'JPEG'
                print('VAL DATASET is ImageNet')

            # test
            self.files = sorted(glob.glob(self.VAL_PATH + "/*." + self.format_val))

        if self.mode == "test":
            if c.Dataset_TEST_mode == 'PARIS':
                self.TEST_PATH = c.TEST_PATH_PARIS
                self.format_test = 'png'
                print('VAL DATASET is PARIS')

            if c.Dataset_TEST_mode == 'ImageNet':
                self.TEST_PATH = c.TEST_PATH_IMAGENET
                self.format_test = 'JPEG'
                print('VAL DATASET is ImageNet')

            # test
            self.files = sorted(glob.glob(self.TEST_PATH + "/*." + self.format_test))

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index])
            image = to_rgb(image)
            item = self.transform(image)
            return item

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files)


if c.Dataset_VAL_mode == 'PARIS':
    cropsize_val = c.cropsize_val_paris
if c.Dataset_VAL_mode == 'ImageNet':
    cropsize_val = c.cropsize_val_imagenet

if c.Dataset_TEST_mode == 'PARIS':
    cropsize_test = c.cropsize_test_paris
if c.Dataset_TEST_mode == 'ImageNet':
    cropsize_test = c.cropsize_test_imagenet

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomCrop(c.cropsize),
    T.ToTensor()
])

transform_val = T.Compose([
    T.CenterCrop(cropsize_val),
    T.ToTensor(),
])

transform_test = T.Compose([
    T.CenterCrop(cropsize_test),
    T.ToTensor(),
])

# Training data loader
trainloader = DataLoader(
    Hinet_Dataset(transforms_=transform, mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=False,
    num_workers=1,
    drop_last=True
)
# Test data loader
valloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=c.shuffle_val,
    pin_memory=False,
    num_workers=1,
    drop_last=True
)

testloader = DataLoader(
    Hinet_Dataset(transforms_=transform_test, mode="test"),
    batch_size=c.batchsize_test,
    shuffle=c.shuffle_test,
    pin_memory=False,
    num_workers=1,
    drop_last=True
)
