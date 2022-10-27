import lmdb
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms



class Dataset(data.Dataset):
    def __init__(self, path_to_lmdb_dir, transform, dataset_type="cropped"):
        self.dataset_type = dataset_type
        self._path_to_lmdb_dir = path_to_lmdb_dir
        self._reader = lmdb.open(path_to_lmdb_dir, lock=False)
        with self._reader.begin() as txn:
            self._length = txn.stat()['entries']
            self._keys = self._keys = [key for key, _ in txn.cursor()]
        self._transform = transform

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        with self._reader.begin() as txn:
            value = txn.get(self._keys[index])

        if self.dataset_type == "cropped":
            from .example_pb2 import Example
        else:
            from .example3_pb2 import Example
        example = Example()

        example.ParseFromString(value)

        image = np.frombuffer(example.image, dtype=np.uint8)
        if self.dataset_type == "cropped":
            image = image.reshape([54, 54, 3])
            image = Image.fromarray(image)
            image = self._transform(image)
        else:
            width_ = example.width
            height_ = example.height
            image = image.reshape((example.height, example.width, 3))
            max_size = 128
            # crop if image is larger than max_size
            if example.width > max_size or example.height > max_size:
                # print(f"cropping image: example.width:{example.width}, example.height:{example.height}")
                if example.width > max_size:
                    image = image[:, :max_size, :]
                    width_ = max_size
                if example.height > max_size:
                    image = image[:max_size, :, :]
                    height_ = max_size
            # # expand image to a proper size
            # image = np.pad(image, (((max_size - height_ + 1) // 2, (max_size - height_ + 1) // 2), ((max_size - width_ + 1) // 2, (max_size - width_ + 1) // 2), (0, 0)), 'median')
            image = Image.fromarray(image)
            image = self._transform(image)
            # print(image.shape)
            # resize image to max_size
            image = transforms.functional.resize(image, max_size)
            # crop the image to a square
            image = transforms.functional.center_crop(image, min(image.shape[1], image.shape[2]))

        length = example.length
        digits = example.digits

        return image, digits

# if __name__ == "__main__":
#     dataset = Dataset("SVHN_lmdb/train.lmdb", None)
#     print("len:", dataset.__len__())