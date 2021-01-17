import os
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        image = self.data_x[index]
        label = self.data_y[index]

        return image, label

    def __len__(self):
        return len(self.data_y)


def create_loaders(train_x, train_y, test_x, test_y, batch_size, channel_norm, val_frac=0.15):

    np.random.seed(0)

    # Sub setting for testing pipeline
    data_x_train = train_x[:8500, :, :]
    data_y_train = train_y[:8500]
    data_x_test = test_x[:1500, :, :]
    data_y_test = test_y[:1500]

    if channel_norm:
        data_x_train = apply_channel_norm(data_x_train)
        data_x_test = apply_channel_norm(data_x_test)

    train_dataset = CustomDataset(data_x_train, data_y_train)
    test_dataset = CustomDataset(data_x_test, data_y_test)

    indices = np.arange(len(data_x_train))
    np.random.shuffle(indices)

    val_count = int(len(indices) * val_frac)
    val_indices, train_indices = indices[:val_count], indices[val_count:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=val_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


def apply_channel_norm(x_data):
    channel_mean = torch.mean(x_data, dim=[0, 2, 3]).view(1, x_data.shape[1], 1, 1)
    channel_std = torch.std(x_data, dim=[0, 2, 3]).view(1, x_data.shape[1], 1, 1)
    x_data = (x_data - channel_mean) / channel_std
    return x_data


def get_mnist_loaders(batch_size, channel_norm=True):
    base_path = '/Users/ravy/Desktop/git/optics/mnist'
    mnist_x_train_path = os.path.join(base_path, "mnist_train_norm.npy")
    mnist_y_train_path = os.path.join(base_path, "train_labels.npy")

    mnist_x_test_path = os.path.join(base_path, "mnist_test_norm.npy")
    mnist_y_test_path = os.path.join(base_path, "test_labels.npy")

    channels = 25
    classes = 10

    mnist_x_train = np.load(mnist_x_train_path)
    mnist_y_train = np.load(mnist_y_train_path)

    mnist_x_test = np.load(mnist_x_test_path)
    mnist_y_test = np.load(mnist_y_test_path)

    mnist_x_train = torch.from_numpy(mnist_x_train.swapaxes(0, 1).swapaxes(1, 2)).float()
    mnist_x_train = mnist_x_train.reshape(-1, channels, 28, 28)

    mnist_x_test = torch.from_numpy(mnist_x_test.swapaxes(0, 1).swapaxes(1, 2)).float()
    mnist_x_test = mnist_x_test.reshape(-1, channels, 28, 28)

    mnist_y_train = torch.from_numpy(mnist_y_train).long()
    mnist_y_train = torch.argmax(mnist_y_train, axis=-1)

    mnist_y_test = torch.from_numpy(mnist_y_test).long()
    mnist_y_test = torch.argmax(mnist_y_test, axis=-1)

    print('Train X Shape', mnist_x_train.shape, 'Y Shape', mnist_y_train.shape)
    print('Test X Shape', mnist_x_test.shape, 'Y Shape', mnist_y_test.shape)

    train_loader, val_loader, test_loader = create_loaders(train_x=mnist_x_train,
                                                           train_y=mnist_y_train,
                                                           test_x=mnist_x_test,
                                                           test_y=mnist_y_test,
                                                           batch_size=batch_size,
                                                           channel_norm=channel_norm)

    label_dictionary = dict(zip(range(10), range(10)))

    return train_loader, val_loader, None, channels, classes, label_dictionary


def get_fashion_loaders(batch_size, channel_norm=True):
    base_path = '/Users/ravy/Desktop/git/optics/fashion/'
    fashion_x_train_path = os.path.join(base_path, "fashion_train_norm.npy")
    fashion_y_train_path = os.path.join(base_path, "fashion_train_labels.npy")

    fashion_x_test_path = os.path.join(base_path, "fashion_test_norm.npy")
    fashion_y_test_path = os.path.join(base_path, "fashion_test_labels.npy")

    channels = 25
    classes = 10

    fashion_x_train = np.load(fashion_x_train_path)
    fashion_y_train = np.load(fashion_y_train_path)

    fashion_x_test = np.load(fashion_x_test_path)
    fashion_y_test = np.load(fashion_y_test_path)

    fashion_x_train = torch.from_numpy(fashion_x_train.swapaxes(1, 2)).float()
    fashion_x_train = fashion_x_train.reshape(-1, channels, 28, 28)

    fashion_x_test = torch.from_numpy(fashion_x_test.swapaxes(1, 2)).float()
    fashion_x_test = fashion_x_test.reshape(-1, channels, 28, 28)

    fashion_y_train = torch.from_numpy(fashion_y_train).long()
    fashion_y_test = torch.from_numpy(fashion_y_test).long()

    print('Train X Shape', fashion_x_train.shape, 'Y Shape', fashion_y_train.shape)
    print('Test X Shape', fashion_x_test.shape, 'Y Shape', fashion_y_test.shape)

    train_loader, val_loader, test_loader = create_loaders(train_x=fashion_x_train,
                                                           train_y=fashion_y_train,
                                                           test_x=fashion_x_test,
                                                           test_y=fashion_y_test,
                                                           batch_size=batch_size,
                                                           channel_norm=channel_norm)

    label_dictionary = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
                        4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker",
                        8: "Bag", 9: "Ankle boot"}

    return train_loader, val_loader, None, channels, classes, label_dictionary


def get_malaria_loaders(batch_size, channel_norm=True):
    base_path = '/Users/ravy/Desktop/git/optics/malaria'

    malaria_x_path = os.path.join(base_path, "malaria_norm.npy")
    malaria_y_path = os.path.join(base_path, "malaria_labels.npy")

    channels = 29
    classes = 2

    malaria_x = np.load(malaria_x_path)
    malaria_y = np.load(malaria_y_path)

    malaria_x = torch.from_numpy(malaria_x).float()
    malaria_y = torch.from_numpy(malaria_y).long()

    malaria_x = torch.cat([malaria_x[:, :, :, 1:10],
                           malaria_x[:, :, :, 11:19],
                           malaria_x[:, :, :, 20:32],
                           malaria_x[:, :, :, 33:42],
                           malaria_x[:, :, :, 43:51],
                           malaria_x[:, :, :, 52:64],
                           malaria_x[:, :, :, 65:74],
                           malaria_x[:, :, :, 75:83],
                           malaria_x[:, :, :, 84:]], axis=-1)

    malaria_x = malaria_x.permute(0, 3, 1, 2)
    print('Train + Test X Shape', malaria_x.shape)
    print('Train + Test Y Shape', malaria_y.shape)

    malaria_x = malaria_x[:, 29:29*2, :, :]

    if channel_norm:
        malaria_x = apply_channel_norm(malaria_x)

    num_train = malaria_x.shape[0]
    indices = list(range(num_train))
    train_dataset = CustomDataset(malaria_x, malaria_y)
    test_dataset = CustomDataset(malaria_x, malaria_y)

    np.random.seed(3)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[:800], indices[800:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=valid_sampler, )

    label_dictionary = {0: "Not Infected", 1: "Infected"}

    return train_loader, val_loader, None, channels, classes, label_dictionary

