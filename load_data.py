import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.datasets as dataset

#define a class to load and process the celeb dataset
class CelebDataset(Dataset):
    def __init__(self, root_dir, batch_size, workers, transform=None):
        self.dataset = dataset.ImageFolder(root=root_dir, transform=transform)        #loading the data from a local directory and applying necessary transforms
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=workers)         #creating dataloader to iterate through batches of fixed size

    #define a function that returns the processed dataset as well as the dataloader in the form of a dictionary
    def get_loader(self):
        return {'dataset': self.dataset, 'data_loader': self.dataloader}


