from torch.utils.data import Dataset

import pandas as pd

class FaceDataset(Dataset):
    """Face dataset."""
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied.
                on a sample.
        """
        self.attributes = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.attributes)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
                                self.attributes.loc[idx, 'id'] + '.jpg')
        image = Image.open(img_name)
        
        # Create a dictionary with the properties of the image
        # and the image itself.
        sample = self.attributes.iloc[0, :].to_dict()
        sample['image'] = image
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        
        return sample
