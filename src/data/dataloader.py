import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



class EmoticDataset(Dataset):
    def __init__(
        self, 
        datapath:str = None,
        emotion_path:str = None,
        img_size = (224, 224),
        train_mode = False,
        emotion_labels = {'Anger' : 0, 'Contempt' : 1, 'Disgust' : 2,
                        'Fear' : 3, 'Happiness' : 4, 'Neutral' : 5,
                        'Sadness' : 6, 'Surprise' : 7}
        ) -> None:
        
        self.data = glob.glob(f'{datapath}/*')
        self.img_size = img_size
        self.emotion_labels = emotion_labels
        
        self._init_augs(train_mode)
        
        if emotion_path is not None:
            with open(emotion_path, 'r') as file:
               data = file.readlines()
            
            self.emotion = dict()    
            for x in data:
                x1, x2 = x.strip().split('\t')
                self.emotion.update({x1:x2})
        else:
            self.emotion = None
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        sample = self.data[index]
        name = sample.split('/')[-1]
        
        img = Image.open(sample)
        img = img.convert("RGB")
        img = self.transform(img)    
        
        # process label
        tmp = name.strip().split('_')
        age, gender = int(tmp[0]), int(tmp[1])
        # age = int(tmp[-2])
        # gender = 1 if tmp[-1].startswith('f') else 0

        emo_name = self.emotion[name]
        emotion = self.emotion_labels[emo_name]
            
        return img, age, gender, emotion
    

    def _init_augs(self, train_mode: bool) -> None:
        if train_mode:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.RandomHorizontalFlip(0.2),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
            
    @staticmethod
    def collate_fn(batch):
        img, age, gender, emotion = zip(*batch)
        
        img =  torch.stack(img)
        age =  torch.Tensor(list(age))
        gender = torch.Tensor(list(gender))
        emotion = torch.Tensor(list(emotion))
        
        return img, [age, gender, emotion]
    

def setup_dataset(datapath, batch_size, num_workers, train_mode, emotion_path):
    train_data = EmoticDataset(datapath, emotion_path=emotion_path, train_mode=True)

    train_dataloader = DataLoader(
        train_data,
        batch_size= batch_size if train_mode else batch_size * 2, 
        shuffle=True if train_mode else False,
        num_workers=num_workers,
        collate_fn=EmoticDataset.collate_fn
    )
    
    return train_data, train_dataloader


if __name__ == '__main__':
    train_data, train_dataloader = setup_dataset('notebook/utkface', batch_size=4, num_workers=1, train_mode=True)
    
    train_sample = next(iter(train_dataloader))
    
    print("\n\nImg: ", train_sample[0].shape)
    print("Age: ", train_sample[1])
    print("Gender: ", train_sample[2])