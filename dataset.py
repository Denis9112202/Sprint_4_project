import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from torchvision import transforms

class FoodDataset(Dataset):
    def __init__(self, dish_df, ingredients_df, images_dir, split, transform=None, scaler=None):
        self.dish_df = dish_df[dish_df['split'] == split].reset_index(drop=True)
        self.ingredients_df = ingredients_df
        self.images_dir = images_dir
        self.transform = transform
        self.scaler = scaler  # Передаём уже обученный скалер

    def __len__(self):
        return len(self.dish_df)

    def __getitem__(self, idx):
        row = self.dish_df.iloc[idx]
        dish_id = row['dish_id']
        img_path = os.path.join(self.images_dir, str(dish_id), 'rgb.png')
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        ingredients_str = row['ingredients']
        if pd.isna(ingredients_str):
            ingredients_ids = []
        else:
            ingredients_ids = [int(x.split('_')[-1]) for x in ingredients_str.split(';')]

        num_ingredients = len(self.ingredients_df)
        ingredients_vector = np.zeros(num_ingredients)
        for ingr_id in ingredients_ids:
            if ingr_id < num_ingredients:
                ingredients_vector[ingr_id] = 1

        mass = self.scaler.transform([[row['total_mass']]])[0, 0]

        calories = row['total_calories']

        return {
            'image': image,
            'ingredients': torch.tensor(ingredients_vector, dtype=torch.float32),
            'mass': torch.tensor([mass], dtype=torch.float32),
            'calories': torch.tensor([calories], dtype=torch.float32),
            'dish_id': dish_id
        }

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
