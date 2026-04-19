import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from scripts.dataset import FoodDataset, get_transforms

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiModalModel(nn.Module):
    def __init__(self, num_ingredients):
        super(MultiModalModel, self).__init__()
        from torchvision.models import resnet18
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 512)

        self.ingr_fc = nn.Sequential(
            nn.Linear(num_ingredients, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.final_fc = nn.Sequential(
            nn.Linear(512 + 256 + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, image, ingredients, mass):
        cnn_features = self.cnn(image)
        ingr_features = self.ingr_fc(ingredients)
        combined = torch.cat([cnn_features, ingr_features, mass], dim=1)
        return self.final_fc(combined)

def train(config):
    set_seed(config['seed'])

    ingredients_df = pd.read_csv(config['ingredients_path'])
    dish_df = pd.read_csv(config['dish_path'])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_mass = dish_df[dish_df['split'] == 'train']['total_mass'].values.reshape(-1, 1)
    scaler.fit(train_mass)

    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    train_dataset = FoodDataset(dish_df, ingredients_df, config['images_dir'], 'train', train_transform, scaler)
    val_dataset = FoodDataset(dish_df, ingredients_df, config['images_dir'], 'test', val_transform, scaler)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = MultiModalModel(len(ingredients_df))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.L1Loss() 

    best_mae = float('inf')

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            image = batch['image'].to(device)
            ingredients = batch['ingredients'].to(device)
            mass = batch['mass'].to(device)
            targets = batch['calories'].to(device)

            outputs = model(image, ingredients, mass)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(device)
                ingredients = batch['ingredients'].to(device)
                mass = batch['mass'].to(device)
                targets = batch['calories'].to(device)

                outputs = model(image, ingredients, mass)
                val_mae += torch.abs(outputs - targets).mean().item()

        val_mae /= len(val_loader)

        print(f'Epoch {epoch+1}/{config["epochs"]}, '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val MAE: {val_mae:.4f}')

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), config['save_path'])
            print(f"Лучшая модель сохранена с MAE: {best_mae:.4f}")


    print(f"\nОбучение завершено. Лучшая валидационная MAE: {best_mae:.4f}")
    return best_mae



def validate_model(config, model=None):
    
    ingredients_df = pd.read_csv(config['ingredients_path'])
    dish_df = pd.read_csv(config['dish_path'])
    test_df = dish_df[dish_df['split'] == 'test'].reset_index(drop=True)

    ingredient_id_to_name = dict(zip(ingredients_df['id'], ingredients_df['ingr']))

    # Создаём словарь dish_id → список названий ингредиентов
    dish_to_ingredients = {}
    for _, row in dish_df.iterrows():
        dish_id = row['dish_id']
        ingredients_str = row['ingredients']

        if pd.isna(ingredients_str):
            ingredient_names = []
        else:
            ingredients_ids = [int(x.split('_')[1]) for x in ingredients_str.split(';')]
            ingredient_names = [
                ingredient_id_to_name.get(ingr_id, f'Unknown_ingredient_{ingr_id}')
                for ingr_id in ingredients_ids
            ]
        dish_to_ingredients[dish_id] = ingredient_names

    mass_train = dish_df[dish_df['split'] == 'train']['total_mass'].values.reshape(-1, 1)
    mass_scaler = StandardScaler()
    mass_scaler.fit(mass_train)

    val_transform = get_transforms(train=False)

    test_dataset = FoodDataset(
        dish_df=test_df,
        ingredients_df=ingredients_df,
        images_dir=config['images_dir'],
        split='test',
        transform=val_transform,
        scaler=mass_scaler
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model is None:
        model = MultiModalModel(len(ingredients_df))
        model.load_state_dict(torch.load(config['save_path'], map_location=device))
    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []
    all_dish_ids = []


    with torch.no_grad(): 
        for batch in test_loader:
            image = batch['image'].to(device)
            ingredients = batch['ingredients'].to(device)
            mass = batch['mass'].to(device)
            targets = batch['calories'].to(device)  # Целевые значения
            dish_ids = batch.get('dish_id', [None] * len(targets))
            outputs = model(image, ingredients, mass)

            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            all_dish_ids.extend(dish_ids)


    mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
    print(f"✅ Финальная MAE на тестовом наборе: {mae:.4f} ккал")

    errors = np.abs(np.array(all_predictions) - np.array(all_targets))
    top_5_worst_indices = np.argsort(errors)[-5:][::-1]

    print("\n" + "="*60)
    print("ТОП‑5 БЛЮД С НАИМЕНЕЕ ТОЧНЫМИ ПРЕДСКАЗАНИЯМИ")
    print("="*60)

    worst_examples = []
    for i, idx in enumerate(top_5_worst_indices, 1):
        dish_id = all_dish_ids[idx]
        pred = all_predictions[idx]
        true = all_targets[idx]
        error = errors[idx]
        ingredient_names = dish_to_ingredients[dish_id]

        print(f"{i}. Блюдо ID: {dish_id}")
        print("   Ингредиенты:")
        for ingredient in ingredient_names:
            print(f"     - {ingredient}")
        print(f"   Предсказание: {pred:.2f} ккал, Истинное значение: {true:.2f} ккал")
        print(f"   Ошибка: {error:.2f} ккал")
        if true != 0:
            relative_error = (error / true) * 100
            print(f"   Относительная ошибка: {relative_error:.2f}%")
        print("-" * 40)

        worst_examples.append({
            'dish_id': dish_id,
            'prediction': pred,
            'true_value': true,
            'error': error
        })

    # Вывод изображений для топ‑5 худших примеров
    print("\n" + "="*60)
    print("ИЗОБРАЖЕНИЯ ДЛЯ ТОП‑5 НАИМЕНЕЕ ТОЧНЫХ ПРЕДСКАЗАНИЙ")
    print("="*60)

    plt.figure(figsize=(15, 12))
    for i, idx in enumerate(top_5_worst_indices, 1):
        dish_id = all_dish_ids[idx]
        img_path = os.path.join(config['images_dir'], str(dish_id), 'rgb.png')

        try:
            img = Image.open(img_path)
            plt.subplot(2, 3, i)
            plt.imshow(img)
            plt.title(f"ID: {dish_id}\nPred: {all_predictions[idx]:.1f} ккал\nTrue: {all_targets[idx]:.1f} ккал")
            plt.axis('off')
        except Exception as e:
            print(f"Ошибка загрузки изображения для блюда {dish_id}: {e}")

    plt.tight_layout()
    plt.show()

    return {
        'mae': mae,
        'top_5_worst': worst_examples,
        'predictions': all_predictions,
        'targets': all_targets,
        'dish_ids': all_dish_ids
    }