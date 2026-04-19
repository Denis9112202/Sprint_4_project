# Конфигурация для обучения модели
config = {
    # Пути к данным
    'ingredients_path': 'data/ingredients.csv',
    'dish_path': 'data/dish.csv',
    'images_dir': 'data/images',

    # Параметры обучения
    'seed': 42,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 50,

    # Сохранение модели
    'save_path': 'best_model.pth',

    # Метрики для отслеживания
    'target_metric': 'MAE',
    'target_threshold': 50  # Целевое значение MAE < 50 ккал
}