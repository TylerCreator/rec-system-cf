import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, ndcg_score, accuracy_score
from sklearn.decomposition import TruncatedSVD, PCA, NMF
from sklearn.neighbors import NearestNeighbors
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import csr_matrix, coo_matrix
from implicit.als import AlternatingLeastSquares
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import openpyxl
from openpyxl.styles import PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.formatting.rule import ColorScaleRule

warnings.filterwarnings("ignore")

# Фиксируем random seed для воспроизводимости
np.random.seed(42)
torch.manual_seed(42)

# ============================ Загрузка и подготовка данных ============================
print("Загрузка данных...")
df = pd.read_csv("calls.csv", sep=";")
df = df[~df['owner'].str.contains('cookies', na=False)]
df = df.sort_values('start_time')

owners = df['owner'].unique()
mids = df['mid'].unique()

train_size = int(len(df) * 0.7)
df_train = df[:train_size]
df_test = df[train_size:]

def build_matrix(df, owners, mids, normalize=False):
    pivot = df.pivot_table(index='owner', columns='mid', values='id', aggfunc='count').fillna(0)
    pivot = pivot.reindex(index=owners, columns=mids, fill_value=0)
    mat = pivot.values
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat = mat / row_sums
    return mat

X_train = build_matrix(df_train, owners, mids, normalize=True)
X_test = build_matrix(df_test, owners, mids, normalize=True)

# ============================ Популярность ============================
def get_popular_services(df, mids):
    counts = df['mid'].value_counts().reindex(mids, fill_value=0)
    counts_max = np.max(counts.values)
    return np.argsort(counts.values/counts_max)[::-1]

popular_services = get_popular_services(df_train, mids)

# ============================ Базовые методы ============================
class KNNRecommender:
    def __init__(self, n_neighbors=3):
        self.model = NearestNeighbors(n_neighbors=n_neighbors+1, metric='euclidean')

    def fit(self, X):
        self.X = X
        self.model.fit(X)
        self.indices = self.model.kneighbors(X, return_distance=False)

    def predict(self):
        preds = np.zeros_like(self.X)
        for i, neighbors in enumerate(self.indices):
            neighbors = neighbors[1:]
            preds[i] = self.X[neighbors].mean(axis=0)
        return preds

class SVDRecommender:
    def __init__(self, model):
        self.model = model

    def fit(self, X):
        self.X = X
        self.embeddings = self.model.fit_transform(X)

    def predict(self):
        return self.model.inverse_transform(self.embeddings)

class ALSRecommender:
    def __init__(self, factors=20, regularization=0.1, iterations=10):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations

    def fit(self, X):
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations
        )
        self.X = csr_matrix(X * 20)
        self.model.fit(self.X)

    def predict(self):
        return self.model.user_factors @ self.model.item_factors.T

class WRMFRecommender:
    def fit(self, X):
        self.X = X
        pass

    def predict(self):
        return self.X

class PopularRecommender:
    def __init__(self, popular_items):
        self.popular_items = popular_items

    def fit(self, X):
        self.X = X
        pass

    def predict(self):
        return np.tile(self.popular_items, (self.X.shape[0], 1))

class RandomRecommender:
    def fit(self, X):
        self.X = X
        pass

    def predict(self):
        return np.random.rand(*self.X.shape)

# ============================ LightFM модели ============================
class LightFMRecommender:
    def __init__(self, loss='warp', no_components=20, epochs=20):
        self.loss = loss
        self.no_components = no_components
        self.epochs = epochs

    def fit(self, df, owners, mids):
        self.dataset = Dataset()
        self.dataset.fit(owners, mids)
        interactions, weights = self.dataset.build_interactions(
            [(row['owner'], row['mid']) for _, row in df.iterrows()]
        )
        self.model = LightFM(loss=self.loss, no_components=self.no_components)
        self.model.fit(interactions, epochs=self.epochs)
        
        # Сохраняем размеры для предсказаний
        self.n_users = len(owners)
        self.n_items = len(mids)

    def predict(self):
        # Создаем все user-item пары
        users = np.arange(self.n_users)
        items = np.arange(self.n_items)
        
        # Создаем сетку всех комбинаций
        users_grid, items_grid = np.meshgrid(users, items, indexing='ij')
        users_flat = users_grid.flatten()
        items_flat = items_grid.flatten()
        
        # Получаем предсказания
        predictions = self.model.predict(users_flat, items_flat)
        return predictions.reshape(self.n_users, self.n_items)

class HybridKNNLightFMRecommender:
    def __init__(self, loss='warp', no_components=20, epochs=20, knn_neighbors=4, alpha=1):
        self.loss = loss
        self.no_components = no_components
        self.epochs = epochs
        self.knn_neighbors = knn_neighbors
        self.alpha = alpha

    def fit(self, df_train, owners, mids, X_train):
        # LightFM
        self.dataset = Dataset()
        self.dataset.fit(owners, mids)
        interactions, weights = self.dataset.build_interactions(
            [(row['owner'], row['mid']) for _, row in df_train.iterrows()]
        )
        self.lightfm_model = LightFM(loss=self.loss, no_components=self.no_components)
        self.lightfm_model.fit(interactions, epochs=self.epochs)
        
        # KNN
        self.X_train = X_train
        self.knn_model = NearestNeighbors(n_neighbors=self.knn_neighbors+1, metric='euclidean')
        self.knn_model.fit(X_train)
        self.indices = self.knn_model.kneighbors(X_train, return_distance=False)
        
        # Сохраняем размеры для предсказаний
        self.n_users = len(owners)
        self.n_items = len(mids)

    def predict(self):
        # LightFM predictions
        users = np.arange(self.n_users)
        items = np.arange(self.n_items)
        users_grid, items_grid = np.meshgrid(users, items, indexing='ij')
        users_flat = users_grid.flatten()
        items_flat = items_grid.flatten()
        lightfm_preds = self.lightfm_model.predict(users_flat, items_flat)
        lightfm_preds = lightfm_preds.reshape(self.n_users, self.n_items)
        
        # KNN predictions
        knn_preds = np.zeros_like(self.X_train)
        for i, neighbors in enumerate(self.indices):
            neighbors = neighbors[1:]
            knn_preds[i] = self.X_train[neighbors].mean(axis=0)
        
        # Combine predictions
        return self.alpha * lightfm_preds + (1 - self.alpha) * knn_preds

# ============================ Нейронные модели ============================
class NCFRecommender:
    def __init__(self, hidden_layers=[64, 32, 16], learning_rate=0.001, epochs=5):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X):
        self.X = X
        self.n_users, self.n_items = X.shape
        
        # Создаем модель
        self.model = NCFModel(self.n_users, self.n_items, self.hidden_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()
        
        # Подготавливаем данные
        user_indices, item_indices, ratings = self._prepare_data(X)
        dataset = TensorDataset(user_indices, item_indices, ratings)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Обучение
        self.model.train()
        for epoch in range(self.epochs):
            for users, items, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(users, items)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def predict(self):
        self.model.eval()
        predictions = np.zeros((self.n_users, self.n_items))
        
        with torch.no_grad():
            for user in range(self.n_users):
                for item in range(self.n_items):
                    user_tensor = torch.tensor([user], dtype=torch.long)
                    item_tensor = torch.tensor([item], dtype=torch.long)
                    pred = self.model(user_tensor, item_tensor).item()
                    predictions[user, item] = pred
        
        return predictions

    def _prepare_data(self, X):
        users, items, ratings = [], [], []
        for user in range(X.shape[0]):
            for item in range(X.shape[1]):
                users.append(user)
                items.append(item)
                ratings.append(1.0 if X[user, item] > 0 else 0.0)
        
        return (torch.tensor(users, dtype=torch.long),
                torch.tensor(items, dtype=torch.long),
                torch.tensor(ratings, dtype=torch.float))

class NCFModel(nn.Module):
    def __init__(self, n_users, n_items, hidden_layers):
        super(NCFModel, self).__init__()
        self.user_embedding = nn.Embedding(n_users, hidden_layers[0])
        self.item_embedding = nn.Embedding(n_items, hidden_layers[0])
        
        layers = []
        input_dim = hidden_layers[0] * 2
        for hidden_dim in hidden_layers[1:]:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        concat = torch.cat([user_emb, item_emb], dim=1)
        return self.mlp(concat).squeeze()

class DeepFMRecommender:
    def __init__(self, embedding_dim=16, hidden_layers=[64, 32], learning_rate=0.001, epochs=5):
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X):
        self.X = X
        self.n_users, self.n_items = X.shape
        
        # Создаем модель
        self.model = DeepFMModel(self.n_users, self.n_items, self.embedding_dim, self.hidden_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()
        
        # Подготавливаем данные
        user_indices, item_indices, ratings = self._prepare_data(X)
        dataset = TensorDataset(user_indices, item_indices, ratings)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Обучение
        self.model.train()
        for epoch in range(self.epochs):
            for users, items, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(users, items)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def predict(self):
        self.model.eval()
        predictions = np.zeros((self.n_users, self.n_items))
        
        with torch.no_grad():
            for user in range(self.n_users):
                for item in range(self.n_items):
                    user_tensor = torch.tensor([user], dtype=torch.long)
                    item_tensor = torch.tensor([item], dtype=torch.long)
                    pred = self.model(user_tensor, item_tensor).item()
                    predictions[user, item] = pred
        
        return predictions

    def _prepare_data(self, X):
        users, items, ratings = [], [], []
        for user in range(X.shape[0]):
            for item in range(X.shape[1]):
                users.append(user)
                items.append(item)
                ratings.append(1.0 if X[user, item] > 0 else 0.0)
        
        return (torch.tensor(users, dtype=torch.long),
                torch.tensor(items, dtype=torch.long),
                torch.tensor(ratings, dtype=torch.float))

class DeepFMModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, hidden_layers):
        super(DeepFMModel, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Factorization Machine part
        self.fm_linear = nn.Linear(embedding_dim * 2, 1)
        
        # Deep part
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*layers)
        
        self.final_layer = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        concat = torch.cat([user_emb, item_emb], dim=1)
        
        # FM part
        fm_out = self.fm_linear(concat)
        
        # Deep part
        deep_out = self.deep(concat)
        
        # Combine
        combined = torch.cat([fm_out, deep_out], dim=1)
        output = self.final_layer(combined)
        return self.sigmoid(output).squeeze()

class SASRecRecommender:
    def __init__(self, embedding_dim=16, n_heads=2, n_layers=2, learning_rate=0.001, epochs=5):
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X):
        self.X = X
        self.n_users, self.n_items = X.shape
        
        # Создаем модель
        self.model = SASRecModel(self.n_users, self.n_items, self.embedding_dim, self.n_heads, self.n_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()
        
        # Подготавливаем данные
        user_indices, item_indices, ratings = self._prepare_data(X)
        dataset = TensorDataset(user_indices, item_indices, ratings)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Обучение
        self.model.train()
        for epoch in range(self.epochs):
            for users, items, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(users, items)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def predict(self):
        self.model.eval()
        predictions = np.zeros((self.n_users, self.n_items))
        
        with torch.no_grad():
            for user in range(self.n_users):
                for item in range(self.n_items):
                    user_tensor = torch.tensor([user], dtype=torch.long)
                    item_tensor = torch.tensor([item], dtype=torch.long)
                    pred = self.model(user_tensor, item_tensor).item()
                    predictions[user, item] = pred
        
        return predictions

    def _prepare_data(self, X):
        users, items, ratings = [], [], []
        for user in range(X.shape[0]):
            for item in range(X.shape[1]):
                users.append(user)
                items.append(item)
                ratings.append(1.0 if X[user, item] > 0 else 0.0)
        
        return (torch.tensor(users, dtype=torch.long),
                torch.tensor(items, dtype=torch.long),
                torch.tensor(ratings, dtype=torch.float))

class SASRecModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, n_heads, n_layers):
        super(SASRecModel, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.final_layer = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Combine embeddings
        combined = user_emb + item_emb
        
        # Apply transformer
        transformer_out = self.transformer(combined.unsqueeze(0))
        
        # Final prediction
        output = self.final_layer(transformer_out.squeeze(0))
        return self.sigmoid(output).squeeze()

# ============================ Оценка ============================
def get_used_services(X):
    return [set(np.where(row > 0)[0]) for row in X]

def get_recommendations(scores, used, popular, k):
    recs = []
    for i, user_scores in enumerate(scores):
        sorted_items = np.argsort(user_scores)[::-1]
        filtered = [item for item in sorted_items if item not in used[i]]
        if len(filtered) < k:
            extra = [item for item in popular if item not in used[i] and item not in filtered]
            filtered += extra
        recs.append(filtered[:k])
    return recs

def evaluate(recs, actual, all_items, scores, k):
    results = {'accuracy': [], 'precision': [], 'recall': [], 'ndcg': []}
    for i in range(len(recs)):
        y_true = [1 if j in actual[i] else 0 for j in range(len(all_items))]
        y_pred = [1 if j in recs[i] else 0 for j in range(len(all_items))]

        rel_true = [1 if j in actual[i] else 0 for j in recs[i]]
        rel_score = [scores[i][j] if j in recs[i] else 0 for j in recs[i]]

        results['accuracy'].append(accuracy_score(y_true, y_pred))
        results['precision'].append(precision_score(y_true, y_pred, zero_division=0))
        results['recall'].append(recall_score(y_true, y_pred, zero_division=0))
        
        if sum(rel_true) > 0 and len(rel_score) > 0:
            try:
                ndcg_val = ndcg_score([rel_true], [rel_score])
                results['ndcg'].append(ndcg_val)
            except:
                results['ndcg'].append(0)
        else:
            results['ndcg'].append(0)

    return {m: np.mean(vals) for m, vals in results.items()}

# ============================ Основной код сравнения ============================
print("Подготовка данных...")
used_train = get_used_services(X_train)
actual_test = get_used_services(X_test)

# Список ВСЕХ моделей для сравнения
models = {
    'KNN': KNNRecommender(),
    'PCA': SVDRecommender(PCA(n_components=10)),
    'NMF': SVDRecommender(NMF(n_components=10)),
    'SVD': SVDRecommender(TruncatedSVD(n_components=10)),
    'ALS': ALSRecommender(),
    'LightFM-WARP': LightFMRecommender(loss='warp', epochs=10),
    'PHCF-BPR': LightFMRecommender(loss='bpr', epochs=5),
    'KNN+LightFM-WARP': HybridKNNLightFMRecommender(loss='warp', epochs=10),
    'KNN+PHCF-BPR': HybridKNNLightFMRecommender(loss='bpr', epochs=5),
    'WRMF': WRMFRecommender(),
    'NCF': NCFRecommender(epochs=3),
    'DeepFM': DeepFMRecommender(epochs=3),
    'SASRec': SASRecRecommender(epochs=3),
    'Popular': PopularRecommender(popular_services),
    'Random': RandomRecommender()
}

print("Обучение и оценка моделей...")
results = {}
k_values = [5, 10, 15]

for name, model in models.items():
    print(f"\n=== {name} ===")
    
    try:
        # Обучение модели
        if name in ['LightFM-WARP', 'PHCF-BPR']:
            model.fit(df_train, owners, mids)
        elif name in ['KNN+LightFM-WARP', 'KNN+PHCF-BPR']:
            model.fit(df_train, owners, mids, X_train)
        else:
            model.fit(X_train)
        
        # Получение предсказаний
        preds = model.predict()
        
        # Оценка на всех k
        model_results = {}
        for k in k_values:
            recs = get_recommendations(preds, used_train, popular_services, k)
            metrics = evaluate(recs, actual_test, mids, preds, k)
            model_results[k] = metrics
            print(f"k={k}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, NDCG={metrics['ndcg']:.4f}")
        
        # Средние метрики
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'ndcg']:
            avg_metrics[metric] = np.mean([model_results[k][metric] for k in k_values])
        
        results[name] = avg_metrics
        
    except Exception as e:
        print(f"Ошибка при обучении {name}: {e}")
        continue

# ============================ Сводная таблица ============================
print("\n" + "="*50)
print("СВОДНАЯ ТАБЛИЦА СРЕДНИХ МЕТРИК ПО МОДЕЛЯМ")
print("="*50)

for name, metrics in results.items():
    print(f"{name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Ndcg: {metrics['ndcg']:.4f}")

# ============================ Экспорт в Excel ============================
print("\nЭкспорт результатов в Excel...")

# Создаем DataFrame
df_results = pd.DataFrame(results).T
df_results.columns = ['Accuracy', 'Precision', 'Recall', 'Ndcg']

# Сортируем по Precision
df_results = df_results.sort_values('Precision', ascending=False)

# Сохраняем в Excel
with pd.ExcelWriter('refactored_complete_comparison_summary.xlsx', engine='openpyxl') as writer:
    df_results.to_excel(writer, sheet_name='Results', index=True)
    
    # Получаем рабочую книгу для форматирования
    workbook = writer.book
    worksheet = writer.sheets['Results']
    
    # Форматирование заголовков
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = openpyxl.styles.Font(bold=True, color="FFFFFF")
    
    # Автоматическая ширина столбцов
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 20)
        worksheet.column_dimensions[column_letter].width = adjusted_width

print("Результаты сохранены в refactored_complete_comparison_summary.xlsx")

# ============================ Визуализация ============================
print("\nСоздание графиков...")

# Создаем фигуру с подграфиками
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# График 1: Precision
axes[0, 0].bar(range(len(results)), [metrics['precision'] for metrics in results.values()])
axes[0, 0].set_title('Precision по моделям', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(range(len(results)))
axes[0, 0].set_xticklabels(list(results.keys()), rotation=45, ha='right')
axes[0, 0].set_ylabel('Precision')
axes[0, 0].grid(True, alpha=0.3)

# График 2: Recall
axes[0, 1].bar(range(len(results)), [metrics['recall'] for metrics in results.values()])
axes[0, 1].set_title('Recall по моделям', fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(range(len(results)))
axes[0, 1].set_xticklabels(list(results.keys()), rotation=45, ha='right')
axes[0, 1].set_ylabel('Recall')
axes[0, 1].grid(True, alpha=0.3)

# График 3: NDCG
axes[1, 0].bar(range(len(results)), [metrics['ndcg'] for metrics in results.values()])
axes[1, 0].set_title('NDCG по моделям', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(range(len(results)))
axes[1, 0].set_xticklabels(list(results.keys()), rotation=45, ha='right')
axes[1, 0].set_ylabel('NDCG')
axes[1, 0].grid(True, alpha=0.3)

# График 4: Accuracy
axes[1, 1].bar(range(len(results)), [metrics['accuracy'] for metrics in results.values()])
axes[1, 1].set_title('Accuracy по моделям', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(range(len(results)))
axes[1, 1].set_xticklabels(list(results.keys()), rotation=45, ha='right')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('refactored_complete_comparison_results.png', dpi=300, bbox_inches='tight')
print("Графики сохранены в refactored_complete_comparison_results.png")

# ============================ Дополнительная визуализация ============================
print("\nСоздание дополнительных графиков...")

# Тепловая карта метрик
plt.figure(figsize=(12, 8))
metrics_matrix = np.array([[metrics['precision'], metrics['recall'], metrics['ndcg'], metrics['accuracy']] 
                          for metrics in results.values()])
sns.heatmap(metrics_matrix.T, 
            xticklabels=list(results.keys()), 
            yticklabels=['Precision', 'Recall', 'NDCG', 'Accuracy'],
            annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Значение метрики'})
plt.title('Тепловая карта метрик по моделям', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('refactored_complete_comparison_heatmap.png', dpi=300, bbox_inches='tight')

# Радарная диаграмма для топ-5 моделей
top_models = sorted(results.items(), key=lambda x: x[1]['precision'], reverse=True)[:5]
model_names = [model[0] for model in top_models]
metrics_names = ['Precision', 'Recall', 'NDCG', 'Accuracy']

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
angles += angles[:1]  # Замыкаем круг

for i, (model_name, metrics) in enumerate(top_models):
    values = [metrics['precision'], metrics['recall'], metrics['ndcg'], metrics['accuracy']]
    values += values[:1]  # Замыкаем круг
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
    ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_names)
ax.set_ylim(0, 1)
ax.set_title('Радарная диаграмма топ-5 моделей', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('refactored_complete_comparison_radar.png', dpi=300, bbox_inches='tight')

print("Дополнительные графики сохранены")

print("\nСравнение завершено!") 