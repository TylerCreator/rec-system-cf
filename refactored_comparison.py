import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
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
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
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
        self.model = AlternatingLeastSquares(factors=20, iterations=10)
        self.X = csr_matrix(X * 20)
        self.model.fit(self.X)

    def predict(self):
        return self.model.user_factors @ self.model.item_factors.T

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
        self.num_items = X.shape[1]

    def predict(self):
        return np.random.rand(self.X.shape[0], self.num_items)

# ============================ LightFM методы ============================
class LightFMRecommender:
    def __init__(self, loss='warp', no_components=20, epochs=20):
        self.loss = loss
        self.no_components = no_components
        self.epochs = epochs

    def fit(self, df, owners, mids):
        dataset = Dataset()
        dataset.fit(owners, mids)
        interactions, _ = dataset.build_interactions([(row['owner'], row['mid']) for _, row in df.iterrows()])
        self.model = LightFM(loss=self.loss, no_components=self.no_components)
        self.model.fit(interactions, epochs=self.epochs, num_threads=4)
        self.dataset = dataset
        self.owners = owners
        self.mids = mids

    def predict(self):
        uid_map = self.dataset.mapping()[0]
        iid_map = self.dataset.mapping()[2]
        num_items = len(iid_map)
        preds = []
        for user in self.owners:
            uid = uid_map[user]
            scores = self.model.predict(uid, np.arange(num_items))
            preds.append(scores)
        return np.array(preds)

class HybridKNNLightFMRecommender:
    def __init__(self, loss='warp', no_components=20, epochs=20, knn_neighbors=4, alpha=1):
        self.loss = loss
        self.no_components = no_components
        self.epochs = epochs
        self.knn_neighbors = knn_neighbors
        self.alpha = alpha

    def fit(self, df_train, owners, mids, X_train):
        knn_model = KNNRecommender(n_neighbors=self.knn_neighbors)
        knn_model.fit(X_train)
        knn_preds = knn_model.predict()

        hybrid_matrix = X_train + self.alpha * knn_preds

        dataset = Dataset()
        dataset.fit(owners, mids)
        interactions, _ = dataset.build_interactions([(owners[i], mids[j]) for i, j in zip(*hybrid_matrix.nonzero())])

        self.model = LightFM(loss=self.loss, no_components=self.no_components)
        self.model.fit(interactions, epochs=self.epochs, num_threads=4)

        self.dataset = dataset
        self.owners = owners
        self.mids = mids

    def predict(self):
        uid_map = self.dataset.mapping()[0]
        iid_map = self.dataset.mapping()[2]
        num_items = len(iid_map)
        preds = []
        for user in self.owners:
            uid = uid_map[user]
            scores = self.model.predict(uid, np.arange(num_items))
            preds.append(scores)
        return np.array(preds)

# ============================ Нейронные методы ============================
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user, item):
        user_emb = self.user_embed(user)
        item_emb = self.item_embed(item)
        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.fc_layers(x).squeeze()

class NCFRecommender:
    def __init__(self, epochs=5, embed_dim=32, device='cpu'):
        self.epochs = epochs
        self.embed_dim = embed_dim
        self.device = device

    def fit(self, df_interactions, owners, mids):
        user2id = {u: i for i, u in enumerate(owners)}
        item2id = {i: j for j, i in enumerate(mids)}

        users = df_interactions['user_id'].map(user2id).values
        items = df_interactions['item_id'].map(item2id).values
        ratings = df_interactions['rating'].values

        self.model = NCF(len(owners), len(mids), self.embed_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        users_tensor = torch.tensor(users, dtype=torch.long).to(self.device)
        items_tensor = torch.tensor(items, dtype=torch.long).to(self.device)
        ratings_tensor = torch.tensor(ratings, dtype=torch.float32).to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            preds = self.model(users_tensor, items_tensor)
            loss = criterion(preds, ratings_tensor)
            loss.backward()
            optimizer.step()

    def predict(self):
        self.model.eval()
        num_users, num_items = self.model.user_embed.num_embeddings, self.model.item_embed.num_embeddings
        with torch.no_grad():
            user_idx = torch.arange(num_users).to(self.device)
            item_idx = torch.arange(num_items).to(self.device)
            preds = torch.zeros((num_users, num_items)).to(self.device)
            for u in user_idx:
                users_batch = u.repeat(num_items)
                preds[u] = self.model(users_batch, item_idx)
        return preds.cpu().numpy()

class DeepFMRecommender:
    def __init__(self, embedding_dim=8, epochs=10, batch_size=256, lr=1e-3, verbose=0):
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.user2id = {}
        self.item2id = {}
        self.feature_columns = None
        self.feature_names = None
        self.model = None

    def fit(self, df_train, owners, mids):
        self.user2id = {user: idx for idx, user in enumerate(owners)}
        self.item2id = {item: idx for idx, item in enumerate(mids)}

        train_df = df_train.copy()
        train_df["user_id"] = train_df["owner"].map(self.user2id)
        train_df["item_id"] = train_df["mid"].map(self.item2id)

        train_df = train_df.dropna(subset=["user_id", "item_id"])
        train_df[["user_id", "item_id"]] = train_df[["user_id", "item_id"]].astype(int)

        num_users = len(owners)
        num_items = len(mids)

        positive_interactions = train_df[["user_id", "item_id"]]
        user_list, item_list, labels = [], [], []
        items_set = set(range(num_items))

        for user in positive_interactions["user_id"].unique():
            pos_items = set(positive_interactions[positive_interactions["user_id"] == user]["item_id"])
            num_pos = len(pos_items)
            num_neg = min(num_pos, len(items_set - pos_items))

            neg_candidates = list(items_set - pos_items)
            if len(neg_candidates) >= num_neg:
                neg_items = np.random.choice(neg_candidates, num_neg, replace=False)
            else:
                neg_items = neg_candidates

            user_list.extend([user] * (num_pos + len(neg_items)))
            item_list.extend(list(pos_items) + list(neg_items))
            labels.extend([1] * num_pos + [0] * len(neg_items))

        train_model_input = {
            "user_id": np.array(user_list),
            "item_id": np.array(item_list)
        }
        labels = np.array(labels)

        self.feature_columns = [
            SparseFeat("user_id", vocabulary_size=num_users, embedding_dim=self.embedding_dim),
            SparseFeat("item_id", vocabulary_size=num_items, embedding_dim=self.embedding_dim)
        ]

        self.feature_names = get_feature_names(self.feature_columns)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DeepFM(
            linear_feature_columns=self.feature_columns,
            dnn_feature_columns=self.feature_columns,
            task="binary",
            device=device
        )

        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.model.fit(
            train_model_input,
            labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            validation_split=0.0
        )

    def predict(self):
        num_users = len(self.user2id)
        num_items = len(self.item2id)

        user_arr = np.repeat(np.arange(num_users), num_items)
        item_arr = np.tile(np.arange(num_items), num_users)

        model_input = {
            "user_id": user_arr,
            "item_id": item_arr
        }

        preds = self.model.predict(model_input, batch_size=self.batch_size).reshape(num_users, num_items)
        return preds

class SASRec(nn.Module):
    def __init__(self, num_items, embed_dim=50, max_seq_len=50, num_heads=1, num_layers=2, dropout=0.2):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, num_items + 1)

    def forward(self, seq):
        seq_len = seq.size(1)
        pos_ids = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand_as(seq)
        seq_emb = self.item_emb(seq) + self.pos_emb(pos_ids)
        mask = (seq == 0)
        transformer_out = self.transformer(seq_emb, src_key_padding_mask=mask)
        logits = self.fc(transformer_out[:, -1, :])
        return logits

class SASRecRecommender:
    def __init__(self, epochs=10, embed_dim=50, max_seq_len=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.epochs = epochs
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.device = device

    def fit(self, df_train, owners, mids):
        self.user2id = {user: idx for idx, user in enumerate(owners)}
        self.item2id = {item: idx+1 for idx, item in enumerate(mids)}

        user_seq = df_train.sort_values('start_time').groupby('owner')['mid'].apply(
            lambda items: [self.item2id[i] for i in items if i in self.item2id])

        self.model = SASRec(num_items=len(mids), embed_dim=self.embed_dim,
                            max_seq_len=self.max_seq_len).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for user, seq in user_seq.items():
                if len(seq) < 2:
                    continue
                seq_input = seq[:-1][-self.max_seq_len:]
                target = seq[-1]
                seq_input = [0]*(self.max_seq_len - len(seq_input)) + seq_input
                seq_tensor = torch.tensor([seq_input], device=self.device)
                target_tensor = torch.tensor([target], device=self.device)

                optimizer.zero_grad()
                logits = self.model(seq_tensor)
                loss = loss_fn(logits, target_tensor)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

    def predict(self):
        self.model.eval()
        num_users = len(self.user2id)
        num_items = len(self.item2id)
        scores = np.zeros((num_users, num_items))

        for user, uidx in self.user2id.items():
            seq = []
            seq_tensor = torch.tensor([[0]*self.max_seq_len], device=self.device)
            with torch.no_grad():
                logits = self.model(seq_tensor)
                scores[uidx, :] = logits.cpu().numpy()[0, 1:]

        return scores

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
        results['ndcg'].append(ndcg_score([rel_true], [rel_score]) if sum(rel_true) > 0 else 0)

    return {m: np.mean(vals) for m, vals in results.items()}

# ============================ Основной код ============================
print("Подготовка данных...")
used_train = get_used_services(X_train)
actual_test = get_used_services(X_test)

# Подготовка данных для нейронных моделей
user_indices, item_indices = X_train.nonzero()
ratings = X_train[user_indices, item_indices]

df_interactions = pd.DataFrame({
    'user_id': np.array(owners)[user_indices],
    'item_id': np.array(mids)[item_indices],
    'rating': ratings
})

# Определение всех моделей
models = {
    'KNN': KNNRecommender(),
    'PCA': SVDRecommender(PCA(n_components=5)),
    'NMF': SVDRecommender(NMF(n_components=5, init='random', random_state=42)),
    'SVD': SVDRecommender(TruncatedSVD(n_components=5)),
    'ALS': ALSRecommender(),
    'LightFM-WARP': LightFMRecommender(loss='warp'),
    'PHCF-BPR': LightFMRecommender(loss='bpr'),
    'KNN+LightFM-WARP': HybridKNNLightFMRecommender(loss='warp'),
    'KNN+PHCF-BPR': HybridKNNLightFMRecommender(loss='bpr'),
    'WRMF': WRMFRecommender(),
    'NCF': NCFRecommender(),
    'DeepFM': DeepFMRecommender(),
    'SASRec': SASRecRecommender(),
    'Popular': PopularRecommender(popular_services),
    'Random': RandomRecommender(),
}

k_values = [5, 10, 15]
results = {model: {k: {} for k in k_values} for model in models.keys()}

print("Обучение и оценка моделей...")
for name, model in models.items():
    print(f"\n=== {name} ===")
    try:
        if name in ['LightFM-WARP', 'PHCF-BPR', 'DeepFM', 'SASRec']:
            model.fit(df_train, owners, mids)
        elif name in ['NCF']:
            model.fit(df_interactions, owners, mids)
        elif name in ['KNN+LightFM-WARP', 'KNN+PHCF-BPR']:
            model.fit(df_train, owners, mids, X_train)
        else:
            model.fit(X_train)

        preds = model.predict()
        for k in k_values:
            recs = get_recommendations(preds, used_train, popular_services, k)
            metrics = evaluate(recs, actual_test, mids, preds, k)
            results[name][k] = metrics
            print(f"k={k}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, NDCG={metrics['ndcg']:.4f}")
    except Exception as e:
        print(f"Ошибка в модели {name}: {e}")
        # Заполняем нулевыми метриками в случае ошибки
        for k in k_values:
            results[name][k] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'ndcg': 0}

# ============================ Сводная таблица ============================
print("\n" + "="*50)
print("СВОДНАЯ ТАБЛИЦА СРЕДНИХ МЕТРИК ПО МОДЕЛЯМ")
print("="*50)

avg_summary = {}
for model in models.keys():
    avg = {metric: np.mean([results[model][k][metric] for k in k_values]) for metric in ['accuracy', 'precision', 'recall', 'ndcg']}
    avg_summary[model] = avg
    print(f"{model}:")
    for m, v in avg.items():
        print(f"  {m.capitalize()}: {v:.4f}")

# ============================ Экспорт в Excel ============================
print("\nЭкспорт результатов в Excel...")
avg_summary_df = pd.DataFrame(avg_summary).T
avg_summary_df.index.name = 'Model'

writer = pd.ExcelWriter("model_comparison_summary.xlsx", engine="openpyxl")

# Средние метрики
avg_summary_df.to_excel(writer, sheet_name="Averages")
ws = writer.sheets["Averages"]

# Добавляем цветовую шкалу
color_rule = ColorScaleRule(start_type='min', start_color='FFFFFF',
                            mid_type='percentile', mid_value=50, mid_color='FFFF99',
                            end_type='max', end_color='63BE7B')

for col in range(2, 6):  # B to E (метрики)
    col_letter = chr(64 + col)
    ws.conditional_formatting.add(f"{col_letter}2:{col_letter}{1+len(avg_summary_df)}", color_rule)

# Метрики по каждому k
for k in k_values:
    k_df = pd.DataFrame({model: results[model][k] for model in models.keys()}).T
    k_df.index.name = 'Model'
    k_df.to_excel(writer, sheet_name=f"k={k}")

writer.close()
print("Результаты сохранены в model_comparison_summary.xlsx")

# ============================ Визуализация ============================
print("\nСоздание графиков...")
for metric in ['accuracy', 'precision', 'recall', 'ndcg']:
    plt.figure(figsize=(15, 6))
    for k in k_values:
        values = [results[model][k][metric] for model in models.keys()]
        plt.bar([f"{model}\nk={k}" for model in models.keys()], values, alpha=0.6, label=f"k={k}")
    plt.title(f"{metric.upper()} by model and k")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric}_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

print("\nСравнение завершено!") 