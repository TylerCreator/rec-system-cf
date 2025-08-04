import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score

# Фиксируем random seed для воспроизводимости
np.random.seed(42)

print("🔍 АНАЛИЗ РАСЧЕТА PRECISION И ПОПУЛЯРНОСТИ")
print("="*60)

# ============================ Загрузка данных ============================
df = pd.read_csv("calls.csv", sep=";")
df = df[~df['owner'].str.contains('cookies', na=False)]
df = df.sort_values('start_time')

owners = df['owner'].unique()
mids = df['mid'].unique()

train_size = int(len(df) * 0.7)
df_train = df[:train_size]
df_test = df[train_size:]

print(f"📊 Общая статистика данных:")
print(f"  - Всего пользователей: {len(owners)}")
print(f"  - Всего сервисов: {len(mids)}")
print(f"  - Тренировочных записей: {len(df_train)}")
print(f"  - Тестовых записей: {len(df_test)}")

# ============================ Анализ популярности ============================
def get_popular_services(df, mids):
    counts = df['mid'].value_counts().reindex(mids, fill_value=0)
    counts_max = np.max(counts.values)
    return np.argsort(counts.values/counts_max)[::-1]

popular_services = get_popular_services(df_train, mids)
popularity_counts = df_train['mid'].value_counts()

print(f"\n📈 Анализ популярности сервисов:")
print(f"  - Самый популярный сервис: {popularity_counts.index[0]} ({popularity_counts.iloc[0]} использований)")
print(f"  - Топ-5 популярных сервисов:")
for i in range(5):
    service = popularity_counts.index[i]
    count = popularity_counts.iloc[i]
    percentage = (count / len(df_train)) * 100
    print(f"    {i+1}. Сервис {service}: {count} раз ({percentage:.1f}% от всех записей)")

# ============================ Функции для анализа ============================
def build_matrix(df, owners, mids, normalize=False):
    pivot = df.pivot_table(index='owner', columns='mid', values='id', aggfunc='count').fillna(0)
    pivot = pivot.reindex(index=owners, columns=mids, fill_value=0)
    mat = pivot.values
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat = mat / row_sums
    return mat

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

# ============================ Подготовка данных ============================
X_train = build_matrix(df_train, owners, mids, normalize=True)
X_test = build_matrix(df_test, owners, mids, normalize=True)

used_train = get_used_services(X_train)
actual_test = get_used_services(X_test)

print(f"\n🔢 Статистика использования сервисов:")
services_per_user_train = [len(used) for used in used_train]
services_per_user_test = [len(actual) for actual in actual_test]

print(f"  - Среднее количество сервисов на пользователя (train): {np.mean(services_per_user_train):.2f}")
print(f"  - Среднее количество сервисов на пользователя (test): {np.mean(services_per_user_test):.2f}")
print(f"  - Медиана сервисов на пользователя (train): {np.median(services_per_user_train):.2f}")
print(f"  - Медиана сервисов на пользователя (test): {np.median(services_per_user_test):.2f}")

# ============================ Анализ Popular модели ============================
print(f"\n🏆 АНАЛИЗ POPULAR МОДЕЛИ:")
print(f"  Popular модель всегда рекомендует одни и те же топ-популярные сервисы")

# Создаем рекомендации Popular модели
popular_predictions = np.tile(popular_services, (len(owners), 1))
popular_recs_k5 = get_recommendations(popular_predictions, used_train, popular_services, 5)
popular_recs_k10 = get_recommendations(popular_predictions, used_train, popular_services, 10)

# Анализируем пересечения с тестовыми данными
print(f"\n📊 Анализ пересечений Popular модели с тестовыми данными:")

hits_k5 = []
hits_k10 = []
precision_k5 = []
precision_k10 = []

for i in range(len(actual_test)):
    # k=5
    recs_5 = set(popular_recs_k5[i][:5])
    actual_5 = actual_test[i]
    hits_5 = len(recs_5.intersection(actual_5))
    hits_k5.append(hits_5)
    precision_k5.append(hits_5 / 5 if len(recs_5) > 0 else 0)
    
    # k=10
    recs_10 = set(popular_recs_k10[i][:10])
    hits_10 = len(recs_10.intersection(actual_5))
    hits_k10.append(hits_10)
    precision_k10.append(hits_10 / 10 if len(recs_10) > 0 else 0)

print(f"  - Среднее количество попаданий при k=5: {np.mean(hits_k5):.3f}")
print(f"  - Среднее количество попаданий при k=10: {np.mean(hits_k10):.3f}")
print(f"  - Средний Precision при k=5: {np.mean(precision_k5):.4f}")
print(f"  - Средний Precision при k=10: {np.mean(precision_k10):.4f}")

# ============================ Анализ почему Popular эффективен ============================
print(f"\n💡 ПОЧЕМУ POPULAR МОДЕЛЬ ЭФФЕКТИВНА:")

# Анализ распределения популярности в тестовых данных
test_service_counts = df_test['mid'].value_counts()
top_5_popular = popularity_counts.head(5).index.tolist()
top_10_popular = popularity_counts.head(10).index.tolist()

test_usage_top5 = sum([test_service_counts.get(service, 0) for service in top_5_popular])
test_usage_top10 = sum([test_service_counts.get(service, 0) for service in top_10_popular])
total_test_usage = len(df_test)

print(f"  1. 📈 Концентрация популярности:")
print(f"     - Топ-5 сервисов покрывают {(test_usage_top5/total_test_usage)*100:.1f}% тестового трафика")
print(f"     - Топ-10 сервисов покрывают {(test_usage_top10/total_test_usage)*100:.1f}% тестового трафика")

# Анализ стабильности популярности между train и test
print(f"\n  2. 🎯 Стабильность популярности между train и test:")
for i, service in enumerate(top_5_popular):
    train_rank = i + 1
    test_count = test_service_counts.get(service, 0)
    test_rank = test_service_counts.index.get_loc(service) + 1 if service in test_service_counts.index else "не в топе"
    print(f"     - Сервис {service}: train_rank={train_rank}, test_rank={test_rank}, test_count={test_count}")

# ============================ Сравнение с Random моделью ============================
print(f"\n🎲 СРАВНЕНИЕ С RANDOM МОДЕЛЬЮ:")

# Random рекомендации
random_predictions = np.random.rand(len(owners), len(mids))
random_recs_k5 = get_recommendations(random_predictions, used_train, popular_services, 5)

random_hits_k5 = []
random_precision_k5 = []

for i in range(len(actual_test)):
    recs_5 = set(random_recs_k5[i][:5])
    actual_5 = actual_test[i]
    hits_5 = len(recs_5.intersection(actual_5))
    random_hits_k5.append(hits_5)
    random_precision_k5.append(hits_5 / 5 if len(recs_5) > 0 else 0)

print(f"  - Random модель - средний Precision при k=5: {np.mean(random_precision_k5):.4f}")
print(f"  - Popular модель - средний Precision при k=5: {np.mean(precision_k5):.4f}")
print(f"  - Улучшение Popular над Random: {(np.mean(precision_k5) / np.mean(random_precision_k5) - 1) * 100:.1f}%")

# ============================ Объяснение формулы Precision ============================
print(f"\n📐 КАК СЧИТАЕТСЯ PRECISION:")
print(f"  Precision = True Positives / (True Positives + False Positives)")
print(f"  Precision = Количество правильных рекомендаций / Общее количество рекомендаций")
print(f"  ")
print(f"  Для k=5: Precision = Попадания из 5 рекомендаций / 5")
print(f"  Для k=10: Precision = Попадания из 10 рекомендаций / 10")

print(f"\n🎯 ВЫВОДЫ:")
print(f"  1. Popular модель эффективна из-за концентрации пользовательских предпочтений")
print(f"  2. Популярные сервисы стабильны между train и test периодами")
print(f"  3. Принцип Парето: малое количество сервисов покрывает большую часть трафика")
print(f"  4. Popular модель показывает высокий Precision, но может иметь низкий Recall")
print(f"  5. Для продакшена Popular хорош как baseline или fallback модель")

# ============================ Пример расчета для одного пользователя ============================
print(f"\n📝 ПРИМЕР РАСЧЕТА ДЛЯ ПОЛЬЗОВАТЕЛЯ №0:")
user_id = 0
actual_services = actual_test[user_id]
popular_recs = set(popular_recs_k5[user_id][:5])

print(f"  - Реальные сервисы пользователя в тесте: {actual_services}")
print(f"  - Рекомендации Popular модели (k=5): {popular_recs}")
print(f"  - Пересечение: {popular_recs.intersection(actual_services)}")
print(f"  - Количество попаданий: {len(popular_recs.intersection(actual_services))}")
print(f"  - Precision для этого пользователя: {len(popular_recs.intersection(actual_services)) / 5:.4f}")