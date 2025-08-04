import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля для научных публикаций
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Устанавливаем шрифты для лучшего качества
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

print("🔬 СОЗДАНИЕ РАСШИРЕННОЙ НАУЧНОЙ ВИЗУАЛИЗАЦИИ")
print("="*60)

# Данные результатов экспериментов
results_data = {
    'PHCF-BPR': {'precision': 0.0696, 'recall': 0.0775, 'ndcg': 0.1729, 'accuracy': 0.9111, 'type': 'LightFM'},
    'KNN+PHCF-BPR': {'precision': 0.0667, 'recall': 0.0780, 'ndcg': 0.1778, 'accuracy': 0.9108, 'type': 'Hybrid'},
    'NCF': {'precision': 0.0538, 'recall': 0.0590, 'ndcg': 0.1628, 'accuracy': 0.9094, 'type': 'Neural'},
    'KNN': {'precision': 0.0474, 'recall': 0.0738, 'ndcg': 0.1483, 'accuracy': 0.9088, 'type': 'Collaborative'},
    'Popular': {'precision': 0.0661, 'recall': 0.0380, 'ndcg': 0.1140, 'accuracy': 0.9110, 'type': 'Baseline'},
    'LightFM-WARP': {'precision': 0.0509, 'recall': 0.0651, 'ndcg': 0.1426, 'accuracy': 0.9094, 'type': 'LightFM'},
    'WRMF': {'precision': 0.0579, 'recall': 0.0182, 'ndcg': 0.1021, 'accuracy': 0.9099, 'type': 'Matrix Factorization'},
    'SASRec': {'precision': 0.0515, 'recall': 0.0158, 'ndcg': 0.0952, 'accuracy': 0.9094, 'type': 'Neural'},
    'KNN+LightFM-WARP': {'precision': 0.0468, 'recall': 0.0611, 'ndcg': 0.1108, 'accuracy': 0.9092, 'type': 'Hybrid'},
    'NMF': {'precision': 0.0263, 'recall': 0.0300, 'ndcg': 0.0589, 'accuracy': 0.9071, 'type': 'Matrix Factorization'},
    'PCA': {'precision': 0.0281, 'recall': 0.0169, 'ndcg': 0.0634, 'accuracy': 0.9071, 'type': 'Matrix Factorization'},
    'DeepFM': {'precision': 0.0351, 'recall': 0.0085, 'ndcg': 0.0490, 'accuracy': 0.9078, 'type': 'Neural'},
    'Random': {'precision': 0.0287, 'recall': 0.0061, 'ndcg': 0.0359, 'accuracy': 0.9074, 'type': 'Baseline'},
    'SVD': {'precision': 0.0216, 'recall': 0.0103, 'ndcg': 0.0423, 'accuracy': 0.9066, 'type': 'Matrix Factorization'},
    'ALS': {'precision': 0.0181, 'recall': 0.0170, 'ndcg': 0.0429, 'accuracy': 0.9066, 'type': 'Matrix Factorization'}
}

# Создаем DataFrame для анализа
df = pd.DataFrame(results_data).T
df.reset_index(inplace=True)
df.rename(columns={'index': 'algorithm'}, inplace=True)

# Преобразуем числовые столбцы
numeric_columns = ['precision', 'recall', 'ndcg', 'accuracy']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col])

# Рассчитываем общий скор
df['overall_score'] = df['precision'] * 0.4 + df['recall'] * 0.3 + df['ndcg'] * 0.3

print("📊 Создание фигуры 1: Комплексный анализ производительности")

# Фигура 1: Комплексный анализ производительности (6 подграфиков)
fig = plt.figure(figsize=(16, 12))

# 1. Основные метрики по алгоритмам
ax1 = plt.subplot(3, 3, 1)
metrics = ['precision', 'recall', 'ndcg']
x = np.arange(len(df))
width = 0.25

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, df[metric], width, label=metric.upper(), alpha=0.8)

plt.xlabel('Алгоритмы')
plt.ylabel('Значение метрики')
plt.title('(a) Основные метрики по алгоритмам')
plt.xticks(x + width, df['algorithm'], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Scatter plot: Precision vs Recall
ax2 = plt.subplot(3, 3, 2)
type_colors = {'LightFM': 'red', 'Neural': 'blue', 'Collaborative': 'green', 
               'Matrix Factorization': 'orange', 'Baseline': 'purple', 'Hybrid': 'brown'}

for algo_type in df['type'].unique():
    subset = df[df['type'] == algo_type]
    plt.scatter(subset['precision'], subset['recall'], 
               label=algo_type, s=100, alpha=0.7, color=type_colors.get(algo_type, 'gray'))

# Добавляем аннотации для топ-3 алгоритмов
top3 = df.nlargest(3, 'overall_score')
for _, row in top3.iterrows():
    plt.annotate(row['algorithm'], (row['precision'], row['recall']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('(b) Precision vs Recall по типам алгоритмов')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# 3. Box plot по типам алгоритмов
ax3 = plt.subplot(3, 3, 3)
# Преобразуем данные для box plot
melted_df = df.melt(id_vars=['algorithm', 'type'], 
                   value_vars=['precision', 'recall', 'ndcg'],
                   var_name='metric', value_name='value')

sns.boxplot(data=melted_df, x='type', y='value', hue='metric')
plt.xticks(rotation=45, ha='right')
plt.title('(c) Распределение метрик по типам')
plt.ylabel('Значение метрики')

# 4. Корреляционная матрица
ax4 = plt.subplot(3, 3, 4)
corr_matrix = df[['precision', 'recall', 'ndcg', 'accuracy']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f')
plt.title('(d) Корреляционная матрица метрик')

# 5. Общий рейтинг
ax5 = plt.subplot(3, 3, 5)
df_sorted = df.sort_values('overall_score', ascending=True)
colors = [type_colors.get(t, 'gray') for t in df_sorted['type']]
plt.barh(range(len(df_sorted)), df_sorted['overall_score'], color=colors, alpha=0.7)
plt.yticks(range(len(df_sorted)), df_sorted['algorithm'])
plt.xlabel('Общий скор')
plt.title('(e) Общий рейтинг алгоритмов')
plt.grid(True, alpha=0.3)

# 6. Radar chart для топ-5
ax6 = plt.subplot(3, 3, 6, projection='polar')
top5 = df.nlargest(5, 'overall_score')
metrics_radar = ['precision', 'recall', 'ndcg', 'accuracy']
angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
angles += angles[:1]

for i, (_, row) in enumerate(top5.iterrows()):
    values = [row[metric] for metric in metrics_radar]
    values += values[:1]
    ax6.plot(angles, values, 'o-', linewidth=2, label=row['algorithm'])
    ax6.fill(angles, values, alpha=0.1)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels([m.upper() for m in metrics_radar])
ax6.set_title('(f) Radar chart топ-5 алгоритмов')
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# 7. PCA анализ
ax7 = plt.subplot(3, 3, 7)
features = df[['precision', 'recall', 'ndcg', 'accuracy']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)

for algo_type in df['type'].unique():
    mask = df['type'] == algo_type
    plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
               label=algo_type, s=100, alpha=0.7, color=type_colors.get(algo_type, 'gray'))

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('(g) PCA: Двумерное представление')
plt.legend()
plt.grid(True, alpha=0.3)

# 8. Статистическое сравнение по типам
ax8 = plt.subplot(3, 3, 8)
type_scores = df.groupby('type')['overall_score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
plt.errorbar(range(len(type_scores)), type_scores['mean'], 
             yerr=type_scores['std'], fmt='o-', capsize=5, capthick=2)
plt.xticks(range(len(type_scores)), type_scores.index, rotation=45, ha='right')
plt.ylabel('Средний общий скор')
plt.title('(h) Сравнение типов алгоритмов')
plt.grid(True, alpha=0.3)

# 9. Efficiency frontier (Precision vs NDCG)
ax9 = plt.subplot(3, 3, 9)
plt.scatter(df['precision'], df['ndcg'], s=df['recall']*1000, 
           c=[type_colors.get(t, 'gray') for t in df['type']], alpha=0.6)

# Добавляем frontier line
df_sorted_prec = df.sort_values('precision')
plt.plot(df_sorted_prec['precision'], df_sorted_prec['ndcg'], '--', alpha=0.5, color='gray')

plt.xlabel('Precision')
plt.ylabel('NDCG')
plt.title('(i) Efficiency Frontier\n(размер = Recall)')
plt.grid(True, alpha=0.3)

# Аннотации для лучших алгоритмов
for _, row in df.nlargest(3, 'overall_score').iterrows():
    plt.annotate(row['algorithm'], (row['precision'], row['ndcg']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.tight_layout()
plt.savefig('scientific_analysis_figure1.png', dpi=300, bbox_inches='tight')
print("✅ Фигура 1 сохранена как 'scientific_analysis_figure1.png'")

print("\n📈 Создание фигуры 2: Статистический анализ")

# Фигура 2: Статистический анализ
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Гистограммы распределений метрик
for i, metric in enumerate(['precision', 'recall', 'ndcg']):
    ax = axes[0, i]
    ax.hist(df[metric], bins=8, alpha=0.7, edgecolor='black')
    ax.axvline(df[metric].mean(), color='red', linestyle='--', label=f'Среднее: {df[metric].mean():.4f}')
    ax.axvline(df[metric].median(), color='blue', linestyle='--', label=f'Медиана: {df[metric].median():.4f}')
    ax.set_xlabel(metric.upper())
    ax.set_ylabel('Частота')
    ax.set_title(f'Распределение {metric.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 2. Q-Q plots для проверки нормальности
for i, metric in enumerate(['precision', 'recall', 'ndcg']):
    ax = axes[1, i]
    stats.probplot(df[metric], dist="norm", plot=ax)
    ax.set_title(f'Q-Q plot для {metric.upper()}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scientific_analysis_figure2.png', dpi=300, bbox_inches='tight')
print("✅ Фигура 2 сохранена как 'scientific_analysis_figure2.png'")

print("\n📊 Создание фигуры 3: Сравнение методологий")

# Фигура 3: Детальное сравнение методологий
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Среднее и доверительные интервалы по типам
ax = axes[0, 0]
type_metrics = df.groupby('type')[['precision', 'recall', 'ndcg']].agg(['mean', 'std'])

types = type_metrics.index
x_pos = np.arange(len(types))
width = 0.25

for i, metric in enumerate(['precision', 'recall', 'ndcg']):
    means = type_metrics[(metric, 'mean')]
    stds = type_metrics[(metric, 'std')]
    ax.bar(x_pos + i*width, means, width, yerr=stds, 
           label=metric.upper(), alpha=0.8, capsize=3)

ax.set_xlabel('Тип алгоритма')
ax.set_ylabel('Значение метрики')
ax.set_title('Среднее ± std по типам алгоритмов')
ax.set_xticks(x_pos + width)
ax.set_xticklabels(types, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Тепловая карта производительности
ax = axes[0, 1]
# Нормализуем данные для лучшей визуализации
metrics_for_heatmap = ['precision', 'recall', 'ndcg', 'overall_score']
df_norm = df.copy()
for metric in metrics_for_heatmap:
    df_norm[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

heatmap_data = df_norm.set_index('algorithm')[metrics_for_heatmap].T
sns.heatmap(heatmap_data, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Нормализованное значение'})
ax.set_title('Нормализованная производительность')
ax.set_xlabel('Алгоритмы')

# 3. Парные корреляции
ax = axes[1, 0]
metrics_corr = df[['precision', 'recall', 'ndcg', 'accuracy']].corr()
# Создаем маску для верхнего треугольника
mask = np.triu(np.ones_like(metrics_corr, dtype=bool))
sns.heatmap(metrics_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.3f', cbar_kws={'label': 'Корреляция'})
ax.set_title('Матрица корреляций метрик')

# 4. Производительность vs сложность (имитация)
ax = axes[1, 1]
# Примерная оценка сложности (относительная)
complexity_scores = {
    'Random': 1, 'Popular': 2, 'KNN': 4, 'PCA': 3, 'NMF': 5, 'SVD': 4,
    'ALS': 6, 'WRMF': 5, 'LightFM-WARP': 7, 'PHCF-BPR': 7,
    'KNN+LightFM-WARP': 8, 'KNN+PHCF-BPR': 8, 'NCF': 9, 'DeepFM': 10, 'SASRec': 10
}

df['complexity'] = df['algorithm'].map(complexity_scores)

for algo_type in df['type'].unique():
    subset = df[df['type'] == algo_type]
    ax.scatter(subset['complexity'], subset['overall_score'], 
              label=algo_type, s=100, alpha=0.7, color=type_colors.get(algo_type, 'gray'))

ax.set_xlabel('Сложность алгоритма (относительная)')
ax.set_ylabel('Общий скор')
ax.set_title('Производительность vs Сложность')
ax.legend()
ax.grid(True, alpha=0.3)

# Добавляем trend line
z = np.polyfit(df['complexity'], df['overall_score'], 1)
p = np.poly1d(z)
ax.plot(df['complexity'], p(df['complexity']), "--", alpha=0.5, color='red')

plt.tight_layout()
plt.savefig('scientific_analysis_figure3.png', dpi=300, bbox_inches='tight')
print("✅ Фигура 3 сохранена как 'scientific_analysis_figure3.png'")

print("\n📋 Создание таблиц для научной статьи")

# Создание статистических таблиц
print("\n📊 Таблица 1: Основные результаты")
results_table = df[['algorithm', 'precision', 'recall', 'ndcg', 'accuracy', 'overall_score', 'type']].copy()
results_table = results_table.sort_values('overall_score', ascending=False)
results_table.index = range(1, len(results_table) + 1)
results_table.columns = ['Алгоритм', 'Precision', 'Recall', 'NDCG', 'Accuracy', 'Общий скор', 'Тип']

# Форматируем числа
for col in ['Precision', 'Recall', 'NDCG', 'Accuracy', 'Общий скор']:
    results_table[col] = results_table[col].round(4)

print(results_table.to_string())

print("\n📊 Таблица 2: Статистика по типам алгоритмов")
type_stats = df.groupby('type')[['precision', 'recall', 'ndcg', 'accuracy', 'overall_score']].agg(['mean', 'std', 'count'])
type_stats.columns = ['_'.join(col).strip() for col in type_stats.columns.values]
type_stats = type_stats.round(4)
print(type_stats.to_string())

print("\n📊 Таблица 3: Корреляционная матрица")
correlation_matrix = df[['precision', 'recall', 'ndcg', 'accuracy']].corr().round(3)
print(correlation_matrix.to_string())

print("\n📊 Таблица 4: Топ-5 алгоритмов по каждой метрике")
for metric in ['precision', 'recall', 'ndcg', 'accuracy']:
    print(f"\nТоп-5 по {metric.upper()}:")
    top5 = df.nlargest(5, metric)[['algorithm', metric, 'type']]
    top5.index = range(1, 6)
    print(top5.to_string())

# Сохраняем таблицы в CSV для дальнейшего использования
results_table.to_csv('scientific_results_table.csv', index=True)
type_stats.to_csv('scientific_type_statistics.csv', index=True)
correlation_matrix.to_csv('scientific_correlation_matrix.csv', index=True)

print(f"\n✅ Анализ завершен! Создано:")
print(f"  - 3 научных рисунка (PNG)")
print(f"  - 3 таблицы результатов (CSV)")
print(f"  - Статистические показатели")

print(f"\n📈 Основные выводы:")
print(f"  1. Лучший алгоритм: {results_table.iloc[0]['Алгоритм']} (скор: {results_table.iloc[0]['Общий скор']:.4f})")
print(f"  2. Лучший тип: {type_stats.loc[type_stats['overall_score_mean'].idxmax()].name}")
print(f"  3. Самая высокая корреляция: {correlation_matrix.abs().unstack().sort_values(ascending=False).iloc[1]:.3f}")

# Создание финального summary plot
plt.figure(figsize=(12, 8))

# Создаем comprehensive plot
gs = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)

# 1. Общий рейтинг
ax1 = plt.subplot(gs[0, :])
top10 = results_table.head(10)
colors = [type_colors.get(t, 'gray') for t in top10['Тип']]
bars = plt.bar(range(len(top10)), top10['Общий скор'], color=colors, alpha=0.8)
plt.xticks(range(len(top10)), top10['Алгоритм'], rotation=45, ha='right')
plt.ylabel('Общий скор')
plt.title('Топ-10 алгоритмов рекомендательных систем', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Добавляем значения на столбцы
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Средние по типам
ax2 = plt.subplot(gs[1, 0])
type_means = df.groupby('type')['overall_score'].mean().sort_values(ascending=False)
colors_type = [type_colors.get(t, 'gray') for t in type_means.index]
plt.bar(range(len(type_means)), type_means.values, color=colors_type, alpha=0.8)
plt.xticks(range(len(type_means)), type_means.index, rotation=45, ha='right')
plt.ylabel('Средний скор')
plt.title('Средние скоры по типам')
plt.grid(True, alpha=0.3)

# 3. Precision vs Recall
ax3 = plt.subplot(gs[1, 1])
for algo_type in df['type'].unique():
    subset = df[df['type'] == algo_type]
    plt.scatter(subset['precision'], subset['recall'], 
               label=algo_type, s=80, alpha=0.7, color=type_colors.get(algo_type, 'gray'))
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision vs Recall')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True, alpha=0.3)

# 4. Метрики лидера
ax4 = plt.subplot(gs[1, 2])
leader = results_table.iloc[0]
metrics_leader = [leader['Precision'], leader['Recall'], leader['NDCG'], leader['Accuracy']]
metric_names = ['Precision', 'Recall', 'NDCG', 'Accuracy']
colors_metrics = ['red', 'blue', 'green', 'orange']
plt.bar(metric_names, metrics_leader, color=colors_metrics, alpha=0.8)
plt.title(f"Метрики лидера\n({leader['Алгоритм']})")
plt.xticks(rotation=45)
plt.ylabel('Значение')
plt.grid(True, alpha=0.3)

plt.suptitle('Комплексный анализ рекомендательных систем', fontsize=16, fontweight='bold')
plt.savefig('scientific_analysis_summary.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Итоговая фигура сохранена как 'scientific_analysis_summary.png'")

print(f"\n🎯 НАУЧНЫЙ АНАЛИЗ ЗАВЕРШЕН!")
print(f"Все файлы готовы для включения в научную статью.")