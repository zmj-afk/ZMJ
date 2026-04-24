import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 加载数据 =====================
df_bike = pd.read_csv("C:/Users/18788/Documents/毕业论文/trips_clean.csv", low_memory=False)
df_weather = pd.read_csv("C:/Users/18788/Documents/毕业论文/芝加哥_真实天气_202411_202510.csv")
df_weather['date'] = pd.to_datetime(df_weather['date'])

# ===================== 2. 构建小时级需求 =====================
df_bike['started_at'] = pd.to_datetime(df_bike['started_at'], errors='coerce')
df_bike['date'] = df_bike['started_at'].dt.date
df_bike['date'] = pd.to_datetime(df_bike['date'])
df_bike['hour'] = df_bike['started_at'].dt.hour

# 目标变量：小时需求量
demand = df_bike.groupby(['date', 'hour']).agg(
    demand=('start_station_id', 'count'),
).reset_index()

# ===================== 3. 融合天气 =====================
data = pd.merge(demand, df_weather, on='date', how='left')
data = data.dropna()

# ===================== 4. 构建四大维度特征 =====================
# 时间特征
data['weekday'] = data['date'].dt.weekday
data['is_weekend'] = (data['weekday'] >= 5).astype(int)
data['is_rush'] = data['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

# 环境特征
data['temp_avg'] = (data['temp_max'] + data['temp_min']) / 2

# 空间特征（用站点活跃度代表）
station_hot = df_bike.groupby(['date', 'hour'])['start_station_id'] \
                     .nunique().reset_index(name='station_count')
data = pd.merge(data, station_hot, on=['date', 'hour'], how='left')

# 运营特征（历史需求强度）
data['demand_lag1'] = data['demand'].shift(1).fillna(0)

# ===================== 5. 四大维度最终特征 =====================
features = [
    # 时间特征
    'hour', 'weekday', 'is_weekend', 'is_rush',
    # 环境特征
    'temp_avg', 'precip_mm', 'wind_max',
    # 空间特征
    'station_count',
    # 运营特征
    'demand_lag1'
]

X = data[features]
y = data['demand']

# ===================== 6. 随机森林模型 =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# 评估指标
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("=" * 60)
print("随机森林模型评估")
print(f"R²  = {r2:.3f}")
print(f"RMSE = {rmse:.2f}")
print(f"MAE  = {mae:.2f}")
print("=" * 60)

# ===================== 7. 特征重要性（四大维度） =====================
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=imp.values, y=imp.index, palette='viridis')
plt.title('四大维度特征重要性排序')
plt.tight_layout()
plt.savefig("1_特征重要性.png", dpi=300)
plt.close()

# ===================== 8. 小时需求趋势图 =====================
hour_mean = data.groupby('hour')['demand'].mean()
plt.figure(figsize=(12, 5))
plt.plot(hour_mean.index, hour_mean.values, marker='o', color='#2E86AB')
plt.title('不同时段共享单车平均需求量')
plt.xticks(range(0, 24))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("2_时段需求趋势.png", dpi=300)
plt.close()

# ===================== 9. 工作日 vs 周末 =====================
plt.figure(figsize=(8, 5))
sns.boxplot(x='is_weekend', y='demand', data=data, palette='Set2')
plt.xticks([0, 1], ['工作日', '周末'])
plt.title('工作日与周末骑行需求对比')
plt.tight_layout()
plt.savefig("3_工作日周末对比.png", dpi=300)
plt.close()

# ===================== 10. 温度与需求关系 =====================
plt.figure(figsize=(10, 5))
sns.scatterplot(x='temp_avg', y='demand', data=data, alpha=0.4, color='#A23B72')
plt.title('温度与共享单车需求关系')
plt.tight_layout()
plt.savefig("4_温度需求关系.png", dpi=300)
plt.close()

# ===================== 11. 降雨对需求影响 =====================
data['is_rain'] = (data['precip_mm'] > 0).astype(int)
plt.figure(figsize=(8, 5))
sns.boxplot(x='is_rain', y='demand', data=data, palette='Set3')
plt.xticks([0, 1], ['无雨', '降雨'])
plt.title('降雨对骑行需求的影响')
plt.tight_layout()
plt.savefig("5_降雨影响.png", dpi=300)
plt.close()

# ===================== 12. 保存最终建模数据 =====================
data[features + ['demand']].to_csv("四大维度建模数据集.csv", index=False, encoding='utf-8-sig')

print("\n✅ 全部完成！已生成：")
print("1. 特征重要性图")
print("2. 时段需求趋势图")
print("3. 工作日/周末对比图")
print("4. 温度-需求关系图")
print("5. 降雨影响图")
print("6. 四大维度建模数据集.csv")