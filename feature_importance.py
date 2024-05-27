import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# データの読み込み
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# 前処理
data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
data = pd.get_dummies(data, drop_first=True)
data = data.fillna(data.mean())

# 特徴量同士の相関を計算
correlation_matrix = data.corr()

# ヒートマップの作成
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.savefig('feature_correlation_matrix.png')
