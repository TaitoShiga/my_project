import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# データの読み込み
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# 前処理
data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
data = pd.get_dummies(data, drop_first=True)
data = data.fillna(data.mean())

# 特徴量とターゲットの分割
X = data.drop(columns=['Survived'])
y = data['Survived']

# 特徴量の数を確認
print(f"元の特徴量の数: {X.shape[1]}")

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレストのモデルを作成
rf = RandomForestClassifier(random_state=42)

# 特徴量選択を行う前のモデルの学習と評価
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy_before = accuracy_score(y_test, y_pred)
print(f"特徴量選択前の正解率: {accuracy_before:.4f}")

# 特徴量重要度を取得
importances = rf.feature_importances_

# 重要度に基づいて特徴量選択を行う
threshold = 0.01
selected_features = X.columns[importances > threshold]
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# 選択した特徴量の数を確認
print(f"選択された特徴量の数: {X_train_selected.shape[1]}")

# 特徴量選択後のモデルの学習と評価
rf.fit(X_train_selected, y_train)
y_pred_selected = rf.predict(X_test_selected)
accuracy_after = accuracy_score(y_test, y_pred_selected)
print(f"特徴量選択後の正解率: {accuracy_after:.4f}")

# 重要度のプロット
plt.figure(figsize=(12, 6))
plt.barh(X.columns, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.savefig('feature_importances.png')
