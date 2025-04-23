# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 设置全局字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征矩阵
y = iris.target  # 目标向量
feature_names = iris.feature_names  # 特征名称
target_names = iris.target_names  # 类别名称

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 进行PCA降维
pca = PCA(n_components=2)  # 设置降维后的主成分数目为2
X_pca = pca.fit_transform(X_scaled)

# 输出降维后的数据形状
print("原始数据形状:", X.shape)
print("降维后数据形状:", X_pca.shape)

# 输出各主成分的方差解释比例
print("各主成分的方差解释比例:", pca.explained_variance_ratio_)
print("前两个主成分的累计方差解释比例之和:", sum(pca.explained_variance_ratio_))

# 可视化降维后的数据
plt.figure(figsize=(8, 6))
for i in range(len(target_names)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_names[i])
plt.title('PCA降维后的鸢尾花数据集')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.legend()
plt.show()

# 分类效果对比实验
# 使用原始数据进行KNN分类
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
knn_original = KNeighborsClassifier(n_neighbors=3)
knn_original.fit(X_train, y_train)
y_pred_original = knn_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)
print("原始四维数据KNN分类的准确率:", accuracy_original)

# 使用降维后的数据进行KNN分类
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_pca.fit(X_pca_train, y_train)
y_pred_pca = knn_pca.predict(X_pca_test)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print("降维后二维数据KNN分类的准确率:", accuracy_pca)