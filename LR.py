import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 读取CSV文件

file_path = './data/complications_final-Copy1.xlsx'
data = pd.read_excel(file_path)

# 处理特征
# 编码性别特征
data['性别'] = data['性别'].map({'女': 0, '男': 1})

# 提取年龄中的数字部分并转换为浮点数
data['年龄'] = data['年龄'].str.extract(r'(\d+)').astype(float)

# 编码口服药物特征

data['糖化血红蛋白'] = data['糖化血红蛋白'].str.rstrip('%').astype(float) / 100.0

data['肌酐'] = data['肌酐'].str.replace(r'[^\d.]', '', regex=True).astype(float)

data['甘油三酯'] = data['甘油三酯'].str.replace(r'[^\d.]', '', regex=True).astype(float)

data['胆固醇'] = data['胆固醇'].str.replace(r'[^\d.]', '', regex=True).astype(float)

data['高密度脂蛋白'] = data['高密度脂蛋白'].str.replace(r'[^\d.]', '', regex=True).astype(float)

data['低密度脂蛋白'] = data['低密度脂蛋白'].str.replace(r'[^\d.]', '', regex=True).astype(float)

data['脂蛋白a'] = data['脂蛋白a'].str.replace(r'[^\d.]', '', regex=True).astype(float)

# 处理目标变量
data['脑梗'] = data['脑梗'].apply(lambda x: 1 if x == 1 else 0)

# 准备特征矩阵 X 和目标向量 y
X = data.drop(['脑梗'], axis=1)  # 特征矩阵
y = data['脑梗']  # 目标向量

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 创建和训练逻辑回归模型

logistic_model = LogisticRegression(random_state=42, max_iter=1000)


logistic_model.fit(X_train, y_train)

# 使用模型进行预测
y_pred = logistic_model.predict(X_test)

# 计算准确率、精确率、召回率和F1值
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 打印结果
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

