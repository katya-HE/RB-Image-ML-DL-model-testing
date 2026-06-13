# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# 1. 读取Excel数据
print("=" * 60)
print("开始多变量逻辑回归分析")
print("=" * 60)

file_path = '/Users/yaya/Desktop/F50.xlsx'
print(f"正在读取文件: {file_path}")

try:
    df = pd.read_excel(file_path, sheet_name=0)
    print("✓ 文件读取成功")
except Exception as e:
    print(f"✗ 读取文件失败: {e}")
    exit()

print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()[:10]}...")  # 显示前10个列名

# 2. 准备自变量(X)和因变量(y)
print("\n" + "-" * 60)
print("准备自变量和因变量")
print("-" * 60)

# 检查必要的列
if 'RB_Status' not in df.columns:
    print("✗ 数据中缺少RB_Status列")
    exit()

y = df['RB_Status']  # 因变量

# 提取特征列 - 从C列开始（索引2）
# 先找出所有可能的特征列
feature_start_idx = 2  # C列是索引2
X = df.iloc[:, feature_start_idx:feature_start_idx+50]  # 取50列

print(f"自变量X的形状: {X.shape}")
print(f"因变量y的形状: {y.shape}")
print(f"\nRB标签分布:")
print(y.value_counts())
print(f"RB比例: {y.mean():.2%}")

# 3. 使用statsmodels进行逻辑回归分析
print("\n" + "=" * 60)
print("使用statsmodels进行逻辑回归分析")
print("=" * 60)

# 添加常数项（截距）
X_const = sm.add_constant(X)
print("正在拟合逻辑回归模型...")

try:
    model = sm.Logit(y, X_const)
    result = model.fit(maxiter=1000, disp=0)
    print("✓ 模型拟合成功")
except Exception as e:
    print(f"✗ 模型拟合失败: {e}")
    print("尝试减少特征数量或增加迭代次数...")
    
    # 尝试减少特征数量
    print("尝试使用前30个特征...")
    X_reduced = df.iloc[:, feature_start_idx:feature_start_idx+30]
    X_const_reduced = sm.add_constant(X_reduced)
    model = sm.Logit(y, X_const_reduced)
    result = model.fit(maxiter=1000, disp=0)
    print("✓ 使用30个特征拟合成功")
    X = X_reduced  # 更新X为减少后的特征

print("\n逻辑回归模型摘要:")
print(result.summary())

# 提取系数和p值
coefficients = result.params  # 系数
p_values = result.pvalues    # p值
odds_ratios = np.exp(coefficients)  # OR值 = exp(系数)

# 创建系数、OR值和p值表
coeff_table = pd.DataFrame({
    '变量': coefficients.index,
    '系数': coefficients.values,
    'OR值': odds_ratios.values,
    'p值': p_values.values
})

# 按p值排序
coeff_table_sorted = coeff_table.sort_values('p值')

print("\n所有变量的系数、OR值和p值（按p值排序）:")
print(coeff_table_sorted.to_string(index=False))

# 4. 使用向后剔除法（递归特征消除RFE）筛选特征
print("\n" + "=" * 60)
print("使用向后剔除法（RFE）筛选重要特征")
print("=" * 60)

# 创建逻辑回归模型用于RFE
## SPSS向后剔除法的逻辑：
#if p_value > 0.05:剔除该变量
    
# VSCode RFE的逻辑：按系数绝对值排序：
#coef_abs = abs(coefficients)剔除 coef_abs 最小的变量

logreg = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)

# 使用RFE选择最重要的特征
num_features_to_select = min(15, X.shape[1])
print(f"计划选择 {num_features_to_select} 个最重要的特征")

rfe = RFE(logreg, n_features_to_select=num_features_to_select)
rfe = rfe.fit(X, y)

# 获取被选中的特征
selected_features = X.columns[rfe.support_].tolist()

print(f"\n✓ RFE选择的 {len(selected_features)} 个重要特征:")
for i, feature in enumerate(selected_features, 1):
    print(f"{i:2d}. {feature}")

# 5. 使用筛选后的特征重新建模
print("\n" + "=" * 60)
print("使用筛选后的特征重新建模")
print("=" * 60)

# 使用选中的特征重新训练模型
X_selected = X[selected_features]
X_selected_const = sm.add_constant(X_selected)

# 拟合新模型
print("正在使用筛选后的特征拟合模型...")
model_selected = sm.Logit(y, X_selected_const)
result_selected = model_selected.fit(maxiter=1000, disp=0)

print("\n筛选后模型的摘要:")
print(result_selected.summary())

# 6. 十倍交叉验证
print("\n" + "=" * 60)
print("十倍交叉验证")
print("=" * 60)

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 打乱数据
indices = np.arange(len(X))
np.random.shuffle(indices)

X_shuffled = X.iloc[indices].reset_index(drop=True)
y_shuffled = y.iloc[indices].reset_index(drop=True)

# 初始化十倍交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 存储每个fold的结果
accuracy_scores = []
auc_scores = []

print("\n开始十倍交叉验证...")
print(f"{'Fold':^6} | {'准确度':^10} | {'AUC':^10}")
print("-" * 30)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_shuffled), 1):
    # 分割训练集和验证集
    X_train, X_val = X_shuffled.iloc[train_idx], X_shuffled.iloc[val_idx]
    y_train, y_val = y_shuffled.iloc[train_idx], y_shuffled.iloc[val_idx]
    
    # 使用筛选的特征进行训练
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    
    # 训练逻辑回归模型
    model_cv = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    model_cv.fit(X_train_selected, y_train)
    
    # 在验证集上进行预测
    y_pred = model_cv.predict(X_val_selected)
    y_pred_proba = model_cv.predict_proba(X_val_selected)[:, 1]
    
    # 计算评估指标
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    accuracy_scores.append(accuracy)
    auc_scores.append(auc)
    
    print(f"{fold:^6} | {accuracy:^10.4f} | {auc:^10.4f}")

print("-" * 30)

# 计算平均性能指标
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

print(f"\n平均准确度: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
print(f"平均AUC: {mean_auc:.4f} (±{std_auc:.4f})")

# 7. 输出最终逻辑回归公式
print("\n" + "=" * 60)
print("最终逻辑回归模型公式")
print("=" * 60)

# 在整个数据集上训练最终模型
X_final = X[selected_features]
final_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
final_model.fit(X_final, y)

# 获取截距和系数
intercept = final_model.intercept_[0]
coefficients_final = final_model.coef_[0]

print("逻辑回归模型公式:")
print("P(RB=1) = 1 / (1 + exp(-z))")
print("\n其中:")

formula_parts = [f"{intercept:.6f}"]
for feature, coef in zip(selected_features, coefficients_final):
    if coef >= 0:
        formula_parts.append(f"+ {coef:.6f}×{feature}")
    else:
        formula_parts.append(f"- {abs(coef):.6f}×{feature}")

formula = "z = " + " ".join(formula_parts)
print(formula)

# 8. 保存结果
print("\n" + "=" * 60)
print("保存结果")
print("=" * 60)

# 保存系数表
coeff_table_sorted.to_csv('/Users/yaya/Desktop/logistic_coefficients.csv', index=False, encoding='utf-8-sig')
print("✓ 系数表已保存到: logistic_coefficients.csv")

# 保存交叉验证结果
cv_results = pd.DataFrame({
    'Fold': range(1, 11),
    '准确度': accuracy_scores,
    'AUC': auc_scores
})
cv_results.to_csv('/Users/yaya/Desktop/cross_validation_results.csv', index=False, encoding='utf-8-sig')
print("✓ 交叉验证结果已保存到: cross_validation_results.csv")

# 保存模型公式
with open('/Users/yaya/Desktop/logistic_formula.txt', 'w', encoding='utf-8') as f:
    f.write("逻辑回归模型公式\n")
    f.write("=" * 40 + "\n\n")
    f.write("P(RB=1) = 1 / (1 + exp(-z))\n\n")
    f.write("其中:\n")
    f.write(formula + "\n\n")
    f.write(f"平均准确度: {mean_accuracy:.4f} (±{std_accuracy:.4f})\n")
    f.write(f"平均AUC: {mean_auc:.4f} (±{std_auc:.4f})\n")

print("✓ 模型公式已保存到: logistic_formula.txt")
print("\n分析完成！")
      