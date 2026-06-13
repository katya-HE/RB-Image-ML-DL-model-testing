import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pickle
import os

# 1. 读取Excel文件
print("步骤1: 读取数据...")
file_path = '/Users/yaya/Desktop/RB_GLCM_features.xlsx'
try:
    # 读取sheet2（筛选出的7个特征）
    df = pd.read_excel(file_path, sheet_name=1)  # sheet2是第二个sheet，索引为1
    print(f"数据读取成功，形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
except Exception as e:
    print(f"读取文件失败: {e}")
    # 如果读取失败，使用你提供的示例数据
    print("使用示例数据...")
    data = {
        'imageid': [f'image_{i:03d}' for i in range(13)] * 2,
        'RB_Status': [0] * 13 + [1] * 13,  # 这里需要补充RB_Status=1的数据
        'glcm_mean': np.random.randn(26),
        'glcm_std': np.random.randn(26) * 100 + 200,
        'glcm_contrast': np.random.randn(26) * 10 + 20,
        'glcm_dissimilarity': np.random.randn(26) * 5 + 15,
        'glcm_homogeneity': np.random.randn(26) * 5 + 215,
        'glcm_asm': np.random.randn(26) * 5000 + 40000,
        'glcm_entropy': np.random.randn(26) * 0.2 + 6.5
    }
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱数据

# 2. 查看数据基本信息
print("\n步骤2: 数据基本信息...")
print(f"数据集形状: {df.shape}")
print(f"RB_Status分布:\n{df['RB_Status'].value_counts()}")
print(f"RB_Status比例:\n{df['RB_Status'].value_counts(normalize=True)}")

# 3. 准备特征和标签
print("\n步骤3: 准备特征和标签...")

# 从你提供的SPSS结果中获取系数
coefficients = {
    'glcm_mean': -15.668,
    'glcm_std': 0.054,
    'glcm_contrast': -2.163,
    'glcm_dissimilarity': 15.991,
    'glcm_homogeneity': 31.611,
    'glcm_asm': 0.008,
    'glcm_entropy': 283.946,
    'intercept': -9242.643
}

# 选择特征（按照SPSS筛选的顺序）
features = ['glcm_mean', 'glcm_std', 'glcm_contrast', 'glcm_dissimilarity', 
            'glcm_homogeneity', 'glcm_asm', 'glcm_entropy']

print(f"使用的特征: {features}")
print(f"特征数量: {len(features)}")

# 检查所有特征是否都存在
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"警告: 以下特征在数据中不存在: {missing_features}")
    # 尝试查找相似的列名
    for f in missing_features:
        similar_cols = [col for col in df.columns if f in col.lower()]
        if similar_cols:
            print(f"  可能匹配的列: {similar_cols}")
else:
    print("所有特征都存在")

X = df[features].values
y = df['RB_Status'].values

print(f"特征矩阵形状: {X.shape}")
print(f"标签形状: {y.shape}")

# 4. 数据重排和分层抽样
print("\n步骤4: 准备十折交叉验证...")

# 使用分层K折交叉验证，确保每折中类别比例相同
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 初始化存储结果的列表
auc_scores = []
accuracy_scores = []
sensitivity_scores = []  # 召回率/真阳性率
specificity_scores = []  # 特异性/真阴性率
fprs = []
tprs = []
thresholds_list = []
fold_predictions = []
fold_true_labels = []

# 5. 十折交叉验证
print("\n步骤5: 开始十折交叉验证...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n--- 第{fold}折 ---")
    
    # 划分训练集和验证集
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"训练集: {X_train.shape}, 类别分布: {np.bincount(y_train)}")
    print(f"验证集: {X_val.shape}, 类别分布: {np.bincount(y_val)}")
    
    # 6. 使用SPSS的系数初始化逻辑回归模型
    print("使用SPSS回归系数初始化模型...")
    
    # 创建逻辑回归模型
    model = LogisticRegression(
        penalty=None,  # 不使用正则化，因为我们使用固定系数
        fit_intercept=True,
        max_iter=1000,
        random_state=42
    )
    
    # 使用SPSS的系数（注意：sklearn的系数顺序可能不同）
    # 我们需要手动设置系数
    model.fit(X_train, y_train)  # 先拟合一次获取系数形状
    
    # 手动设置系数和截距
    coef_array = np.array([coefficients[f] for f in features]).reshape(1, -1)
    model.coef_ = coef_array
    model.intercept_ = np.array([coefficients['intercept']])
    
    print(f"模型系数: {model.coef_}")
    print(f"模型截距: {model.intercept_}")
    
    # 7. 预测
    y_pred_prob = model.predict_proba(X_val)[:, 1]  # 预测概率
    y_pred = (y_pred_prob >= 0.5).astype(int)  # 使用0.5作为阈值
    
    # 8. 计算性能指标
    # AUC
    auc = roc_auc_score(y_val, y_pred_prob)
    auc_scores.append(auc)
    
    # 准确率
    accuracy = accuracy_score(y_val, y_pred)
    accuracy_scores.append(accuracy)
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    
    # 灵敏度（召回率）
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivity_scores.append(sensitivity)
    
    # 特异度
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity_scores.append(specificity)
    
    # ROC曲线数据
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
    fprs.append(fpr)
    tprs.append(tpr)
    thresholds_list.append(thresholds)
    
    # 存储预测结果
    fold_predictions.append(y_pred_prob)
    fold_true_labels.append(y_val)
    
    print(f"AUC: {auc:.4f}")
    print(f"准确率: {accuracy:.4f}")
    print(f"灵敏度: {sensitivity:.4f}")
    print(f"特异度: {specificity:.4f}")
    print(f"混淆矩阵: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

# 9. 计算平均性能指标
print("\n" + "="*50)
print("十折交叉验证结果总结")
print("="*50)

print(f"\nAUC (10折平均): {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"AUC范围: {min(auc_scores):.4f} - {max(auc_scores):.4f}")
print(f"准确率 (10折平均): {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"灵敏度 (10折平均): {np.mean(sensitivity_scores):.4f} ± {np.std(sensitivity_scores):.4f}")
print(f"特异度 (10折平均): {np.mean(specificity_scores):.4f} ± {np.std(specificity_scores):.4f}")

# 10. 绘制ROC曲线
print("\n步骤6: 绘制ROC曲线...")

plt.figure(figsize=(10, 8))

# 绘制每折的ROC曲线
for i in range(n_splits):
    plt.plot(fprs[i], tprs[i], lw=1, alpha=0.3, label=f'Fold {i+1} (AUC = {auc_scores[i]:.3f})')

# 绘制平均ROC曲线（微平均）
from numpy import interp

# 微平均ROC曲线
all_fpr = np.unique(np.concatenate([fpr for fpr in fprs]))
mean_tpr = np.zeros_like(all_fpr)

for i in range(n_splits):
    mean_tpr += interp(all_fpr, fprs[i], tprs[i])

mean_tpr /= n_splits

# 计算微平均AUC
from sklearn.metrics import auc as auc_calc
mean_auc = auc_calc(all_fpr, mean_tpr)

# 绘制平均ROC曲线
plt.plot(all_fpr, mean_tpr, color='red', lw=2, 
         label=f'平均ROC (AUC = {mean_auc:.3f})', linestyle='--')

# 绘制对角线
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (1 - 特异度)')
plt.ylabel('真阳性率 (灵敏度)')
plt.title('十折交叉验证 ROC 曲线\n(使用SPSS逻辑回归系数)')
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, alpha=0.3)

# 11. 保存结果到桌面
print("\n步骤7: 保存结果...")

# 创建结果字典
results = {
    'auc_scores': auc_scores,
    'accuracy_scores': accuracy_scores,
    'sensitivity_scores': sensitivity_scores,
    'specificity_scores': specificity_scores,
    'mean_auc': np.mean(auc_scores),
    'std_auc': np.std(auc_scores),
    'mean_accuracy': np.mean(accuracy_scores),
    'std_accuracy': np.std(accuracy_scores),
    'model_coefficients': coefficients,
    'features_used': features,
    'cross_validation_folds': n_splits
}

# 保存ROC曲线图片
desktop_path = '/Users/yaya/Desktop'
roc_curve_path = os.path.join(desktop_path, 'ROC_Curve_10fold_CV.png')
plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
print(f"ROC曲线已保存到: {roc_curve_path}")

# 保存结果到Excel
results_df = pd.DataFrame({
    'Fold': list(range(1, n_splits + 1)),
    'AUC': auc_scores,
    'Accuracy': accuracy_scores,
    'Sensitivity': sensitivity_scores,
    'Specificity': specificity_scores
})

# 添加平均值行
mean_row = pd.DataFrame({
    'Fold': ['Mean'],
    'AUC': [np.mean(auc_scores)],
    'Accuracy': [np.mean(accuracy_scores)],
    'Sensitivity': [np.mean(sensitivity_scores)],
    'Specificity': [np.mean(specificity_scores)]
})

results_df = pd.concat([results_df, mean_row], ignore_index=True)

excel_path = os.path.join(desktop_path, 'CV_Validation_Results.xlsx')
results_df.to_excel(excel_path, index=False)
print(f"交叉验证结果已保存到: {excel_path}")

# 保存详细结果到pickle文件
pickle_path = os.path.join(desktop_path, 'CV_Validation_Results.pkl')
with open(pickle_path, 'wb') as f:
    pickle.dump(results, f)
print(f"详细结果已保存到: {pickle_path}")

# 12. 显示最终总结
print("\n" + "="*50)
print("最终验证结果总结")
print("="*50)
print(f"使用SPSS逻辑回归系数:")
for feature, coef in coefficients.items():
    print(f"  {feature}: {coef}")

print(f"\n十折交叉验证平均性能:")
print(f"  AUC: {np.mean(auc_scores):.4f} (±{np.std(auc_scores):.4f})")
print(f"  准确率: {np.mean(accuracy_scores):.4f} (±{np.std(accuracy_scores):.4f})")
print(f"  灵敏度: {np.mean(sensitivity_scores):.4f} (±{np.std(sensitivity_scores):.4f})")
print(f"  特异度: {np.mean(specificity_scores):.4f} (±{np.std(specificity_scores):.4f})")

print(f"\n结果文件已保存到桌面:")
print(f"  1. ROC曲线图片: {roc_curve_path}")
print(f"  2. Excel结果文件: {excel_path}")
print(f"  3. Pickle详细结果: {pickle_path}")

# 显示ROC曲线
plt.show()