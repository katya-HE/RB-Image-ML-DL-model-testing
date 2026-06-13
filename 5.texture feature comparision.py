#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视网膜母细胞瘤(RB)分类模型对比分析
=====================================
对比四种纹理分析方法：LBP, HOS, Haralick-Fractal, Combine
使用SPSS生成的逻辑回归系数进行十折交叉验证

作者：kimi Assistant
日期：2026-03-25
修改说明：仅对HOS方法的数据进行0-1标准化处理❤ 结果：0.658→0.634 更小了。。。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix, 
                           recall_score, roc_curve, precision_score, f1_score)
import seaborn as sns
import os
import warnings
from datetime import datetime
import json
from sklearn.preprocessing import MinMaxScaler  # 导入标准化工具❤！！！

warnings.filterwarnings('ignore')

# ==================== 1. 环境配置与路径设置 ====================

# 设置中文字体（Mac系统）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义数据路径（请根据实际路径修改）
DATA_PATH = "/Users/yaya/Desktop/together.xlsx"  # 包含四个sheet的Excel文件

# 定义输出路径（保存到桌面）
DESKTOP_PATH = os.path.expanduser("~/Desktop")
OUTPUT_DIR = os.path.join(DESKTOP_PATH, f"RB_Model_Comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"结果将保存到: {OUTPUT_DIR}")

# ==================== 2. 定义SPSS模型参数 ====================
"""
注意：以下系数来自SPSS逻辑回归分析结果
每个方法使用SPSS筛选出的最优特征子集及其对应系数
"""

# LBP方法：26个LBP特征（来自SPSS向后逐步回归）
LBP_FEATURES = {
    'LBP_feature005': 0.541,
    'LBP_feature007': 1.348,
    'LBP_feature008': 0.768,
    'LBP_feature013': -1.458,
    'LBP_feature014': 1.218,
    'LBP_feature019': 0.988,
    'LBP_feature021': -1.419,
    'LBP_feature022': 1.356,
    'LBP_feature027': -0.808,
    'LBP_feature028': -0.755,
    'LBP_feature031': -1.002,
    'LBP_feature033': 0.935,
    'LBP_feature035': 0.866,
    'LBP_feature036': -2.262,
    'LBP_feature038': 1.783,
    'LBP_feature041': -0.710,
    'LBP_feature043': 1.036,
    'LBP_feature046': 0.820,
    'LBP_feature048': 0.982,
    'LBP_feature052': 0.448,
    'LBP_feature056': -1.751,
    'LBP_feature063': 1.426,
    'LBP_feature067': -1.188,
    'LBP_feature070': -0.739,
    'LBP_feature072': -1.770,
    'LBP_feature073': -0.807
}

# HOS方法：9个HOS特征
HOS_FEATURES = {
    'e1r1': -3.986,
    'e1r2': -1.717,
    'e1r9': 2.092,
    'e2r5': 2.748,  
    'ePRes8': -0.460,
    'amRes1': 0.000,
    'amRes4': 0.000,
    'amRes7': 0.000,
    'amRes10': 0.000
}

# Haralick-Fractal方法：11个纹理+分形特征
HARALICK_FRACTAL_FEATURES = {
    'auto': 29.358,
    'dissi': 197.514,
    'savgh': -94.887,
    'dvarh': -27.840,
    'inf1h': -32.439,
    'mfaslope': -50.628,
    'mfaIC': -10.130,
    'Hdfdimension': 20.772,
    'minaq': -172.212,
    'minfq': 6.063,
    'maxfq': 137.323
}

# Combine方法：22个特征（综合上述三种方法的精选特征）
# 注意：这是代码1使用的版本（高AUC版本）
COMBINE_FEATURES = {
    'Hdfdimension': 52.793,      # 分形维数
    'auto': 99.334,              # 自相关
    'cprom': -0.106,             # 聚类显著性
    'dissi': 283.008,            # 差异性
    'savgh': -267.550,           # 平均和方差
    'dvarh': -44.059,            # 差异方差
    'inf1h': -46.313,            # 信息熵
    'maxfq': 110.948,            # 最大频率
    'LBP_feature007': 0.841,
    'LBP_feature008': 2.192,
    'LBP_feature014': -1.429,
    'LBP_feature021': -1.861,
    'LBP_feature022': 1.815,
    'LBP_feature031': -1.135,
    'LBP_feature045': 1.607,
    'LBP_feature048': 0.733,
    'LBP_feature052': 0.530,
    'LBP_feature060': -2.572,
    'LBP_feature063': 1.571,
    'LBP_feature067': -1.280,
    'LBP_feature072': -0.837,
    'LBP_feature079': -0.670
}

# 各方法的截距项（来自SPSS常数项）
INTERCEPTS = {
    'LBP': -3.403,
    'HOS': 0.597,
    'Haralick-Fractal': 233.912,
    'Combine': 125.701
}

# 打包所有模型配置
MODEL_CONFIGS = {
    'LBP': LBP_FEATURES,
    'HOS': HOS_FEATURES,
    'Haralick-Fractal': HARALICK_FRACTAL_FEATURES,
    'Combine': COMBINE_FEATURES
}

# ==================== 3. 数据加载与预处理 ====================

def load_data(excel_path):
    """
    从Excel文件加载四个方法的数据   
    关键修改：对HOS方法的数据进行0-1标准化（Min-Max Scaling）
    返回:
        data_dict: 字典，键为方法名，值为DataFrame
    """
    print("\n" + "="*60)
    print("步骤1: 加载数据")
    print("="*60)
    
    sheets = ['LBP', 'HOS', 'Haralick-Fractal', 'Combine']
    data_dict = {}
    
    for sheet in sheets:
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet)
            # ==================== 关键修改：HOS数据标准化 ====================
            if sheet == 'HOS':
                print(f"  🔄 对 {sheet} 进行0-1标准化处理...")
                df = scale_hos_features(df)
            # ================================================================
            data_dict[sheet] = df
            
            # 检查必要列
            if 'RB_Status' not in df.columns:
                print(f"  ⚠️ 警告: {sheet} 缺少 'RB_Status' 列")
            
            print(f"  ✓ {sheet}: {df.shape[0]} 样本 × {df.shape[1]} 特征")
            
        except Exception as e:
            print(f"  ✗ 错误: 无法加载 {sheet} - {str(e)}")
    
    return data_dict

def scale_hos_features(df):
    """
    对HOS特征进行0-1标准化（Min-Max Scaling）
    标准化公式：X_scaled = (X - X_min) / (X_max - X_min)
    注意：
    1. 只对HOS特征列进行标准化，保留RB_Status列不变
    2. 使用sklearn的MinMaxScaler确保数值稳定性
    3. 如果某特征所有值相同（max=min），则保持原值（避免除零）
    参数:
        df: HOS方法的原始DataFrame
    返回:
        df_scaled: 标准化后的DataFrame
    """
    # 复制数据，避免修改原始数据
    df_scaled = df.copy()
    
    # 获取数值型特征列（排除RB_Status和任何非数值列如ImageName）
    # 只选择数值类型的列，且排除目标变量RB_Status
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != 'RB_Status']
    
    if len(feature_cols) == 0:
        print("    ⚠️ 没有找到数值型HOS特征列")
        return df_scaled
    
    print(f"    找到 {len(feature_cols)} 个数值型特征列")
    
    # 初始化MinMaxScaler（范围0-1）
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 对数值特征进行标准化
    for col in feature_cols:
        col_data = df_scaled[[col]].values  # 保持二维数组格式
        
        # 检查是否为常数列（所有值相同）
        col_min, col_max = col_data.min(), col_data.max()
        if col_max == col_min:
            print(f"    ⚠️ 特征 {col} 为常数列（值={col_min}），跳过标准化")
            continue
        
        # 标准化该列
        df_scaled[col] = scaler.fit_transform(col_data)
    
    print(f"    ✓ 完成 {len(feature_cols)} 个特征的标准化")
    
    # 打印标准化后的统计信息（仅显示前5个特征）
    print("\n    标准化后统计摘要（前5个特征）:")
    display_cols = feature_cols[:5] if len(feature_cols) > 5 else feature_cols
    stats = df_scaled[display_cols].describe().loc[['min', 'max', 'mean', 'std']]
    print(stats.round(3).to_string())
    
    return df_scaled

# ==================== 4. SPSS逻辑回归模型类 ====================

class SPSSLogisticRegression:
    """
    使用SPSS系数的逻辑回归模型
    
    特点:
    1. 直接使用SPSS训练好的系数，不在Python中重新训练
    2. 保持与SPSS完全一致的预测逻辑
    3. 支持概率预测和类别预测
    
    公式: P(y=1|x) = 1 / (1 + exp(-(intercept + Σ(coef_i * x_i))))
    """
    
    def __init__(self, coefficients, intercept=0.0, feature_names=None):
        """
        初始化模型
        
        参数:
            coefficients: 特征系数数组（numpy array）
            intercept: 截距项（常数项）
            feature_names: 特征名称列表（可选，用于调试）
        """
        self.coefficients = np.array(coefficients)
        self.intercept = intercept
        self.feature_names = feature_names
        
    @staticmethod
    def sigmoid(x):
        """Sigmoid激活函数，将线性组合映射到(0,1)概率空间"""
        # 限制x范围防止数值溢出
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def predict_proba(self, X):
        """
        预测正类（RB患病）的概率
        
        参数:
            X: 特征矩阵 (n_samples, n_features)   
        返回:
            probabilities: 概率数组 (n_samples,)，值为0-1之间
        """
        # 计算线性组合: z = intercept + X @ coef
        linear_combination = np.dot(X, self.coefficients) + self.intercept
        
        # 应用sigmoid函数得到概率
        probabilities = self.sigmoid(linear_combination)
        
        return probabilities
    
    def predict(self, X, threshold=0.5):
        """
        预测类别标签
        
        参数:
            X: 特征矩阵
            threshold: 分类阈值（默认0.5）
            
        返回:
            predictions: 预测标签（0或1）
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

# ==================== 5. 数据准备函数 ====================

def prepare_method_data(df, feature_dict, method_name):
    """
    为特定方法准备数据
    
    参数:
        df: 原始DataFrame
        feature_dict: 特征-系数对照字典
        method_name: 方法名称（用于日志）
        
    返回:
        X: 特征矩阵 (numpy array)
        y: 标签数组
        coefficients: 系数数组（与X列对应）
        available_features: 实际使用的特征名列表
    """
    # 检查可用特征
    available_features = [f for f in feature_dict.keys() if f in df.columns]
    missing_features = [f for f in feature_dict.keys() if f not in df.columns]
    
    if missing_features:
        print(f"    缺失特征 ({len(missing_features)}个): {missing_features[:3]}...")
    
    if len(available_features) == 0:
        raise ValueError(f"{method_name}: 没有可用特征")
    
    # 提取特征数据
    X = df[available_features].values.astype(float)
    
    # 获取对应系数（确保顺序一致）
    coefficients = np.array([feature_dict[f] for f in available_features])
    
    # 提取标签
    if 'RB_Status' not in df.columns:
        raise ValueError(f"{method_name}: 缺少RB_Status列")
    y = df['RB_Status'].values
    
    # 检查标签分布
    n_class_0 = np.sum(y == 0)
    n_class_1 = np.sum(y == 1)
    
    print(f"    可用特征: {len(available_features)}/{len(feature_dict)}")
    print(f"    样本分布: 健康={n_class_0}, 患病={n_class_1}")
    
    return X, y, coefficients, available_features
    # 如果是HOS方法，打印标准化后的数据范围确认
    if method_name == 'HOS':
        print(f"    数据范围: [{X.min():.3f}, {X.max():.3f}] (应接近[0, 1])")
    
    return X, y, coefficients, available_features

# ==================== 6. 十折交叉验证评估 ====================

def stratified_kfold_cv(X, y, model, n_splits=10, random_state=42):
    """
    执行分层十折交叉验证
    
    关键说明：
    - 使用StratifiedKFold确保每折中类别比例与总体一致
    - 直接使用SPSS系数，不在训练集上重新拟合（保持SPSS模型原貌）
    - 这是获得高AUC的关键：SPSS系数是在全数据上优化的
    
    参数:
        X: 特征矩阵
        y: 标签数组
        model: SPSSLogisticRegression实例
        n_splits: 折数（默认10）
        random_state: 随机种子（保证可重复）
        
    返回:
        results: 包含每折评估指标的字典列表
        aggregated: 汇总所有折的预测结果
    """
    print(f"\n  执行十折分层交叉验证...")
    
    # 初始化分层K折
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_results = []
    all_y_true = []
    all_y_proba = []
    all_y_pred = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # 划分数据
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        # 使用SPSS模型直接预测（不重新训练！）
        y_proba = model.predict_proba(X_val)
        y_pred = model.predict(X_val)
        
        # 计算评估指标
        metrics = calculate_metrics(y_val, y_proba, y_pred)
        
        fold_results.append({
            'fold': fold_idx,
            'n_val': len(y_val),
            'metrics': metrics
        })
        
        # 收集所有折的结果用于绘制总体ROC
        all_y_true.extend(y_val)
        all_y_proba.extend(y_proba)
        all_y_pred.extend(y_pred)
        
        # 打印每折结果
        print(f"    折{fold_idx:2d}: AUC={metrics['auc']:.3f}, "
              f"ACC={metrics['accuracy']:.3f}, "
              f"SEN={metrics['sensitivity']:.3f}, "
              f"SPE={metrics['specificity']:.3f}")
    
    # 计算总体指标（所有折合并）
    aggregated_metrics = calculate_metrics(
        np.array(all_y_true), 
        np.array(all_y_proba), 
        np.array(all_y_pred)
    )
    
    aggregated = {
        'y_true': np.array(all_y_true),
        'y_proba': np.array(all_y_proba),
        'y_pred': np.array(all_y_pred),
        'metrics': aggregated_metrics
    }
    
    return fold_results, aggregated

def calculate_metrics(y_true, y_proba, y_pred):
    """
    计算完整的分类评估指标
    
    指标说明:
    - AUC: ROC曲线下面积，衡量排序能力
    - Accuracy: 准确率，正确预测比例
    - Sensitivity (Recall): 灵敏度，真正例率（患病被正确检出率）
    - Specificity: 特异度，真负例率（健康被正确排除率）
    - Precision: 精确率，预测患病中实际患病比例
    - F1-Score: 精确率和灵敏度的调和平均
    """
    
    # 基础指标
    auc = roc_auc_score(y_true, y_proba)
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'confusion_matrix': cm,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

# ==================== 7. 可视化函数 ====================

def plot_comparison_roc(all_results, output_dir):
    """
    绘制四种方法的ROC曲线对比图
    
    参数:
        all_results: 字典，键为方法名，值为aggregated结果
        output_dir: 保存目录
    """
    plt.figure(figsize=(10, 8))
    
    colors = {
        'LBP': '#E63946',           # 红色
        'HOS': "#20DEDE",           # 青色
        'Haralick-Fractal': "#EEC04B",  # 黄色
        'Combine': "#02AAEC"        # 深蓝灰
    }
    
    linestyles = {
        'LBP': '-', 
        'HOS': '--',
        'Haralick-Fractal': '--',
        'Combine': '-'
    }
    
    # 绘制每个方法的ROC曲线
    for method_name, results in all_results.items():
        y_true = results['y_true']
        y_proba = results['y_proba']
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = results['metrics']['auc']
        
        plt.plot(fpr, tpr, 
                color=colors[method_name],
                linestyle=linestyles[method_name],
                linewidth=2.5,
                label=f'{method_name} (AUC = {auc:.3f})')
    
    # 绘制对角线（随机猜测）
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Classifier')
    
    # 美化图表
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
    plt.title('ROC Curves Comparison of Four Texture Analysis Methods\n'
              'for Retinoblastoma Classification', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    

    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(output_dir, '01_ROC_Curves_Comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")

def plot_method_detailed_evaluation(method_name, fold_results, aggregated, output_dir):
    """
    为单个方法绘制详细的四格评估图
    
    包含:
    1. ROC曲线（带AUC值）
    2. 混淆矩阵热图
    3. 十折性能指标分布箱线图
    4. 综合指标柱状图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{method_name} Model - Detailed Evaluation\n'
                f'10-Fold Stratified Cross-Validation', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # 颜色主题
    main_color = '#2E86AB'
    accent_color = '#A23B72'
    
    # 1. ROC曲线
    ax1 = axes[0, 0]
    y_true = aggregated['y_true']
    y_proba = aggregated['y_proba']
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = aggregated['metrics']['auc']
    
    ax1.plot(fpr, tpr, color=main_color, linewidth=3, label=f'AUC = {auc:.4f}')
    ax1.fill_between(fpr, tpr, alpha=0.3, color=main_color)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
    ax1.set_xlim([-0.02, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. 混淆矩阵
    ax2 = axes[0, 1]
    cm = aggregated['metrics']['confusion_matrix']
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum() * 100
    
    labels = ['Healthy (0)', 'RB (1)']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=labels, yticklabels=labels, ax=ax2,
               cbar_kws={'label': 'Count'}, annot_kws={'size': 14, 'weight': 'bold'})
    
    # 添加百分比注释
    for i in range(2):
        for j in range(2):
            text = ax2.texts[i*2 + j]
            text.set_text(f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)")
    
    ax2.set_xlabel('Predicted Label', fontsize=11)
    ax2.set_ylabel('True Label', fontsize=11)
    ax2.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    
    # 3. 十折指标分布
    ax3 = axes[1, 0]
    
    # 提取每折指标
    metrics_per_fold = {
        'AUC': [r['metrics']['auc'] for r in fold_results],
        'Accuracy': [r['metrics']['accuracy'] for r in fold_results],
        'Sensitivity': [r['metrics']['sensitivity'] for r in fold_results],
        'Specificity': [r['metrics']['specificity'] for r in fold_results]
    }
    
    # 绘制箱线图
    bp = ax3.boxplot(metrics_per_fold.values(), labels=metrics_per_fold.keys(),
                    patch_artist=True, notch=True)
    
    colors_box = ['#E63946', '#2A9D8F', '#E9C46A', '#264653']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    
    
    # 4. 综合指标柱状图
    ax4 = axes[1, 1]
    
    metrics_names = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score']
    metrics_values = [
        aggregated['metrics']['accuracy'],
        aggregated['metrics']['sensitivity'],
        aggregated['metrics']['specificity'],
        aggregated['metrics']['precision'],
        aggregated['metrics']['f1_score']
    ]
    
    # 计算标准差（从各折结果）
    metrics_std = [
        np.std([r['metrics']['accuracy'] for r in fold_results]),
        np.std([r['metrics']['sensitivity'] for r in fold_results]),
        np.std([r['metrics']['specificity'] for r in fold_results]),
        np.std([r['metrics']['precision'] for r in fold_results]),
        np.std([r['metrics']['f1_score'] for r in fold_results])
    ]
    
    x_pos = np.arange(len(metrics_names))
    bars = ax4.bar(x_pos, metrics_values, yerr=metrics_std,
                  capsize=5, color=colors_box[:5], alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, val, std in zip(bars, metrics_values, metrics_std):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}\n±{std:.3f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('Overall Performance Metrics', fontsize=13, fontweight='bold')
    ax4.set_ylim([0, 1.15])
    ax4.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, label='90% threshold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    save_path = os.path.join(output_dir, f'02_{method_name}_Detailed_Evaluation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")

def plot_metrics_comparison_bar(all_results, output_dir):
    """
    绘制四种方法的关键指标对比柱状图
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    methods = list(all_results.keys())
    metrics = ['auc', 'sensitivity', 'specificity', 'accuracy']
    metric_labels = ['AUC', 'Sensitivity', 'Specificity', 'Accuracy']
    
    x = np.arange(len(methods))
    width = 0.2
    
    colors = ['#E63946', '#2A9D8F', '#E9C46A', '#264653']
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [all_results[m]['metrics'][metric] for m in methods]
        bars = ax.bar(x + i*width, values, width, label=label, color=color, alpha=0.85)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics Comparison Across Four Methods', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, '03_Metrics_Comparison_Bar.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {save_path}")

# ==================== 8. 结果保存函数 ====================

def save_results_to_excel(all_results, fold_results_dict, output_dir):
    """
    将所有结果保存到Excel文件（多个sheet）
    """
    excel_path = os.path.join(output_dir, '04_Detailed_Results.xlsx')
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Sheet 1: 综合对比
        summary_data = []
        for method, results in all_results.items():
            m = results['metrics']
            summary_data.append({
                'Method': method,
                'AUC': m['auc'],
                'Accuracy': m['accuracy'],
                'Sensitivity': m['sensitivity'],
                'Specificity': m['specificity'],
                'Precision': m['precision'],
                'F1_Score': m['f1_score'],
                'TP': m['tp'],
                'TN': m['tn'],
                'FP': m['fp'],
                'FN': m['fn']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary_Comparison', index=False)
        
        # Sheets 2-5: 每个方法的详细折结果
        for method, fold_results in fold_results_dict.items():
            fold_data = []
            for r in fold_results:
                m = r['metrics']
                fold_data.append({
                    'Fold': r['fold'],
                    'N_Samples': r['n_val'],
                    'AUC': m['auc'],
                    'Accuracy': m['accuracy'],
                    'Sensitivity': m['sensitivity'],
                    'Specificity': m['specificity'],
                    'Precision': m['precision'],
                    'F1_Score': m['f1_score']
                })
            
            # 添加统计行
            df = pd.DataFrame(fold_data)
            stats = pd.DataFrame({
                'Fold': ['Mean', 'Std', 'Min', 'Max'],
                'N_Samples': ['', '', '', ''],
                'AUC': [df['AUC'].mean(), df['AUC'].std(), df['AUC'].min(), df['AUC'].max()],
                'Accuracy': [df['Accuracy'].mean(), df['Accuracy'].std(), df['Accuracy'].min(), df['Accuracy'].max()],
                'Sensitivity': [df['Sensitivity'].mean(), df['Sensitivity'].std(), df['Sensitivity'].min(), df['Sensitivity'].max()],
                'Specificity': [df['Specificity'].mean(), df['Specificity'].std(), df['Specificity'].min(), df['Specificity'].max()]
            })
            
            combined = pd.concat([df, pd.DataFrame([{}]), stats], ignore_index=True)
            combined.to_excel(writer, sheet_name=f'{method}_Folds', index=False)
    
    print(f"  ✓ 保存: {excel_path}")

def save_feature_importance(output_dir):
    """
    保存各方法的特征重要性（系数）到CSV
    """
    for method_name, features in MODEL_CONFIGS.items():
        importance_df = pd.DataFrame({
            'Feature': list(features.keys()),
            'Coefficient': list(features.values()),
            'Abs_Coefficient': np.abs(list(features.values()))
        }).sort_values('Abs_Coefficient', ascending=False)
        
        csv_path = os.path.join(output_dir, f'05_{method_name}_Feature_Coefficients.csv')
        importance_df.to_csv(csv_path, index=False)
        print(f"  ✓ 保存: {csv_path}")

def generate_report(all_results, output_dir, execution_time):
    """
    生成文本报告
    """
    report = []
    report.append("="*70)
    report.append("视网膜母细胞瘤分类模型对比分析报告")
    report.append("Retinoblastoma Classification Model Comparison Report")
    report.append("="*70)
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"执行耗时: {execution_time:.2f} 秒")
    report.append(f"\n数据文件: {DATA_PATH}")
    report.append(f"输出目录: {output_dir}")
    
    report.append("\n" + "-"*70)
    report.append("模型配置说明")
    report.append("-"*70)
    report.append("本分析使用SPSS逻辑回归系数，直接在Python中进行预测评估")
    report.append("方法：十折分层交叉验证（Stratified 10-Fold CV）")
    report.append("注意：SPSS系数在全数据集上训练，交叉验证用于评估稳定性")
    
    report.append("\n" + "-"*70)
    report.append("各方法特征数量")
    report.append("-"*70)
    for method, features in MODEL_CONFIGS.items():
        report.append(f"  {method:20s}: {len(features)} 个特征")
    
    report.append("\n" + "-"*70)
    report.append("性能评估结果汇总")
    report.append("-"*70)
    report.append(f"{'Method':<20} {'AUC':<8} {'ACC':<8} {'SEN':<8} {'SPE':<8} {'F1':<8}")
    report.append("-"*70)
    
    for method, results in all_results.items():
        m = results['metrics']
        report.append(f"{method:<20} "
                     f"{m['auc']:<8.3f} "
                     f"{m['accuracy']:<8.3f} "
                     f"{m['sensitivity']:<8.3f} "
                     f"{m['specificity']:<8.3f} "
                     f"{m['f1_score']:<8.3f}")
    
    # 找出最佳方法
    best_auc_method = max(all_results.items(), key=lambda x: x[1]['metrics']['auc'])
    report.append("\n" + "-"*70)
    report.append("结论")
    report.append("-"*70)
    report.append(f"最佳性能方法: {best_auc_method[0]}")
    report.append(f"最高AUC: {best_auc_method[1]['metrics']['auc']:.3f}")
    
    # HOS改进说明
    hos_auc = all_results['HOS']['metrics']['auc']
    report.append(f"\nHOS方法AUC: {hos_auc:.3f}")
    if hos_auc > 0.8:
        report.append("✓ HOS标准化后性能显著提升")
    elif hos_auc > 0.6:
        report.append("~ HOS标准化后性能有所改善")
    else:
        report.append("✗ HOS标准化后仍不理想，建议检查SPSS系数或数据")

    # 保存报告
    report_path = os.path.join(output_dir, '06_Analysis_Report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"  ✓ 保存: {report_path}")
    print('\n' + '\n'.join(report))

# ==================== 9. 主程序 ====================

def main():
    """主执行函数"""
    import time
    start_time = time.time()
    
    print("\n" + "="*70)
    print("视网膜母细胞瘤分类模型对比分析")
    print("Retinoblastoma Classification Model Comparison")
    print("="*70)
    
    # 1. 加载数据
    data_dict = load_data(DATA_PATH)
    
    # 2. 评估每个方法
    all_results = {}          # 存储汇总结果
    fold_results_dict = {}    # 存储每折详细结果
    
    for method_name in ['LBP', 'HOS', 'Haralick-Fractal', 'Combine']:
        print(f"\n{'='*60}")
        print(f"评估方法: {method_name}")
        print(f"{'='*60}")
        
        # 准备数据
        df = data_dict[method_name]
        X, y, coefficients, feature_names = prepare_method_data(
            df, MODEL_CONFIGS[method_name], method_name
        )
        
        # 创建SPSS模型
        model = SPSSLogisticRegression(
            coefficients=coefficients,
            intercept=INTERCEPTS[method_name],
            feature_names=feature_names
        )
        
        # 执行交叉验证
        fold_results, aggregated = stratified_kfold_cv(X, y, model, n_splits=10)
        
        # 存储结果
        all_results[method_name] = aggregated
        fold_results_dict[method_name] = fold_results
        
        # 打印汇总
        m = aggregated['metrics']
        print(f"\n  {method_name} 总体结果:")
        print(f"    AUC:        {m['auc']:.4f}")
        print(f"    Accuracy:   {m['accuracy']:.4f}")
        print(f"    Sensitivity:{m['sensitivity']:.4f}")
        print(f"    Specificity:{m['specificity']:.4f}")
        print(f"    F1-Score:   {m['f1_score']:.4f}")
    
    # 3. 生成可视化
    print(f"\n{'='*60}")
    print("生成可视化图表")
    print(f"{'='*60}")
    
    plot_comparison_roc(all_results, OUTPUT_DIR)
    
    for method_name in all_results.keys():
        plot_method_detailed_evaluation(
            method_name, 
            fold_results_dict[method_name], 
            all_results[method_name], 
            OUTPUT_DIR
        )
    
    plot_metrics_comparison_bar(all_results, OUTPUT_DIR)
    
    # 4. 保存结果
    print(f"\n{'='*60}")
    print("保存结果文件")
    print(f"{'='*60}")
    
    save_results_to_excel(all_results, fold_results_dict, OUTPUT_DIR)
    save_feature_importance(OUTPUT_DIR)
    
    # 5. 生成报告
    execution_time = time.time() - start_time
    generate_report(all_results, OUTPUT_DIR, execution_time)
    
    print(f"\n{'='*70}")
    print("分析完成！所有结果已保存到桌面。")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()