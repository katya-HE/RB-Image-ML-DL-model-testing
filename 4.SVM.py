#SVM——support vector machine：找到一个高维最好决策边界→支持向量（最近的点）离边界距离最远large margin
# nonlinear；监督学习；低纬度数据升维找超平面；正-负-决策超平面hyperplane；
# 导入必要的库
import pandas as pd  # 用于读取Excel文件和处理数据表格
import numpy as np  # 用于数值计算和数组操作
import os  # 用于操作系统路径和文件操作
from pathlib import Path  # 用于处理文件路径，比os.path更现代
import matplotlib.pyplot as plt  # 用于绘制图表和可视化
import seaborn as sns  # 基于matplotlib的统计可视化库，用于美化图表
from sklearn.model_selection import cross_val_predict, StratifiedKFold  # 交叉验证相关工具
from sklearn.preprocessing import StandardScaler  # 特征缩放工具
from sklearn.svm import SVC  # 支持向量机分类器
from sklearn.metrics import (  # 评估指标函数
    roc_auc_score, roc_curve, confusion_matrix, 
    classification_report, accuracy_score
)
import joblib  # 用于保存和加载Python对象（如模型）

# 主函数：程序入口
def main():
    # 定义Excel文件路径，替换成你的实际文件路径
    file_path = '/Users/yaya/Desktop/Final.xlsx'
    
    # 获取用户桌面路径，并创建输出文件夹
    desktop = Path.home() / 'Desktop'  # 获取当前用户的桌面路径
    output_dir = desktop / 'SVM_CrossValidation_Results'  # 在桌面创建结果文件夹
    os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建
    
    # 打印输出目录路径，方便用户查找结果
    print(f"结果将保存到: {output_dir}")
    
    # 1. 加载数据
    print("\n" + "="*60)  # 打印分隔线
    print("步骤1: 加载数据")  # 显示当前步骤
    print("="*60)
    
    df = pd.read_excel(file_path)  # 从Excel文件读取数据到DataFrame
    
    # 显示数据基本信息，帮助用户了解数据结构
    print(f"数据形状: {df.shape}")  # 打印数据的行数和列数
    print(f"列名: {df.columns.tolist()}")  # 打印所有列的名称
    print("\n数据前5行:")  # 打印表头
    print(df.head())  # 显示前5行数据
    
    # 显示类别分布，检查数据是否平衡
    print("\n类别分布:")  # 打印表头
    print(df['RB_Status'].value_counts())  # 统计RB_Status列中0和1的数量
    
    # 2. 准备特征和标签
    print("\n" + "="*60)  # 打印分隔线
    print("步骤2: 准备特征矩阵X和标签向量y")  # 显示当前步骤
    print("="*60)
    
    X = df.iloc[:, 2:].values  # 提取特征：从第3列开始到最后一列的所有数据
    # iloc[:, 2:] 表示选择所有行，从索引2开始的列
    # .values 将DataFrame转换为numpy数组
    
    y = df['RB_Status'].values  # 提取标签：RB_Status列的所有值
    # 这将因变量（是否有RB）作为模型的预测目标
    
    print(f"特征矩阵X形状: {X.shape}")  # 打印特征矩阵的维度
    print(f"标签向量y形状: {y.shape}")  # 打印标签向量的维度
    
    # 3. 特征缩放
    print("\n" + "="*60)  # 打印分隔线
    print("步骤3: 特征缩放（Standardization）")  # 显示当前步骤
    print("="*60)
    
    scaler = StandardScaler()  # 创建StandardScaler对象，用于标准化特征
    # StandardScaler会将每个特征转换为均值为0，标准差为1的分布
    
    X_scaled = scaler.fit_transform(X)  # 拟合数据并转换
    # fit_transform先计算每个特征的均值和标准差，然后进行转换
    print("特征缩放完成！")  # 提示缩放完成
    
    # 保存scaler对象，以便后续对新数据进行相同的转换
    scaler_path = output_dir / 'scaler.pkl'  # 定义scaler保存路径
    joblib.dump(scaler, scaler_path)  # 将scaler对象保存到文件
    print(f"Scaler已保存到: {scaler_path}")  # 提示保存位置
    
    # 4. 创建SVM模型
    print("\n" + "="*60)  # 打印分隔线
    print("步骤4: 创建SVM分类器")  # 显示当前步骤
    print("="*60)
    
    model = SVC(  # 创建支持向量机分类器对象
        kernel='rbf',  # 使用RBF径向基函数核，适合非线性分类
        C=1.0,  # 正则化参数，控制分类错误的惩罚程度
        gamma='scale',  # 核函数系数，'scale'表示使用1/(n_features * X.var())
        probability=True,  # 启用概率估计，用于计算AUC和ROC曲线
        random_state=42  # 随机种子，确保结果可重复
    )
    print("SVM模型创建完成！")  # 提示模型创建完成
    print(f"核函数: {model.kernel}")  # 打印核函数类型
    print(f"正则化参数C: {model.C}")  # 打印C参数值
    
    # 5. 设置10折交叉验证
    print("\n" + "="*60)  # 打印分隔线
    print("步骤5: 设置10折分层交叉验证")  # 显示当前步骤
    print("="*60)
    
    cv = StratifiedKFold(  # 创建分层K折交叉验证对象
        n_splits=10,  # 分成10折
        shuffle=True,  # 在分割前打乱数据顺序
        random_state=42  # 随机种子，确保每次分割一致
    )
    # StratifiedKFold会保持每折中类别比例与原数据集一致
    
    print(f"交叉验证折数: {cv.n_splits}")  # 打印折数
    print(" StratifiedKFold确保每折都保持类别平衡")  # 解释StratifiedKFold的作用
    
    # 6. 执行交叉验证并获取预测
    print("\n" + "="*60)  # 打印分隔线
    print("步骤6: 执行交叉验证并获取预测结果")  # 显示当前步骤
    print("="*60)
    
    print("正在进行交叉验证，这可能需要几分钟...")  # 提示可能需要等待
    
    # 获取预测概率（用于计算AUC）
    y_pred_proba = cross_val_predict(  # 使用交叉验证进行预测
        model,  # 使用的模型
        X_scaled,  # 标准化后的特征
        y,  # 真实标签
        cv=cv,  # 交叉验证策略
        method='predict_proba'  # 预测概率值
    )[:, 1]  # 取第二列（正类的概率）
    
    # 获取预测类别（用于计算混淆矩阵）
    y_pred = cross_val_predict(  # 使用交叉验证进行预测
        model,  # 使用的模型
        X_scaled,  # 标准化后的特征
        y,  # 真实标签
        cv=cv,  # 交叉验证策略
        method='predict'  # 预测类别标签
    )
    
    print("交叉验证预测完成！")  # 提示预测完成
    
    # 7. 计算评估指标
    print("\n" + "="*60)  # 打印分隔线
    print("步骤7: 计算模型评估指标")  # 显示当前步骤
    print("="*60)
    
    # 计算准确率：正确预测的比例
    accuracy = accuracy_score(y, y_pred)  # 比较真实标签和预测标签
    print(f"准确率 (Accuracy): {accuracy:.4f}")  # 打印准确率
    
    # 计算AUC：ROC曲线下面积，衡量分类器性能
    auc_score = roc_auc_score(y, y_pred_proba)  # 使用真实标签和预测概率
    print(f"AUC (ROC曲线下面积): {auc_score:.4f}")  # 打印AUC值
    
    # 生成混淆矩阵：真实标签 vs 预测标签的表格
    cm = confusion_matrix(y, y_pred)  # 计算混淆矩阵
    print("\n混淆矩阵 (Confusion Matrix):")  # 打印表头
    print(cm)  # 打印矩阵
    
    # 打印混淆矩阵解释
    print("\n混淆矩阵解读:")
    print(f"真阴性 (TN): {cm[0,0]} - 实际无RB，预测无RB")  # 解释左上角
    print(f"假阳性 (FP): {cm[0,1]} - 实际无RB，预测有RB")  # 解释右上角
    print(f"假阴性 (FN): {cm[1,0]} - 实际有RB，预测无RB")  # 解释左下角
    print(f"真阳性 (TP): {cm[1,1]} - 实际有RB，预测有RB")  # 解释右下角
    
    # 生成分类报告：包含精确率、召回率、F1分数
    print("\n分类报告 (Classification Report):")  # 打印表头
    report = classification_report(  # 生成分类报告
        y,  # 真实标签
        y_pred,  # 预测标签
        target_names=['No RB', 'RB']  # 类别名称
    )
    print(report)  # 打印报告
    
    # 8. 保存模型
    print("\n" + "="*60)  # 打印分隔线
    print("步骤8: 保存训练好的模型")  # 显示当前步骤
    print("="*60)
    
    # 在整个数据集上重新训练模型（用于实际应用）
    model.fit(X_scaled, y)  # 使用所有数据训练最终模型
    
    model_path = output_dir / 'svm_model.pkl'  # 定义模型保存路径
    joblib.dump(model, model_path)  # 将模型对象保存到文件
    print(f"最终模型已保存到: {model_path}")  # 提示保存位置
    
    # 9. 绘制并保存ROC曲线
    print("\n" + "="*60)  # 打印分隔线
    print("步骤9: 绘制ROC曲线")  # 显示当前步骤
    print("="*60)
    
    plt.figure(figsize=(10, 8))  # 创建新图形，设置大小
    
    # 计算ROC曲线的假阳性率和真阳性率
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)  # 计算ROC曲线数据
    
    # 绘制ROC曲线
    plt.plot(  # 绘制线条
        fpr, tpr,  # x轴是假阳性率，y轴是真阳性率
        label=f'ROC曲线 (AUC = {auc_score:.3f})',  # 标签显示AUC值
        color='darkorange',  # 线条颜色
        linewidth=2  # 线条宽度
    )
    
    # 绘制对角线（随机猜测的基线）
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测基线', alpha=0.5)  # 绘制虚线
    
    # 设置图形属性
    plt.xlim([0.0, 1.0])  # x轴范围
    plt.ylim([0.0, 1.05])  # y轴范围
    plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)  # x轴标签
    plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)  # y轴标签
    plt.title('SVM分类器ROC曲线 - 视网膜母细胞瘤检测', fontsize=14)  # 图形标题
    plt.legend(loc="lower right")  # 图例位置
    plt.grid(True, alpha=0.3)  # 显示网格
    
    # 保存图形
    roc_path = output_dir / 'roc_curve.png'  # 定义ROC曲线保存路径
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')  # 保存为高分辨率PNG
    plt.close()  # 关闭图形，释放内存
    print(f"ROC曲线已保存到: {roc_path}")  # 提示保存位置
    
    # 10. 绘制并保存混淆矩阵热图
    print("\n" + "="*60)  # 打印分隔线
    print("步骤10: 绘制混淆矩阵热图")  # 显示当前步骤
    print("="*60)
    
    plt.figure(figsize=(8, 6))  # 创建新图形，设置大小
    
    # 绘制热图
    sns.heatmap(  # seaborn的热图函数
        cm,  # 混淆矩阵数据
        annot=True,  # 在每个单元格中显示数值
        fmt='d',  # 数值格式为整数
        cmap='Blues',  # 颜色映射
        xticklabels=['无RB', '有RB'],  # x轴标签
        yticklabels=['无RB', '有RB'],  # y轴标签
        cbar_kws={'label': '样本数量'}  # 颜色条标签
    )
    
    # 设置图形属性
    plt.xlabel('预测标签', fontsize=12)  # x轴标签
    plt.ylabel('真实标签', fontsize=12)  # y轴标签
    plt.title('混淆矩阵 - 视网膜母细胞瘤检测', fontsize=14)  # 图形标题
    
    # 保存图形
    cm_path = output_dir / 'confusion_matrix.png'  # 定义混淆矩阵保存路径
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')  # 保存为高分辨率PNG
    plt.close()  # 关闭图形，释放内存
    print(f"混淆矩阵热图已保存到: {cm_path}")  # 提示保存位置
    
    # 11. 保存评估指标到文本文件
    print("\n" + "="*60)  # 打印分隔线
    print("步骤11: 保存详细评估指标")  # 显示当前步骤
    print("="*60)
    
    metrics_path = output_dir / 'evaluation_metrics.txt'  # 定义评估指标保存路径
    
    # 打开文件并写入评估结果
    with open(metrics_path, 'w', encoding='utf-8') as f:  # 以写入模式打开文件，使用UTF-8编码
        f.write("="*60 + "\n")  # 写入分隔线
        f.write("SVM分类器评估报告 - 视网膜母细胞瘤检测\n")  # 写入标题
        f.write("="*60 + "\n\n")  # 写入分隔线和空行
        
        f.write("模型配置:\n")  # 写入章节标题
        f.write(f"  核函数: {model.kernel}\n")  # 写入核函数类型
        f.write(f"  正则化参数 C: {model.C}\n")  # 写入C参数值
        f.write(f"  交叉验证折数: 10\n")  # 写入交叉验证折数
        f.write(f"  样本总数: {len(y)}\n")  # 写入样本总数
        f.write(f"  特征数量: {X.shape[1]}\n\n")  # 写入特征数量
        
        f.write("性能指标:\n")  # 写入章节标题
        f.write(f"  准确率 (Accuracy): {accuracy:.4f}\n")  # 写入准确率
        f.write(f"  AUC (ROC曲线下面积): {auc_score:.4f}\n\n")  # 写入AUC
        
        f.write("混淆矩阵:\n")  # 写入章节标题
        f.write(str(cm) + "\n\n")  # 写入混淆矩阵
        
        # 计算并写入灵敏度（召回率）和特异度
        tn, fp, fn, tp = cm.ravel()  # 将混淆矩阵展平为四个值
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 计算灵敏度
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 计算特异度
        
        f.write(f"灵敏度 (Sensitivity/Recall): {sensitivity:.4f}\n")  # 写入灵敏度
        f.write(f"特异度 (Specificity): {specificity:.4f}\n\n")  # 写入特异度
        
        f.write("分类报告:\n")  # 写入章节标题
        f.write(report + "\n")  # 写入分类报告
    
    print(f"评估指标已保存到: {metrics_path}")  # 提示保存位置
    
    # 12. 完成提示
    print("\n" + "="*60)  # 打印分隔线
    print("✅ 所有任务完成！")  # 显示完成消息
    print("="*60)
    print(f"所有结果已保存到桌面文件夹: {output_dir}")  # 提示结果位置
    print("\n文件清单:")  # 打印文件清单标题
    print(f"  1. 训练好的模型: svm_model.pkl")  # 列出模型文件
    print(f"  2. 特征缩放器: scaler.pkl")  # 列出缩放器文件
    print(f"  3. ROC曲线图: roc_curve.png")  # 列出ROC图
    print(f"  4. 混淆矩阵图: confusion_matrix.png")  # 列出混淆矩阵图
    print(f"  5. 评估报告: evaluation_metrics.txt")  # 列出评估报告

# 当脚本直接运行时执行main函数
if __name__ == '__main__':  # 检查是否是直接运行（不是被导入）
    main()  # 调用主函数