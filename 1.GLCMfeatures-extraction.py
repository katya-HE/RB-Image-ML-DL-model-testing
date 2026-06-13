# 步骤1: 导入必要的库
import os
import cv2 # OpenCV库，用于图像处理：读取、转换、调整图像大小
import numpy as np #数值计算：矩阵运算、数学计算
import pandas as pd #数据处理与保存：DataFrame操作，保存为Excel文件
from tqdm import tqdm  # 进度条显示
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# 步骤2: 定义GLCM核心函数
def fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5):
    """
    快速计算灰度共生矩阵(GLCM) - 核心函数
    计算每个像素邻域内的灰度共生矩阵
    """
    mi, ma = vmin, vmax
    ks = kernel_size
    h, w = img.shape

    # 1. 灰度量化: 将0-255范围量化为nbit个等级
    # 例如nbit=16时，0-15→0, 16-31→1, ..., 240-255→15
    bins = np.linspace(mi, ma+1, nbit+1)
    gl1 = np.digitize(img, bins) - 1  # 主像素矩阵
    gl2 = np.append(gl1[:, 1:], gl1[:, -1:], axis=1)  # 右偏移1像素（计算水平方向共生对）

    # 2. 初始化四维GLCM数组: [灰度级i, 灰度级j, 图像高, 图像宽]
    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    
    # 3. 统计每个灰度对(i,j)的出现位置
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1 == i) & (gl2 == j))
            glcm[i, j, mask] = 1

    # 4. 滑动窗口卷积: 统计每个像素邻域内的灰度对频率
    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i, j] = cv2.filter2D(glcm[i, j], -1, kernel)

    return glcm.astype(np.float32)


# 步骤3: 定义GLCM纹理特征提取函数
def fast_glcm_mean(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM均值: 反映平均灰度水平"""
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit)**2
    return mean

def fast_glcm_std(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM标准差: 反映灰度变化程度"""
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit)**2

    std2 = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            std2 += (glcm[i, j] * i - mean)**2
    return np.sqrt(std2)

def fast_glcm_contrast(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM对比度: 反映纹理清晰度和沟纹深浅"""
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    cont = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            cont += glcm[i, j] * (i - j)**2
    return cont

def fast_glcm_dissimilarity(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM相异性: 度量纹理差异程度"""
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    diss = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            diss += glcm[i, j] * np.abs(i - j)
    return diss

def fast_glcm_homogeneity(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM同质性: 度量纹理局部均匀性"""
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    homo = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            homo += glcm[i, j] / (1. + (i - j)**2)
    return homo

def fast_glcm_ASM(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM角二阶矩(ASM)和能量: 度量纹理均匀性"""
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    asm = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            asm += glcm[i, j]**2
    return asm, np.sqrt(asm)

def fast_glcm_max(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM最大值: 反映最突出的纹理模式"""
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    return np.max(glcm, axis=(0, 1))

def fast_glcm_entropy(img, vmin=0, vmax=255, nbit=8, ks=5):
    """GLCM熵: 度量纹理复杂性和信息量"""
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    pnorm = glcm / np.sum(glcm, axis=(0, 1)) + 1. / ks**2
    return np.sum(-pnorm * np.log(pnorm), axis=(0, 1))

# 步骤4: 图像预处理函数
def preprocess_retina_image(img_path, target_size=(512, 512)):
    """
    专用预处理视网膜图像
    - 读取RGB图像
    - 转换为灰度图
    - 调整大小（从4000x4000降到512x512，节省95%计算量）
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ⚠️  无法读取: {os.path.basename(img_path)}")
        return None
    
    # RGB转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 调整大小（使用INTER_AREA插值，适合缩小图像）
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    
    return resized

# 步骤5: 批量特征提取函数
def extract_all_glcm_features(img, nbit=16, ks=15):
    """
    提取所有GLCM纹理特征（返回全局平均值）
    参数已按您的要求设置: nbit=16, ks=15
    将0-255的灰度值量化为16个等级（每个等级16个灰度值）。减少计算量，但保留足够纹理信息。
    ks=15：滑动窗口大小为15×15。这是非常大的窗口，会严重模糊纹理信息。
    target_size=(512, 512)：将4000×4000的图像缩小到512×512。这丢失了大量细节信息。 
    建议使用更高的nbit（如32或64），更小的ks（如5或7），并保持更高的图像分辨率。
    根本原因总结：
    特征提取参数不当：窗口过大(15×15)量化过粗(16级)，丢失纹理细节
    特征未标准化：不同特征尺度差异巨大，影响逻辑回归
    系数使用不当：SPSS系数可能对应标准化后的特征，你直接用于原始特征
    可能的数据问题：检查两类图像是否真的有明显纹理差异
    建议先尝试：
    修改GLCM参数：nbit=32, ks=7
    在交叉验证中添加特征标准化
    使用重新训练的逻辑回归，而不是固定SPSS系数
    # 在交叉验证前添加
    from sklearn.preprocessing import StandardScaler
    # 保存原始系数
    original_coefficients = coefficients.copy()
    # 重新拟合模型时标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    # 使用标准化后的数据重新计算系数（或使用原始系数）
    检查特征是否真的能区分两类样本
    """
    features = {}
    
    # 计算各个特征并取全局平均值
    features['glcm_mean'] = np.mean(fast_glcm_mean(img, nbit=nbit, ks=ks))
    features['glcm_std'] = np.mean(fast_glcm_std(img, nbit=nbit, ks=ks))
    features['glcm_contrast'] = np.mean(fast_glcm_contrast(img, nbit=nbit, ks=ks))
    features['glcm_dissimilarity'] = np.mean(fast_glcm_dissimilarity(img, nbit=nbit, ks=ks))
    features['glcm_homogeneity'] = np.mean(fast_glcm_homogeneity(img, nbit=nbit, ks=ks))
    
    asm, energy = fast_glcm_ASM(img, nbit=nbit, ks=ks)
    features['glcm_asm'] = np.mean(asm)
    features['glcm_energy'] = np.mean(energy)
    
    features['glcm_max'] = np.mean(fast_glcm_max(img, nbit=nbit, ks=ks))
    features['glcm_entropy'] = np.mean(fast_glcm_entropy(img, nbit=nbit, ks=ks))
    
    return features

# 步骤6: 主处理流程
def main():
    # ==================== 配置参数 ====================
    base_path = '/Users/yaya/Desktop/255RB_510NORMAL'
    health_path = os.path.join(base_path, 'health')
    rb_path = os.path.join(base_path, 'RB')
    output_excel = '/Users/yaya/Desktop/RB_GLCM_features.xlsx'
    
    # GLCM参数（按您的要求）
    NBIT = 16  # 量化等级
    KS = 15    # 滑动窗口大小
    TARGET_SIZE = (512, 512)  # 图像缩放尺寸（从4000x4000缩小）
    
    print("=" * 60)
    print("视网膜母细胞瘤GLCM纹理特征提取系统")
    print("=" * 60)
    
    # ==================== 路径检查 ====================
    if not os.path.exists(base_path):
        print(f"❌ 错误: 找不到主文件夹\n   {base_path}")
        return
    
    if not os.path.exists(health_path):
        print(f"❌ 错误: 找不到健康图像文件夹\n   {health_path}")
        return
    
    if not os.path.exists(rb_path):
        print(f"❌ 错误: 找不到病变图像文件夹\n   {rb_path}")
        return
    
    # ==================== 批量处理 ====================
    all_records = []
    
    # 1. 处理健康图像 (标签=0)
    print("\n【阶段1】处理健康图像文件夹...")
    health_files = [f for f in os.listdir(health_path) if f.lower().endswith(('.jpg', '.jpeg'))]
    print(f"   发现 {len(health_files)} 张图像")
    
    for filename in tqdm(health_files, desc="健康图像处理进度", ncols=100):
        img_path = os.path.join(health_path, filename)
        image_id = os.path.splitext(filename)[0]  # 提取文件名作为ID
        
        # 预处理
        img_processed = preprocess_retina_image(img_path, TARGET_SIZE)
        if img_processed is None:
            continue
        
        # 提取特征
        try:
            features = extract_all_glcm_features(img_processed, nbit=NBIT, ks=KS)
            record = {
                'imageid': image_id,
                'RB状态': 0,  # 0=健康
            }
            record.update(features)
            all_records.append(record)
        except Exception as e:
            print(f"  ⚠️  特征提取失败: {filename} - {e}")
            continue
    
    # 2. 处理病变图像 (标签=1)
    print("\n【阶段2】处理病变图像文件夹...")
    rb_files = [f for f in os.listdir(rb_path) if f.lower().endswith(('.jpg', '.jpeg'))]
    print(f"   发现 {len(rb_files)} 张图像")
    
    for filename in tqdm(rb_files, desc="病变图像处理进度", ncols=100):
        img_path = os.path.join(rb_path, filename)
        image_id = os.path.splitext(filename)[0]
        
        # 预处理
        img_processed = preprocess_retina_image(img_path, TARGET_SIZE)
        if img_processed is None:
            continue
        
        # 提取特征
        try:
            features = extract_all_glcm_features(img_processed, nbit=NBIT, ks=KS)
            record = {
                'imageid': image_id,
                'RB状态': 1,  # 1=病变
            }
            record.update(features)
            all_records.append(record)
        except Exception as e:
            print(f"  ⚠️  特征提取失败: {filename} - {e}")
            continue
    
    # ==================== 保存结果 ====================
    if all_records:
        print("\n【阶段3】保存特征到Excel...")
        df = pd.DataFrame(all_records)
        
        # 按imageid排序
        df = df.sort_values('imageid').reset_index(drop=True)
        
        # 保存到Excel
        df.to_excel(output_excel, index=False, engine='openpyxl')
        
        print(f"✅ 完成！特征文件已保存:\n   {output_excel}")
        
        # 显示统计信息
        print("\n" + "=" * 60)
        print("数据集统计摘要")
        print("=" * 60)
        print(f"总图像数:  {len(df):>6} 张")
        print(f"健康图像:  {len(df[df['RB状态'] == 0]):>6} 张 (标签=0)")
        print(f"病变图像:  {len(df[df['RB状态'] == 1]):>6} 张 (标签=1)")
        print(f"特征数量:  {len(df.columns) - 2:>6} 个")
        print("=" * 60)
        
        # 显示特征列名
        print("\n提取的特征列:")
        feature_cols = [col for col in df.columns if col not in ['imageid', 'RB状态']]
        for i, col in enumerate(feature_cols, 1):
            print(f"{i:2d}. {col}")
        
        # 显示数据预览
        print("\n数据预览 (前5行):")
        print(df.head().to_string())
        
    else:
        print("❌ 错误: 没有成功处理任何图像，请检查文件路径和格式")

# 步骤7: 程序入口
if __name__ == '__main__':
    main()