import numpy as np
from scipy.stats import norm


def calculate_proportion_sample_size(p_A=0.09, p_B=0.07, alpha=0.05, power=0.8, kappa=1.0):
    """
    计算比例类AB测试的最小样本量（基于比例类公式）

    参数:
        p_A: 实验组比例（默认9%）
        p_B: 对照组比例（默认7%）
        alpha: 显著性水平（默认0.05）
        power: 统计功效（默认0.8）
        kappa: 样本比例 n_A/n_B（默认1.0，即1:1分配）

    返回:
        n_B: B组（对照组）所需最小样本量
        n_A: A组（实验组）所需最小样本量
        total: 总样本量
    """
    # 计算效应值（绝对提升）
    effect_size = abs(p_A - p_B)

    # 计算Z值
    z_alpha = norm.ppf(1 - alpha / 2)  # 双侧检验
    z_beta = norm.ppf(power)  # 统计功效

    # 计算联合方差项
    variance_component = (
            p_A * (1 - p_A) / kappa +
            p_B * (1 - p_B)
    )

    # 计算B组样本量
    n_B = variance_component * ((z_alpha + z_beta) / effect_size) ** 2

    # 计算A组样本量
    n_A = kappa * n_B

    # 向上取整
    n_B = int(np.ceil(n_B))
    n_A = int(np.ceil(n_A))
    total = n_A + n_B

    return {
        '对照组比例': p_B,
        '实验组比例': p_A,
        '绝对提升': effect_size,
        '相对提升': (p_A - p_B) / p_B * 100,
        'B组样本量(n_B)': n_B,
        'A组样本量(n_A)': n_A,
        '总样本量': total,
        '统计功效': power,
        '显著性水平': alpha,
        '样本比例(κ)': kappa,
        'Z_alpha': z_alpha,
        'Z_beta': z_beta
    }


# 示例：默认情况（7%提升到9%）
default_result = calculate_proportion_sample_size()
print("\n【默认比例测试（7%→9%）】")
for key, value in default_result.items():
    if key in ['B组样本量(n_B)', 'A组样本量(n_A)', '总样本量']:
        print(f"{key}: {value}")
    elif key not in ['Z_alpha', 'Z_beta']:
        print(f"{key}: {value:.4f}")

# 不同场景对比
print("\n\n【不同场景样本量对比】")

# 场景1：更小的提升（7%→7.5%）
small_effect = calculate_proportion_sample_size(p_A=0.075, p_B=0.07)
print(f"\n7%→7.5%（小效果提升）：需总样本量 {small_effect['总样本量']}人")

# 场景2：更大的提升（7%→11%）
large_effect = calculate_proportion_sample_size(p_A=0.11, p_B=0.07)
print(f"7%→11%（大效果提升）：需总样本量 {large_effect['总样本量']}人")

# 场景3：不同样本比例（2:1分配）
kappa_effect = calculate_proportion_sample_size(kappa=2.0)
print(
    f"7%→9%（2:1分配）：需总样本量 {kappa_effect['总样本量']}人（A组:B组={kappa_effect['A组样本量(n_A)']}:{kappa_effect['B组样本量(n_B)']}）")