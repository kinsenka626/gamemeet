import numpy as np
import scipy.stats as stats


def proportion_test(N1, X1, N2, X2, alpha=0.05, alternative='two-sided'):
    """
    比例差异检验函数（基于比例类假设检验公式）

    参数:
        N1: 对照组总样本量
        X1: 对照组成功量
        N2: 实验组总样本量
        X2: 实验组成功量
        alpha: 显著性水平 (默认0.05)
        alternative: 检验类型 ('two-sided', 'greater', 'less')

    返回:
        dict: 包含所有统计量和分析结果的字典
    """
    # 计算比例
    p1 = X1 / N1
    p2 = X2 / N2
    p_pool = (X1 + X2) / (N1 + N2)

    # 1. 计算Z值和p值 (使用合并比例的方差)
    SE_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / N1 + 1 / N2))
    Z_pool = (p1 - p2) / SE_pool

    # 计算p值（根据检验类型）
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(Z_pool)))
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(Z_pool)
    else:  # 'less'
        p_value = stats.norm.cdf(Z_pool)

    # 2. 计算统计功效 (根据图片公式)
    SE_separate = np.sqrt(p1 * (1 - p1) / N1 + p2 * (1 - p2) / N2)
    Z_separate = (p1 - p2) / SE_separate

    if alternative == 'two-sided':
        z_critical = stats.norm.ppf(1 - alpha / 2)
        power = (stats.norm.cdf(Z_separate - z_critical) +
                 stats.norm.cdf(-Z_separate - z_critical))
    else:
        z_critical = stats.norm.ppf(1 - alpha)
        if alternative == 'greater':
            power = 1 - stats.norm.cdf(z_critical - Z_separate)
        else:  # 'less'
            power = stats.norm.cdf(-z_critical - Z_separate)

    # 3. 计算置信区间
    diff = p1 - p2
    if alternative == 'two-sided':
        z_ci = stats.norm.ppf(1 - alpha / 2)
    else:
        z_ci = stats.norm.ppf(1 - alpha)

    SE_ci = np.sqrt(p1 * (1 - p1) / N1 + p2 * (1 - p2) / N2)

    if alternative == 'two-sided':
        ci_low = diff - z_ci * SE_ci
        ci_high = diff + z_ci * SE_ci
    elif alternative == 'greater':
        ci_low = diff - z_ci * SE_ci
        ci_high = np.inf
    else:  # 'less'
        ci_low = -np.inf
        ci_high = diff + z_ci * SE_ci

    # 结果解释
    if p_value < alpha:
        conclusion = "拒绝原假设：两组留存率存在显著差异"
    else:
        conclusion = "无法拒绝原假设：两组留存率无显著差异"

    # 返回结果字典
    return {
        '对照组比例': p1,
        '实验组比例': p2,
        '比例差异': diff,
        'Z值': Z_pool,
        'p值': p_value,
        '统计功效': power,
        '置信区间': (ci_low, ci_high),
        '结论': conclusion,
        '显著性水平': alpha,
        '检验类型': alternative
    }


# 使用示例
if __name__ == "__main__":
    # 用户提供的数据
    result = proportion_test(
        N1=2792, X1=1042,  # 对照组 (A)
        N2=3350, X2=1209,  # 实验组 (B)
        alpha=0.05,
        alternative='two-sided'
    )

    # 打印结果
    print("比例差异检验结果:")
    print(f"对照组比例: {result['对照组比例']:.4f}")
    print(f"实验组比例: {result['实验组比例']:.4f}")
    print(f"比例差异: {result['比例差异']:.4f}")
    print(f"Z值: {result['Z值']:.4f}")
    print(f"p值: {result['p值']:.6f}")
    print(f"统计功效: {result['统计功效']:.4f}")
    print(
        f"{int((1 - result['显著性水平']) * 100)}% 置信区间: ({result['置信区间'][0]:.4f}, {result['置信区间'][1]:.4f})")
    print(f"结论: {result['结论']}")