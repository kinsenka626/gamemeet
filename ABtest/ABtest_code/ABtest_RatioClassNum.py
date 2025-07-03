import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.power import TTestIndPower


def analyze_arpu_ab_test(file_path):
    """
    分析两组ARPU数据的AB测试结果

    参数:
        file_path: Excel文件路径

    返回:
        包含所有统计结果的字典
    """
    # 1. 读取Excel数据
    df = pd.read_excel(file_path)

    # 2. 预处理数据 - 删除汇总行和空行
    # 根据原始数据格式，有效数据行中"初始事件发生时间"列不为空
    df = df[df['初始事件发生时间'].notna()]

    # 3. 将ARPU列转换为数值类型
    df['ARPU'] = pd.to_numeric(df['ARPU'], errors='coerce')

    # 4. 分组数据
    group1 = df[df['app版本'] == '1.2.7']['ARPU'].dropna()
    group2 = df[df['app版本'] == '1.2.6']['ARPU'].dropna()

    # 5. 描述性统计分析
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # 6. 独立样本t检验（假设方差齐性）
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)

    # 7. 计算置信区间
    diff_mean = mean1 - mean2
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    se_diff = pooled_std * np.sqrt(1 / n1 + 1 / n2)
    ci_low = diff_mean - stats.t.ppf(0.975, n1 + n2 - 2) * se_diff
    ci_high = diff_mean + stats.t.ppf(0.975, n1 + n2 - 2) * se_diff

    # 8. 计算统计功效
    effect_size = np.abs(diff_mean) / pooled_std
    power_analysis = TTestIndPower()
    power = power_analysis.solve_power(
        effect_size=effect_size,
        nobs1=n1,
        alpha=0.05,
        ratio=n2 / n1
    )

    # 9. 结果解释
    if p_value < 0.05:
        conclusion = f"拒绝原假设：版本 {group1.name if mean1 > mean2 else group2.name} 的ARPU显著较高"
    else:
        conclusion = "无法拒绝原假设：两组ARPU无显著差异"

    # 10. 汇总结果
    return {
        "Group1": {
            "name": "1.2.7",
            "sample_size": n1,
            "mean": mean1,
            "std": std1
        },
        "Group2": {
            "name": "1.2.6",
            "sample_size": n2,
            "mean": mean2,
            "std": std2
        },
        "t_statistic": t_stat,
        "p_value": p_value,
        "mean_difference": diff_mean,
        "effect_size": effect_size,
        "confidence_interval": (ci_low, ci_high),
        "statistical_power": power,
        "conclusion": conclusion
    }


def main():
    # 文件路径 - 请替换为您的实际文件路径
    file_path = "yy_data.xlsx"

    # 执行分析
    results = analyze_arpu_ab_test(file_path)

    # 打印结果
    print("=" * 60)
    print("AB测试结果 - ARPU分析")
    print("=" * 60)

    # 打印描述性统计
    print(f"\n【描述性统计】")
    print(f"版本 {results['Group1']['name']} (N={results['Group1']['sample_size']}):")
    print(f"  均值 = {results['Group1']['mean']:.4f}, 标准差 = {results['Group1']['std']:.4f}")
    print(f"版本 {results['Group2']['name']} (N={results['Group2']['sample_size']}):")
    print(f"  均值 = {results['Group2']['mean']:.4f}, 标准差 = {results['Group2']['std']:.4f}")

    # 打印假设检验结果
    print(f"\n【假设检验】")
    print(f"t统计量 = {results['t_statistic']:.4f}")
    print(f"p值 = {results['p_value']:.6f}")
    if results['p_value'] < 0.05:
        print(f"结果在α=0.05水平显著 ({'*' * 10}显著{'*' * 10})")
    else:
        print(f"结果不显著 (p > 0.05)")

    # 打印效应量和置信区间
    print(f"\n【效应量与置信区间】")
    print(f"均值差异 = {results['mean_difference']:.4f}")
    print(f"效应量(Cohen's d) = {results['effect_size']:.4f}")
    print(f"95%置信区间: [{results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f}]")

    # 打印统计功效
    print(f"\n【统计功效】")
    print(f"统计功效 = {results['statistical_power']:.4f}")
    if results['statistical_power'] < 0.8:
        print("注意: 统计功效 < 0.8，检测差异的能力较低，建议增加样本量")

    # 打印结论
    print(f"\n【结论】")
    print(results['conclusion'])


if __name__ == "__main__":
    main()