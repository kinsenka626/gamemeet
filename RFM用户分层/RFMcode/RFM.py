import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime


def load_and_preprocess(file_path):
    """数据预处理"""
    df = pd.read_excel(file_path)

    # 处理M（消费金额）
    df['total_spent'] = pd.to_numeric(df['total_spent'], errors='coerce').fillna(0).clip(lower=0)

    # 处理R（最近购买时间）
    current_date = datetime(2025, 7, 1)
    df['march_last_purchase'] = pd.to_datetime(df['march_last_purchase'], errors='coerce')
    df['R_days'] = (current_date - df['march_last_purchase']).dt.days
    df['R_days'] = df['R_days'].fillna(df['R_days'].max() + 1).clip(lower=0)

    # 处理F（购买频率）
    df['march_purchase_count'] = pd.to_numeric(df['march_purchase_count'],
                                               errors='coerce').fillna(0).clip(lower=0)

    return df


def rfm_clustering(df):
    """执行RFM聚类"""
    df[['R_cluster', 'F_cluster', 'M_cluster']] = 0
    active_customers = df[df['total_spent'] > 0].copy()

    # R值聚类（数值越小越好）
    kmeans_R = KMeans(n_clusters=5, random_state=42)
    active_customers['R_cluster'] = kmeans_R.fit_predict(active_customers[['R_days']])

    # F值聚类（数值越大越好）
    kmeans_F = KMeans(n_clusters=5, random_state=42)
    active_customers['F_cluster'] = kmeans_F.fit_predict(active_customers[['march_purchase_count']])

    # M值聚类
    kmeans_M = KMeans(n_clusters=7, random_state=42)
    active_customers['M_cluster'] = kmeans_M.fit_predict(active_customers[['total_spent']]) + 1

    df.update(active_customers)
    return df


def generate_rfm_report(df):
    """生成分析报告"""
    df['RFM_group'] = df.apply(lambda x: f"R{x['R_cluster']}F{x['F_cluster']}M{x['M_cluster']}", axis=1)

    report = df.groupby('RFM_group').agg(
        customer_count=('RFM_group', 'size'),
        avg_R_days=('R_days', 'mean'),
        avg_F=('march_purchase_count', 'mean'),
        avg_M=('total_spent', 'mean'),
        total_spent=('total_spent', 'sum')
    ).reset_index()

    total_customers = len(df)
    total_spending = report['total_spent'].sum()

    report['customer_ratio'] = report['customer_count'] / total_customers
    report['spending_ratio'] = report['total_spent'] / total_spending

    # 保留原始数值列用于Excel计算
    excel_report = report.copy()

    # 添加格式化展示列
    formatting = {
        'customer_ratio': '{:.2%}',
        'spending_ratio': '{:.2%}',
        'avg_R_days': '{:.1f}天',
        'avg_F': '{:.1f}次',
        'avg_M': '¥{:.2f}',
        'total_spent': '¥{:.2f}'
    }

    for col, fmt in formatting.items():
        excel_report[f'{col}_formatted'] = excel_report[col].apply(lambda x: fmt.format(x))

    return excel_report


def save_to_excel(df, report, output_path):
    """保存结果到Excel"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 保存完整数据集
        df.to_excel(writer, sheet_name='原始数据', index=False)

        # 保存分析报告
        report.to_excel(writer, sheet_name='分析报告', index=False)

        # 获取工作表对象进行格式设置
        workbook = writer.book
        worksheet = writer.sheets['分析报告']

        # 设置列宽
        column_widths = {
            'A': 15, 'B': 12, 'C': 15,
            'D': 12, 'E': 12, 'F': 15,
            'G': 12, 'H': 12, 'I': 12
        }
        for col, width in column_widths.items():
            worksheet.column_dimensions[col].width = width

        # 添加标题
        worksheet['A1'] = 'RFM客户分析报告'
        worksheet.merge_cells('A1:I1')
        title_cell = worksheet['A1']
        title_cell.font = Font(bold=True, size=14)
        title_cell.alignment = Alignment(horizontal='center')


if __name__ == "__main__":
    input_path = "payRFM_trash_0605.xlsx"
    output_path = "RFM分析报告.xlsx"

    # 处理流程
    df = load_and_preprocess(input_path)
    df = rfm_clustering(df)
    report = generate_rfm_report(df)
    print(report.to_markdown(index=False, stralign='center', numalign='center'))
    # 保存Excel
    save_to_excel(df, report, output_path)
    print(f"分析结果已保存至：{output_path}")