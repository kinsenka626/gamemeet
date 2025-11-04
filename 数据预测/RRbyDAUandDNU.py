import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    with open(config_file, 'r', encoding='utf-8') as f:
        config.read_file(f)

    settings = config['Settings']
    return {
        'file_name': settings.get('file_name'),
        'sheet_name': settings.get('sheet_name'),
        'columns': settings.get('columns').split(','),
        'index_col': settings.get('index_col'),
        'skip_rows': settings.getint('skip_rows'),
        'min_days': settings.getint('min_days'),
        'max_days': settings.getint('max_days'),
        'k_range': [float(x) for x in settings.get('k_range').split(',')],
        'p_range': [float(x) for x in settings.get('p_range').split(',')],
        'kernel_size': settings.getint('kernel_size'),
    }

def read_excel_data(file_name, sheet_name, columns, skip_rows=0, index_col=None):
    try:
        df = pd.read_excel(file_name, sheet_name=sheet_name, usecols=columns, skiprows=skip_rows)

        # 去除列名中的前后空格
        df.columns = df.columns.str.strip()

        # 确保列名存在
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少列: {', '.join(missing_columns)}")

        if index_col:
            df[index_col] = pd.to_datetime(df[index_col], errors='coerce')
            df.set_index(index_col, inplace=True)

        return df
    except Exception as e:
        print(f"Error reading Excel data: {e}")
        return None


# 计算卷积
def convolve_with_kernel(f_t, kernel_t, min_days=20, max_days=300):
    result = np.zeros_like(f_t)
    for i in range(min_days, max_days):
        kernel_range = max(0, i - len(kernel_t) + 1)
        result[i] = np.sum(f_t[kernel_range:i + 1] * kernel_t[i - kernel_range::-1])
    return result


# 误差函数
def error_function(params, f_t, h_t, time_series, min_days=20, max_days=300):
    p, k = params
    kernel_t = k * (time_series[:240] + 1) ** p
    h_t_pred = convolve_with_kernel(f_t, kernel_t, min_days, max_days)
    return np.sum((h_t[min_days:max_days] - h_t_pred[min_days:max_days]) ** 2)


# 主函数
def main():
    # 从配置文件加载配置
    config = read_config('config.ini')
    # 读取数据
    data = read_excel_data(
        file_name=config['file_name'],
        sheet_name=config['sheet_name'],
        columns=config['columns'],
        skip_rows=config['skip_rows'],
        index_col=config['index_col']
    )

    time_series = (data.index.values - data.index.values[0]).astype('timedelta64[D]').astype(int)
    f_t = data[config['columns'][1]].values
    h_t = data[config['columns'][2]].values

    # 网格搜索的范围调整
    k_values = np.linspace(config['k_range'][0], config['k_range'][1], 100)
    p_values = np.linspace(config['p_range'][0], config['p_range'][1], 100)

    best_params = None
    min_error = float('inf')

    # 网格搜索
    for p in p_values:
        for k in k_values:
            params = [p, k]
            error = error_function(params, f_t, h_t, time_series, config['min_days'], config['max_days'])
            if error < min_error:
                min_error = error
                best_params = params

    # 输出最佳参数
    p_best, k_best = best_params
    kernel_best = k_best * (time_series[:config['kernel_size']] + 1) ** p_best
    h_t_best = convolve_with_kernel(f_t, kernel_best, min_days=config['min_days'], max_days=config['max_days'])
    plt.plot(time_series, h_t, label="真实 h(t)", color='blue')
    plt.plot(time_series, h_t_best, label="估计 h(t)", linestyle='dashed', color='red')
    plt.legend()
    plt.title('DAU与DNU逆卷积')
    plt.xlabel('时间（日）')
    plt.ylabel('DAU')
    plt.show()

    print(kernel_best[1:30])
    print(f"最佳幂函数参数： k = {k_best}, p = {p_best}")


# 调用主函数
if __name__ == "__main__":
    main()
