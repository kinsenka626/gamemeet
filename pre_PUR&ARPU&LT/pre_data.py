import numpy as np


class MultiNewtonInterpolation:
    """
    双变量牛顿插值法 - 支持lt/arpu/PUR三指标预测
    原理：基于差商递推公式构建插值多项式[3](@ref)
    """

    def __init__(self, nodes):
        self.nodes = nodes
        self.x = np.array([node["x"] for node in nodes])
        self.y = np.array([node["y"] for node in nodes])
        # 分离目标指标
        self.lt_data = np.array([node["lt"] for node in nodes])
        self.arpu_data = np.array([node["arpu"] for node in nodes])
        self.pur_data = np.array([node["PUR"] for node in nodes])

        # 构建各指标的差商表
        self.F_lt = self._calc_divided_diff(self.x, self.y, self.lt_data)
        self.F_arpu = self._calc_divided_diff(self.x, self.y, self.arpu_data)
        self.F_pur = self._calc_divided_diff(self.x, self.y, self.pur_data)

    def _calc_divided_diff(self, x, y, z):
        """计算双变量差商矩阵[3,5](@ref)"""
        n = len(x)
        F = np.zeros((n, n))
        F[:, 0] = z

        # 递推计算差商
        for j in range(1, n):
            for i in range(j, n):
                dx = x[i] - x[i - j]
                dy = y[i] - y[i - j]
                F[i][j] = (F[i][j - 1] - F[i - 1][j - 1]) / (dx + dy)
        return F

    def predict(self, x_target, y_target):
        """预测目标点(x_target, y_target)的三项指标"""
        # LT预测 (强相关指标)
        lt_pred = self._evaluate_poly(x_target, y_target, self.x, self.y, self.F_lt)
        # ARPU预测
        arpu_pred = self._evaluate_poly(x_target, y_target, self.x, self.y, self.F_arpu)
        # PUR预测
        pur_pred = self._evaluate_poly(x_target, y_target, self.x, self.y, self.F_pur)

        return {
            "lt": max(1.5, min(3.0, lt_pred)),  # 约束在合理范围
            "arpu": max(0.2, min(0.8, arpu_pred)),
            "PUR": max(0.01, min(0.1, pur_pred))
        }

    def _evaluate_poly(self, x_target, y_target, x, y, F):
        """牛顿插值多项式求值[4](@ref)"""
        result = F[0][0]
        prod_term = 1.0

        for i in range(1, len(x)):
            # 计算双变量基函数乘积
            dx_term = (x_target - x[i - 1])
            dy_term = (y_target - y[i - 1])
            prod_term *= (dx_term + dy_term)

            result += F[i][i] * prod_term

        return result


# ===== 使用示例 =====
if __name__ == "__main__":
    nodes = [
        {"x": 23.348, "y": 59, "lt": 2.2856, "arpu": 0.3675, "PUR": 0.0356},
        {"x": 29.292, "y": 67, "lt": 2.2798, "arpu": 0.3485, "PUR": 0.0359}
    ]

    # 初始化插值器
    try:
        interpolator = MultiNewtonInterpolation(nodes)
        # 预测中间点 (例：x=2.6, y=63)
        prediction = interpolator.predict(2.6, 63)
        print(f"预测结果 (x=2.6, y=63):")
        print(f"  LTV = {prediction['lt']:.4f}")
        print(f"  ARPU = {prediction['arpu']:.4f}")
        print(f"  PUR = {prediction['PUR']:.4f}")
    except Exception as e:
        print(f"警告: {str(e)}")
        print("※ 数据点不足！牛顿插值至少需要3个点，当前结果仅为线性外推")
        # 应急线性插值
        x_ratio = (2.6 - 2.3348) / (2.9292 - 2.3348)
        lt_pred = 2.2856 + x_ratio * (2.2798 - 2.2856)
        arpu_pred = 0.3675 + x_ratio * (0.3485 - 0.3675)
        pur_pred = 0.0356 + x_ratio * (0.0359 - 0.0356)
        print(f"应急线性预测: LTV={lt_pred:.4f}, ARPU={arpu_pred:.4f}, PUR={pur_pred:.4f}")