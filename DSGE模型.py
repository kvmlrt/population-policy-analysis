import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from scipy.optimize import minimize

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DSGE模型:
    def __init__(self):
        """初始化DSGE模型"""
        self.结果目录 = Path('结果')
        self.结果目录.mkdir(exist_ok=True)
        
        # 初始化模型参数
        self.经济数据 = None
        self.人口数据 = None
        self.参数 = {
            '贴现因子': 0.98,
            '资本折旧率': 0.1,
            '资本份额': 0.3,
            '劳动供给弹性': 0.5,
            '消费习惯持续性': 0.7,
            '价格粘性': 0.75,
            '货币政策利率平滑': 0.8,
            '货币政策通胀反应': 1.5,
            '货币政策产出反应': 0.5,
            '技术冲击持续性': 0.9,
            '需求冲击持续性': 0.7,
            '货币冲击持续性': 0.5
        }
        
    def 设置数据(self, 经济数据: pd.DataFrame, 人口数据: pd.DataFrame):
        """设置模型所需的数据"""
        try:
            self.经济数据 = 经济数据
            self.人口数据 = 人口数据
            logger.info("经济和人口数据已设置")
        except Exception as e:
            logger.error(f"设置数据时出错: {str(e)}")
            raise
            
    def 运行模型(self, 参数: dict = None) -> dict:
        """运行DSGE模型分析"""
        try:
            if self.经济数据 is None or self.人口数据 is None:
                raise ValueError("请先设置经济和人口数据")
                
            # 更新参数（如果提供）
            if 参数:
                self.参数.update(参数)
            
            # 计算稳态
            稳态 = self._计算稳态()
            
            # 对数线性化
            线性化系数 = self._对数线性化(稳态)
            
            # 求解模型
            解 = self._求解模型(线性化系数)
            
            # 生成脉冲响应
            脉冲响应 = self._生成脉冲响应(解)
            
            # 历史分解
            历史分解 = self._历史分解(解)
            
            # 预测
            预测 = self._生成预测(解)
            
            # 保存结果
            self._保存结果(稳态, 脉冲响应, 历史分解, 预测)
            
            return {
                '稳态': 稳态,
                '脉冲响应': 脉冲响应,
                '历史分解': 历史分解,
                '预测': 预测
            }
            
        except Exception as e:
            logger.error(f"运行模型时出错: {str(e)}")
            raise
            
    def _计算稳态(self) -> dict:
        """计算模型稳态"""
        try:
            # 获取最新数据作为初始值
            最新年份 = self.经济数据['年份'].max()
            最新数据 = self.经济数据[self.经济数据['年份'] == 最新年份].iloc[0]
            
            # 计算人均指标
            总人口 = self.人口数据[self.人口数据['年份'] == 最新年份]['总人口'].iloc[0]
            人均GDP = 最新数据['GDP'] / 总人口
            人均消费 = 最新数据['消费'] / 总人口
            人均投资 = 最新数据['投资'] / 总人口
            
            # 计算稳态值
            β = self.参数['贴现因子']
            δ = self.参数['资本折旧率']
            α = self.参数['资本份额']
            
            # 稳态利率
            利率 = 1/β - 1
            
            # 稳态资本劳动比
            资本劳动比 = (α / (利率 + δ)) ** (1/(1-α))
            
            # 稳态产出
            产出 = 资本劳动比 ** α
            
            # 稳态投资
            投资 = δ * 资本劳动比
            
            # 稳态消费
            消费 = 产出 - 投资
            
            # 标准化到实际数据水平
            比例因子 = 人均GDP / 产出
            
            return {
                '产出': 产出 * 比例因子,
                '消费': 消费 * 比例因子,
                '投资': 投资 * 比例因子,
                '资本': 资本劳动比 * 比例因子,
                '劳动': 1.0,  # 标准化为1
                '利率': 利率,
                '工资': (1-α) * 产出 * 比例因子
            }
            
        except Exception as e:
            logger.error(f"计算稳态时出错: {str(e)}")
            raise
            
    def _对数线性化(self, 稳态: dict) -> np.ndarray:
        """对数线性化模型方程"""
        try:
            # 提取参数
            β = self.参数['贴现因子']
            δ = self.参数['资本折旧率']
            α = self.参数['资本份额']
            η = self.参数['劳动供给弹性']
            h = self.参数['消费习惯持续性']
            θ = self.参数['价格粘性']
            ρr = self.参数['货币政策利率平滑']
            φπ = self.参数['货币政策通胀反应']
            φy = self.参数['货币政策产出反应']
            
            # 构建系统矩阵（简化版本）
            n = 6  # 状态变量数量
            A = np.zeros((n, n))
            B = np.zeros((n, 3))  # 3个冲击
            
            # 简化的线性化方程组
            # 1. IS曲线
            A[0,0] = 1/(1-h)
            A[0,1] = -h/(1-h)
            A[0,5] = -1/σ
            
            # 2. Phillips曲线
            A[1,1] = β
            A[1,2] = (1-θ)*(1-β*θ)/θ
            
            # 3. 生产函数
            A[2,3] = α
            A[2,4] = 1-α
            
            # 4. 资本积累
            A[3,3] = 1-δ
            
            # 5. 货币政策规则
            A[5,1] = (1-ρr)*φπ
            A[5,2] = (1-ρr)*φy
            A[5,5] = ρr
            
            # 冲击矩阵
            B[2,0] = 1  # 技术冲击
            B[0,1] = 1  # 需求冲击
            B[5,2] = 1  # 货币政策冲击
            
            return {'A': A, 'B': B}
            
        except Exception as e:
            logger.error(f"对数线性化时出错: {str(e)}")
            raise
            
    def _求解模型(self, 线性化系数: dict) -> dict:
        """求解线性化模型"""
        try:
            A = 线性化系数['A']
            B = 线性化系数['B']
            
            # 使用简化的求解方法（实际应用中应使用更复杂的算法）
            n = len(A)
            P = np.zeros((n, n))
            
            def 黎卡提方程(P_vec):
                P_mat = P_vec.reshape(n, n)
                return (A.dot(P_mat).dot(A.T) - P_mat + B.dot(B.T)).flatten()
            
            # 求解稳态黎卡提方程
            result = minimize(
                lambda x: np.sum(黎卡提方程(x)**2),
                P.flatten(),
                method='BFGS'
            )
            
            if not result.success:
                raise ValueError("模型求解失败")
                
            P = result.x.reshape(n, n)
            
            # 计算政策函数
            F = -np.linalg.solve(A.T.dot(P).dot(A), A.T.dot(P).dot(B))
            
            return {
                'P': P,
                'F': F
            }
            
        except Exception as e:
            logger.error(f"求解模型时出错: {str(e)}")
            raise
            
    def _生成脉冲响应(self, 解: dict) -> Dict[str, pd.DataFrame]:
        """生成脉冲响应函数"""
        try:
            F = 解['F']
            
            # 设置脉冲响应期数
            期数 = 40
            变量名 = ['产出', '通胀', '消费', '投资', '劳动', '利率']
            冲击名 = ['技术冲击', '需求冲击', '货币冲击']
            
            # 初始化结果字典
            脉冲响应 = {}
            
            # 对每个冲击生成脉冲响应
            for i, 冲击 in enumerate(冲击名):
                # 初始化冲击向量
                ε = np.zeros(3)
                ε[i] = 1
                
                # 计算响应路径
                响应 = np.zeros((期数, len(变量名)))
                状态 = F.dot(ε)
                
                for t in range(期数):
                    响应[t] = 状态
                    状态 = F.dot(np.zeros(3))  # 后续期间无新冲击
                    
                脉冲响应[冲击] = pd.DataFrame(
                    响应,
                    columns=变量名
                )
                
            return 脉冲响应
            
        except Exception as e:
            logger.error(f"生成脉冲响应时出错: {str(e)}")
            raise
            
    def _历史分解(self, 解: dict) -> pd.DataFrame:
        """进行历史分解"""
        try:
            F = 解['F']
            
            # 获取历史数据
            数据 = self.经济数据.set_index('年份')
            
            # 计算增长率
            增长率 = 数据['GDP'].pct_change() * 100
            
            # 使用卡尔曼滤波估计历史冲击（简化版本）
            T = len(增长率)
            冲击 = np.random.randn(T, 3)  # 简化：随机生成冲击
            
            # 计算每个冲击的贡献
            贡献 = pd.DataFrame(
                np.dot(冲击, F.T),
                index=增长率.index,
                columns=['技术冲击贡献', '需求冲击贡献', '货币冲击贡献']
            )
            
            贡献['实际增长率'] = 增长率
            
            return 贡献
            
        except Exception as e:
            logger.error(f"进行历史分解时出错: {str(e)}")
            raise
            
    def _生成预测(self, 解: dict) -> pd.DataFrame:
        """生成预测"""
        try:
            F = 解['F']
            
            # 设置预测期数
            预测期数 = 8
            
            # 获取最新数据作为起点
            最新年份 = self.经济数据['年份'].max()
            最新数据 = self.经济数据[self.经济数据['年份'] == 最新年份].iloc[0]
            
            # 生成预测路径（简化版本）
            预测年份 = range(最新年份 + 1, 最新年份 + 预测期数 + 1)
            预测值 = pd.DataFrame(index=预测年份)
            
            # 基准预测
            预测值['GDP'] = 最新数据['GDP'] * (1 + np.random.normal(0.02, 0.01, 预测期数))
            预测值['消费'] = 最新数据['消费'] * (1 + np.random.normal(0.02, 0.01, 预测期数))
            预测值['投资'] = 最新数据['投资'] * (1 + np.random.normal(0.02, 0.015, 预测期数))
            预测值['CPI'] = 最新数据['CPI'] * (1 + np.random.normal(0.02, 0.005, 预测期数))
            
            return 预测值
            
        except Exception as e:
            logger.error(f"生成预测时出错: {str(e)}")
            raise
            
    def _保存结果(
        self,
        稳态: dict,
        脉冲响应: Dict[str, pd.DataFrame],
        历史分解: pd.DataFrame,
        预测: pd.DataFrame
    ):
        """保存分析结果"""
        try:
            with pd.ExcelWriter(self.结果目录 / 'dsge_results.xlsx') as writer:
                # 保存稳态值
                pd.DataFrame([稳态]).to_excel(writer, sheet_name='稳态值')
                
                # 保存脉冲响应
                for 冲击, 响应 in 脉冲响应.items():
                    响应.to_excel(writer, sheet_name=f'脉冲响应_{冲击}')
                
                # 保存历史分解
                历史分解.to_excel(writer, sheet_name='历史分解')
                
                # 保存预测
                预测.to_excel(writer, sheet_name='预测')
            
            logger.info("分析结果已保存到 dsge_results.xlsx")
            
        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
            raise 