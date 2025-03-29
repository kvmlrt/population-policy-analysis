import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from scipy.stats import norm, t, multivariate_normal

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class 蒙特卡洛模型:
    def __init__(self):
        """初始化蒙特卡洛模型"""
        self.结果目录 = Path('结果')
        self.结果目录.mkdir(exist_ok=True)
        
        # 初始化模型参数
        self.经济数据 = None
        self.行业数据 = None
        self.模拟次数 = 10000
        self.置信水平 = 0.95
        
    def 设置数据(self, 经济数据: pd.DataFrame, 行业数据: pd.DataFrame):
        """设置模型所需的数据"""
        try:
            self.经济数据 = 经济数据
            self.行业数据 = 行业数据
            logger.info("经济和行业数据已设置")
        except Exception as e:
            logger.error(f"设置数据时出错: {str(e)}")
            raise
            
    def 运行模型(self, 参数: dict = None) -> dict:
        """运行蒙特卡洛模型分析"""
        try:
            if self.经济数据 is None or self.行业数据 is None:
                raise ValueError("请先设置经济和行业数据")
                
            # 更新参数（如果提供）
            if 参数:
                self.模拟次数 = 参数.get('模拟次数', self.模拟次数)
                self.置信水平 = 参数.get('置信水平', self.置信水平)
            
            # 计算历史波动率
            波动率 = self._计算波动率()
            
            # 生成情景
            情景 = self._生成情景(波动率)
            
            # 计算风险值
            风险值 = self._计算风险值(情景)
            
            # 压力测试
            压力测试 = self._进行压力测试(情景)
            
            # 敏感性分析
            敏感性 = self._进行敏感性分析(情景)
            
            # 保存结果
            self._保存结果(波动率, 情景, 风险值, 压力测试, 敏感性)
            
            return {
                '波动率': 波动率,
                '情景': 情景,
                '风险值': 风险值,
                '压力测试': 压力测试,
                '敏感性': 敏感性
            }
            
        except Exception as e:
            logger.error(f"运行模型时出错: {str(e)}")
            raise
            
    def _计算波动率(self) -> pd.DataFrame:
        """计算历史波动率"""
        try:
            # 计算经济指标的波动率
            经济波动率 = pd.DataFrame()
            for 列 in ['GDP', '消费', '投资', 'CPI']:
                if 列 in self.经济数据.columns:
                    增长率 = self.经济数据[列].pct_change()
                    经济波动率[列] = [
                        增长率.std(),
                        增长率.skew(),
                        增长率.kurt()
                    ]
            经济波动率.index = ['标准差', '偏度', '峰度']
            
            # 计算行业产出的波动率
            最新年份 = self.行业数据['年份'].max()
            行业数据 = self.行业数据[self.行业数据['年份'] == 最新年份]
            行业波动率 = pd.DataFrame()
            
            for 行业 in 行业数据['行业'].unique():
                行业数据_单个 = self.行业数据[self.行业数据['行业'] == 行业]
                增长率 = 行业数据_单个['产出'].pct_change()
                行业波动率[行业] = [
                    增长率.std(),
                    增长率.skew(),
                    增长率.kurt()
                ]
            行业波动率.index = ['标准差', '偏度', '峰度']
            
            return {
                '经济波动率': 经济波动率,
                '行业波动率': 行业波动率
            }
            
        except Exception as e:
            logger.error(f"计算波动率时出错: {str(e)}")
            raise
            
    def _生成情景(self, 波动率: dict) -> Dict[str, pd.DataFrame]:
        """生成蒙特卡洛情景"""
        try:
            情景 = {}
            
            # 生成经济指标的情景
            经济波动率 = 波动率['经济波动率']
            经济情景 = pd.DataFrame()
            
            for 列 in 经济波动率.columns:
                # 使用t分布生成随机数，以更好地捕捉尾部风险
                自由度 = 5  # 较小的自由度会产生更厚的尾部
                随机数 = t.rvs(df=自由度, size=self.模拟次数)
                
                # 调整随机数以匹配目标矩
                标准差 = 经济波动率.loc['标准差', 列]
                调整后随机数 = 随机数 * 标准差
                
                经济情景[列] = 调整后随机数
            
            情景['经济情景'] = 经济情景
            
            # 生成行业产出的情景
            行业波动率 = 波动率['行业波动率']
            行业情景 = pd.DataFrame()
            
            # 考虑行业间相关性
            行业数 = len(行业波动率.columns)
            相关系数 = np.full((行业数, 行业数), 0.3)  # 假设所有行业间有0.3的相关性
            np.fill_diagonal(相关系数, 1)
            
            # 生成多元正态分布的随机数
            标准差向量 = 行业波动率.loc['标准差'].values
            协方差矩阵 = np.outer(标准差向量, 标准差向量) * 相关系数
            
            随机数 = multivariate_normal.rvs(
                mean=np.zeros(行业数),
                cov=协方差矩阵,
                size=self.模拟次数
            )
            
            行业情景 = pd.DataFrame(
                随机数,
                columns=行业波动率.columns
            )
            
            情景['行业情景'] = 行业情景
            
            return 情景
            
        except Exception as e:
            logger.error(f"生成情景时出错: {str(e)}")
            raise
            
    def _计算风险值(self, 情景: Dict[str, pd.DataFrame]) -> dict:
        """计算在险价值(VaR)和期望短缺(ES)"""
        try:
            风险值 = {}
            
            # 计算经济指标的风险值
            经济情景 = 情景['经济情景']
            经济风险值 = pd.DataFrame()
            
            for 列 in 经济情景.columns:
                数据 = 经济情景[列]
                VaR = np.percentile(数据, (1 - self.置信水平) * 100)
                ES = 数据[数据 <= VaR].mean()
                
                经济风险值[列] = [VaR, ES]
            
            经济风险值.index = ['VaR', 'ES']
            风险值['经济风险'] = 经济风险值
            
            # 计算行业风险值
            行业情景 = 情景['行业情景']
            行业风险值 = pd.DataFrame()
            
            for 列 in 行业情景.columns:
                数据 = 行业情景[列]
                VaR = np.percentile(数据, (1 - self.置信水平) * 100)
                ES = 数据[数据 <= VaR].mean()
                
                行业风险值[列] = [VaR, ES]
            
            行业风险值.index = ['VaR', 'ES']
            风险值['行业风险'] = 行业风险值
            
            return 风险值
            
        except Exception as e:
            logger.error(f"计算风险值时出错: {str(e)}")
            raise
            
    def _进行压力测试(self, 情景: Dict[str, pd.DataFrame]) -> dict:
        """进行压力测试"""
        try:
            压力测试 = {}
            
            # 定义压力情景
            压力情景 = {
                '轻度压力': -1,  # 1个标准差
                '中度压力': -2,  # 2个标准差
                '重度压力': -3   # 3个标准差
            }
            
            # 对经济指标进行压力测试
            经济情景 = 情景['经济情景']
            经济压力测试 = pd.DataFrame()
            
            for 列 in 经济情景.columns:
                标准差 = 经济情景[列].std()
                压力结果 = {
                    级别: 经济情景[列].mean() + 倍数 * 标准差
                    for 级别, 倍数 in 压力情景.items()
                }
                经济压力测试[列] = 压力结果.values()
            
            经济压力测试.index = 压力情景.keys()
            压力测试['经济压力测试'] = 经济压力测试
            
            # 对行业进行压力测试
            行业情景 = 情景['行业情景']
            行业压力测试 = pd.DataFrame()
            
            for 列 in 行业情景.columns:
                标准差 = 行业情景[列].std()
                压力结果 = {
                    级别: 行业情景[列].mean() + 倍数 * 标准差
                    for 级别, 倍数 in 压力情景.items()
                }
                行业压力测试[列] = 压力结果.values()
            
            行业压力测试.index = 压力情景.keys()
            压力测试['行业压力测试'] = 行业压力测试
            
            return 压力测试
            
        except Exception as e:
            logger.error(f"进行压力测试时出错: {str(e)}")
            raise
            
    def _进行敏感性分析(self, 情景: Dict[str, pd.DataFrame]) -> dict:
        """进行敏感性分析"""
        try:
            敏感性 = {}
            
            # 对经济指标进行敏感性分析
            经济情景 = 情景['经济情景']
            经济敏感性 = pd.DataFrame()
            
            # 计算相关系数矩阵
            经济相关系数 = 经济情景.corr()
            
            # 计算弹性（使用简单的线性回归系数近似）
            基准列 = 'GDP'
            if 基准列 in 经济情景.columns:
                for 列 in 经济情景.columns:
                    if 列 != 基准列:
                        斜率 = np.polyfit(经济情景[基准列], 经济情景[列], 1)[0]
                        经济敏感性[列] = [斜率]
            
            经济敏感性.index = ['对GDP的弹性']
            敏感性['经济敏感性'] = {
                '相关系数': 经济相关系数,
                '弹性': 经济敏感性
            }
            
            # 对行业进行敏感性分析
            行业情景 = 情景['行业情景']
            行业敏感性 = pd.DataFrame()
            
            # 计算行业间相关系数
            行业相关系数 = 行业情景.corr()
            
            # 计算对总体经济的敏感性
            if 'GDP' in 经济情景.columns:
                for 列 in 行业情景.columns:
                    斜率 = np.polyfit(经济情景['GDP'], 行业情景[列], 1)[0]
                    行业敏感性[列] = [斜率]
            
            行业敏感性.index = ['对GDP的弹性']
            敏感性['行业敏感性'] = {
                '相关系数': 行业相关系数,
                '弹性': 行业敏感性
            }
            
            return 敏感性
            
        except Exception as e:
            logger.error(f"进行敏感性分析时出错: {str(e)}")
            raise
            
    def _保存结果(
        self,
        波动率: dict,
        情景: Dict[str, pd.DataFrame],
        风险值: dict,
        压力测试: dict,
        敏感性: dict
    ):
        """保存分析结果"""
        try:
            with pd.ExcelWriter(self.结果目录 / 'mc_results.xlsx') as writer:
                # 保存波动率
                波动率['经济波动率'].to_excel(writer, sheet_name='经济波动率')
                波动率['行业波动率'].to_excel(writer, sheet_name='行业波动率')
                
                # 保存情景分析结果（仅保存部分样本）
                样本数 = min(1000, self.模拟次数)
                情景['经济情景'].head(样本数).to_excel(writer, sheet_name='经济情景样本')
                情景['行业情景'].head(样本数).to_excel(writer, sheet_name='行业情景样本')
                
                # 保存风险值
                风险值['经济风险'].to_excel(writer, sheet_name='经济风险值')
                风险值['行业风险'].to_excel(writer, sheet_name='行业风险值')
                
                # 保存压力测试结果
                压力测试['经济压力测试'].to_excel(writer, sheet_name='经济压力测试')
                压力测试['行业压力测试'].to_excel(writer, sheet_name='行业压力测试')
                
                # 保存敏感性分析结果
                敏感性['经济敏感性']['相关系数'].to_excel(writer, sheet_name='经济相关系数')
                敏感性['经济敏感性']['弹性'].to_excel(writer, sheet_name='经济弹性')
                敏感性['行业敏感性']['相关系数'].to_excel(writer, sheet_name='行业相关系数')
                敏感性['行业敏感性']['弹性'].to_excel(writer, sheet_name='行业弹性')
            
            logger.info("分析结果已保存到 mc_results.xlsx")
            
        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
            raise 