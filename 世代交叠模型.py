import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from scipy.optimize import fsolve

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class 世代交叠模型:
    def __init__(self):
        """初始化世代交叠模型"""
        self.结果目录 = Path('结果')
        self.结果目录.mkdir(exist_ok=True)
        
        # 初始化模型参数
        self.人口数据 = None
        self.经济数据 = None
        self.社保数据 = None
        
    def 设置数据(self, 人口数据: pd.DataFrame, 经济数据: pd.DataFrame, 社保数据: pd.DataFrame):
        """设置模型所需的数据"""
        try:
            self.人口数据 = 人口数据
            self.经济数据 = 经济数据
            self.社保数据 = 社保数据
            logger.info("人口、经济和社保数据已设置")
        except Exception as e:
            logger.error(f"设置数据时出错: {str(e)}")
            raise
            
    def 运行模型(self, 参数: dict) -> dict:
        """运行世代交叠模型分析"""
        try:
            if self.人口数据 is None or self.经济数据 is None or self.社保数据 is None:
                raise ValueError("请先设置所有必要的数据")
                
            # 获取模型参数
            退休年龄 = 参数.get('退休年龄', 65)
            最大年龄 = 参数.get('最大年龄', 100)
            贴现率 = 参数.get('贴现率', 0.98)
            风险厌恶度 = 参数.get('风险厌恶度', 2.0)
            社保缴费率 = 参数.get('社保缴费率', 0.2)
            
            # 计算人口结构
            人口结构 = self._计算人口结构(退休年龄, 最大年龄)
            
            # 计算生命周期消费和储蓄
            消费储蓄 = self._计算生命周期消费储蓄(
                人口结构, 贴现率, 风险厌恶度, 社保缴费率
            )
            
            # 计算养老金缺口
            养老金缺口 = self._计算养老金缺口(人口结构, 社保缴费率)
            
            # 计算宏观经济影响
            宏观影响 = self._计算宏观经济影响(消费储蓄, 养老金缺口)
            
            # 保存结果
            self._保存结果(人口结构, 消费储蓄, 养老金缺口, 宏观影响)
            
            return {
                '人口结构': 人口结构,
                '消费储蓄': 消费储蓄,
                '养老金缺口': 养老金缺口,
                '宏观影响': 宏观影响
            }
            
        except Exception as e:
            logger.error(f"运行模型时出错: {str(e)}")
            raise
            
    def _计算人口结构(self, 退休年龄: int, 最大年龄: int) -> pd.DataFrame:
        """计算人口结构"""
        try:
            最新年份 = self.人口数据['年份'].max()
            最新数据 = self.人口数据[self.人口数据['年份'] == 最新年份]
            
            # 构建年龄结构
            年龄范围 = range(20, 最大年龄 + 1)  # 从20岁开始
            人口结构 = pd.DataFrame(index=年龄范围)
            
            # 计算各年龄段人口比例
            总人口 = 最新数据['总人口'].iloc[0]
            劳动人口 = 最新数据['劳动年龄人口'].iloc[0]
            老年人口 = 最新数据['老年人口'].iloc[0]
            
            # 简化处理：假设人口在各年龄段均匀分布
            工作年龄段 = range(20, 退休年龄)
            退休年龄段 = range(退休年龄, 最大年龄 + 1)
            
            人口结构['人口数'] = 0
            人口结构.loc[工作年龄段, '人口数'] = 劳动人口 / len(工作年龄段)
            人口结构.loc[退休年龄段, '人口数'] = 老年人口 / len(退休年龄段)
            
            # 计算抚养比
            人口结构['是否退休'] = 人口结构.index >= 退休年龄
            人口结构['抚养比'] = (人口结构[人口结构['是否退休']]['人口数'].sum() / 
                              人口结构[~人口结构['是否退休']]['人口数'].sum())
            
            logger.info("人口结构计算完成")
            return 人口结构
            
        except Exception as e:
            logger.error(f"计算人口结构时出错: {str(e)}")
            raise
            
    def _计算生命周期消费储蓄(
        self, 
        人口结构: pd.DataFrame,
        贴现率: float,
        风险厌恶度: float,
        社保缴费率: float
    ) -> pd.DataFrame:
        """计算生命周期消费和储蓄"""
        try:
            # 获取经济参数
            最新年份 = self.经济数据['年份'].max()
            最新数据 = self.经济数据[self.经济数据['年份'] == 最新年份]
            人均GDP = 最新数据['GDP'].iloc[0] / self.人口数据[self.人口数据['年份'] == 最新年份]['总人口'].iloc[0]
            
            # 初始化结果DataFrame
            消费储蓄 = 人口结构.copy()
            
            # 计算工作期收入（简化假设：收入为人均GDP）
            消费储蓄['收入'] = 0
            消费储蓄.loc[~消费储蓄['是否退休'], '收入'] = 人均GDP
            
            # 计算养老金收入（简化假设：为工作期收入的一定比例）
            养老金比例 = 0.4  # 假设养老金为工作期收入的40%
            消费储蓄.loc[消费储蓄['是否退休'], '收入'] = 人均GDP * 养老金比例
            
            # 计算社保缴费
            消费储蓄['社保缴费'] = 0
            消费储蓄.loc[~消费储蓄['是否退休'], '社保缴费'] = 消费储蓄['收入'] * 社保缴费率
            
            # 计算可支配收入
            消费储蓄['可支配收入'] = 消费储蓄['收入'] - 消费储蓄['社保缴费']
            
            # 使用效用最大化计算最优消费
            def 效用函数(消费, 收入, 风险厌恶):
                if 风险厌恶 == 1:
                    return np.log(消费)
                else:
                    return (消费 ** (1 - 风险厌恶) - 1) / (1 - 风险厌恶)
                    
            def 最优消费(收入, 贴现率, 风险厌恶):
                return 收入 * (1 - 贴现率 ** (1/风险厌恶))
            
            # 计算消费和储蓄
            消费储蓄['消费'] = 消费储蓄['可支配收入'].apply(
                lambda x: 最优消费(x, 贴现率, 风险厌恶度)
            )
            消费储蓄['储蓄'] = 消费储蓄['可支配收入'] - 消费储蓄['消费']
            
            # 计算效用水平
            消费储蓄['效用'] = 消费储蓄['消费'].apply(
                lambda x: 效用函数(x, x, 风险厌恶度)
            )
            
            logger.info("生命周期消费和储蓄计算完成")
            return 消费储蓄
            
        except Exception as e:
            logger.error(f"计算生命周期消费储蓄时出错: {str(e)}")
            raise
            
    def _计算养老金缺口(self, 人口结构: pd.DataFrame, 社保缴费率: float) -> dict:
        """计算养老金缺口"""
        try:
            最新年份 = self.社保数据['年份'].max()
            最新数据 = self.社保数据[self.社保数据['年份'] == 最新年份]
            
            # 计算当前养老金收支
            当前收入 = 最新数据['养老金收入'].iloc[0]
            当前支出 = 最新数据['养老金支出'].iloc[0]
            当前缺口 = 当前支出 - 当前收入
            
            # 计算未来养老金缺口
            工作人口 = 人口结构[~人口结构['是否退休']]['人口数'].sum()
            退休人口 = 人口结构[人口结构['是否退休']]['人口数'].sum()
            
            # 获取人均GDP
            人均GDP = (self.经济数据[self.经济数据['年份'] == 最新年份]['GDP'].iloc[0] / 
                    self.人口数据[self.人口数据['年份'] == 最新年份]['总人口'].iloc[0])
            
            # 计算未来收支
            预计收入 = 工作人口 * 人均GDP * 社保缴费率
            预计支出 = 退休人口 * 人均GDP * 0.4  # 假设养老金为工作期收入的40%
            预计缺口 = 预计支出 - 预计收入
            
            # 计算缺口占GDP比重
            GDP = self.经济数据[self.经济数据['年份'] == 最新年份]['GDP'].iloc[0]
            当前缺口率 = 当前缺口 / GDP * 100
            预计缺口率 = 预计缺口 / GDP * 100
            
            return {
                '当前缺口': 当前缺口,
                '预计缺口': 预计缺口,
                '当前缺口率': 当前缺口率,
                '预计缺口率': 预计缺口率
            }
            
        except Exception as e:
            logger.error(f"计算养老金缺口时出错: {str(e)}")
            raise
            
    def _计算宏观经济影响(self, 消费储蓄: pd.DataFrame, 养老金缺口: dict) -> dict:
        """计算宏观经济影响"""
        try:
            # 计算总量指标
            总人口 = 消费储蓄['人口数'].sum()
            总消费 = (消费储蓄['消费'] * 消费储蓄['人口数']).sum()
            总储蓄 = (消费储蓄['储蓄'] * 消费储蓄['人口数']).sum()
            
            # 获取GDP数据
            最新年份 = self.经济数据['年份'].max()
            GDP = self.经济数据[self.经济数据['年份'] == 最新年份]['GDP'].iloc[0]
            
            # 计算比率
            消费率 = 总消费 / GDP * 100
            储蓄率 = 总储蓄 / GDP * 100
            
            # 计算养老金缺口对经济的影响
            缺口影响 = 养老金缺口['预计缺口率'] * 0.5  # 假设缺口的一半转化为GDP损失
            
            # 计算人均指标
            人均消费 = 总消费 / 总人口
            人均储蓄 = 总储蓄 / 总人口
            
            return {
                '消费率': 消费率,
                '储蓄率': 储蓄率,
                '人均消费': 人均消费,
                '人均储蓄': 人均储蓄,
                'GDP影响': -缺口影响  # 负值表示损失
            }
            
        except Exception as e:
            logger.error(f"计算宏观经济影响时出错: {str(e)}")
            raise
            
    def _保存结果(
        self,
        人口结构: pd.DataFrame,
        消费储蓄: pd.DataFrame,
        养老金缺口: dict,
        宏观影响: dict
    ):
        """保存分析结果"""
        try:
            with pd.ExcelWriter(self.结果目录 / 'olg_results.xlsx') as writer:
                # 保存人口结构
                人口结构.to_excel(writer, sheet_name='人口结构')
                
                # 保存消费储蓄
                消费储蓄.to_excel(writer, sheet_name='消费储蓄')
                
                # 保存养老金缺口
                pd.DataFrame([养老金缺口]).to_excel(writer, sheet_name='养老金缺口')
                
                # 保存宏观影响
                pd.DataFrame([宏观影响]).to_excel(writer, sheet_name='宏观影响')
            
            logger.info("分析结果已保存到 olg_results.xlsx")
            
        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
            raise 