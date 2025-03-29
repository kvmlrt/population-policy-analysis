import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from shap import KernelExplainer
from .人口智能体 import 人口智能体
from .经济智能体 import 经济智能体
from .社保智能体 import 社保智能体
from .外部冲击智能体 import 外部冲击智能体

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class 多智能体系统:
    """多智能体系统类，负责协调各个智能体的交互"""
    
    def __init__(self, 初始数据: Dict[str, Any]):
        """初始化多智能体系统
        
        Args:
            初始数据: 包含各个智能体初始数据的字典
        """
        self.人口智能体 = 人口智能体(初始数据.get('人口数据', {}))
        self.经济智能体 = 经济智能体(初始数据.get('经济数据', {}))
        self.社保智能体 = 社保智能体(初始数据.get('社保数据', {}))
        self.外部冲击智能体 = 外部冲击智能体()
        
        # 初始化结果存储
        self.模拟结果 = {
            '年份': [],
            '总人口': [],
            'GDP': [],
            '社保收入': [],
            '社保支出': [],
            '城镇人口': [],
            '农村人口': [],
            '儿童人口': [],
            '劳动年龄人口': [],
            '老年人口': []
        }
    
    def 模拟步进(self, 外部冲击: Dict[str, float] = None) -> Dict[str, Any]:
        """执行一步模拟
        
        Args:
            外部冲击: 外部冲击影响系数字典
            
        Returns:
            更新后的状态字典
        """
        try:
            # 1. 获取当前状态
            人口状态 = self.人口智能体.获取状态()
            经济状态 = self.经济智能体.获取状态()
            社保状态 = self.社保智能体.获取状态()
            
            # 2. 计算中间变量
            人均GDP = 经济状态['GDP'] / 人口状态['总人口']
            社保覆盖率 = 社保状态['参保人数'] / 人口状态['劳动年龄人口']
            
            # 3. 获取外部冲击影响
            外部影响 = self.外部冲击智能体.获取影响系数() if 外部冲击 else {}
            
            # 4. 更新人口智能体
            产业升级率 = 外部影响.get('产业升级率', 0.0)
            新人口状态 = self.人口智能体.更新状态(
                人均GDP=人均GDP,
                社保覆盖率=社保覆盖率,
                产业升级率=产业升级率
            )
            
            # 5. 更新经济智能体
            新经济状态 = self.经济智能体.更新状态(
                人口数据=新人口状态,
                社保缴费率=社保状态['缴费率'],
                教育水平=新人口状态['教育水平'],
                外部冲击=外部影响
            )
            
            # 6. 更新社保智能体
            新社保状态 = self.社保智能体.更新状态(
                GDP=新经济状态['GDP'],
                人口数据=新人口状态,
                经济数据=新经济状态
            )
            
            # 7. 返回更新后的状态
            return {
                '人口': 新人口状态,
                '经济': 新经济状态,
                '社保': 新社保状态
            }
            
        except Exception as e:
            logger.error(f"模拟步进时出错: {str(e)}")
            raise
    
    def 模拟(self, 步数: int) -> pd.DataFrame:
        """执行多步模拟
        
        Args:
            步数: 模拟步数
            
        Returns:
            包含模拟结果的DataFrame
        """
        try:
            # 重置结果存储
            self.模拟结果 = {
                '年份': [],
                '总人口': [],
                'GDP': [],
                '社保收入': [],
                '社保支出': [],
                '城镇人口': [],
                '农村人口': [],
                '儿童人口': [],
                '劳动年龄人口': [],
                '老年人口': []
            }
            
            # 记录初始状态
            初始人口状态 = self.人口智能体.获取状态()
            初始经济状态 = self.经济智能体.获取状态()
            初始社保状态 = self.社保智能体.获取状态()
            
            self.模拟结果['年份'].append(2024)
            self.模拟结果['总人口'].append(初始人口状态['总人口'])
            self.模拟结果['GDP'].append(初始经济状态['GDP'])
            self.模拟结果['社保收入'].append(初始社保状态['社保收入'])
            self.模拟结果['社保支出'].append(初始社保状态['社保支出'])
            self.模拟结果['城镇人口'].append(初始人口状态['城镇人口'])
            self.模拟结果['农村人口'].append(初始人口状态['总人口'] - 初始人口状态['城镇人口'])
            self.模拟结果['儿童人口'].append(初始人口状态['儿童人口'])
            self.模拟结果['劳动年龄人口'].append(初始人口状态['劳动年龄人口'])
            self.模拟结果['老年人口'].append(初始人口状态['老年人口'])
            
            # 执行多步模拟
            for i in range(步数):
                新状态 = self.模拟步进()
                
                self.模拟结果['年份'].append(2024 + i + 1)
                self.模拟结果['总人口'].append(新状态['人口']['总人口'])
                self.模拟结果['GDP'].append(新状态['经济']['GDP'])
                self.模拟结果['社保收入'].append(新状态['社保']['社保收入'])
                self.模拟结果['社保支出'].append(新状态['社保']['社保支出'])
                self.模拟结果['城镇人口'].append(新状态['人口']['城镇人口'])
                self.模拟结果['农村人口'].append(新状态['人口']['总人口'] - 新状态['人口']['城镇人口'])
                self.模拟结果['儿童人口'].append(新状态['人口']['儿童人口'])
                self.模拟结果['劳动年龄人口'].append(新状态['人口']['劳动年龄人口'])
                self.模拟结果['老年人口'].append(新状态['人口']['老年人口'])
            
            # 转换为DataFrame
            return pd.DataFrame(self.模拟结果)
            
        except Exception as e:
            logger.error(f"多步模拟出错: {str(e)}")
            raise

    def 模型评估(self, 测试数据: dict) -> Dict:
        """
        评估模型预测效果
        参数:
            测试数据: dict, 包含'人口'、'经济'、'社保'三个DataFrame
        返回:
            Dict: 包含各项评估指标
        """
        try:
            # 获取测试数据的年份范围
            人口数据 = 测试数据['人口']
            经济数据 = 测试数据['经济']
            社保数据 = 测试数据['社保']
            
            年份列表 = sorted(set(人口数据['年份']) & set(经济数据['年份']) & set(社保数据['年份']))
            
            # 运行模拟
            模拟结果 = self.模拟(len(年份列表))
            
            # 构建实际数据序列
            实际数据 = {
                '总人口': [],
                'GDP': [],
                '社保收入': [],
                '社保支出': []
            }
            
            for 年份 in 年份列表:
                实际数据['总人口'].append(人口数据[人口数据['年份'] == 年份].iloc[0]['总人口'])
                实际数据['GDP'].append(经济数据[经济数据['年份'] == 年份].iloc[0]['GDP'])
                实际数据['社保收入'].append(社保数据[社保数据['年份'] == 年份].iloc[0]['收入'])
                实际数据['社保支出'].append(社保数据[社保数据['年份'] == 年份].iloc[0]['支出'])
            
            # 计算评估指标
            评估结果 = {}
            
            # 1. 计算各指标的RMSE
            for 指标 in ['总人口', 'GDP', '社保收入', '社保支出']:
                rmse = np.sqrt(mean_squared_error(
                    实际数据[指标],
                    模拟结果[指标][:len(年份列表)]
                ))
                评估结果[f'{指标}_RMSE'] = rmse
                
                # 计算相对误差
                相对误差 = np.mean(np.abs(
                    (np.array(实际数据[指标]) - np.array(模拟结果[指标][:len(年份列表)])) /
                    np.array(实际数据[指标])
                ))
                评估结果[f'{指标}_相对误差'] = 相对误差
            
            # 2. 计算方向准确率
            for 指标 in ['GDP', '社保收入']:
                实际变化 = np.sign(np.diff(实际数据[指标]))
                预测变化 = np.sign(np.diff(模拟结果[指标][:len(年份列表)]))
                方向准确率 = np.mean(实际变化 == 预测变化)
                评估结果[f'{指标}_方向准确率'] = 方向准确率
            
            # 3. 计算相关系数
            for 指标 in ['总人口', 'GDP', '社保收入', '社保支出']:
                相关系数 = np.corrcoef(
                    实际数据[指标],
                    模拟结果[指标][:len(年份列表)]
                )[0, 1]
                评估结果[f'{指标}_相关系数'] = 相关系数
            
            return 评估结果
            
        except Exception as e:
            logger.error(f"模型评估出错: {str(e)}")
            raise 