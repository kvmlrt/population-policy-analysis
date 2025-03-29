import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class 人口智能体:
    """人口智能体类，负责模拟人口变化"""
    
    def __init__(self, 初始数据: Dict[str, Any]):
        """初始化人口智能体
        
        Args:
            初始数据: 包含初始人口数据的字典
        """
        self.状态 = {
            '总人口': 初始数据.get('总人口', 140000),  # 万人
            '城镇人口': 初始数据.get('城镇人口', 90000),  # 万人
            '儿童人口': 初始数据.get('儿童人口', 25000),  # 万人
            '劳动年龄人口': 初始数据.get('劳动年龄人口', 90000),  # 万人
            '老年人口': 初始数据.get('老年人口', 25000),  # 万人
            '教育水平': 初始数据.get('教育水平', 0.5),  # 0-1之间的指数
            '生育率': 初始数据.get('生育率', 0.012),  # 每年每人
            '死亡率': 初始数据.get('死亡率', 0.007),  # 每年每人
            '城镇化率': 初始数据.get('城镇化率', 0.64),  # 0-1之间
            '生育率弹性': 初始数据.get('生育率弹性', 0.3),  # 对经济和政策的敏感度
        }
        
        # 验证数据一致性
        总人口 = self.状态['总人口']
        年龄结构和 = (self.状态['儿童人口'] + self.状态['劳动年龄人口'] + 
                     self.状态['老年人口'])
        if abs(总人口 - 年龄结构和) > 1:  # 允许1万人的误差
            logger.warning(f"人口总数({总人口})与年龄结构之和({年龄结构和})不匹配")
    
    def 获取状态(self) -> Dict[str, Any]:
        """获取当前状态
        
        Returns:
            包含当前状态的字典
        """
        return self.状态.copy()
    
    def 更新状态(self, 人均GDP: float, 社保覆盖率: float, 产业升级率: float = 0.02) -> Dict[str, Any]:
        """更新人口状态
        
        Args:
            人均GDP: 人均国内生产总值（万元/人）
            社保覆盖率: 社会保障覆盖率（0-1之间）
            产业升级率: 产业升级速度（0-1之间）
            
        Returns:
            更新后的状态字典
        """
        try:
            # 1. 计算生育率调整
            生育率基准 = self.状态['生育率']
            生育率弹性 = self.状态['生育率弹性']
            GDP影响 = np.log(1 + 人均GDP) * 生育率弹性
            社保影响 = 社保覆盖率 * 0.2  # 社保覆盖提高生育意愿
            新生育率 = 生育率基准 * (1 + GDP影响 + 社保影响)
            新生育率 = max(0.008, min(0.02, 新生育率))  # 限制在合理范围内
            
            # 2. 计算人口变化
            出生人口 = self.状态['劳动年龄人口'] * 新生育率
            死亡人口 = (self.状态['总人口'] * self.状态['死亡率'] * 
                       (1 - 社保覆盖率 * 0.1))  # 社保覆盖降低死亡率
            
            # 3. 更新年龄结构
            新儿童人口 = (self.状态['儿童人口'] + 出生人口 - 
                         self.状态['儿童人口'] * 0.067)  # 每年约15分之一进入劳动年龄
            新劳动年龄人口 = (self.状态['劳动年龄人口'] + 
                             self.状态['儿童人口'] * 0.067 - 
                             self.状态['劳动年龄人口'] * 0.022)  # 每年约45分之一退休
            新老年人口 = (self.状态['老年人口'] + 
                         self.状态['劳动年龄人口'] * 0.022 - 
                         死亡人口)
            
            # 4. 更新城镇化率
            城镇化增长 = 0.01 * (1 + 产业升级率)  # 基础增长率受产业升级影响
            新城镇化率 = min(0.85, self.状态['城镇化率'] + 城镇化增长)
            
            # 5. 更新教育水平
            教育水平增长 = 0.02 * (1 + 产业升级率)  # 教育水平提升受产业升级影响
            新教育水平 = min(0.9, self.状态['教育水平'] + 教育水平增长)
            
            # 6. 更新状态
            新状态 = {
                '总人口': 新儿童人口 + 新劳动年龄人口 + 新老年人口,
                '城镇人口': (新儿童人口 + 新劳动年龄人口 + 新老年人口) * 新城镇化率,
                '儿童人口': 新儿童人口,
                '劳动年龄人口': 新劳动年龄人口,
                '老年人口': 新老年人口,
                '教育水平': 新教育水平,
                '生育率': 新生育率,
                '死亡率': self.状态['死亡率'],
                '城镇化率': 新城镇化率,
                '生育率弹性': self.状态['生育率弹性']
            }
            
            self.状态 = 新状态
            return 新状态
            
        except Exception as e:
            logger.error(f"更新人口状态时出错: {str(e)}")
            raise 