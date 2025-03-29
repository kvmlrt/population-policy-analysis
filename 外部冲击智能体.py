import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class 外部冲击智能体:
    """外部冲击智能体，用于模拟外部事件对系统的影响"""
    
    def __init__(self):
        """初始化外部冲击智能体"""
        self.影响系数 = {
            '产业升级率': 0.02,  # 默认每年产业升级率
            'GDP增长率调整': 0.0,  # GDP增长率的调整系数
            '人口流动率': 0.0,  # 人口流动影响系数
            '社保覆盖率调整': 0.0  # 社保覆盖率的调整系数
        }
    
    def 设置影响系数(self, 新系数: Dict[str, float]) -> None:
        """设置新的影响系数
        
        Args:
            新系数: 包含各项影响系数的字典
        """
        self.影响系数.update(新系数)
        logger.info(f"更新外部冲击影响系数: {self.影响系数}")
    
    def 获取影响系数(self) -> Dict[str, float]:
        """获取当前的影响系数
        
        Returns:
            当前的影响系数字典
        """
        return self.影响系数.copy()
    
    def 重置影响系数(self) -> None:
        """重置所有影响系数为默认值"""
        self.影响系数 = {
            '产业升级率': 0.02,
            'GDP增长率调整': 0.0,
            '人口流动率': 0.0,
            '社保覆盖率调整': 0.0
        }
        logger.info("重置外部冲击影响系数为默认值") 