import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class 投入产出模型:
    def __init__(self):
        """初始化投入产出模型"""
        self.结果目录 = Path('结果')
        self.结果目录.mkdir(exist_ok=True)
        
        # 初始化模型参数
        self.行业数据 = None
        self.经济数据 = None
        self.直接消耗系数 = None
        self.完全消耗系数 = None
        self.劳动力系数 = None
        self.基准冲击 = None
        self.基准效应 = None
        
    def 设置数据(self, 行业数据: pd.DataFrame, 经济数据: pd.DataFrame):
        """设置模型所需的数据"""
        try:
            self.行业数据 = 行业数据
            self.经济数据 = 经济数据
            logger.info("行业和经济数据已设置")
        except Exception as e:
            logger.error(f"设置数据时出错: {str(e)}")
            raise
            
    def 运行模型(self, 参数: dict) -> dict:
        """运行投入产出模型分析"""
        try:
            if self.行业数据 is None or self.经济数据 is None:
                raise ValueError("请先设置行业和经济数据")
                
            # 计算系数
            self._计算直接消耗系数()
            self._计算完全消耗系数()
            self._计算劳动力系数()
            
            # 计算乘数
            产出乘数 = self._计算乘数()
            
            # 计算关联效应
            前向关联, 后向关联 = self._计算关联效应()
            
            # 识别关键部门
            关键部门 = self._识别关键部门(前向关联, 后向关联)
            
            # 模拟政策冲击
            self._模拟政策冲击(参数.get('政策冲击', {}))
            
            # 保存结果
            self._保存结果()
            
            return {
                '产出乘数': 产出乘数,
                '前向关联': 前向关联,
                '后向关联': 后向关联,
                '关键部门': 关键部门,
                '政策效应': self.基准效应
            }
            
        except Exception as e:
            logger.error(f"运行模型时出错: {str(e)}")
            raise
            
    def _计算直接消耗系数(self):
        """计算直接消耗系数"""
        try:
            最新年份 = self.行业数据['年份'].max()
            最新数据 = self.行业数据[self.行业数据['年份'] == 最新年份]
            
            # 构建投入产出矩阵
            行业列表 = 最新数据['行业'].unique()
            n = len(行业列表)
            self.直接消耗系数 = pd.DataFrame(
                np.random.random((n, n)) * 0.1,  # 示例：使用随机数据
                index=行业列表,
                columns=行业列表
            )
            
            # 确保系数合理
            self.直接消耗系数 = self.直接消耗系数 / self.直接消耗系数.sum(axis=0)
            
            logger.info("直接消耗系数计算完成")
            
        except Exception as e:
            logger.error(f"计算直接消耗系数时出错: {str(e)}")
            raise
            
    def _计算完全消耗系数(self):
        """计算完全消耗系数"""
        try:
            if self.直接消耗系数 is None:
                raise ValueError("请先计算直接消耗系数")
                
            # 计算列昂惕夫逆矩阵
            n = len(self.直接消耗系数)
            单位矩阵 = np.eye(n)
            self.完全消耗系数 = pd.DataFrame(
                np.linalg.inv(单位矩阵 - self.直接消耗系数.values),
                index=self.直接消耗系数.index,
                columns=self.直接消耗系数.columns
            )
            
            logger.info("完全消耗系数计算完成")
            
        except Exception as e:
            logger.error(f"计算完全消耗系数时出错: {str(e)}")
            raise
            
    def _计算劳动力系数(self):
        """计算劳动力系数"""
        try:
            最新年份 = self.行业数据['年份'].max()
            最新数据 = self.行业数据[self.行业数据['年份'] == 最新年份]
            
            # 计算每单位产出的就业人数
            self.劳动力系数 = pd.Series(
                最新数据['就业人数'].values / 最新数据['产出'].values,
                index=最新数据['行业']
            )
            
            logger.info("劳动力系数计算完成")
            
        except Exception as e:
            logger.error(f"计算劳动力系数时出错: {str(e)}")
            raise
            
    def _计算乘数(self) -> pd.Series:
        """计算产出乘数"""
        try:
            if self.完全消耗系数 is None:
                raise ValueError("请先计算完全消耗系数")
                
            # 计算每个部门的产出乘数（列和）
            产出乘数 = pd.Series(
                self.完全消耗系数.sum(axis=0),
                index=self.完全消耗系数.columns
            )
            
            logger.info("产出乘数计算完成")
            return 产出乘数
            
        except Exception as e:
            logger.error(f"计算产出乘数时出错: {str(e)}")
            raise
            
    def _计算关联效应(self) -> Tuple[pd.Series, pd.Series]:
        """计算前向和后向关联效应"""
        try:
            if self.完全消耗系数 is None:
                raise ValueError("请先计算完全消耗系数")
                
            n = len(self.完全消耗系数)
            
            # 计算后向关联效应
            后向关联 = pd.Series(
                self.完全消耗系数.sum(axis=0) / n,
                index=self.完全消耗系数.columns
            )
            
            # 计算前向关联效应
            前向关联 = pd.Series(
                self.完全消耗系数.sum(axis=1) / n,
                index=self.完全消耗系数.index
            )
            
            logger.info("关联效应计算完成")
            return 前向关联, 后向关联
            
        except Exception as e:
            logger.error(f"计算关联效应时出错: {str(e)}")
            raise
            
    def _识别关键部门(self, 前向关联: pd.Series, 后向关联: pd.Series) -> Dict[str, List[str]]:
        """识别关键部门"""
        try:
            # 计算平均值作为判断标准
            前向平均 = 前向关联.mean()
            后向平均 = 后向关联.mean()
            
            # 分类部门
            关键部门 = []
            一般部门 = []
            前向主导 = []
            后向主导 = []
            
            for 行业 in 前向关联.index:
                前向值 = 前向关联[行业]
                后向值 = 后向关联[行业]
                
                if 前向值 > 前向平均 and 后向值 > 后向平均:
                    关键部门.append(行业)
                elif 前向值 > 前向平均:
                    前向主导.append(行业)
                elif 后向值 > 后向平均:
                    后向主导.append(行业)
                else:
                    一般部门.append(行业)
            
            return {
                '关键部门': 关键部门,
                '前向主导': 前向主导,
                '后向主导': 后向主导,
                '一般部门': 一般部门
            }
            
        except Exception as e:
            logger.error(f"识别关键部门时出错: {str(e)}")
            raise
            
    def _模拟政策冲击(self, 政策参数: dict):
        """模拟政策冲击"""
        try:
            if self.完全消耗系数 is None:
                raise ValueError("请先计算完全消耗系数")
                
            # 获取基准产出
            最新年份 = self.行业数据['年份'].max()
            基准产出 = self.行业数据[self.行业数据['年份'] == 最新年份].set_index('行业')['产出']
            
            # 设置政策冲击
            self.基准冲击 = pd.Series(0, index=基准产出.index)
            for 行业, 冲击 in 政策参数.items():
                if 行业 in self.基准冲击.index:
                    self.基准冲击[行业] = 冲击
            
            # 计算政策效应
            self.基准效应 = pd.Series(
                np.dot(self.完全消耗系数, self.基准冲击 * 基准产出),
                index=基准产出.index
            )
            
            logger.info("政策冲击模拟完成")
            
        except Exception as e:
            logger.error(f"模拟政策冲击时出错: {str(e)}")
            raise
            
    def _保存结果(self):
        """保存分析结果"""
        try:
            with pd.ExcelWriter(self.结果目录 / 'io_results.xlsx') as writer:
                # 保存系数
                if self.直接消耗系数 is not None:
                    self.直接消耗系数.to_excel(writer, sheet_name='直接消耗系数')
                if self.完全消耗系数 is not None:
                    self.完全消耗系数.to_excel(writer, sheet_name='完全消耗系数')
                if self.劳动力系数 is not None:
                    self.劳动力系数.to_frame('劳动力系数').to_excel(writer, sheet_name='劳动力系数')
                
                # 保存政策冲击结果
                if self.基准冲击 is not None and self.基准效应 is not None:
                    pd.DataFrame({
                        '政策冲击': self.基准冲击,
                        '政策效应': self.基准效应
                    }).to_excel(writer, sheet_name='政策分析')
            
            logger.info("分析结果已保存到 io_results.xlsx")
            
        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
            raise 