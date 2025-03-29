import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, save_dir: str = 'visualizations'):
        """初始化可视化器"""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_industry_structure(self, data: pd.DataFrame, year: int):
        """绘制产业结构图"""
        try:
            plt.figure(figsize=(12, 6))
            
            # 获取指定年份的产业占比数据
            year_data = data[data['年份'] == year]
            industry_shares = {
                '第一产业': year_data['第一产业占比'].iloc[0],
                '第二产业': year_data['第二产业占比'].iloc[0],
                '第三产业': year_data['第三产业占比'].iloc[0]
            }
            
            # 绘制饼图
            plt.pie(industry_shares.values(), labels=industry_shares.keys(), autopct='%1.1f%%')
            plt.title(f'{year}年产业结构')
            
            # 保存图片
            plt.savefig(self.save_dir / f'industry_structure_{year}.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制产业结构图时出错: {str(e)}")
            raise
    
    def plot_employment_trends(self, data: pd.DataFrame):
        """绘制就业趋势图"""
        try:
            plt.figure(figsize=(12, 6))
            
            # 绘制各产业就业人数趋势
            plt.plot(data['年份'], data['第一产业就业人数'], label='第一产业')
            plt.plot(data['年份'], data['第二产业就业人数'], label='第二产业')
            plt.plot(data['年份'], data['第三产业就业人数'], label='第三产业')
            
            plt.title('各产业就业人数变化趋势')
            plt.xlabel('年份')
            plt.ylabel('就业人数')
            plt.legend()
            plt.grid(True)
            
            # 保存图片
            plt.savefig(self.save_dir / 'employment_trends.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制就业趋势图时出错: {str(e)}")
            raise
    
    def plot_sector_classification(self, sector_data: pd.DataFrame):
        """绘制部门分类散点图"""
        try:
            plt.figure(figsize=(10, 8))
            
            # 为不同类型的部门设置不同的颜色
            colors = {
                '关键部门': 'red',
                '最终需求导向型': 'blue',
                '中间投入导向型': 'green',
                '一般部门': 'gray'
            }
            
            # 绘制散点图
            for sector_type in colors:
                mask = sector_data['部门类型'] == sector_type
                plt.scatter(
                    sector_data.loc[mask, '后向关联'],
                    sector_data.loc[mask, '前向关联'],
                    c=colors[sector_type],
                    label=sector_type,
                    alpha=0.6
                )
            
            plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
            plt.axvline(x=1, color='k', linestyle='--', alpha=0.3)
            
            plt.title('部门分类散点图')
            plt.xlabel('后向关联')
            plt.ylabel('前向关联')
            plt.legend()
            plt.grid(True)
            
            # 保存图片
            plt.savefig(self.save_dir / 'sector_classification.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制部门分类散点图时出错: {str(e)}")
            raise
    
    def plot_policy_effects(self, results: dict):
        """绘制政策效应对比图"""
        try:
            plt.figure(figsize=(15, 6))
            
            # 准备数据
            industries = results['baseline_effect'].index
            x = np.arange(len(industries))
            width = 0.35
            
            # 绘制柱状图
            plt.bar(x - width/2, results['baseline_effect'], width, label='基准情景')
            plt.bar(x + width/2, results['targeted_effect'], width, label='目标情景')
            
            plt.title('政策效应对比')
            plt.xlabel('行业')
            plt.ylabel('效应大小')
            plt.xticks(x, industries, rotation=45, ha='right')
            plt.legend()
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(self.save_dir / 'policy_effects.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制政策效应对比图时出错: {str(e)}")
            raise
    
    def plot_multiplier_effects(self, output_multipliers: pd.Series, income_multipliers: pd.Series):
        """绘制乘数效应图"""
        try:
            plt.figure(figsize=(12, 6))
            
            # 准备数据
            industries = output_multipliers.index
            x = np.arange(len(industries))
            width = 0.35
            
            # 绘制柱状图
            plt.bar(x - width/2, output_multipliers, width, label='产出乘数')
            plt.bar(x + width/2, income_multipliers, width, label='收入乘数')
            
            plt.title('行业乘数效应')
            plt.xlabel('行业')
            plt.ylabel('乘数大小')
            plt.xticks(x, industries, rotation=45, ha='right')
            plt.legend()
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(self.save_dir / 'multiplier_effects.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制乘数效应图时出错: {str(e)}")
            raise
    
    def create_all_visualizations(self, industry_data: pd.DataFrame, io_results: dict):
        """创建所有可视化图表"""
        try:
            # 绘制产业结构图
            latest_year = industry_data['年份'].max()
            self.plot_industry_structure(industry_data, latest_year)
            
            # 绘制就业趋势图
            self.plot_employment_trends(industry_data)
            
            # 绘制部门分类散点图
            self.plot_sector_classification(io_results['sector_classification'])
            
            # 绘制政策效应对比图
            self.plot_policy_effects(io_results)
            
            # 绘制乘数效应图
            self.plot_multiplier_effects(
                io_results['output_multipliers'],
                io_results['income_multipliers']
            )
            
            logger.info("所有可视化图表已生成")
            
        except Exception as e:
            logger.error(f"创建可视化图表时出错: {str(e)}")
            raise 