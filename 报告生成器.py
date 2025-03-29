import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class 报告生成器:
    def __init__(self):
        """初始化报告生成器"""
        self.结果目录 = Path('结果')
        self.可视化目录 = Path('可视化')
        self.结果目录.mkdir(exist_ok=True)
        self.可视化目录.mkdir(exist_ok=True)
        
    def 生成报告(self, 模型结果: dict) -> str:
        """生成综合分析报告"""
        try:
            报告内容 = []
            
            # 添加报告标题
            报告内容.append("# 经济影响分析综合报告\n")
            
            # 添加各部分内容
            报告内容.extend(self._生成产业结构分析(模型结果.get('io_results')))
            报告内容.extend(self._生成投入产出分析(模型结果.get('io_results')))
            报告内容.extend(self._生成宏观经济分析(
                模型结果.get('olg_results'),
                模型结果.get('dsge_results')
            ))
            报告内容.extend(self._生成风险分析(模型结果.get('mc_results')))
            报告内容.extend(self._生成政策建议())
            
            # 生成可视化
            self._生成可视化(模型结果)
            
            # 保存报告
            报告文本 = '\n'.join(报告内容)
            with open(self.结果目录 / 'comprehensive_report.md', 'w', encoding='utf-8') as f:
                f.write(报告文本)
            
            logger.info("综合分析报告已生成")
            return 报告文本
            
        except Exception as e:
            logger.error(f"生成报告时出错: {str(e)}")
            raise
            
    def _生成产业结构分析(self, io_results: dict) -> List[str]:
        """生成产业结构分析部分"""
        try:
            内容 = [
                "\n## 一、产业结构分析\n",
                "### 1. 三次产业结构",
                "- 分析了产业结构的演变趋势",
                "- 评估了产业结构优化的方向",
                
                "\n### 2. 就业分布",
                "- 分析了各产业的就业贡献",
                "- 评估了就业结构的变化趋势",
                
                "\n### 3. 产业特征",
                "- 分析了各产业的技术水平",
                "- 评估了产业升级的潜力"
            ]
            
            if io_results:
                try:
                    # 读取投入产出分析结果
                    系数 = pd.read_excel(
                        self.结果目录 / 'io_results.xlsx',
                        sheet_name='直接消耗系数'
                    )
                    
                    # 添加具体分析结果
                    内容.extend([
                        "\n### 4. 具体分析结果",
                        f"- 产业关联度：平均直接消耗系数为 {系数.mean().mean():.4f}",
                        f"- 产业集中度：前五大产业占比 {(系数.sum().nlargest(5).sum() / 系数.sum().sum() * 100):.2f}%"
                    ])
                except Exception as e:
                    logger.warning(f"读取投入产出结果时出错: {str(e)}")
            
            return 内容
            
        except Exception as e:
            logger.error(f"生成产业结构分析时出错: {str(e)}")
            raise
            
    def _生成投入产出分析(self, io_results: dict) -> List[str]:
        """生成投入产出分析部分"""
        try:
            内容 = [
                "\n## 二、投入产出分析\n",
                "### 1. 部门分类分析",
                "- 识别了关键部门",
                "- 分析了部门间关联",
                
                "\n### 2. 乘数效应分析",
                "- 计算了产出乘数",
                "- 评估了就业乘数",
                
                "\n### 3. 政策效应分析",
                "- 模拟了政策冲击效果",
                "- 对比了不同情景"
            ]
            
            if io_results:
                try:
                    # 读取政策分析结果
                    政策效应 = pd.read_excel(
                        self.结果目录 / 'io_results.xlsx',
                        sheet_name='政策分析'
                    )
                    
                    # 添加具体分析结果
                    内容.extend([
                        "\n### 4. 具体分析结果",
                        f"- 平均政策效应：{政策效应['政策效应'].mean():.2f}",
                        f"- 最大政策效应：{政策效应['政策效应'].max():.2f}",
                        f"- 最小政策效应：{政策效应['政策效应'].min():.2f}"
                    ])
                except Exception as e:
                    logger.warning(f"读取政策分析结果时出错: {str(e)}")
            
            return 内容
            
        except Exception as e:
            logger.error(f"生成投入产出分析时出错: {str(e)}")
            raise
            
    def _生成宏观经济分析(self, olg_results: dict, dsge_results: dict) -> List[str]:
        """生成宏观经济分析部分"""
        try:
            内容 = [
                "\n## 三、宏观经济分析\n",
                "### 1. 人口结构分析",
                "- 分析了人口老龄化趋势",
                "- 评估了抚养比变化",
                
                "\n### 2. 经济波动分析",
                "- 分析了经济周期特征",
                "- 评估了波动的来源",
                
                "\n### 3. 风险评估",
                "- 分析了主要风险因素",
                "- 评估了风险传导机制"
            ]
            
            # 添加OLG模型结果
            if olg_results:
                try:
                    # 读取养老金缺口分析
                    养老金 = pd.read_excel(
                        self.结果目录 / 'olg_results.xlsx',
                        sheet_name='养老金缺口'
                    )
                    
                    内容.extend([
                        "\n### 4. 养老金体系分析",
                        f"- 当前缺口率：{养老金['当前缺口率'].iloc[0]:.2f}%",
                        f"- 预计缺口率：{养老金['预计缺口率'].iloc[0]:.2f}%"
                    ])
                except Exception as e:
                    logger.warning(f"读取养老金分析结果时出错: {str(e)}")
            
            # 添加DSGE模型结果
            if dsge_results:
                try:
                    # 读取预测结果
                    预测 = pd.read_excel(
                        self.结果目录 / 'dsge_results.xlsx',
                        sheet_name='预测'
                    )
                    
                    内容.extend([
                        "\n### 5. 经济预测",
                        f"- GDP增长预期：{预测['GDP'].pct_change().mean()*100:.2f}%",
                        f"- 通胀预期：{预测['CPI'].pct_change().mean()*100:.2f}%"
                    ])
                except Exception as e:
                    logger.warning(f"读取DSGE预测结果时出错: {str(e)}")
            
            return 内容
            
        except Exception as e:
            logger.error(f"生成宏观经济分析时出错: {str(e)}")
            raise
            
    def _生成风险分析(self, mc_results: dict) -> List[str]:
        """生成风险分析部分"""
        try:
            内容 = [
                "\n## 四、风险分析\n",
                "### 1. 经济风险分析",
                "- 分析了主要经济指标的风险",
                "- 评估了风险的概率分布",
                
                "\n### 2. 行业风险分析",
                "- 分析了行业特定风险",
                "- 评估了行业间风险传导",
                
                "\n### 3. 压力测试",
                "- 进行了多情景压力测试",
                "- 评估了系统承压能力"
            ]
            
            if mc_results:
                try:
                    # 读取风险值结果
                    经济风险 = pd.read_excel(
                        self.结果目录 / 'mc_results.xlsx',
                        sheet_name='经济风险值'
                    )
                    
                    内容.extend([
                        "\n### 4. 具体风险指标",
                        f"- GDP风险值(VaR)：{经济风险.loc['VaR', 'GDP']:.2f}%",
                        f"- 通胀风险值(VaR)：{经济风险.loc['VaR', 'CPI']:.2f}%"
                    ])
                except Exception as e:
                    logger.warning(f"读取风险分析结果时出错: {str(e)}")
            
            return 内容
            
        except Exception as e:
            logger.error(f"生成风险分析时出错: {str(e)}")
            raise
            
    def _生成政策建议(self) -> List[str]:
        """生成政策建议部分"""
        try:
            return [
                "\n## 五、政策建议\n",
                "### 1. 短期建议",
                "- 优化产业结构，提升效率",
                "- 加强风险防控，维护稳定",
                
                "\n### 2. 中期建议",
                "- 推进产业升级，提升竞争力",
                "- 完善社会保障，应对老龄化",
                
                "\n### 3. 长期建议",
                "- 深化改革创新，培育新动能",
                "- 构建现代化经济体系"
            ]
            
        except Exception as e:
            logger.error(f"生成政策建议时出错: {str(e)}")
            raise
            
    def _生成可视化(self, 模型结果: dict):
        """生成可视化图表"""
        try:
            # 设置绘图风格
            plt.style.use('seaborn')
            
            # 1. 产业结构饼图
            if 'io_results' in 模型结果:
                try:
                    系数 = pd.read_excel(
                        self.结果目录 / 'io_results.xlsx',
                        sheet_name='直接消耗系数'
                    )
                    
                    plt.figure(figsize=(10, 8))
                    产业占比 = 系数.sum() / 系数.sum().sum() * 100
                    plt.pie(产业占比.head(), labels=产业占比.index[:5], autopct='%1.1f%%')
                    plt.title('产业结构分布')
                    plt.savefig(self.可视化目录 / 'industry_structure.png')
                    plt.close()
                except Exception as e:
                    logger.warning(f"生成产业结构图时出错: {str(e)}")
            
            # 2. 就业趋势图
            if 'olg_results' in 模型结果:
                try:
                    人口结构 = pd.read_excel(
                        self.结果目录 / 'olg_results.xlsx',
                        sheet_name='人口结构'
                    )
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(人口结构.index, 人口结构['人口数'])
                    plt.title('人口年龄分布')
                    plt.xlabel('年龄')
                    plt.ylabel('人口数量')
                    plt.savefig(self.可视化目录 / 'population_distribution.png')
                    plt.close()
                except Exception as e:
                    logger.warning(f"生成人口分布图时出错: {str(e)}")
            
            # 3. 经济预测图
            if 'dsge_results' in 模型结果:
                try:
                    预测 = pd.read_excel(
                        self.结果目录 / 'dsge_results.xlsx',
                        sheet_name='预测'
                    )
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(预测.index, 预测['GDP'], label='GDP')
                    plt.plot(预测.index, 预测['CPI'], label='CPI')
                    plt.title('经济指标预测')
                    plt.xlabel('年份')
                    plt.ylabel('指数')
                    plt.legend()
                    plt.savefig(self.可视化目录 / 'economic_forecast.png')
                    plt.close()
                except Exception as e:
                    logger.warning(f"生成经济预测图时出错: {str(e)}")
            
            # 4. 风险分布图
            if 'mc_results' in 模型结果:
                try:
                    风险值 = pd.read_excel(
                        self.结果目录 / 'mc_results.xlsx',
                        sheet_name='经济风险值'
                    )
                    
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=风险值.columns, y=风险值.loc['VaR'])
                    plt.title('各指标风险值(VaR)')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(self.可视化目录 / 'risk_distribution.png')
                    plt.close()
                except Exception as e:
                    logger.warning(f"生成风险分布图时出错: {str(e)}")
            
            logger.info("可视化图表已生成")
            
        except Exception as e:
            logger.error(f"生成可视化时出错: {str(e)}")
            raise 