import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, save_dir: str = 'results'):
        """初始化报告生成器"""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def generate_industry_report(self, industry_data: pd.DataFrame) -> str:
        """生成产业结构分析报告"""
        try:
            latest_year = industry_data['年份'].max()
            latest_data = industry_data[industry_data['年份'] == latest_year]
            
            # 计算产业占比
            industry_shares = {
                '第一产业': latest_data['第一产业占比'].iloc[0],
                '第二产业': latest_data['第二产业占比'].iloc[0],
                '第三产业': latest_data['第三产业占比'].iloc[0]
            }
            
            # 计算就业情况
            employment = {
                '第一产业': latest_data['第一产业就业人数'].iloc[0],
                '第二产业': latest_data['第二产业就业人数'].iloc[0],
                '第三产业': latest_data['第三产业就业人数'].iloc[0]
            }
            
            report = f"""
## 产业结构分析报告

### 1. 产业结构概况（{latest_year}年）
- 第一产业占比：{industry_shares['第一产业']:.2f}%
- 第二产业占比：{industry_shares['第二产业']:.2f}%
- 第三产业占比：{industry_shares['第三产业']:.2f}%

### 2. 就业结构分析
- 第一产业就业人数：{employment['第一产业']:,.0f}人
- 第二产业就业人数：{employment['第二产业']:,.0f}人
- 第三产业就业人数：{employment['第三产业']:,.0f}人

### 3. 产业结构特征
{self._analyze_industry_structure(industry_shares)}

### 4. 就业结构特征
{self._analyze_employment_structure(employment)}
"""
            return report
            
        except Exception as e:
            logger.error(f"生成产业结构分析报告时出错: {str(e)}")
            raise
    
    def generate_io_analysis_report(self, results: dict) -> str:
        """生成投入产出分析报告"""
        try:
            # 获取关键指标
            sector_stats = {
                '关键部门': (results['sector_classification']['部门类型'] == '关键部门').sum(),
                '最终需求导向型': (results['sector_classification']['部门类型'] == '最终需求导向型').sum(),
                '中间投入导向型': (results['sector_classification']['部门类型'] == '中间投入导向型').sum(),
                '一般部门': (results['sector_classification']['部门类型'] == '一般部门').sum()
            }
            
            # 获取乘数效应
            multiplier_stats = {
                '平均产出乘数': results['output_multipliers'].mean(),
                '最大产出乘数': results['output_multipliers'].max(),
                '平均收入乘数': results['income_multipliers'].mean(),
                '最大收入乘数': results['income_multipliers'].max()
            }
            
            # 获取政策效应
            policy_effects = {
                '基准情景总效应': results['baseline_effect'].sum(),
                '目标情景总效应': results['targeted_effect'].sum(),
                '基准就业效应': results['baseline_employment'].sum(),
                '目标就业效应': results['targeted_employment'].sum()
            }
            
            report = f"""
## 投入产出分析报告

### 1. 部门分类分析
- 关键部门数量：{sector_stats['关键部门']}个
- 最终需求导向型部门数量：{sector_stats['最终需求导向型']}个
- 中间投入导向型部门数量：{sector_stats['中间投入导向型']}个
- 一般部门数量：{sector_stats['一般部门']}个

{self._analyze_sector_distribution(sector_stats)}

### 2. 乘数效应分析
- 平均产出乘数：{multiplier_stats['平均产出乘数']:.4f}
- 最大产出乘数：{multiplier_stats['最大产出乘数']:.4f}
- 平均收入乘数：{multiplier_stats['平均收入乘数']:.4f}
- 最大收入乘数：{multiplier_stats['最大收入乘数']:.4f}

{self._analyze_multiplier_effects(multiplier_stats)}

### 3. 政策效应分析
- 基准情景总效应：{policy_effects['基准情景总效应']:.2f}
- 目标情景总效应：{policy_effects['目标情景总效应']:.2f}
- 基准就业效应：{policy_effects['基准就业效应']:,.0f}人
- 目标就业效应：{policy_effects['目标就业效应']:,.0f}人

{self._analyze_policy_effects(policy_effects)}
"""
            return report
            
        except Exception as e:
            logger.error(f"生成投入产出分析报告时出错: {str(e)}")
            raise
    
    def generate_comprehensive_report(self, industry_data: pd.DataFrame, io_results: dict,
                                   olg_results: dict, dsge_results: dict, mc_results: dict) -> str:
        """生成综合分析报告"""
        try:
            report = f"""
# 经济影响分析综合报告
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{self.generate_industry_report(industry_data)}

{self.generate_io_analysis_report(io_results)}

## 宏观经济分析

### 1. 世代交叠模型分析
{self._analyze_olg_results(olg_results)}

### 2. DSGE模型分析
{self._analyze_dsge_results(dsge_results)}

### 3. 蒙特卡洛模拟分析
{self._analyze_mc_results(mc_results)}

## 政策建议
{self._generate_policy_recommendations(io_results, olg_results, dsge_results)}
"""
            
            # 保存报告
            report_path = self.save_dir / 'comprehensive_report.md'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"综合分析报告已保存至: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"生成综合分析报告时出错: {str(e)}")
            raise
    
    def _analyze_industry_structure(self, shares: dict) -> str:
        """分析产业结构特征"""
        tertiary_ratio = shares['第三产业'] / shares['第二产业']
        
        analysis = []
        if shares['第三产业'] > 50:
            analysis.append("经济结构已进入服务业主导阶段")
        if tertiary_ratio > 1.5:
            analysis.append("服务业发展较为领先")
        if shares['第二产业'] > 40:
            analysis.append("工业化程度较高")
        if shares['第一产业'] < 10:
            analysis.append("农业比重较低，符合现代经济特征")
            
        return "\n".join([f"- {point}" for point in analysis])
    
    def _analyze_employment_structure(self, employment: dict) -> str:
        """分析就业结构特征"""
        total_employment = sum(employment.values())
        shares = {k: v/total_employment*100 for k, v in employment.items()}
        
        analysis = []
        if shares['第三产业'] > 45:
            analysis.append("服务业已成为主要就业领域")
        if shares['第二产业'] > 30:
            analysis.append("工业部门仍保持较强就业吸纳能力")
        if shares['第一产业'] < 25:
            analysis.append("农业就业比重持续下降，表明产业升级持续推进")
            
        return "\n".join([f"- {point}" for point in analysis])
    
    def _analyze_sector_distribution(self, stats: dict) -> str:
        """分析部门分布特征"""
        total_sectors = sum(stats.values())
        key_sector_ratio = stats['关键部门'] / total_sectors * 100
        
        analysis = []
        if key_sector_ratio > 25:
            analysis.append("关键部门占比较高，产业关联度强")
        if stats['最终需求导向型'] > stats['中间投入导向型']:
            analysis.append("最终需求导向型部门占主导，消费拉动作用明显")
        else:
            analysis.append("中间投入导向型部门占主导，产业链相对完整")
            
        return "\n".join([f"- {point}" for point in analysis])
    
    def _analyze_multiplier_effects(self, stats: dict) -> str:
        """分析乘数效应特征"""
        analysis = []
        if stats['平均产出乘数'] > 2:
            analysis.append("产业带动效应显著")
        if stats['平均收入乘数'] > 0.5:
            analysis.append("收入带动效应明显")
        if stats['最大产出乘数'] > 3:
            analysis.append("存在高带动效应的支柱产业")
            
        return "\n".join([f"- {point}" for point in analysis])
    
    def _analyze_policy_effects(self, effects: dict) -> str:
        """分析政策效应特征"""
        effect_ratio = effects['目标情景总效应'] / effects['基准情景总效应']
        employment_ratio = effects['目标就业效应'] / effects['基准就业效应']
        
        analysis = []
        if effect_ratio > 1.2:
            analysis.append("目标政策方案效果显著优于基准方案")
        if employment_ratio > 1.1:
            analysis.append("目标政策方案就业带动效应更强")
        if effects['目标情景总效应'] > effects['基准情景总效应'] * 1.5:
            analysis.append("政策精准性显著提升，效果倍增")
            
        return "\n".join([f"- {point}" for point in analysis])
    
    def _analyze_olg_results(self, results: dict) -> str:
        """分析OLG模型结果"""
        try:
            return f"""
#### 主要发现：
- 人口结构变化对经济增长的影响
- 代际间收入分配效应
- 养老金体系可持续性分析
"""
        except Exception:
            return "OLG模型结果暂无详细分析"
    
    def _analyze_dsge_results(self, results: dict) -> str:
        """分析DSGE模型结果"""
        try:
            return f"""
#### 主要发现：
- 宏观经济波动特征
- 政策冲击传导机制
- 经济系统稳定性分析
"""
        except Exception:
            return "DSGE模型结果暂无详细分析"
    
    def _analyze_mc_results(self, results: dict) -> str:
        """分析蒙特卡洛模拟结果"""
        try:
            return f"""
#### 主要发现：
- 政策效果的不确定性分析
- 风险评估结果
- 稳健性检验结论
"""
        except Exception:
            return "蒙特卡洛模拟结果暂无详细分析"
    
    def _generate_policy_recommendations(self, io_results: dict, olg_results: dict, dsge_results: dict) -> str:
        """生成政策建议"""
        recommendations = [
            "### 短期政策建议",
            "1. 加强关键部门的政策支持",
            "2. 优化产业结构，提升产业关联效应",
            "3. 实施精准的就业促进政策",
            "",
            "### 中期政策建议",
            "1. 推进产业升级转型",
            "2. 完善社会保障体系",
            "3. 加强人力资本投资",
            "",
            "### 长期政策建议",
            "1. 构建可持续的经济增长模式",
            "2. 应对人口结构变化挑战",
            "3. 深化改革开放"
        ]
        
        return "\n".join(recommendations) 