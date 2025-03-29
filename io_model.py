import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from .visualization import Visualizer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IOModel:
    def __init__(self, params: Dict[str, Any]):
        """
        初始化投入产出模型
        params: 模型参数字典
        """
        self.params = params
        self.industry_data = None
        self.economic_data = None
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        self.visualizer = Visualizer()
        
        # 初始化结果变量
        self.direct_coefficients = None
        self.complete_coefficients = None
        self.labor_coefficients = None
        self.output_multipliers = None
        self.employment_multipliers = None
        
        # 验证参数
        self._validate_params()
        
    def _validate_params(self):
        """验证模型参数"""
        required_params = ['simulation_years']
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"缺少必要参数: {param}")
    
    def set_industry_data(self, data: pd.DataFrame):
        """设置行业数据"""
        try:
            # 确保数据为数值类型
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in data.columns:
                if col not in numeric_cols and col != '年份':
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            self.industry_data = data
            logger.info("行业数据已设置")
        except Exception as e:
            logger.error(f"设置行业数据时出错: {str(e)}")
            raise
    
    def set_economic_data(self, data: pd.DataFrame):
        """设置经济数据"""
        try:
            self.economic_data = data
            logger.info("经济数据已设置")
        except Exception as e:
            logger.error(f"设置经济数据时出错: {str(e)}")
            raise
    
    def _calculate_direct_coefficients(self) -> pd.DataFrame:
        """计算直接消耗系数"""
        try:
            if self.industry_data is None:
                raise ValueError("未设置行业数据")
            
            # 获取最新年份的数据
            latest_year = self.industry_data['年份'].max()
            io_table = self.industry_data[self.industry_data['年份'] == latest_year].copy()
            
            # 删除年份列和总就业人数列
            exclude_cols = ['年份', '总就业人数']
            numeric_cols = [col for col in io_table.columns if col not in exclude_cols]
            io_table = io_table[numeric_cols]
            
            # 确保数据不为空
            if io_table.empty:
                raise ValueError("行业数据为空")
            
            # 计算总投入
            total_input = io_table.iloc[0]
            
            # 避免除以零
            total_input = total_input.replace(0, np.nan)
            
            # 创建方阵
            n = len(numeric_cols)
            direct_coef = pd.DataFrame(
                np.zeros((n, n)),
                index=numeric_cols,
                columns=numeric_cols
            )
            
            # 填充直接消耗系数
            for i in range(n):
                for j in range(n):
                    if total_input[j] != 0:
                        direct_coef.iloc[i, j] = io_table.iloc[0, i] / total_input[j]
            
            # 填充缺失值
            direct_coef = direct_coef.fillna(0)
            
            return direct_coef
            
        except Exception as e:
            logger.error(f"计算直接消耗系数时出错: {str(e)}")
            raise
    
    def _calculate_complete_coefficients(self, direct_coef: pd.DataFrame) -> pd.DataFrame:
        """计算完全消耗系数"""
        try:
            # 确保输入数据不为空
            if direct_coef.empty:
                raise ValueError("直接消耗系数矩阵为空")
            
            # 创建单位矩阵
            n = len(direct_coef)
            identity = pd.DataFrame(
                np.eye(n),
                index=direct_coef.index,
                columns=direct_coef.columns
            )
            
            # 计算列昂惕夫逆矩阵
            try:
                # 使用泰勒级数展开近似计算
                complete_coef = identity.copy()
                power_matrix = direct_coef.copy()
                for _ in range(5):  # 使用5阶近似
                    complete_coef = complete_coef + power_matrix
                    power_matrix = power_matrix.dot(direct_coef)
            except Exception as e:
                logger.error(f"计算完全消耗系数时出错: {str(e)}")
                raise
            
            return complete_coef
            
        except Exception as e:
            logger.error(f"计算完全消耗系数时出错: {str(e)}")
            raise
    
    def _calculate_labor_coefficients(self) -> pd.Series:
        """计算劳动力系数"""
        try:
            if self.industry_data is None:
                raise ValueError("未设置行业数据")
            
            # 获取最新年份的数据
            latest_year = self.industry_data['年份'].max()
            latest_data = self.industry_data[self.industry_data['年份'] == latest_year].copy()
            
            # 获取行业列（排除年份和总就业人数列）
            industry_cols = [col for col in latest_data.columns if col not in ['年份', '总就业人数']]
            
            # 计算每个行业的就业人数占比
            total_employment = latest_data[industry_cols].iloc[0].sum()
            labor_coef = latest_data[industry_cols].iloc[0] / total_employment
            
            return labor_coef
            
        except Exception as e:
            logger.error(f"计算劳动力系数时出错: {str(e)}")
            raise
    
    def _calculate_multipliers(self) -> Tuple[pd.Series, pd.Series]:
        """计算乘数效应"""
        try:
            if self.complete_coefficients is None:
                raise ValueError("未计算完全消耗系数")
            
            # 计算产出乘数（列和）
            output_multipliers = self.complete_coefficients.sum()
            
            # 计算收入乘数
            if self.labor_coefficients is not None:
                income_multipliers = self.labor_coefficients.dot(self.complete_coefficients)
            else:
                income_multipliers = pd.Series(0, index=self.complete_coefficients.columns)
            
            return output_multipliers, income_multipliers
            
        except Exception as e:
            logger.error(f"计算乘数效应时出错: {str(e)}")
            raise
    
    def _calculate_linkage_effects(self) -> Tuple[pd.Series, pd.Series]:
        """计算前向和后向关联效应"""
        try:
            if self.complete_coefficients is None:
                raise ValueError("未计算完全消耗系数")
            
            # 计算后向关联效应（列和）
            backward_linkage = self.complete_coefficients.sum()
            backward_linkage = backward_linkage / backward_linkage.mean()
            
            # 计算前向关联效应（行和）
            forward_linkage = self.complete_coefficients.sum(axis=1)
            forward_linkage = forward_linkage / forward_linkage.mean()
            
            return backward_linkage, forward_linkage
            
        except Exception as e:
            logger.error(f"计算关联效应时出错: {str(e)}")
            raise
    
    def _identify_key_sectors(self) -> pd.DataFrame:
        """识别关键部门"""
        try:
            backward_linkage, forward_linkage = self._calculate_linkage_effects()
            
            # 创建部门分类
            sectors = pd.DataFrame({
                '后向关联': backward_linkage,
                '前向关联': forward_linkage
            })
            
            # 添加部门类型
            sectors['部门类型'] = '一般部门'
            sectors.loc[(sectors['后向关联'] > 1) & (sectors['前向关联'] > 1), '部门类型'] = '关键部门'
            sectors.loc[(sectors['后向关联'] > 1) & (sectors['前向关联'] <= 1), '部门类型'] = '最终需求导向型'
            sectors.loc[(sectors['后向关联'] <= 1) & (sectors['前向关联'] > 1), '部门类型'] = '中间投入导向型'
            
            return sectors
            
        except Exception as e:
            logger.error(f"识别关键部门时出错: {str(e)}")
            raise
    
    def _simulate_policy_shock(self, complete_coef: pd.DataFrame, shock_scenario: str = 'baseline') -> Tuple[pd.Series, pd.Series]:
        """模拟政策冲击"""
        try:
            # 获取最新年份的数据
            latest_year = self.industry_data['年份'].max()
            latest_data = self.industry_data[self.industry_data['年份'] == latest_year]
            
            # 获取行业列
            industry_cols = [col for col in latest_data.columns if col not in ['年份', '总就业人数']]
            baseline_output = latest_data[industry_cols].iloc[0]
            
            # 根据不同情景设置冲击
            if shock_scenario == 'baseline':
                # 基准情景：所有部门产出增长5%
                shock = baseline_output * self.params.get('shock_size', 0.05)
            else:
                # 目标情景：重点部门产出增长10%，其他部门增长3%
                shock = baseline_output.copy()
                key_sectors = ['制造业', '信息技术服务业', '科学研究和技术服务业']
                for sector in shock.index:
                    if any(key in sector for key in key_sectors):
                        shock[sector] *= self.params.get('targeted_shock_size', 0.10)
                    else:
                        shock[sector] *= self.params.get('shock_size', 0.03)
            
            # 计算总效应
            total_effect = complete_coef.dot(shock)
            
            return shock, total_effect
            
        except Exception as e:
            logger.error(f"模拟政策冲击时出错: {str(e)}")
            raise
    
    def run_analysis(self) -> Dict[str, Any]:
        """运行模型分析"""
        try:
            if self.industry_data is None:
                raise ValueError("未设置行业数据")
            
            # 计算各类系数
            self.direct_coefficients = self._calculate_direct_coefficients()
            self.complete_coefficients = self._calculate_complete_coefficients(self.direct_coefficients)
            self.labor_coefficients = self._calculate_labor_coefficients()
            
            # 计算乘数效应
            self.output_multipliers, self.income_multipliers = self._calculate_multipliers()
            
            # 识别关键部门
            self.sector_classification = self._identify_key_sectors()
            
            # 模拟政策冲击
            self.baseline_shock, self.baseline_effect = self._simulate_policy_shock(self.complete_coefficients, 'baseline')
            self.targeted_shock, self.targeted_effect = self._simulate_policy_shock(self.complete_coefficients, 'targeted')
            
            # 计算就业效应
            self.baseline_employment = self.labor_coefficients * self.baseline_effect
            self.targeted_employment = self.labor_coefficients * self.targeted_effect
            
            # 整理结果
            results = {
                'direct_coefficients': self.direct_coefficients,
                'complete_coefficients': self.complete_coefficients,
                'labor_coefficients': self.labor_coefficients,
                'output_multipliers': self.output_multipliers,
                'income_multipliers': self.income_multipliers,
                'sector_classification': self.sector_classification,
                'baseline_shock': self.baseline_shock,
                'baseline_effect': self.baseline_effect,
                'baseline_employment': self.baseline_employment,
                'targeted_shock': self.targeted_shock,
                'targeted_effect': self.targeted_effect,
                'targeted_employment': self.targeted_employment
            }
            
            # 保存结果
            self._save_results(results)
            logger.info("模型模拟完成")
            
            return results
            
        except Exception as e:
            logger.error(f"模型分析过程中出错: {str(e)}")
            raise
    
    def _save_results(self, results: Dict[str, Any]):
        """保存分析结果并生成可视化"""
        try:
            # 创建Excel写入器
            with pd.ExcelWriter(self.results_dir / 'io_results.xlsx') as writer:
                # 保存各类系数
                results['direct_coefficients'].to_excel(writer, sheet_name='直接消耗系数')
                results['complete_coefficients'].to_excel(writer, sheet_name='完全消耗系数')
                results['labor_coefficients'].to_frame('劳动力系数').to_excel(writer, sheet_name='劳动力系数')
                
                # 保存乘数效应
                pd.DataFrame({
                    '产出乘数': results['output_multipliers'],
                    '收入乘数': results['income_multipliers']
                }).to_excel(writer, sheet_name='乘数效应')
                
                # 保存部门分类
                results['sector_classification'].to_excel(writer, sheet_name='部门分类')
                
                # 保存政策冲击结果
                pd.DataFrame({
                    '基准冲击': results['baseline_shock'],
                    '基准效应': results['baseline_effect'],
                    '基准就业效应': results['baseline_employment'],
                    '目标冲击': results['targeted_shock'],
                    '目标效应': results['targeted_effect'],
                    '目标就业效应': results['targeted_employment']
                }).to_excel(writer, sheet_name='政策冲击结果')
                
                # 保存结果汇总
                summary = pd.DataFrame({
                    '指标': [
                        '总产出乘数',
                        '平均产出乘数',
                        '最大产出乘数',
                        '总收入乘数',
                        '平均收入乘数',
                        '最大收入乘数',
                        '关键部门数量',
                        '最终需求导向型部门数量',
                        '中间投入导向型部门数量',
                        '一般部门数量',
                        '基准情景总效应',
                        '目标情景总效应',
                        '基准情景就业效应',
                        '目标情景就业效应'
                    ],
                    '值': [
                        results['output_multipliers'].sum(),
                        results['output_multipliers'].mean(),
                        results['output_multipliers'].max(),
                        results['income_multipliers'].sum(),
                        results['income_multipliers'].mean(),
                        results['income_multipliers'].max(),
                        (results['sector_classification']['部门类型'] == '关键部门').sum(),
                        (results['sector_classification']['部门类型'] == '最终需求导向型').sum(),
                        (results['sector_classification']['部门类型'] == '中间投入导向型').sum(),
                        (results['sector_classification']['部门类型'] == '一般部门').sum(),
                        results['baseline_effect'].sum(),
                        results['targeted_effect'].sum(),
                        results['baseline_employment'].sum(),
                        results['targeted_employment'].sum()
                    ]
                })
                summary.to_excel(writer, sheet_name='结果汇总', index=False)
            
            # 生成可视化
            if self.industry_data is not None:
                self.visualizer.create_all_visualizations(self.industry_data, results)
            
            logger.info("结果和可视化已保存")
            
        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
            raise

    def save_results(self, file_path: str):
        """保存分析结果"""
        try:
            # 创建Excel写入器
            with pd.ExcelWriter(file_path) as writer:
                # 保存各类系数
                if self.direct_coefficients is not None:
                    self.direct_coefficients.to_excel(writer, sheet_name='直接消耗系数')
                if self.complete_coefficients is not None:
                    self.complete_coefficients.to_excel(writer, sheet_name='完全消耗系数')
                if self.labor_coefficients is not None:
                    self.labor_coefficients.to_frame('劳动力系数').to_excel(writer, sheet_name='劳动力系数')
                
                # 保存政策冲击结果
                if hasattr(self, 'baseline_shock') and hasattr(self, 'baseline_effect'):
                    pd.DataFrame({
                        '基准冲击': self.baseline_shock,
                        '基准效应': self.baseline_effect,
                        '基准就业效应': self.baseline_employment,
                        '目标冲击': self.targeted_shock,
                        '目标效应': self.targeted_effect,
                        '目标就业效应': self.targeted_employment
                    }).to_excel(writer, sheet_name='政策冲击结果')
            
            logger.info(f"结果已保存到: {file_path}")
            
        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
            raise

def main():
    """主函数"""
    try:
        # 设置模型参数
        params = {
            'n_sectors': 42,
            'base_year': 2023,
            'price_elasticity': 0.5,
            'demand_elasticity': 0.75,
            # TODO: 添加其他必要参数
        }
        
        # 初始化模型
        model = IOModel(params)
        
        # 读取数据
        data_dir = Path('data')
        io_data = pd.read_excel(data_dir / 'io_table_processed.xlsx')
        labor_data = pd.read_excel(data_dir / 'labor_processed.xlsx')
        
        # 加载数据
        model.set_industry_data(io_data)
        model.set_economic_data(labor_data)
        
        # 计算系数
        results = model.run_analysis()
        
        # 保存结果
        model.save_results('results.xlsx')
        
        logger.info("投入产出模型模拟完成")
        
    except Exception as e:
        logger.error(f"投入产出模型模拟过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 