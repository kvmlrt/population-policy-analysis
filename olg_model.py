import numpy as np
import pandas as pd
from typing import Dict, Any
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OLGModel:
    def __init__(self, params: Dict[str, Any]):
        """
        初始化OLG模型
        
        Args:
            params: 模型参数字典
        """
        self.params = params
        self.population_data = None
        self.economic_data = None
        self.pension_data = None
        
        # 验证参数
        self._validate_params()
        
        # 初始化模型参数
        self.n_generations = params['n_generations']
        self.time_periods = params['time_periods']
        self.discount_rate = params['discount_rate']
        self.retirement_age = params['retirement_age']
        self.max_age = params['max_age']
        self.social_security_tax_rate = params['social_security_tax_rate']
        self.capital_share = params['capital_share']
        self.depreciation_rate = params['depreciation_rate']
        
        # 初始化状态变量
        self.capital_stock = None
        self.labor_supply = None
        self.consumption = None
        self.savings = None
        self.pension_benefits = None
        
    def _validate_params(self):
        """验证模型参数"""
        required_params = [
            'n_generations',
            'time_periods',
            'discount_rate',
            'retirement_age',
            'max_age',
            'social_security_tax_rate',
            'capital_share',
            'depreciation_rate'
        ]
        
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"缺少必要参数: {param}")
    
    def set_population_data(self, data: pd.DataFrame):
        """设置人口数据"""
        self.population_data = data
        logger.info("人口数据已设置")
    
    def set_economic_data(self, data: pd.DataFrame):
        """设置经济数据"""
        self.economic_data = data
        logger.info("经济数据已设置")
    
    def set_pension_data(self, data: pd.DataFrame):
        """设置养老金数据"""
        self.pension_data = data
        logger.info("养老金数据已设置")
    
    def _initialize_state_variables(self):
        """初始化状态变量"""
        self.capital_stock = np.zeros(self.time_periods)
        self.labor_supply = np.zeros((self.n_generations, self.time_periods))
        self.consumption = np.zeros((self.n_generations, self.time_periods))
        self.savings = np.zeros((self.n_generations, self.time_periods))
        self.pension_benefits = np.zeros(self.time_periods)
    
    def _calculate_production(self, t: int) -> float:
        """计算生产函数"""
        return (self.capital_stock[t] ** self.capital_share) * \
               (np.sum(self.labor_supply[:, t]) ** (1 - self.capital_share))
    
    def _calculate_wage_rate(self, t: int) -> float:
        """计算工资率"""
        return (1 - self.capital_share) * self._calculate_production(t) / \
               np.sum(self.labor_supply[:, t])
    
    def _calculate_interest_rate(self, t: int) -> float:
        """计算利率"""
        return self.capital_share * self._calculate_production(t) / \
               self.capital_stock[t] - self.depreciation_rate
    
    def _calculate_pension_benefits(self, t: int):
        """计算养老金收益"""
        working_population = np.sum(self.labor_supply[:-1, t])
        retired_population = self.labor_supply[-1, t]
        total_wage = working_population * self._calculate_wage_rate(t)
        
        self.pension_benefits[t] = self.social_security_tax_rate * total_wage / retired_population
    
    def _calculate_utility(self, c: float) -> float:
        """计算效用函数"""
        if c <= 0:
            return float('-inf')
        return np.log(c)
    
    def run_simulation(self) -> Dict[str, pd.DataFrame]:
        """运行模型模拟"""
        try:
            # 检查数据是否已设置
            if any(data is None for data in [self.population_data, self.economic_data, self.pension_data]):
                raise ValueError("请先设置所有必要的数据")
            
            # 初始化状态变量
            self._initialize_state_variables()
            
            # 设置初始值
            self.capital_stock[0] = self.economic_data['资本存量'].iloc[0]
            
            # 主循环
            for t in range(self.time_periods - 1):
                # 计算各代人的劳动供给
                for g in range(self.n_generations):
                    if g < self.n_generations - 1:  # 工作人口
                        self.labor_supply[g, t] = 1.0
                    else:  # 退休人口
                        self.labor_supply[g, t] = 0.0
                
                # 计算工资和利率
                wage_rate = self._calculate_wage_rate(t)
                interest_rate = self._calculate_interest_rate(t)
                
                # 计算养老金
                self._calculate_pension_benefits(t)
                
                # 计算各代人的消费和储蓄
                for g in range(self.n_generations):
                    if g < self.n_generations - 1:  # 工作人口
                        income = wage_rate * (1 - self.social_security_tax_rate)
                        self.consumption[g, t] = income * 0.8  # 假设消费80%的收入
                        self.savings[g, t] = income * 0.2  # 储蓄20%的收入
                    else:  # 退休人口
                        self.consumption[g, t] = self.pension_benefits[t]
                        self.savings[g, t] = 0
                
                # 更新资本存量
                self.capital_stock[t + 1] = np.sum(self.savings[:, t]) * (1 + interest_rate)
            
            # 整理结果
            results = {
                'capital_stock': pd.DataFrame({
                    '时期': range(self.time_periods),
                    '资本存量': self.capital_stock
                }),
                'labor_supply': pd.DataFrame(self.labor_supply.T, columns=[f'代际{i+1}' for i in range(self.n_generations)]),
                'consumption': pd.DataFrame(self.consumption.T, columns=[f'代际{i+1}' for i in range(self.n_generations)]),
                'savings': pd.DataFrame(self.savings.T, columns=[f'代际{i+1}' for i in range(self.n_generations)]),
                'pension_benefits': pd.DataFrame({
                    '时期': range(self.time_periods),
                    '养老金收益': self.pension_benefits
                })
            }
            
            logger.info("模型模拟完成")
            return results
            
        except Exception as e:
            logger.error(f"模型模拟过程中出错: {str(e)}")
            raise
    
    def save_results(self, filepath: str):
        """保存模型结果"""
        try:
            results = self.run_simulation()
            
            # 使用ExcelWriter保存多个数据框到不同的工作表
            with pd.ExcelWriter(filepath) as writer:
                for sheet_name, df in results.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"结果已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
            raise

def main():
    """主函数"""
    try:
        # 设置模型参数
        params = {
            'n_generations': 3,
            'time_periods': 30,
            'discount_rate': 0.05,
            'retirement_age': 65,
            'max_age': 100,
            'social_security_tax_rate': 0.2,
            'capital_share': 0.3,
            'depreciation_rate': 0.05
        }
        
        # 初始化模型
        model = OLGModel(params)
        
        # 读取数据
        data_dir = Path('data')
        population_data = pd.read_excel(data_dir / 'population_processed.xlsx')
        economic_data = pd.read_excel(data_dir / 'economic_processed.xlsx')
        
        # 设置参数
        model.set_population_data(population_data)
        model.set_economic_data(economic_data)
        
        # 运行模拟
        results = model.run_simulation()
        
        # 保存结果
        model.save_results('results.xlsx')
        
        logger.info("OLG模型模拟完成")
        
    except Exception as e:
        logger.error(f"OLG模型模拟过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 