import numpy as np
import pandas as pd
from typing import Dict, Any
import logging
from pathlib import Path
from scipy import stats

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonteCarloModel:
    def __init__(self, params: Dict[str, Any]):
        """
        初始化蒙特卡洛模拟模型
        
        Args:
            params: 模型参数字典
        """
        self.params = params
        self.economic_data = None
        self.population_data = None
        
        # 验证参数
        self._validate_params()
        
        # 初始化模型参数
        self.n_simulations = params['n_simulations']  # 模拟次数
        self.time_periods = params['time_periods']  # 时间周期
        self.confidence_level = params['confidence_level']  # 置信水平
        np.random.seed(params['seed'])  # 设置随机数种子
        
        # 初始化结果变量
        self.gdp_growth = None
        self.fiscal_deficit = None
        self.employment_rate = None
        
    def _validate_params(self):
        """验证模型参数"""
        required_params = [
            'n_simulations',
            'time_periods',
            'confidence_level',
            'seed'
        ]
        
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"缺少必要参数: {param}")
    
    def set_economic_data(self, data: pd.DataFrame):
        """设置经济数据"""
        self.economic_data = data
        logger.info("经济数据已设置")
    
    def set_population_data(self, data: pd.DataFrame):
        """设置人口数据"""
        self.population_data = data
        logger.info("人口数据已设置")
    
    def _initialize_result_variables(self):
        """初始化结果变量"""
        self.gdp_growth = np.zeros((self.n_simulations, self.time_periods))
        self.fiscal_deficit = np.zeros((self.n_simulations, self.time_periods))
        self.employment_rate = np.zeros((self.n_simulations, self.time_periods))
    
    def _generate_gdp_growth(self):
        """生成GDP增长率"""
        # 从历史数据估计参数
        historical_growth = self.economic_data['gdp增长率'].values
        mu = np.mean(historical_growth)
        sigma = np.std(historical_growth)
        
        # 生成随机增长率
        for i in range(self.n_simulations):
            self.gdp_growth[i] = np.random.normal(mu, sigma, self.time_periods)
    
    def _generate_fiscal_deficit(self):
        """生成财政赤字率"""
        # 假设财政赤字率服从截断正态分布
        for i in range(self.n_simulations):
            self.fiscal_deficit[i] = stats.truncnorm(
                -2, 2,  # 截断范围：[-2, 2]个标准差
                loc=0.03,  # 均值3%
                scale=0.01  # 标准差1%
            ).rvs(self.time_periods)
    
    def _generate_employment_rate(self):
        """生成就业率"""
        # 假设就业率服从Beta分布
        for i in range(self.n_simulations):
            self.employment_rate[i] = stats.beta(
                a=80,  # 形状参数，控制分布的形状
                b=20,  # 形状参数，控制分布的形状
                loc=0.8,  # 位置参数，最小值
                scale=0.2  # 尺度参数，范围大小
            ).rvs(self.time_periods)
    
    def _calculate_confidence_intervals(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """计算置信区间"""
        alpha = 1 - self.confidence_level
        lower = alpha / 2
        upper = 1 - lower
        
        mean = np.mean(data, axis=0)
        lower_bound = np.percentile(data, lower * 100, axis=0)
        upper_bound = np.percentile(data, upper * 100, axis=0)
        
        return {
            'mean': mean,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    def run_simulations(self) -> Dict[str, pd.DataFrame]:
        """运行蒙特卡洛模拟"""
        try:
            # 检查数据是否已设置
            if any(data is None for data in [self.economic_data, self.population_data]):
                raise ValueError("请先设置所有必要的数据")
            
            # 初始化结果变量
            self._initialize_result_variables()
            
            # 生成模拟数据
            self._generate_gdp_growth()
            self._generate_fiscal_deficit()
            self._generate_employment_rate()
            
            # 计算置信区间
            gdp_ci = self._calculate_confidence_intervals(self.gdp_growth)
            fiscal_ci = self._calculate_confidence_intervals(self.fiscal_deficit)
            employment_ci = self._calculate_confidence_intervals(self.employment_rate)
            
            # 整理结果
            results = {
                'gdp_growth': pd.DataFrame({
                    '时期': range(self.time_periods),
                    '平均值': gdp_ci['mean'],
                    '下界': gdp_ci['lower_bound'],
                    '上界': gdp_ci['upper_bound']
                }),
                'fiscal_deficit': pd.DataFrame({
                    '时期': range(self.time_periods),
                    '平均值': fiscal_ci['mean'],
                    '下界': fiscal_ci['lower_bound'],
                    '上界': fiscal_ci['upper_bound']
                }),
                'employment_rate': pd.DataFrame({
                    '时期': range(self.time_periods),
                    '平均值': employment_ci['mean'],
                    '下界': employment_ci['lower_bound'],
                    '上界': employment_ci['upper_bound']
                })
            }
            
            logger.info("蒙特卡洛模拟完成")
            return results
            
        except Exception as e:
            logger.error(f"蒙特卡洛模拟过程中出错: {str(e)}")
            raise
    
    def save_results(self, filepath: str):
        """保存模型结果"""
        try:
            results = self.run_simulations()
            
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
            'n_simulations': 10000,
            'time_periods': 30,
            'confidence_level': 0.95,
            'seed': 42
        }
        
        # 初始化模型
        model = MonteCarloModel(params)
        
        # 设置经济数据
        economic_data = pd.DataFrame({
            'gdp增长率': np.random.normal(0.03, 0.01, 10000)
        })
        model.set_economic_data(economic_data)
        
        # 设置人口数据
        population_data = pd.DataFrame({
            'population': np.random.randint(1000000, 10000000, 10000)
        })
        model.set_population_data(population_data)
        
        # 运行模拟
        results = model.run_simulations()
        
        # 保存结果
        model.save_results('results.xlsx')
        
    except Exception as e:
        logger.error(f"蒙特卡洛模拟过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 