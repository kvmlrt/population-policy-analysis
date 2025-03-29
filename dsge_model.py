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

class DSGEModel:
    def __init__(self, params: Dict[str, Any]):
        """
        初始化DSGE模型
        
        Args:
            params: 模型参数字典
        """
        self.params = params
        self.economic_data = None
        
        # 验证参数
        self._validate_params()
        
        # 初始化模型参数
        self.beta = params['beta']  # 贴现因子
        self.alpha = params['alpha']  # 资本份额
        self.delta = params['delta']  # 资本折旧率
        self.rho = params['rho']  # 技术冲击持续性
        self.sigma = params['sigma']  # 技术冲击标准差
        self.phi = params['phi']  # 劳动供给弹性
        self.eta = params['eta']  # 货币政策对通胀的反应
        self.time_periods = params['time_periods']  # 模拟期数
        
        # 初始化状态变量
        self.output = None
        self.consumption = None
        self.investment = None
        self.capital = None
        self.labor = None
        self.technology = None
        self.inflation = None
        self.interest_rate = None
        
    def _validate_params(self):
        """验证模型参数"""
        required_params = [
            'beta',
            'alpha',
            'delta',
            'rho',
            'sigma',
            'phi',
            'eta',
            'time_periods'
        ]
        
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"缺少必要参数: {param}")
    
    def set_economic_data(self, data: pd.DataFrame):
        """设置经济数据"""
        self.economic_data = data
        logger.info("经济数据已设置")
    
    def _initialize_state_variables(self):
        """初始化状态变量"""
        self.output = np.zeros(self.time_periods)
        self.consumption = np.zeros(self.time_periods)
        self.investment = np.zeros(self.time_periods)
        self.capital = np.zeros(self.time_periods)
        self.labor = np.zeros(self.time_periods)
        self.technology = np.zeros(self.time_periods)
        self.inflation = np.zeros(self.time_periods)
        self.interest_rate = np.zeros(self.time_periods)
    
    def _generate_technology_shocks(self):
        """生成技术冲击"""
        shocks = np.random.normal(0, self.sigma, self.time_periods)
        self.technology[0] = shocks[0]
        for t in range(1, self.time_periods):
            self.technology[t] = self.rho * self.technology[t-1] + shocks[t]
    
    def _calculate_steady_state(self):
        """计算稳态值"""
        # 稳态利率
        r_ss = 1/self.beta - 1
        
        # 稳态资本劳动比
        k_l_ss = ((r_ss + self.delta)/(self.alpha))**(1/(self.alpha-1))
        
        # 稳态产出劳动比
        y_l_ss = k_l_ss**self.alpha
        
        # 稳态投资劳动比
        i_l_ss = self.delta * k_l_ss
        
        # 稳态消费劳动比
        c_l_ss = y_l_ss - i_l_ss
        
        return {
            'r_ss': r_ss,
            'k_l_ss': k_l_ss,
            'y_l_ss': y_l_ss,
            'i_l_ss': i_l_ss,
            'c_l_ss': c_l_ss
        }
    
    def run_simulation(self) -> Dict[str, pd.DataFrame]:
        """运行模型模拟"""
        try:
            # 检查数据是否已设置
            if self.economic_data is None:
                raise ValueError("请先设置经济数据")
            
            # 初始化状态变量
            self._initialize_state_variables()
            
            # 生成技术冲击
            self._generate_technology_shocks()
            
            # 获取稳态值
            steady_state = self._calculate_steady_state()
            
            # 设置初始值
            self.capital[0] = self.economic_data['资本存量'].iloc[0]
            self.labor[0] = 1.0  # 标准化劳动供给
            
            # 主循环
            for t in range(self.time_periods - 1):
                # 计算产出
                self.output[t] = np.exp(self.technology[t]) * \
                                (self.capital[t]**self.alpha) * \
                                (self.labor[t]**(1-self.alpha))
                
                # 计算投资
                self.investment[t] = self.output[t] * steady_state['i_l_ss'] / steady_state['y_l_ss']
                
                # 计算消费
                self.consumption[t] = self.output[t] - self.investment[t]
                
                # 更新资本存量
                self.capital[t+1] = (1-self.delta)*self.capital[t] + self.investment[t]
                
                # 计算通货膨胀率（简化版）
                self.inflation[t] = (self.output[t]/self.output[t-1] - 1) if t > 0 else 0
                
                # 计算名义利率（Taylor规则）
                self.interest_rate[t] = max(0, steady_state['r_ss'] + self.eta * self.inflation[t])
                
                # 更新劳动供给
                self.labor[t+1] = self.labor[t] * (1 + 0.1 * (self.output[t]/self.output[t-1] - 1)) \
                                 if t > 0 else self.labor[t]
            
            # 整理结果
            results = {
                'output': pd.DataFrame({
                    '时期': range(self.time_periods),
                    '产出': self.output
                }),
                'consumption_investment': pd.DataFrame({
                    '时期': range(self.time_periods),
                    '消费': self.consumption,
                    '投资': self.investment
                }),
                'capital_labor': pd.DataFrame({
                    '时期': range(self.time_periods),
                    '资本': self.capital,
                    '劳动': self.labor
                }),
                'technology': pd.DataFrame({
                    '时期': range(self.time_periods),
                    '技术水平': self.technology
                }),
                'inflation_interest': pd.DataFrame({
                    '时期': range(self.time_periods),
                    '通货膨胀率': self.inflation,
                    '利率': self.interest_rate
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
            'T': 10,
            # TODO: 添加其他必要参数
        }
        
        # 初始化模型
        model = DSGEModel(params)
        
        # 读取数据
        data_dir = Path('data')
        household_data = pd.read_excel(data_dir / 'household_processed.xlsx')
        firm_data = pd.read_excel(data_dir / 'firm_processed.xlsx')
        government_data = pd.read_excel(data_dir / 'government_processed.xlsx')
        
        # 设置参数
        model.set_household_parameters(household_data)
        model.set_firm_parameters(firm_data)
        model.set_government_parameters(government_data)
        
        # 运行模拟
        results = model.simulate()
        
        # 保存结果
        model.save_results('results')
        
        logger.info("DSGE模型模拟完成")
        
    except Exception as e:
        logger.error(f"DSGE模型模拟过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 