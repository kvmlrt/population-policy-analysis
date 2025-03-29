import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class 数据处理器:
    def __init__(self):
        """初始化数据处理器"""
        # 获取当前文件所在目录
        当前目录 = Path(__file__).parent.parent
        self.数据目录 = 当前目录 / '数据'
        self.处理后数据目录 = 当前目录 / '处理后数据'
        self.数据目录.mkdir(exist_ok=True)
        self.处理后数据目录.mkdir(exist_ok=True)
        
        # 定义不同类型指标的标准化方法
        self.标准化方法 = {
            'GDP类': MinMaxScaler(),  # 适用于总量指标
            '比率类': StandardScaler(),  # 适用于比率指标
            '增长类': RobustScaler()  # 适用于增长率指标
        }
    
    def 数据质量验证(self, 数据: pd.DataFrame, 列名: str) -> bool:
        """
        验证数据质量
        返回: bool, 数据是否通过验证
        """
        try:
            # 1. 检查缺失值
            缺失比例 = 数据[列名].isnull().mean()
            if 缺失比例 > 0.1:  # 缺失值超过10%
                logger.warning(f"{列名}列缺失值比例过高: {缺失比例:.2%}")
                return False
            
            # 2. 检查异常值
            z_scores = np.abs(stats.zscore(数据[列名].dropna()))
            异常值比例 = (z_scores > 3).mean()  # 3个标准差以外视为异常
            if 异常值比例 > 0.05:  # 异常值超过5%
                logger.warning(f"{列名}列异常值比例过高: {异常值比例:.2%}")
                return False
            
            # 3. 检查数据范围
            if 数据[列名].min() < 0 and 列名 not in ['增长率', '变化率']:
                logger.warning(f"{列名}列存在异常负值")
                return False
            
            # 4. 检查数据趋势
            if len(数据) > 1:
                变化率 = 数据[列名].pct_change()
                if abs(变化率.mean()) > 0.5:  # 平均变化率超过50%
                    logger.warning(f"{列名}列数据变化率异常")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"数据质量验证出错: {str(e)}")
            return False
    
    def 标准化数据(self, 数据: pd.DataFrame, 列名: str, 指标类型: str) -> pd.Series:
        """
        根据指标类型选择合适的标准化方法
        """
        try:
            if 指标类型 not in self.标准化方法:
                raise ValueError(f"未知的指标类型: {指标类型}")
            
            标准化器 = self.标准化方法[指标类型]
            return pd.Series(
                标准化器.fit_transform(数据[列名].values.reshape(-1, 1)).flatten(),
                index=数据.index
            )
            
        except Exception as e:
            logger.error(f"数据标准化出错: {str(e)}")
            raise
    
    def 处理所有数据(self) -> dict:
        """处理所有数据"""
        try:
            # 处理各类数据
            行业数据 = self._处理行业数据()
            经济数据 = self._处理经济数据()
            人口数据 = self._处理人口数据()
            社保数据 = self._处理社保数据()
            
            # 保存处理后的数据
            行业数据.to_excel(self.处理后数据目录 / '行业数据.xlsx', index=False)
            经济数据.to_excel(self.处理后数据目录 / '经济数据.xlsx', index=False)
            人口数据.to_excel(self.处理后数据目录 / '人口数据.xlsx', index=False)
            社保数据.to_excel(self.处理后数据目录 / '社保数据.xlsx', index=False)
            
            # 生成行业分析报告
            self._生成行业分析报告(行业数据)
            
            logger.info(f"行业数据处理完成，共有{len(行业数据['行业'].unique())}个行业，{len(行业数据['年份'].unique())}年数据")
            logger.info("数据预处理完成")
            
            return {
                '行业': 行业数据,
                '经济': 经济数据,
                '人口': 人口数据,
                '社保': 社保数据
            }
            
        except Exception as e:
            logger.error(f"数据处理过程中出错: {str(e)}")
            raise
    
    def _处理行业数据(self) -> pd.DataFrame:
        """处理行业数据"""
        try:
            # 读取行业数据
            数据 = pd.read_csv(self.数据目录 / '行业数据.csv')
            
            # 标准化列名
            列名映射 = {
                'year': '年份',
                'industry': '行业',
                'output': '产出',
                'employment': '就业人数',
                'investment': '投资',
                'value_added': '增加值'
            }
            数据.rename(columns=列名映射, inplace=True)
            
            # 确保年份列为整数
            数据['年份'] = 数据['年份'].astype(int)
            
            # 计算行业占比
            年度总量 = 数据.groupby('年份')['产出'].sum().reset_index()
            数据 = 数据.merge(年度总量, on='年份', suffixes=('', '_总量'))
            数据['占比'] = 数据['产出'] / 数据['产出_总量'] * 100
            
            # 计算增长率
            数据['增长率'] = 数据.groupby('行业')['产出'].pct_change() * 100
            
            # 计算劳动生产率
            数据['劳动生产率'] = 数据['增加值'] / 数据['就业人数']
            
            return 数据
            
        except Exception as e:
            logger.error(f"处理行业数据时出错: {str(e)}")
            raise
    
    def _处理经济数据(self) -> pd.DataFrame:
        """处理经济数据"""
        try:
            数据 = pd.read_csv(self.数据目录 / '经济数据.csv')
            
            # 标准化列名
            列名映射 = {
                'year': '年份',
                'gdp': 'GDP',
                'consumption': '消费',
                'investment': '投资',
                'export': '出口',
                'import': '进口',
                'cpi': 'CPI',
                'ppi': 'PPI'
            }
            数据.rename(columns=列名映射, inplace=True)
            
            # 数据质量验证
            for 列名 in 数据.columns:
                if 列名 != '年份' and not self.数据质量验证(数据, 列名):
                    logger.warning(f"{列名}列数据质量验证未通过，将进行修正")
                    # 处理异常值
                    数据[列名] = self._处理异常值(数据[列名])
            
            # 差异化标准化处理
            标准化数据 = pd.DataFrame({'年份': 数据['年份']})
            
            # GDP类指标标准化
            for 列名 in ['GDP', '消费', '投资', '出口', '进口']:
                标准化数据[f'{列名}_标准化'] = self.标准化数据(数据, 列名, 'GDP类')
            
            # 比率类指标标准化
            for 列名 in ['CPI', 'PPI']:
                标准化数据[f'{列名}_标准化'] = self.标准化数据(数据, 列名, '比率类')
            
            # 计算并标准化增长指标
            数据['GDP增长率'] = 数据['GDP'].pct_change() * 100
            标准化数据['GDP增长率_标准化'] = self.标准化数据(数据, 'GDP增长率', '增长类')
            
            # 合并原始数据和标准化数据
            结果 = pd.merge(数据, 标准化数据, on='年份')
            
            return 结果
            
        except Exception as e:
            logger.error(f"处理经济数据时出错: {str(e)}")
            raise
    
    def _处理异常值(self, 序列: pd.Series) -> pd.Series:
        """处理异常值"""
        try:
            # 计算上下四分位数
            Q1 = 序列.quantile(0.25)
            Q3 = 序列.quantile(0.75)
            IQR = Q3 - Q1
            
            # 定义异常值边界
            下界 = Q1 - 1.5 * IQR
            上界 = Q3 + 1.5 * IQR
            
            # 将异常值替换为边界值
            序列 = 序列.clip(lower=下界, upper=上界)
            
            return 序列
            
        except Exception as e:
            logger.error(f"处理异常值时出错: {str(e)}")
            raise
    
    def _处理人口数据(self) -> pd.DataFrame:
        """处理人口数据"""
        try:
            数据 = pd.read_csv(self.数据目录 / '人口数据.csv')
            
            # 计算生育率和死亡率
            数据['生育率'] = (数据['儿童人口'].shift(-1) - 数据['儿童人口']) / 数据['总人口'] + 0.015  # 假设基础死亡率1.5%
            数据['死亡率'] = 数据['老年人口'] / 数据['总人口'] * 0.05  # 假设老年人口死亡率5%
            
            # 计算城镇化率
            数据['城镇化率'] = 数据['城镇人口'] / 数据['总人口']
            
            # 计算教育水平（假设逐年提高）
            基准教育年限 = 9.0  # 2015年基准
            数据['教育水平'] = [基准教育年限 + 0.1 * i for i in range(len(数据))]
            
            # 数据质量验证
            for 列名 in ['总人口', '劳动年龄人口', '老年人口', '儿童人口', '城镇人口', '农村人口']:
                if not self.数据质量验证(数据, 列名):
                    logger.warning(f"{列名}列数据质量验证未通过，将进行修正")
                    数据[列名] = self._处理异常值(数据[列名])
            
            return 数据
            
        except Exception as e:
            logger.error(f"处理人口数据时出错: {str(e)}")
            raise
    
    def _处理社保数据(self) -> pd.DataFrame:
        """处理社保数据"""
        try:
            数据 = pd.read_csv(self.数据目录 / '社保数据.csv')
            
            # 计算总收入和支出
            数据['收入'] = 数据['养老金收入'] + 数据['医保收入'] + 数据['失业保险收入']
            数据['支出'] = 数据['养老金支出'] + 数据['医保支出'] + 数据['失业保险支出']
            
            # 计算参保人数（假设覆盖率逐年提高）
            基准覆盖率 = 0.8  # 2015年基准
            数据['参保人数'] = pd.read_csv(self.数据目录 / '人口数据.csv')['总人口'] * \
                          [基准覆盖率 + 0.01 * i for i in range(len(数据))]
            
            # 数据质量验证
            for 列名 in ['收入', '支出', '参保人数']:
                if not self.数据质量验证(数据, 列名):
                    logger.warning(f"{列名}列数据质量验证未通过，将进行修正")
                    数据[列名] = self._处理异常值(数据[列名])
            
            return 数据
            
        except Exception as e:
            logger.error(f"处理社保数据时出错: {str(e)}")
            raise
    
    def _生成行业分析报告(self, 数据: pd.DataFrame):
        """生成行业分析报告"""
        try:
            报告内容 = []
            最新年份 = 数据['年份'].max()
            
            # 计算行业集中度
            最新数据 = 数据[数据['年份'] == 最新年份].sort_values('产出', ascending=False)
            前十行业 = 最新数据.head(10)
            
            报告内容.append(f"# {最新年份}年行业分析报告\n")
            
            # 产出前十行业
            报告内容.append("## 一、产出前十行业")
            for _, 行 in 前十行业.iterrows():
                报告内容.append(f"- {行['行业']}: {行['产出']:,.2f}亿元，占比{行['占比']:.2f}%")
            
            # 增长最快的行业
            增长最快 = 最新数据.nlargest(5, '增长率')
            报告内容.append("\n## 二、增长最快的行业")
            for _, 行 in 增长最快.iterrows():
                报告内容.append(f"- {行['行业']}: 增长率{行['增长率']:.2f}%")
            
            # 就业贡献最大的行业
            就业最多 = 最新数据.nlargest(5, '就业人数')
            报告内容.append("\n## 三、就业贡献最大的行业")
            for _, 行 in 就业最多.iterrows():
                报告内容.append(f"- {行['行业']}: {行['就业人数']:,.0f}人")
            
            # 劳动生产率最高的行业
            效率最高 = 最新数据.nlargest(5, '劳动生产率')
            报告内容.append("\n## 四、劳动生产率最高的行业")
            for _, 行 in 效率最高.iterrows():
                报告内容.append(f"- {行['行业']}: {行['劳动生产率']:.2f}万元/人")
            
            # 保存报告
            with open(self.处理后数据目录 / '行业分析报告.md', 'w', encoding='utf-8') as f:
                f.write('\n'.join(报告内容))
            
            logger.info("行业分析报告已保存")
            
        except Exception as e:
            logger.error(f"生成行业分析报告时出错: {str(e)}")
            raise 