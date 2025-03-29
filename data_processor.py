import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.processed_data_dir = Path('processed_data')
        self.processed_data_dir.mkdir(exist_ok=True)
        
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        # 将列名转换为小写并去除空格
        df.columns = df.columns.str.lower().str.strip()
        # 替换可能的年份列名
        year_columns = ['year', '年份', '年', 'date', '日期']
        for col in df.columns:
            if any(year in col.lower() for year in year_columns):
                df = df.rename(columns={col: '年份'})
                break
        return df
    
    def _ensure_year_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保数据框有年份列，并统一年份列的数据类型"""
        if '年份' not in df.columns:
            # 尝试从索引获取年份
            if isinstance(df.index, pd.DatetimeIndex):
                df['年份'] = df.index.year
            else:
                # 如果没有年份信息，添加默认年份
                df['年份'] = range(2012, 2012 + len(df))
        
        # 确保年份列为整数类型
        df['年份'] = pd.to_numeric(df['年份'], errors='coerce').astype('Int64')
        return df
    
    def _calculate_capital_stock(self, gdp_data: pd.Series) -> pd.Series:
        """计算资本存量"""
        # 使用永续盘存法估算资本存量
        # K(t) = K(t-1) * (1-δ) + I(t)
        # 其中δ为折旧率，I(t)为投资
        depreciation_rate = 0.05  # 假设5%的折旧率
        investment_rate = 0.45  # 假设45%的投资率
        
        # 初始化资本存量（假设为GDP的3倍）
        initial_capital = gdp_data.iloc[0] * 3
        
        # 计算资本存量序列
        capital_stock = [initial_capital]
        for i in range(1, len(gdp_data)):
            investment = gdp_data.iloc[i] * investment_rate
            capital = capital_stock[-1] * (1 - depreciation_rate) + investment
            capital_stock.append(capital)
        
        return pd.Series(capital_stock, index=gdp_data.index)
    
    def process_population_data(self) -> pd.DataFrame:
        """处理人口数据"""
        try:
            # 读取年龄结构数据
            age_structure = pd.read_excel(self.data_dir / '年龄结构.xlsx')
            age_structure = self._standardize_column_names(age_structure)
            age_structure = self._ensure_year_column(age_structure)
            
            # 读取出生率数据
            birth_rate = pd.read_excel(self.data_dir / '人口出生率.xls')
            birth_rate = self._standardize_column_names(birth_rate)
            birth_rate = self._ensure_year_column(birth_rate)
            
            # 读取死亡率数据
            death_rate = pd.read_excel(self.data_dir / '自然死亡率2012-2023.xls')
            death_rate = self._standardize_column_names(death_rate)
            death_rate = self._ensure_year_column(death_rate)
            
            # 合并数据
            population_data = pd.merge(age_structure, birth_rate, on='年份', how='outer')
            population_data = pd.merge(population_data, death_rate, on='年份', how='outer')
            
            # 按年份排序
            population_data = population_data.sort_values('年份')
            
            # 保存处理后的数据
            population_data.to_excel(self.processed_data_dir / 'processed_population_data.xlsx', index=False)
            return population_data
            
        except Exception as e:
            logger.error(f"处理人口数据时出错: {str(e)}")
            raise
    
    def process_economic_data(self) -> pd.DataFrame:
        """处理经济数据"""
        try:
            # 读取GDP数据
            gdp_data = pd.read_excel(self.data_dir / '各地区GDP总值.xls')
            gdp_data = self._standardize_column_names(gdp_data)
            gdp_data = self._ensure_year_column(gdp_data)
            
            # 计算全国GDP总值（排除年份列）
            numeric_cols = gdp_data.select_dtypes(include=[np.number]).columns
            numeric_cols = numeric_cols[numeric_cols != '年份']
            gdp_total = gdp_data[numeric_cols].sum(axis=1)
            
            # 计算GDP增长率
            gdp_growth = gdp_total.pct_change()
            
            # 计算资本存量
            capital_stock = self._calculate_capital_stock(gdp_total)
            
            # 读取就业数据
            employment_data = pd.read_excel(self.data_dir / '就业劳动力.xlsx')
            employment_data = self._standardize_column_names(employment_data)
            employment_data = self._ensure_year_column(employment_data)
            
            # 整理经济数据
            economic_data = pd.DataFrame({
                '年份': gdp_data['年份'],
                'gdp总值': gdp_total,
                'gdp增长率': gdp_growth,
                '资本存量': capital_stock
            })
            
            # 如果存在就业人数列，添加到经济数据中
            if '就业人数' in employment_data.columns:
                economic_data = pd.merge(
                    economic_data,
                    employment_data[['年份', '就业人数']],
                    on='年份',
                    how='left'
                )
            
            # 按年份排序
            economic_data = economic_data.sort_values('年份')
            
            # 保存处理后的数据
            economic_data.to_excel(self.processed_data_dir / 'processed_economic_data.xlsx', index=False)
            return economic_data
            
        except Exception as e:
            logger.error(f"处理经济数据时出错: {str(e)}")
            raise
    
    def process_social_security_data(self) -> pd.DataFrame:
        """处理社会保障数据"""
        try:
            # 读取养老金数据
            pension_data = pd.read_excel(self.data_dir / '养老金统筹比例.xls')
            pension_data = self._standardize_column_names(pension_data)
            pension_data = self._ensure_year_column(pension_data)
            
            # 按年份排序
            pension_data = pension_data.sort_values('年份')
            
            # 保存处理后的数据
            pension_data.to_excel(self.processed_data_dir / 'processed_pension_data.xlsx', index=False)
            return pension_data
            
        except Exception as e:
            logger.error(f"处理社会保障数据时出错: {str(e)}")
            raise
    
    def process_industry_data(self) -> pd.DataFrame:
        """处理行业数据"""
        try:
            # 读取行业数据
            industry_data = pd.read_excel(self.data_dir / '行业分城镇非私营单位(1).xls', skiprows=[0])
            
            # 重命名列
            industry_data.columns = ['指标'] + [str(year) + '年' for year in range(2023, 2014, -1)]
            
            # 删除空行和全为零的行
            industry_data = industry_data.dropna(how='all')
            industry_data = industry_data[~(industry_data.iloc[:, 1:] == 0).all(axis=1)]
            
            # 设置指标为索引
            industry_data = industry_data.set_index('指标')
            
            # 转置数据，使年份成为索引
            industry_data = industry_data.transpose()
            
            # 重置索引，将年份变为列
            industry_data = industry_data.reset_index()
            industry_data.columns.name = None
            
            # 重命名年份列
            industry_data = industry_data.rename(columns={'index': '年份'})
            
            # 提取年份数字
            industry_data['年份'] = industry_data['年份'].str.extract('(\d+)').astype(int)
            
            # 确保所有数据列为数值类型，并处理异常值
            for col in industry_data.columns:
                if col != '年份':
                    # 转换为数值类型
                    industry_data[col] = pd.to_numeric(industry_data[col], errors='coerce')
                    
                    # 处理异常值（超过3个标准差的值）
                    mean = industry_data[col].mean()
                    std = industry_data[col].std()
                    industry_data.loc[abs(industry_data[col] - mean) > 3*std, col] = mean
            
            # 计算行业占比
            industry_data_pct = industry_data.copy()
            for year in industry_data['年份'].unique():
                year_data = industry_data[industry_data['年份'] == year]
                total = year_data.drop('年份', axis=1).sum(axis=1).iloc[0]
                for col in industry_data.columns:
                    if col != '年份':
                        industry_data_pct.loc[industry_data_pct['年份'] == year, col + '_占比'] = \
                            year_data[col].iloc[0] / total * 100
            
            # 计算年度变化率
            industry_data_growth = industry_data.copy()
            for col in industry_data.columns:
                if col != '年份':
                    industry_data_growth[col + '_增长率'] = industry_data[col].pct_change() * 100
            
            # 合并所有指标
            result_data = pd.merge(industry_data, industry_data_pct.drop(industry_data.columns[1:], axis=1), on='年份')
            result_data = pd.merge(result_data, industry_data_growth.drop(industry_data.columns[1:], axis=1), on='年份')
            
            # 计算总就业人数和总量指标
            result_data['总就业人数'] = result_data[industry_data.columns[1:]].sum(axis=1)
            result_data['行业数量'] = len(industry_data.columns[1:])
            
            # 按年份排序
            result_data = result_data.sort_values('年份')
            
            # 填充缺失值
            result_data = result_data.ffill().bfill()
            
            # 添加行业分类
            industry_categories = {
                '第一产业': ['农林牧渔业'],
                '第二产业': ['采矿业', '制造业', '电力热力燃气及水生产和供应业', '建筑业'],
                '第三产业': ['批发和零售业', '交通运输仓储和邮政业', '住宿和餐饮业', '信息传输软件和信息技术服务业',
                         '金融业', '房地产业', '租赁和商务服务业', '科学研究和技术服务业', '水利环境和公共设施管理业',
                         '居民服务修理和其他服务业', '教育', '卫生和社会工作', '文化体育和娱乐业',
                         '公共管理社会保障和社会组织']
            }
            
            # 计算产业分类就业人数
            for category, industries in industry_categories.items():
                category_cols = [col for col in industry_data.columns[1:]
                               if any(ind in col for ind in industries)]
                result_data[f'{category}就业人数'] = result_data[category_cols].sum(axis=1)
                result_data[f'{category}占比'] = result_data[f'{category}就业人数'] / result_data['总就业人数'] * 100
            
            # 保存处理后的数据
            result_data.to_excel(self.processed_data_dir / 'processed_industry_data.xlsx', index=False)
            
            # 保存行业分析报告
            self._save_industry_analysis(result_data)
            
            logger.info(f"行业数据处理完成，共有{len(industry_data.columns)-1}个行业，{len(industry_data)}年数据")
            return result_data
            
        except Exception as e:
            logger.error(f"处理行业数据时出错: {str(e)}")
            raise
    
    def _save_industry_analysis(self, data: pd.DataFrame):
        """保存行业分析报告"""
        try:
            with pd.ExcelWriter(self.processed_data_dir / 'industry_analysis.xlsx') as writer:
                # 基础统计信息
                stats = pd.DataFrame({
                    '指标': ['总就业人数', '行业数量', '第一产业占比', '第二产业占比', '第三产业占比'],
                    '最新值': [
                        data['总就业人数'].iloc[-1],
                        data['行业数量'].iloc[-1],
                        data['第一产业占比'].iloc[-1],
                        data['第二产业占比'].iloc[-1],
                        data['第三产业占比'].iloc[-1]
                    ],
                    '平均值': [
                        data['总就业人数'].mean(),
                        data['行业数量'].mean(),
                        data['第一产业占比'].mean(),
                        data['第二产业占比'].mean(),
                        data['第三产业占比'].mean()
                    ],
                    '变化趋势': [
                        '上升' if data['总就业人数'].iloc[-1] > data['总就业人数'].iloc[0] else '下降',
                        '不变',
                        '上升' if data['第一产业占比'].iloc[-1] > data['第一产业占比'].iloc[0] else '下降',
                        '上升' if data['第二产业占比'].iloc[-1] > data['第二产业占比'].iloc[0] else '下降',
                        '上升' if data['第三产业占比'].iloc[-1] > data['第三产业占比'].iloc[0] else '下降'
                    ]
                })
                stats.to_excel(writer, sheet_name='基础统计', index=False)
                
                # 行业结构变化
                structure = data[[col for col in data.columns if '占比' in col and '产业' not in col]].describe()
                structure.to_excel(writer, sheet_name='行业结构')
                
                # 增长率分析
                growth = data[[col for col in data.columns if '增长率' in col]].describe()
                growth.to_excel(writer, sheet_name='增长率分析')
                
            logger.info("行业分析报告已保存")
            
        except Exception as e:
            logger.error(f"保存行业分析报告时出错: {str(e)}")
            raise
    
    def process_all_data(self) -> Dict[str, pd.DataFrame]:
        """处理所有数据"""
        try:
            processed_data = {
                'population': self.process_population_data(),
                'economic': self.process_economic_data(),
                'social_security': self.process_social_security_data(),
                'industry': self.process_industry_data()
            }
            logger.info("数据预处理完成")
            return processed_data
            
        except Exception as e:
            logger.error(f"数据预处理过程中出错: {str(e)}")
            raise

def main():
    """主函数"""
    try:
        # 初始化数据处理器
        processor = DataProcessor()
        
        # 处理所有数据
        processed_data = processor.process_all_data()
        
        logger.info("数据预处理完成")
        
    except Exception as e:
        logger.error(f"数据预处理过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 