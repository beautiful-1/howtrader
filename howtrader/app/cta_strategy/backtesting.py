from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Callable
from itertools import product
from functools import lru_cache
from time import time
import multiprocessing
import random
import traceback
from typing import Type, Dict, List
import numpy as np
from pandas import DataFrame
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from deap import creator, base, tools, algorithms

from howtrader.trader.constant import (Direction, Offset, Exchange,
                                       Interval, Status)
from howtrader.trader.database import get_database, BaseDatabase
from howtrader.trader.object import OrderData, TradeData, BarData, TickData
from howtrader.trader.utility import round_to
from decimal import Decimal
from howtrader.event import Event, EventEngine

database: BaseDatabase = get_database()

from .base import (
    BacktestingMode,
    EngineType,
    STOPORDER_PREFIX,
    StopOrder,
    StopOrderStatus,
    INTERVAL_DELTA_MAP
)
from .template import CtaTemplate

# Set deap algo
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class OptimizationSetting:
    """
    Setting for runnning optimization.
    """

    def __init__(self):
        """"""
        self.params = {}
        self.target_name = ""

    def add_parameter(
            self, name: str, start: float, end: float = None, step: float = None
    ):
        """"""
        if not end and not step:
            self.params[name] = [start]
            return

        if start >= end:
            print("start value should be greater than end value")
            return

        if step <= 0:
            print("step value should be greater than zero")
            return

        value = start
        value_list = []

        while value <= end:
            value_list.append(value)
            value += step

        self.params[name] = value_list

    def set_target(self, target_name: str):
        """"""
        self.target_name = target_name

    def generate_setting(self):
        """"""
        keys = self.params.keys()
        values = self.params.values()
        products = list(product(*values))

        settings = []
        for p in products:
            setting = dict(zip(keys, p))
            settings.append(setting)

        return settings

    def generate_setting_ga(self):
        """"""
        settings_ga = []
        settings = self.generate_setting()
        for d in settings:
            param = [tuple(i) for i in d.items()]
            settings_ga.append(param)
        return settings_ga


class BacktestingEngine:
    """"""

    engine_type: EngineType = EngineType.BACKTESTING
    gateway_name: str = "BACKTESTING"

    def __init__(self, event_engine: EventEngine = None) -> None:
        """"""
        self.vt_symbol: str = ""
        self.symbol: str = ""
        # 交易所信息
        self.exchange: Exchange = None
        # 开始时间
        self.start: datetime = None
        # 结束时间
        self.end: datetime = None
        # 费率
        self.rate: float = 0
        # 滑点
        self.slippage: float = 0
        # 没收合约的数量
        self.size: float = 1
        # 代表最小价格变动
        self.pricetick: float = 0
        # 初始资本
        self.capital: float = 1_000_000
        # 回测模式，是基于k线还是tick
        self.mode: BacktestingMode = BacktestingMode.BAR
        # 一年的交易天数
        self.annual_days: int = 365
        # 代表是否是反向策略
        self.inverse: bool = False
        # 使用的策略类
        self.strategy_class: Type[CtaTemplate] = None
        # 策略实例
        self.strategy: CtaTemplate = None
        # tick数据
        self.tick: TickData
        # k线数据
        self.bar: BarData
        # 当前时间
        self.datetime: datetime = None
        # K线周期
        self.interval: Interval = None
        # 回测天数
        self.days: int = 0
        # 回调函数
        self.callback: Callable = None
        # 历史数据
        self.history_data: list = []
        # 止损单数量
        self.stop_order_count: int = 0
        # 代表止损单字典，记录止损单的信息。
        self.stop_orders: Dict[str, StopOrder] = {}
        # 当前活跃的止损单
        self.active_stop_orders: Dict[str, StopOrder] = {}
        # 限价单数量
        self.limit_order_count: int = 0
        # 限价单字典
        self.limit_orders: Dict[str, OrderData] = {}
        # 活跃限价单字典
        self.active_limit_orders: Dict[str, OrderData] = {}
        # 交易数量
        self.trade_count: int = 0
        # 交易记录的字典
        self.trades: Dict[str, TradeData] = {}

        self.logs: list = []
        # 每日回测结果字典。
        self.daily_results: Dict[date, DailyResult] = {}
        # 每日回测结果的DataFrame。
        self.daily_df: DataFrame = None
        self.event_engine: EventEngine = event_engine

    def clear_data(self) -> None:
        """
        Clear all data of last backtesting.
        """
        self.strategy = None
        self.tick = None
        self.bar = None
        self.datetime = None

        self.stop_order_count = 0
        self.stop_orders.clear()
        self.active_stop_orders.clear()

        self.limit_order_count = 0
        self.limit_orders.clear()
        self.active_limit_orders.clear()

        self.trade_count = 0
        self.trades.clear()

        self.logs.clear()
        self.daily_results.clear()

    def set_parameters(
            self,
            vt_symbol: str,
            interval: Interval,
            start: datetime,
            rate: float,
            slippage: float,
            size: float,
            pricetick: float,
            capital: int = 0,
            end: datetime = None,
            mode: BacktestingMode = BacktestingMode.BAR,
            inverse: bool = False,
            annual_days: int = 365
    ):
        """"""
        self.mode = mode
        self.vt_symbol = vt_symbol
        self.interval = interval
        self.rate = rate
        self.slippage = slippage
        self.size = size
        self.pricetick = pricetick
        self.start = start

        self.symbol, exchange_str = self.vt_symbol.split(".")
        self.exchange = Exchange(exchange_str)

        self.capital = capital
        self.end = end
        self.mode = mode
        self.inverse = inverse
        self.annual_days = annual_days

    # Type[CtaTemplate] 表示这个参数应该是一个 CtaTemplate 类或其子类的类型。
    def add_strategy(self, strategy_class: Type[CtaTemplate], setting: dict) -> None:
        """"""
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, self.vt_symbol, setting
        )

    def load_data(self) -> None:
        """"""
        self.output("start loading historical data")

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            self.output("start date should be less than end date")
            return

        self.history_data.clear()  # Clear previously loaded history data

        # Load 30 days of data each time and allow for progress update
        progress_delta = timedelta(days=30)
        total_delta = self.end - self.start
        interval_delta = INTERVAL_DELTA_MAP[self.interval]

        start = self.start
        end = self.start + progress_delta
        progress = 0

        while start < self.end:
            end = min(end, self.end)  # Make sure end time stays within set range

            if self.mode == BacktestingMode.BAR:
                data = load_bar_data(
                    self.symbol,
                    self.exchange,
                    self.interval,
                    start,
                    end
                )
            else:
                data = load_tick_data(
                    self.symbol,
                    self.exchange,
                    start,
                    end
                )

            self.history_data.extend(data)

            progress += progress_delta / total_delta
            progress = min(progress, 1)
            progress_bar = "#" * int(progress * 10)
            self.output(f"loading progress：{progress_bar} [{progress:.0%}]")

            start = end + interval_delta
            end += (progress_delta + interval_delta)

        self.output(f"loading data finished, total counts：{len(self.history_data)}")

    def run_backtesting(self) -> None:
        """"""
        if self.mode == BacktestingMode.BAR:
            func = self.new_bar
        else:
            func = self.new_tick

        self.strategy.on_init()

        # Use the first [days] of history data for initializing strategy
        day_count: int = 0
        ix: int = 0

        for ix, data in enumerate(self.history_data):
            if self.datetime and data.datetime.day != self.datetime.day:
                day_count += 1
                if day_count >= self.days:
                    break

            self.datetime = data.datetime

            try:
                self.callback(data)
            except Exception:
                self.output("raise exception, stop backtesting")
                self.output(traceback.format_exc())
                return

        self.strategy.inited = True
        self.output("initialize strategy")

        self.strategy.on_start()
        self.strategy.trading = True
        self.output("start backtesting")

        # Use the rest of history data for running backtesting
        for data in self.history_data[ix:]:
            try:
                func(data)
            except Exception:
                self.output("raise exception, stop backtesting")
                self.output(traceback.format_exc())
                return

        self.strategy.on_stop()
        self.output("finish backtesting")

    def calculate_result(self):
        """"""
        self.output("start calculating pnl")
        # 检查是否存在交易订单
        if not self.trades:
            self.output("there is no trades，can't calculate")
            return

        # Add trade data into daily reuslt
        # 遍历所有交易记录.
        for trade in self.trades.values():
            # 获取日期
            d: date = trade.datetime.date()
            # self.daily_results[d]是一个字典，获取到字典以后，将这个交易添加当日的交易结果中
            daily_result: DailyResult = self.daily_results[d]
            daily_result.add_trade(trade)

        # Calculate daily result by iteration.
        # 初始化前一日的收盘价。
        pre_close = 0
        # 初始化起始仓位。
        start_pos = 0
        # 遍历每日结果
        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                # 前一日的收盘价
                pre_close,
                # 起始仓位
                start_pos,
                # 合约大小
                self.size,
                # 手续费率
                self.rate,
                # 滑点
                self.slippage,
                # 是否反向持仓
                self.inverse
            )

            pre_close = daily_result.close_price
            start_pos = daily_result.end_pos

        # Generate dataframe
        # defaultdict 是 Python 中的一个数据结构，它可以在创建字典时指定默认值的类型
        results = defaultdict(list)
        # 再次遍历每日结果。
        for daily_result in self.daily_results.values():
            for key, value in daily_result.__dict__.items():
                # 将每日结果对象的属性名作为键，属性值作为值，添加到results字典中。
                results[key].append(value)

        # from_dict 函数允许你从一个字典创建一个 DataFrame

        # .set_index("date"): 这一部分是对 DataFrame 的进一步操作。
        # set_index 方法用于将 DataFrame 的一列设置为索引列，这里的参数是 "date"，意味着将 DataFrame 中名为 "date" 的列设置为索引。
        # 索引是 DataFrame 中用于标识和访问行的标签。

        # 可以通过使用，self.daily_df.loc["2023-08-14"] 将返回 "2023-08-14" 这一行的数据。
        self.daily_df = DataFrame.from_dict(results).set_index("date")

        self.output("finish calculating pnl ")
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True):
        """"""
        self.output("start calculating strategy's performance")

        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df is None:
            # Set all statistics to 0 if no trade.
            start_date: str = ""
            end_date: str = ""
            total_days: int = 0
            profit_days: int = 0
            loss_days: int = 0
            end_balance: float = 0
            max_drawdown: float = 0
            max_ddpercent: float = 0
            max_drawdown_duration: int = 0
            total_net_pnl: float = 0
            daily_net_pnl: float = 0
            total_commission: float = 0
            daily_commission: float = 0
            total_slippage: float = 0
            daily_slippage: float = 0
            total_turnover: float = 0
            daily_turnover: float = 0
            total_trade_count: int = 0
            daily_trade_count: int = 0
            total_return: float = 0
            annual_return: float = 0
            daily_return: float = 0
            return_std: float = 0
            sharpe_ratio: float = 0
            return_drawdown_ratio: float = 0
        else:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital
            df["return"] = np.log(df["balance"] / df["balance"].shift(1)).fillna(0)
            df["highlevel"] = (
                df["balance"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )
            df["drawdown"] = df["balance"] - df["highlevel"]
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100

            # Calculate statistics value
            start_date = df.index[0]
            end_date = df.index[-1]

            total_days: int = len(df)
            profit_days: int = len(df[df["net_pnl"] > 0])
            loss_days: int = len(df[df["net_pnl"] < 0])

            end_balance = df["balance"].iloc[-1]
            max_drawdown = df["drawdown"].min()
            max_ddpercent = df["ddpercent"].min()
            max_drawdown_end = df["drawdown"].idxmin()

            if isinstance(max_drawdown_end, date):
                max_drawdown_start = df["balance"][:max_drawdown_end].idxmax()
                max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days
            else:
                max_drawdown_duration = 0

            total_net_pnl = df["net_pnl"].sum()
            daily_net_pnl = total_net_pnl / total_days

            total_commission = df["commission"].sum()
            daily_commission = total_commission / total_days

            total_slippage = df["slippage"].sum()
            daily_slippage = total_slippage / total_days

            total_turnover = df["turnover"].sum()
            daily_turnover = total_turnover / total_days

            total_trade_count = df["trade_count"].sum()
            daily_trade_count = total_trade_count / total_days

            total_return = (end_balance / self.capital - 1) * 100
            annual_return = total_return / total_days * self.annual_days
            daily_return = df["return"].mean() * 100
            return_std = df["return"].std() * 100

            if return_std:
                sharpe_ratio = daily_return / return_std * np.sqrt(365)
            else:
                sharpe_ratio = 0

            return_drawdown_ratio = -total_return / max_ddpercent

        # Output
        if output:
            self.output("-" * 30)
            self.output(f"start date：\t{start_date}")
            self.output(f"end date：\t{end_date}")

            self.output(f"total days【总回测天数】：\t{total_days}")
            self.output(f"profit days【盈利的交易日数】：\t{profit_days}")
            self.output(f"loss days【亏损的交易日数】：\t{loss_days}")

            self.output(f"capital【初始资本】：\t{self.capital:,.2f}")
            self.output(f"end balance【回测结束时的资本余额】：\t{end_balance:,.2f}")

            self.output(f"total return【总回报率】：\t{total_return:,.2f}%")
            self.output(f"annual return【年化回报率】：\t{annual_return:,.2f}%")
            self.output(f"max drawdown【最大回撤金额】: \t{max_drawdown:,.2f}")
            self.output(f"max drawdown percent【最大回撤百分比】: \t{max_ddpercent:,.2f}%")
            self.output(f"max drawdown duration【最大回撤持续的天数】: \t{max_drawdown_duration}")

            self.output(f"total net pnl【总净利润】：\t{total_net_pnl:,.2f}")
            self.output(f"total commission【总手续费】：\t{total_commission:,.2f}")
            self.output(f"total slippage【总滑点成本】：\t{total_slippage:,.2f}")
            self.output(f"total turnover【总成交额】：\t{total_turnover:,.2f}")
            self.output(f"total trade count【总交易次数】：\t{total_trade_count}")

            self.output(f"daily net pnl【每日净利润的平均值】：\t{daily_net_pnl:,.2f}")
            self.output(f"daily commission【每日手续费的平均值】：\t{daily_commission:,.2f}")
            self.output(f"daily slippage【每日滑点成本的平均值】：\t{daily_slippage:,.2f}")
            self.output(f"daily turnover【每日成交额的平均值】：\t{daily_turnover:,.2f}")
            self.output(f"daily trade count【每日平均交易次数】：\t{daily_trade_count}")

            self.output(f"daily return【每日平均回报率】：\t{daily_return:,.2f}%")
            self.output(f"return std【回报率的标准差，用于衡量波动性】：\t{return_std:,.2f}%")
            self.output(f"Sharpe Ratio【夏普比率，用于衡量每单位风险所获得的超额回报】：\t{sharpe_ratio:,.2f}")
            self.output(f"return drawdown ratio【回报率与回撤比率】：\t{return_drawdown_ratio:,.2f}")

        statistics = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
        }

        # Filter potential error infinite value
        # items() 方法返回一个包含字典中所有键值对的可迭代对象。
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        self.output("finish calculating strategy's performance")
        return statistics

    def print_trades(self, trades: Dict[str, TradeData]):
        df = DataFrame(columns=['gateway_name', 'symbol', 'exchange', 'orderid', 'tradeid',
                                'direction', 'offset', 'price', 'volume', 'datetime'])

        # 遍历数据字典，将每个TradeData对象的属性提取出来，添加为一行数据
        for key, trade_data in trades.items():
            df.loc[key] = [trade_data.gateway_name, trade_data.symbol, trade_data.exchange,
                           trade_data.orderid, trade_data.tradeid, trade_data.direction,
                           trade_data.offset, float(trade_data.price), float(trade_data.volume),
                           trade_data.datetime]

        # 输出DataFrame
        return df

    def show_chart(self, df: DataFrame = None):
        """"""
        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        table_df = self.print_trades(self.trades)
        # Check for init DataFrame
        if df is None:
            return
        # 创建一个子图对象 fig，该对象包含了四个子图，分别用于绘制余额曲线、回撤曲线、每日盈亏柱状图和盈亏分布直方图。
        # make_subplots 函数允许在同一个图表中绘制多个子图。
        fig = make_subplots(
            # 创建了一个四行一列的子图布局
            rows=4,
            cols=1,
            # 设置了每个子图的标题。
            subplot_titles=["Balance", "Drawdown", "Daily Pnl", "Pnl Distribution", "Table-Trades"],
            # 设置了子图之间的垂直间距。
            vertical_spacing=0.06
        )

        # 创建子图布局，用于显示表格
        fig2 = make_subplots(
            rows=1, cols=1,
        )

        # 接下来，分别创建了四个图表的数据和样式：
        # go 是 Plotly Python 库中的一个子模块，它包含了用于创建各种图表类型的函数。
        # go.Scatter 函数用于创建散点图，并且可以用于绘制折线图。它允许你指定横轴和纵轴的数据，以及图表的样式。
        balance_line = go.Scatter(
            x=df.index,
            y=df["balance"],
            mode="lines",
            name="Balance"
        )
        drawdown_scatter = go.Scatter(
            x=df.index,
            y=df["drawdown"],
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="Drawdown"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Daily Pnl")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")
        table_tredes = go.Table(
            header=dict(values=list(table_df.columns)),
            cells=dict(
                values=[table_df.gateway_name, table_df.symbol, table_df.exchange, table_df.orderid, table_df.tradeid,
                        table_df.direction, table_df.offset, table_df.price, table_df.volume, table_df.datetime])
        )
        # 方法将上述图表添加到子图中，并指定子图的行列位置。
        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)
        # 设置图表的高度和宽度。
        fig.update_layout(height=1000, width=1000)
        fig.show()

        fig2.add_trace(table_tredes)
        fig2.update_layout(height=1000, showlegend=False)
        fig2.show()

    def run_optimization(self, optimization_setting: OptimizationSetting, output=True):
        """
        target name: end_balance, max_drawdown, max_ddpercent, max_drawdown_duration, total_net_pnl
        daily_net_pnl, total_commission, daily_commission, total_slippage, daily_slippage, total_turnover, daily_turnover
        total_trade_count, daily_trade_count, total_return, annual_return, daily_return, return_std, sharpe_ratio, return_drawdown_ratio
        """
        # Get optimization setting and target
        settings = optimization_setting.generate_setting()
        target_name = optimization_setting.target_name

        if not settings:
            self.output("produce parameters are empty, please check your parameters")
            return

        if not target_name:
            self.output("optimized target is not set, please check your target name")
            return

        # Use multiprocessing pool for running backtesting with different setting
        # Force to use spawn method to create new process (instead of fork on Linux)
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(multiprocessing.cpu_count())

        results = []
        for setting in settings:
            result = (pool.apply_async(optimize, (
                target_name,
                self.strategy_class,
                setting,
                self.vt_symbol,
                self.interval,
                self.start,
                self.rate,
                self.slippage,
                self.size,
                self.pricetick,
                self.capital,
                self.end,
                self.mode,
                self.inverse
            )))
            results.append(result)

        pool.close()
        pool.join()

        # Sort results and output
        result_values = [result.get() for result in results]
        result_values.sort(reverse=True, key=lambda result: result[1])

        if output:
            for value in result_values:
                msg = f"parameter：{value[0]}, target：{value[1]}"
                self.output(msg)

        return result_values

    def run_ga_optimization(self, optimization_setting: OptimizationSetting, population_size=100, ngen_size=30,
                            output=True) -> list:
        """
        target name: end_balance, max_drawdown, max_ddpercent, max_drawdown_duration, total_net_pnl
        daily_net_pnl, total_commission, daily_commission, total_slippage, daily_slippage, total_turnover, daily_turnover
        total_trade_count, daily_trade_count, total_return, annual_return, daily_return, return_std, sharpe_ratio, return_drawdown_ratio
        """
        # Get optimization setting and target
        settings = optimization_setting.generate_setting_ga()
        target_name = optimization_setting.target_name

        if not settings:
            self.output("produce parameters are empty, please check your parameters")
            return

        if not target_name:
            self.output("optimized target is not set, please check your target name: ")
            return

        # Define parameter generation function
        def generate_parameter():
            """"""
            return random.choice(settings)

        def mutate_individual(individual, indpb):
            """"""
            size = len(individual)
            paramlist = generate_parameter()
            for i in range(size):
                if random.random() < indpb:
                    individual[i] = paramlist[i]
            return individual,

        # Create ga object function
        global ga_target_name
        global ga_strategy_class
        global ga_setting
        global ga_vt_symbol
        global ga_interval
        global ga_start
        global ga_rate
        global ga_slippage
        global ga_size
        global ga_pricetick
        global ga_capital
        global ga_end
        global ga_mode
        global ga_inverse

        ga_target_name = target_name
        ga_strategy_class = self.strategy_class
        ga_setting = settings[0]
        ga_vt_symbol = self.vt_symbol
        ga_interval = self.interval
        ga_start = self.start
        ga_rate = self.rate
        ga_slippage = self.slippage
        ga_size = self.size
        ga_pricetick = self.pricetick
        ga_capital = self.capital
        ga_end = self.end
        ga_mode = self.mode
        ga_inverse = self.inverse

        # Set up genetic algorithem
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, generate_parameter)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate_individual, indpb=1)
        toolbox.register("evaluate", ga_optimize)
        toolbox.register("select", tools.selNSGA2)

        total_size = len(settings)
        pop_size = population_size  # number of individuals in each generation
        lambda_ = pop_size  # number of children to produce at each generation
        mu = int(pop_size * 0.8)  # number of individuals to select for the next generation

        cxpb = 0.95  # probability that an offspring is produced by crossover
        mutpb = 1 - cxpb  # probability that an offspring is produced by mutation
        ngen = ngen_size  # number of generation

        pop = toolbox.population(pop_size)
        hof = tools.ParetoFront()  # end result of pareto front

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        np.set_printoptions(suppress=True)
        stats.register("mean", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # Multiprocessing is not supported yet.
        # pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # toolbox.register("map", pool.map)

        # Run ga optimization
        self.output(f"total size：{total_size}")
        self.output(f"population size：{pop_size}")
        self.output(f"selected next generation size：{mu}")
        self.output(f"number of generation：{ngen}")
        self.output(f"probability of crossover：{cxpb:.0%}")
        self.output(f"probability of mutation：{mutpb:.0%}")

        start = time()

        algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu,
            lambda_,
            cxpb,
            mutpb,
            ngen,
            stats,
            halloffame=hof
        )

        end = time()
        cost = int((end - start))

        self.output(f"finish optimization，cost {cost} seconds")

        # Return result list
        results = []

        for parameter_values in hof:
            setting = dict(parameter_values)
            target_value = ga_optimize(parameter_values)[0]
            results.append((setting, target_value, {}))

        return results

    def update_daily_close(self, price: float):
        """"""
        d = self.datetime.date()

        daily_result = self.daily_results.get(d, None)
        if daily_result:
            daily_result.close_price = price
        else:
            self.daily_results[d] = DailyResult(d, price)

    def new_bar(self, bar: BarData):
        """"""
        self.bar = bar
        self.datetime = bar.datetime

        self.cross_limit_order()
        self.cross_stop_order()
        self.strategy.on_bar(bar)

        self.update_daily_close(bar.close_price)

    def new_tick(self, tick: TickData):
        """"""
        self.tick = tick
        self.datetime = tick.datetime

        self.cross_limit_order()
        self.cross_stop_order()
        self.strategy.on_tick(tick)

        self.update_daily_close(tick.last_price)

    def cross_limit_order(self):
        """
        Cross limit order with last bar/tick data.
        """
        if self.mode == BacktestingMode.BAR:
            long_cross_price = self.bar.low_price
            short_cross_price = self.bar.high_price
            long_best_price = self.bar.open_price
            short_best_price = self.bar.open_price
        else:
            long_cross_price = self.tick.ask_price_1
            short_cross_price = self.tick.bid_price_1
            long_best_price = long_cross_price
            short_best_price = short_cross_price

        for order in list(self.active_limit_orders.values()):
            # Push order update with status "not traded" (pending).
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.on_order(order)

            # Check whether limit orders can be filled.
            long_cross = (Direction.LONG == order.direction and order.price >= long_cross_price > 0)
            short_cross = (
                    Direction.SHORT == order.direction and order.price <= short_cross_price and short_cross_price > 0)

            if not long_cross and not short_cross:
                continue

            # Push order udpate with status "all traded" (filled).
            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.on_order(order)

            self.active_limit_orders.pop(order.vt_orderid)

            # Push trade update
            self.trade_count += 1

            if long_cross:
                trade_price = min(order.price, Decimal(str(long_best_price)))
                pos_change = order.volume
            else:
                trade_price = max(order.price, Decimal(str(short_best_price)))
                pos_change = -order.volume

            trade = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            self.strategy.pos += pos_change
            self.strategy.on_trade(trade)

            self.trades[trade.vt_tradeid] = trade

    def cross_stop_order(self):
        """
        Cross stop order with last bar/tick data.
        """
        if self.mode == BacktestingMode.BAR:
            long_cross_price = self.bar.high_price
            short_cross_price = self.bar.low_price
            long_best_price = self.bar.open_price
            short_best_price = self.bar.open_price
        else:
            long_cross_price = self.tick.last_price
            short_cross_price = self.tick.last_price
            long_best_price = long_cross_price
            short_best_price = short_cross_price

        for stop_order in list(self.active_stop_orders.values()):
            # Check whether stop order can be triggered.
            long_cross = (
                    stop_order.direction == Direction.LONG
                    and stop_order.price <= long_cross_price
            )

            short_cross = (
                    stop_order.direction == Direction.SHORT
                    and stop_order.price >= short_cross_price
            )

            if not long_cross and not short_cross:
                continue

            # Create order data.
            self.limit_order_count += 1

            order: OrderData = OrderData(
                symbol=self.symbol,
                exchange=self.exchange,
                orderid=str(self.limit_order_count),
                direction=stop_order.direction,
                offset=stop_order.offset,
                price=stop_order.price,
                volume=stop_order.volume,
                traded=stop_order.volume,
                status=Status.ALLTRADED,
                gateway_name=self.gateway_name,
                datetime=self.datetime
            )

            self.limit_orders[order.vt_orderid] = order

            # Create trade data.
            if long_cross:
                trade_price = max(stop_order.price, Decimal(str(long_best_price)))
                pos_change = order.volume
            else:
                trade_price = min(stop_order.price, Decimal(str(short_best_price)))
                pos_change = -order.volume

            self.trade_count += 1

            trade = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            self.trades[trade.vt_tradeid] = trade

            # Update stop order.
            stop_order.vt_orderids.append(order.vt_orderid)
            stop_order.status = StopOrderStatus.TRIGGERED

            if stop_order.stop_orderid in self.active_stop_orders:
                self.active_stop_orders.pop(stop_order.stop_orderid)

            # Push update to strategy.
            self.strategy.on_stop_order(stop_order)
            self.strategy.on_order(order)

            self.strategy.pos += pos_change
            self.strategy.on_trade(trade)

    def load_bar(
            self,
            vt_symbol: str,
            days: int,
            interval: Interval,
            callback: Callable,
            use_database: bool
    ) -> List[BarData]:
        """"""
        self.days = days
        self.callback = callback
        return []

    def load_tick(self, vt_symbol: str, days: int, callback: Callable) -> List[TickData]:
        """"""
        self.days = days
        self.callback = callback
        return []

    def send_order(
            self,
            strategy: CtaTemplate,
            direction: Direction,
            offset: Offset,
            price: Decimal,
            volume: Decimal,
            stop: bool,
            lock: bool,
            net: bool,
            maker: bool = False
    ) -> List[str]:
        """"""
        price = round_to(price, Decimal(str(self.pricetick)))
        if stop:
            vt_orderid: str = self.send_stop_order(direction, offset, price, volume)
        else:
            vt_orderid: str = self.send_limit_order(direction, offset, price, volume)
        return [vt_orderid]

    def send_stop_order(
            self,
            direction: Direction,
            offset: Offset,
            price: Decimal,
            volume: Decimal
    ) -> str:
        """"""
        self.stop_order_count += 1

        stop_order: StopOrder = StopOrder(
            vt_symbol=self.vt_symbol,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            datetime=self.datetime,
            stop_orderid=f"{STOPORDER_PREFIX}.{self.stop_order_count}",
            strategy_name=self.strategy.strategy_name,
        )

        self.active_stop_orders[stop_order.stop_orderid] = stop_order
        self.stop_orders[stop_order.stop_orderid] = stop_order

        return stop_order.stop_orderid

    def send_limit_order(
            self,
            direction: Direction,
            offset: Offset,
            price: Decimal,
            volume: Decimal
    ) -> str:
        """"""
        self.limit_order_count += 1

        order: OrderData = OrderData(
            symbol=self.symbol,
            exchange=self.exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            gateway_name=self.gateway_name,
            datetime=self.datetime
        )

        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order

        return order.vt_orderid

    def cancel_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        """
        Cancel order by vt_orderid.
        """
        if vt_orderid.startswith(STOPORDER_PREFIX):
            self.cancel_stop_order(strategy, vt_orderid)
        else:
            self.cancel_limit_order(strategy, vt_orderid)

    def cancel_stop_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        """"""
        if vt_orderid not in self.active_stop_orders:
            return None
        stop_order = self.active_stop_orders.pop(vt_orderid)

        stop_order.status = StopOrderStatus.CANCELLED
        self.strategy.on_stop_order(stop_order)

    def cancel_limit_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        """"""
        if vt_orderid not in self.active_limit_orders:
            return None
        order = self.active_limit_orders.pop(vt_orderid)

        order.status = Status.CANCELLED
        self.strategy.on_order(order)

    def cancel_all(self, strategy: CtaTemplate) -> None:
        """
        Cancel all orders, both limit and stop.
        """
        vt_orderids = list(self.active_limit_orders.keys())
        for vt_orderid in vt_orderids:
            self.cancel_limit_order(strategy, vt_orderid)

        stop_orderids = list(self.active_stop_orders.keys())
        for vt_orderid in stop_orderids:
            self.cancel_stop_order(strategy, vt_orderid)

    def write_log(self, msg: str, strategy: CtaTemplate = None) -> None:
        """
        Write log message.
        """
        msg = f"{self.datetime}\t{msg}"
        self.logs.append(msg)

    def send_email(self, msg: str, strategy: CtaTemplate = None) -> None:
        """
        Send email to default receiver.
        """
        pass

    def sync_strategy_data(self, strategy: CtaTemplate) -> None:
        """
        Sync strategy data into json file.
        """
        pass

    def get_engine_type(self) -> EngineType:
        """
        Return engine type.
        """
        return self.engine_type

    def get_pricetick(self, strategy: CtaTemplate) -> float:
        """
        Return contract pricetick data.
        """
        return self.pricetick

    def put_strategy_event(self, strategy: CtaTemplate) -> None:
        """
        Put an event to update strategy status.
        """
        pass

    def output(self, msg) -> None:
        """
        Output message of backtesting engine.
        """
        print(f"{datetime.now()}\t{msg}")

    def get_all_trades(self) -> List[TradeData]:
        """
        Return all trade data of current backtesting result.
        """
        return list(self.trades.values())

    def get_all_orders(self) -> List[OrderData]:
        """
        Return all limit order data of current backtesting result.
        """
        return list(self.limit_orders.values())

    def get_all_daily_results(self) -> List["DailyResult"]:
        """
        Return all daily result data.
        """
        return list(self.daily_results.values())


class DailyResult:
    """"""

    def __init__(self, date: date, close_price: float):
        """"""
        # 这是报告的日期，表示交易日的日期。
        self.date = date
        # 这是当日的收盘价格，通常是股票或其他资产的收盘价格。
        self.close_price = close_price
        # 这是前一交易日的收盘价格，用于计算持仓盈亏等指标。
        self.pre_close = 0
        # 这是一个存储了在该交易日内发生的交易的列表。通常，每个交易都是一个对象，包含有关交易的详细信息，如交易价格、数量、方向等。
        self.trades = []
        # 是记录了在该交易日内发生的总交易次数。
        self.trade_count = 0
        # 这是当日的交易开始仓位，表示当日交易开始前的持仓数量。
        self.start_pos = 0
        # 结束仓位
        self.end_pos = 0
        # 这是交易的总成交额，表示在该交易日内的所有交易的成交额之和。
        self.turnover = 0
        # 这是交易的总手续费，表示在该交易日内的所有交易的手续费之和。
        self.commission = 0
        # 这是交易的总滑点成本，表示在该交易日内的所有交易的滑点成本之和。
        self.slippage = 0
        # 这是交易盈亏，表示在该交易日内的所有交易的盈亏总和。
        self.trading_pnl = 0
        # 这是持仓盈亏，表示在该交易日内的所有持仓在期间内的盈亏总和。
        self.holding_pnl = 0
        # 这是总盈亏，表示在该交易日内的所有交易和持仓的盈亏总和。
        self.total_pnl = 0
        # 这是净盈亏，表示在该交易日内的所有交易和持仓的盈亏总和，减去了手续费和滑点成本。
        self.net_pnl = 0

    def add_trade(self, trade: TradeData):
        """"""
        self.trades.append(trade)

    def calculate_pnl(
            self,
            pre_close: float,
            start_pos: float,
            size: float,
            rate: float,
            slippage: float,
            inverse: bool
    ):
        """"""
        # If no pre_close provided on the first day,
        # use value 1 to avoid zero division error
        if pre_close:
            self.pre_close = pre_close
        else:
            self.pre_close = 1

        # Holding pnl is the pnl from holding position at day start
        self.start_pos = start_pos
        self.end_pos = start_pos

        if not inverse:  # For normal contract
            self.holding_pnl = self.start_pos * (self.close_price - self.pre_close) * size
        else:  # For crypto currency inverse contract
            self.holding_pnl = self.start_pos * (1 / self.pre_close - 1 / self.close_price) * size

        # Trading pnl is the pnl from new trade during the day
        self.trade_count = len(self.trades)

        for trade in self.trades:
            if trade.direction == Direction.LONG:
                pos_change = float(trade.volume)
            else:
                pos_change = -float(trade.volume)

            self.end_pos += pos_change

            # For normal contract
            if not inverse:
                turnover = float(trade.volume) * size * float(trade.price)
                self.trading_pnl += pos_change * (self.close_price - float(trade.price)) * size
                self.slippage += float(trade.volume) * size * slippage
            # For cryptocurrency inverse contract
            else:
                turnover = float(trade.volume) * size / float(trade.price)
                self.trading_pnl += pos_change * (1 / float(trade.price) - 1 / self.close_price) * size
                self.slippage += float(trade.volume) * size * slippage / (float(trade.price) ** 2)

            self.turnover += turnover
            self.commission += turnover * rate

        # Net pnl takes account of commission and slippage cost
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission - self.slippage


def optimize(
        target_name: str,
        strategy_class: CtaTemplate,
        setting: dict,
        vt_symbol: str,
        interval: Interval,
        start: datetime,
        rate: float,
        slippage: float,
        size: float,
        pricetick: float,
        capital: int,
        end: datetime,
        mode: BacktestingMode,
        inverse: bool
):
    """
    Function for running in multiprocessing.pool
    """
    engine = BacktestingEngine()

    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=interval,
        start=start,
        rate=rate,
        slippage=slippage,
        size=size,
        pricetick=pricetick,
        capital=capital,
        end=end,
        mode=mode,
        inverse=inverse
    )

    engine.add_strategy(strategy_class, setting)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    statistics = engine.calculate_statistics(output=False)

    target_value = statistics[target_name]
    return (str(setting), target_value, statistics)


@lru_cache(maxsize=1000000)
def _ga_optimize(parameter_values: tuple):
    """"""
    setting = dict(parameter_values)

    result = optimize(
        ga_target_name,
        ga_strategy_class,
        setting,
        ga_vt_symbol,
        ga_interval,
        ga_start,
        ga_rate,
        ga_slippage,
        ga_size,
        ga_pricetick,
        ga_capital,
        ga_end,
        ga_mode,
        ga_inverse
    )
    return (result[1],)


def ga_optimize(parameter_values: list):
    """"""
    return _ga_optimize(tuple(parameter_values))


"""
由于装饰器 @lru_cache 的存在，如果多次调用 load_bar_data 函数使用相同的参数，那么第一次调用会执行数据库查询，
并将结果缓存起来。后续对相同参数的调用将直接从缓存中获取结果，而不会再次执行数据库查询，从而提高了查询效率。
这对于需要频繁查询相同数据的场景非常有用。
"""


@lru_cache(maxsize=999)
def load_bar_data(
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime
):
    """"""
    return database.load_bar_data(
        symbol, exchange, interval, start, end
    )


@lru_cache(maxsize=999)
def load_tick_data(
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime
):
    """"""
    return database.load_tick_data(
        symbol, exchange, start, end
    )


# GA related global value
ga_end = None
ga_mode = None
ga_target_name = None
ga_strategy_class = None
ga_setting = None
ga_vt_symbol = None
ga_interval = None
ga_start = None
ga_rate = None
ga_slippage = None
ga_size = None
ga_pricetick = None
ga_capital = None
