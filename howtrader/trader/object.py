"""
Basic data structure used for general trading function in the trading platform.
"""

from dataclasses import dataclass
from datetime import datetime
from logging import INFO
from decimal import Decimal
import pandas as pd
from .constant import Direction, Exchange, Interval, Offset, Status, Product, OptionType, OrderType

ACTIVE_STATUSES = {Status.SUBMITTING, Status.NOTTRADED, Status.PARTTRADED}  # define the active status set.


@dataclass
class BaseData:
    """
    Any data object needs a gateway_name as source
    and should inherit base data.
    """

    gateway_name: str


@dataclass
class TickData(BaseData):
    """
    Tick data contains information about:
        * last trade in market
        * orderbook snapshot
        * intraday market statistics.
    """

    symbol: str
    exchange: Exchange
    datetime: datetime

    name: str = ""
    # 当前 Tick 时刻的成交量，表示该时刻的成交数量。
    volume: float = 0
    # 当前 Tick 时刻的成交额，表示该时刻的成交总价值。
    turnover: float = 0
    # 当前 Tick 时刻的持仓量，表示尚未平仓的合约数量。
    open_interest: float = 0
    # 最新成交价，表示最后一笔成交的价格。
    last_price: float = 0
    # 最后一笔成交的成交量。
    last_volume: float = 0
    # 当前交易品种的涨停价格。
    limit_up: float = 0
    # 当前交易品种的跌停价格。
    limit_down: float = 0
    # 当前 Tick 时刻的开盘价，即该时刻的第一笔成交的价格。
    open_price: float = 0
    # 当前 Tick 时刻的最高价，表示该时刻的最高成交价格。
    high_price: float = 0
    # 当前 Tick 时刻的最低价，表示该时刻的最低成交价格。
    low_price: float = 0
    # 当前 Tick 时刻的昨收盘价，即上一个交易日的收盘价格。
    pre_close: float = 0
    # 当前的买入报价，通常有五档报价，这些变量分别表示第 1 到第 5 档的买入价格。
    bid_price_1: float = 0
    bid_price_2: float = 0
    bid_price_3: float = 0
    bid_price_4: float = 0
    bid_price_5: float = 0
    # 当前的卖出报价，通常有五档报价，这些变量分别表示第 1 到第 5 档的卖出价格。
    ask_price_1: float = 0
    ask_price_2: float = 0
    ask_price_3: float = 0
    ask_price_4: float = 0
    ask_price_5: float = 0
    # 当前的买入报价对应的成交量，表示每一档买入报价对应的成交数量。
    bid_volume_1: float = 0
    bid_volume_2: float = 0
    bid_volume_3: float = 0
    bid_volume_4: float = 0
    bid_volume_5: float = 0
    # 当前的卖出报价对应的成交量，表示每一档卖出报价对应的成交数量。
    ask_volume_1: float = 0
    ask_volume_2: float = 0
    ask_volume_3: float = 0
    ask_volume_4: float = 0
    ask_volume_5: float = 0

    localtime: datetime = None

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class BarData(BaseData):
    """
    Candlestick bar data of a certain trading period.
    """

    symbol: str
    exchange: Exchange
    datetime: datetime

    interval: Interval = None
    volume: float = 0
    turnover: float = 0
    open_interest: float = 0
    open_price: float = 0
    high_price: float = 0
    low_price: float = 0
    close_price: float = 0

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class OrderData(BaseData):
    """
    Order data contains information for tracking lastest status
    of a specific order.
    """

    symbol: str
    exchange: Exchange
    orderid: str
    # 订单类型，默认是限价单（OrderType.LIMIT），还可以是市价单等。
    type: OrderType = OrderType.LIMIT
    # 单的买卖方向，可以是多头（Direction.LONG）、空头（Direction.SHORT）或者不确定。
    direction: Direction = None
    # 订单的开平仓标志，可以是开仓（Offset.OPEN）、平仓（Offset.CLOSE）或者不确定。
    offset: Offset = Offset.NONE
    # 订单的价格，如果是市价单则为0。
    price: Decimal = Decimal("0")
    volume: Decimal = Decimal("0")
    # 订单已成交的数量。
    traded: Decimal = Decimal("0")

    traded_price: Decimal = Decimal("0")
    # 订单的状态，例如提交中（Status.SUBMITTING）、已撤销（Status.CANCELLED）等。
    status: Status = Status.SUBMITTING
    # 订单的创建时间。
    datetime: datetime = datetime.now()
    # 订单的最后更新时间。
    update_time: datetime = datetime.now()
    # 订单的参考信息。
    reference: str = ""
    # 如果订单被拒绝，这里记录拒绝的原因。
    rejected_reason: str = ""  # Order Rejected Reason

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_orderid: str = f"{self.gateway_name}.{self.orderid}"

    def is_active(self) -> bool:
        """
        Check if the order is active.
        """
        return self.status in ACTIVE_STATUSES

    def create_cancel_request(self) -> "CancelRequest":
        """
        Create cancel request object from order.
        """
        return CancelRequest(
            orderid=self.orderid, symbol=self.symbol, exchange=self.exchange
        )

    def create_query_request(self) -> "OrderQueryRequest":
        """
        Create OrderQueryRequest for updating the order when the order hasn't updated for a long time.
        you can config the update interval in vt_setting.json file, config the value "update_interval"
        """
        return OrderQueryRequest(orderid=self.orderid, symbol=self.symbol, exchange=self.exchange)


@dataclass
class TradeData(BaseData):
    """
    Trade data contains information of a fill of an order. One order
    can have several trade fills.
    """
    """
    `TradeData` 和 `TickData` 都是在量化交易中用于记录市场数据和交易信息的数据结构，但它们之间有一些关键的区别：

    1. **记录的信息不同**：
        - `TickData`：表示市场上的每一笔报价，通常包括买入价、卖出价、成交价、成交量等信息。`TickData` 记录了市场的瞬时状态，可以用来观察市场的细微波动。
        - `TradeData`：表示一笔实际的成交交易，包括交易的时间、方向（买入或卖出）、成交价、成交量等信息。`TradeData` 记录了交易的具体细节，通常是在订单成交后生成。
    
    2. **应用场景不同**：
        - `TickData`：主要用于短期交易策略和高频交易策略，因为它提供了更频繁的市场数据更新，可以捕捉市场瞬时的价格变化。
        - `TradeData`：主要用于记录实际的交易事件，对于风险管理、交易分析、交易报告和账户监控非常重要。它记录了每笔成交的关键信息。
    
    3. **数据频率不同**：
        - `TickData`：通常以较高的频率更新，每次市场报价变动都会生成新的 `TickData`。
        - `TradeData`：以实际交易发生的频率生成，一般来说，只有在订单成交时才会生成 `TradeData`。
    
    4. **信息粒度不同**：
        - `TickData`：提供更精细的市场数据，可以用于分析市场深度、盘口变化等。
        - `TradeData`：提供更高层次的交易信息，可以用于分析成交价格、成交量、交易方向等。
    
    总的来说，`TickData` 主要用于市场数据分析和短期交易策略，而 `TradeData` 主要用于交易事件的记录和交易策略的管理。
    在实际应用中，这两者可以结合使用，以满足不同交易策略和需求的数据获取和分析。Ï
    """
    # 交易的标的物，通常是一个金融产品的代码，比如股票代码或期货合约代码
    symbol: str
    # 交易所或市场的名称，标识了交易发生的地点。
    exchange: Exchange
    """
    订单id
    """
    orderid: str
    # 交易的唯一标识符或编号。
    tradeid: str = ""
    direction: Direction = None
    # 交易的开平仓类型，可以是开仓、平仓等。
    offset: Offset = Offset.NONE
    # 交易的价格，即成交价。
    price: Decimal = Decimal("0")
    # 交易的数量，表示成交的数量。
    volume: Decimal = Decimal("0")
    # 交易发生的时间戳。
    datetime: datetime = None

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_orderid: str = f"{self.gateway_name}.{self.orderid}"
        self.vt_tradeid: str = f"{self.gateway_name}.{self.tradeid}"


@dataclass
class PositionData(BaseData):
    """
    Positon data is used for tracking each individual position holding.
    这个数据类通常用于跟踪和管理某个特定交易品种的仓位信息，包括持仓数量、盈亏情况等。
    """

    symbol: str
    exchange: Exchange
    # 交易方向，表示是多头仓位（Direction.LONG）还是空头仓位（Direction.SHORT）。
    direction: Direction
    # 仓位数量，表示持有的合约或股票数量。
    volume: float = 0
    # 冻结数量，表示被冻结的仓位数量，通常是由于下单或其他原因而被暂时冻结的数量。
    frozen: float = 0
    # 仓位平均价格，表示持有该仓位的平均成本价格。
    price: float = 0
    # 强平价格，表示如果市场价格达到这个价格，该仓位将被自动平仓以限制损失。
    liquidation_price: float = 0
    # 杠杆倍数，表示该仓位的杠杆比例。
    leverage: int = 1
    # 仓位的盈亏情况，表示持有该仓位时的浮动盈亏
    pnl: float = 0
    # 昨仓数量，表示昨天持有的仓位数量。
    yd_volume: float = 0
    # 这个数据类的初始化函数 __post_init__ 主要用于设置一些辅助属性：
    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        # 使用 vt_symbol 和 direction 构建的字符串，表示仓位的唯一标识。
        self.vt_positionid: str = f"{self.vt_symbol}.{self.direction.value}"


@dataclass
class AccountData(BaseData):
    """
    Account data contains information about balance, frozen and
    available.
    """

    accountid: str

    balance: float = 0
    frozen: float = 0

    def __post_init__(self) -> None:
        """"""
        self.available: float = self.balance - self.frozen
        self.vt_accountid: str = f"{self.gateway_name}.{self.accountid}"


@dataclass
class LogData(BaseData):
    """
    Log data is used for recording log messages on GUI or in log files.
    """

    msg: str
    level: int = INFO

    def __post_init__(self) -> None:
        """"""
        self.time: datetime = datetime.now()


@dataclass
class ContractData(BaseData):
    """
    Contract data contains basic information about each contract traded.
    这个类定义了一个名为 ContractData 的数据类，用于表示交易合约（或产品）的基本信息。
    """


    symbol: str
    exchange: Exchange
    name: str
    product: Product
    #  合约规模，表示每个合约的数量或份额。
    size: Decimal
    # 价格最小变动单位，表示合约价格的最小变动。
    pricetick: Decimal
    # 最小交易金额，表示下单的最小金额要求，通常是 价格 * 数量 >= 最小金额。
    min_notional: Decimal = Decimal("1")  # order's value, price * amount >= min_notional
    # 最小下单数量，表示下单的最小数量要求，某些交易所需要指定最小下单数量。
    min_size: Decimal = Decimal("1")  # place minimum order size, 最小的下单数量，okx使用.
    # 合约最小交易数量，表示交易合约的最小交易数量要求。
    min_volume: Decimal = Decimal("1")  # minimum trading volume of the contract, 下单精度要求.
    # 是否支持止损单，表示交易所是否支持设置止损单。
    stop_supported: bool = False  # whether server supports stop order
    # 是否使用净仓位，表示交易网关是否使用净仓位模式。
    net_position: bool = False  # whether gateway uses net position volume
    # 是否提供历史数据，表示交易网关是否提供该合约的历史数据。
    history_data: bool = False  # whether gateway provides bar history data

    option_strike: float = 0
    option_underlying: str = ""  # vt_symbol of underlying contract
    option_type: OptionType = None
    option_listed: datetime = None
    option_expiry: datetime = None
    option_portfolio: str = ""
    option_index: str = ""  # for identifying options with same strike price

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class PremiumRateData(BaseData):
    """
    PremiumRate
    """
    symbol: str
    exchange: Exchange
    next_funding_time: datetime
    updated_datetime: datetime
    last_funding_rate: Decimal = Decimal("0")
    interest_rate: Decimal = Decimal("0")

    def __post_init__(self):
        """"""
        self.vt_symbol = f"{self.symbol}.{self.exchange.value}"


@dataclass
class OriginalKlineData(BaseData):
    """
    exchange kline data
    """
    symbol: str
    exchange: Exchange
    interval: Interval
    kline_df: pd.DataFrame
    klines: list

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class QuoteData(BaseData):
    """
    Quote data contains information for tracking lastest status
    of a specific quote.

    用于表示行情数据，通常用于跟踪特定行情的最新状态。以下是这个类的各个成员变量：
    """

    symbol: str
    exchange: Exchange
    # 表示行情的唯一标识符，通常是一个字符串。
    quoteid: str
    # 表示买入价格，通常是一个浮点数，表示当前的买入报价。
    bid_price: float = 0.0
    # 表示买入数量，通常是一个整数，表示当前的买入报价对应的数量。
    bid_volume: int = 0
    # 表示卖出价格，通常是一个浮点数，表示当前的卖出报价。
    ask_price: float = 0.0
    # 表示卖出数量，通常是一个整数，表示当前的卖出报价对应的数量。
    ask_volume: int = 0
    bid_offset: Offset = Offset.NONE
    ask_offset: Offset = Offset.NONE
    # 表示行情状态，通常是一个枚举值，用于表示行情的状态，比如正在提交、已成交等。
    status: Status = Status.SUBMITTING
    datetime: datetime = None
    # 表示行情数据的参考信息，通常是一个字符串，用于保存与行情数据相关的参考信息。
    reference: str = ""

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_quoteid: str = f"{self.gateway_name}.{self.quoteid}"

    def is_active(self) -> bool:
        """
        Check if the quote is active.
        """
        return self.status in ACTIVE_STATUSES

    def create_cancel_request(self) -> "CancelRequest":
        """
        Create cancel request object from quote.
        """
        req: CancelRequest = CancelRequest(
            orderid=self.quoteid, symbol=self.symbol, exchange=self.exchange
        )
        return req


@dataclass
class SubscribeRequest:
    """
    Request sending to specific gateway for subscribing tick data update.
    """

    symbol: str
    exchange: Exchange

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class OrderRequest:
    """
    Request sending to specific gateway for creating a new order.
    """

    symbol: str
    exchange: Exchange
    direction: Direction
    type: OrderType
    volume: Decimal
    price: Decimal = Decimal("0")
    offset: Offset = Offset.NONE
    reference: str = ""

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"

    def create_order_data(self, orderid: str, gateway_name: str) -> OrderData:
        """
        Create order data from request.
        """
        order: OrderData = OrderData(
            symbol=self.symbol,
            exchange=self.exchange,
            orderid=orderid,
            type=self.type,
            direction=self.direction,
            offset=self.offset,
            price=self.price,
            volume=self.volume,
            reference=self.reference,
            gateway_name=gateway_name,
        )
        return order


@dataclass
class CancelRequest:
    """
    Request sending to specific gateway for canceling an existing order.
    """

    orderid: str
    symbol: str
    exchange: Exchange

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class OrderQueryRequest:
    """
    Query the existing order status
    """
    orderid: str
    symbol: str
    exchange: Exchange

    def __post_init__(self):
        """"""
        self.vt_symbol = f"{self.symbol}.{self.exchange.value}"


@dataclass
class HistoryRequest:
    """
    Request sending to specific gateway for querying history data.
    """

    symbol: str
    exchange: Exchange
    start: datetime
    end: datetime = None
    interval: Interval = None
    limit: int = 1000

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class QuoteRequest:
    """
    Request sending to specific gateway for creating a new quote.
    """

    symbol: str
    exchange: Exchange
    bid_price: float
    bid_volume: int
    ask_price: float
    ask_volume: int
    bid_offset: Offset = Offset.NONE
    ask_offset: Offset = Offset.NONE
    reference: str = ""

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"

    def create_quote_data(self, quoteid: str, gateway_name: str) -> QuoteData:
        """
        Create quote data from request.
        """
        quote: QuoteData = QuoteData(
            symbol=self.symbol,
            exchange=self.exchange,
            quoteid=quoteid,
            bid_price=self.bid_price,
            bid_volume=self.bid_volume,
            ask_price=self.ask_price,
            ask_volume=self.ask_volume,
            bid_offset=self.bid_offset,
            ask_offset=self.ask_offset,
            reference=self.reference,
            gateway_name=gateway_name,
        )
        return quote


class GridPositionCalculator(object):
    """
    用来计算网格头寸的平均价格
    Use for calculating the grid position's average price.
    :param grid_step: 网格间隙.
    """

    def __init__(self, grid_step: float = 1.0):
        # 初始化self.pos为Decimal类型的0，表示当前头寸数量。
        self.pos: Decimal = Decimal("0")
        # 初始化self.avg_price为Decimal类型的0，表示当前头寸的平均价格。
        self.avg_price: Decimal = Decimal("0")
        # 将传入的grid_step参数转换为Decimal类型并赋值给self.grid_step，表示网格间隙。
        self.grid_step: Decimal = Decimal(str(grid_step))

    def update_position(self, trade: TradeData):

        previous_pos = self.pos
        previous_avg = self.avg_price
        # 如果交易方向是多头（买入）。
        if trade.direction == Direction.LONG:
            # 将当前头寸数量加上交易的成交量，表示多头交易头寸的增加。
            self.pos += trade.volume
            # 如果当前头寸数量为0，将平均价格设为0。
            if self.pos == Decimal("0"):
                self.avg_price = Decimal("0")
            else:
                # 如果之前的头寸数量为0，将平均价格设置为当前交易的价格。
                if previous_pos == Decimal("0"):
                    self.avg_price = trade.price

                elif previous_pos > 0:
                    # 算多头头寸的平均价格，考虑了之前的头寸数量、平均价格和当前交易的成交量。
                    self.avg_price = (previous_pos * previous_avg + trade.volume * trade.price) / abs(self.pos)
                # 如果之前的头寸数量和当前头寸数量都为负数，表示之前是空头头寸。
                elif previous_pos < 0 and self.pos < 0:
                    # 计算空头头寸的平均价格，考虑了之前的平均价格、当前交易的成交量和网格间隙。
                    self.avg_price = (previous_avg * abs(self.pos) - (
                            trade.price - previous_avg) * trade.volume - trade.volume * self.grid_step) / abs(
                        self.pos)
                # 如果之前的头寸数量为负数而当前头寸数量为正数，表示进行了多头平仓和空头开仓。
                elif previous_pos < 0 < self.pos:
                    self.avg_price = trade.price
        # 如果交易方向是空头（卖出），与多头情况类似，但处理的是空头的头寸。
        elif trade.direction == Direction.SHORT:
            self.pos -= trade.volume

            if self.pos == Decimal("0"):
                self.avg_price = Decimal("0")
            else:

                if previous_pos == Decimal("0"):
                    self.avg_price = trade.price

                elif previous_pos < 0:
                    self.avg_price = (abs(previous_pos) * previous_avg + trade.volume * trade.price) / abs(self.pos)

                elif previous_pos > 0 and self.pos > 0:
                    self.avg_price = (previous_avg * self.pos - (
                            trade.price - previous_avg) * trade.volume + trade.volume * self.grid_step) / abs(
                        self.pos)

                elif previous_pos > 0 > self.pos:
                    self.avg_price = trade.price
