"""
General constant enums used in the trading platform.
"""

from enum import Enum
import pytz
from tzlocal import get_localzone_name

LOCAL_TZ = pytz.timezone(get_localzone_name())  # pytz.timezone("Asia/Shanghai")

"""
这个枚举类的作用是为了提供一个标准的方式来表示订单、交易或仓位的交易方向，使代码更具可读性和一致性。
例如，当你处理一个交易记录时，可以使用 Direction.LONG 表示这是一个多头交易，而使用 Direction.SHORT 表示这是一个空头交易。
同样，它还可以用于表示持仓的方向，以便更清晰地了解多头、空头或净头寸的情况。
"""


class Direction(Enum):
    """
    Direction of order/trade/position.
    """
    LONG = "Long"
    SHORT = "Short"
    # 枚举成员 NET 代表净头寸方向，通常表示净头寸，即多头仓位减去空头仓位。它的值是字符串 "Net"。
    NET = "Net"


"""
这个枚举类的作用是为了提供一个标准的方式来表示订单或交易的开平方向，使代码更具可读性和一致性。
例如，当你创建一个订单时，可以使用 Offset.OPEN 来表示这是一个开仓订单，而使用 Offset.CLOSE 表示这是一个平仓订单。
枚举类还可以防止使用无效或错误的开平方向值，因为它只允许从预定义的成员中选择一个值。
"""


class Offset(Enum):
    """
    Offset of order/trade.
    """
    NONE = ""
    # 枚举成员 OPEN 代表开仓方向，即买入或卖出以建立新仓位。它的值是字符串 "OPEN"。
    OPEN = "OPEN"
    # 枚举成员 CLOSE 代表平仓方向，即买入或卖出以平掉已有仓位。它的值是字符串 "CLOSE"。
    CLOSE = "CLOSE"
    # 枚举成员 CLOSETODAY 代表平今方向，即买入或卖出以平掉今天建立的仓位。它的值是字符串 "CLOSETODAY"。
    CLOSETODAY = "CLOSETODAY"
    # 枚举成员 CLOSEYESTERDAY 也代表平今方向，与 CLOSETODAY 具有相同的值，可能是一个错误，应该是 "CLOSEYESTERDAY"。
    CLOSEYESTERDAY = "CLOSEYESTERDAY"


"""
这个枚举类的作用是为了提供一个标准的方式来表示订单的不同状态，使代码更具可读性和一致性。当你处理订单时，
可以使用这些枚举成员来明确订单的状态，从而更容易进行逻辑控制和错误处理。
例如，你可以检查订单是否已全部成交，如果是，就执行一些特定的操作，如果不是，则执行其他操作。
"""


class Status(Enum):
    """
    Order status.
    """
    # SUBMITTING 代表订单正在提交的状态，通常表示订单已经被发送到交易所或交易系统，但还没有被完全接受或处理。它的值是字符串 "SUBMITTING"。
    SUBMITTING = "SUBMITTING"
    # 枚举成员 NOTTRADED 代表订单未成交的状态，即订单尚未被执行，没有任何成交记录。它的值是字符串 "NOTTRADED"
    NOTTRADED = "NOTTRADED"
    # 枚举成员 PARTTRADED 代表订单部分成交，即订单已经成交了一部分，但还有未成交的部分。它的值是字符串 "PARTTRADED"。
    PARTTRADED = "PARTTRADED"
    # 枚举成员 ALLTRADED 代表订单全部成交，即订单已经完全成交，没有未成交的部分。它的值是字符串 "ALLTRADED"。
    ALLTRADED = "ALLTRADED"
    # 枚举成员 CANCELLED 代表订单已被取消，即订单在执行之前被撤销。它的值是字符串 "CANCELLED"。
    CANCELLED = "CANCELLED"
    # 枚举成员 REJECTED 代表订单被拒绝，即订单由于某种原因未被接受或执行。它的值是字符串 "REJECTED"。
    REJECTED = "REJECTED"


class Product(Enum):
    """
    Product class.
    """
    SPOT = "SPOT"
    FUTURES = "FUTURES"
    OPTION = "OPTION"
    EQUITY = "EQUITY"
    INDEX = "INDEX"
    FOREX = "FOREX"
    ETF = "ETF"
    BOND = "BOND"
    WARRANT = "WARRANT"
    SPREAD = "SPREAD"
    FUND = "FUND"


"""
这个枚举类的作用是为了提供一个标准的方式来表示订单的不同类型，使代码更具可读性和一致性。
当你创建订单时，可以使用这些枚举成员来明确订单的类型，从而更容易理解订单的性质和行为。
"""


class OrderType(Enum):
    """
    Order type.
    """

    # 枚举成员 LIMIT 代表限价订单，即订单的价格是一个限定的价格。这意味着订单只有在达到或超过指定价格时才会成交。
    LIMIT = "LIMIT"
    # 枚举成员 TAKER 代表吃单（Taker）订单，即市价单，订单会以市场上当前的最优价格立即成交，不考虑价格。这通常是一种立即执行订单的类型。
    TAKER = "TAKER"
    # 枚举成员 MAKER 代表挂单（Maker）订单，即限价单，订单将等待市场价格达到或超过指定价格才会成交。这通常是一种需要等待特定价格的订单类型。
    MAKER = "MAKER"
    # 枚举成员 STOP 代表止损订单，即当市场价格达到或低于指定价格时，订单会自动触发并成交。这通常用于风险管理，以限制损失。
    STOP = "STOP"
    # 枚举成员 FAK 代表即时成交或取消（Fill or Kill，FAK）订单，即订单要么立即全部成交，要么在成交之前立即取消。
    FAK = "FAK"
    # 枚举成员 FOK 代表全部成交或取消（Fill or Kill，FOK）订单，即订单要么立即全部成交，要么完全取消。
    FOK = "FOK"
    # 枚举成员 RFQ 代表询价（Request for Quote，RFQ）订单，即订单需要先向市场询价，然后再决定是否成交。这通常用于某些场景中，如外汇交易。
    RFQ = "RFQ"


class OptionType(Enum):
    """
    Option type.
    """
    CALL = "CALL"
    PUT = "PUT"


class Exchange(Enum):
    """
    Exchange.
    """
    # CryptoCurrency
    FTX = "FTX"
    OKX = "OKX"  # previous OKEX
    BITFINEX = "BITFINEX"
    BINANCE = "BINANCE"
    BYBIT = "BYBIT"  # bybit.com

    # Special Function
    LOCAL = "LOCAL"  # For local generated data

    # Chinese
    CFFEX = "CFFEX"  # China Financial Futures Exchange
    SHFE = "SHFE"  # Shanghai Futures Exchange
    CZCE = "CZCE"  # Zhengzhou Commodity Exchange
    DCE = "DCE"  # Dalian Commodity Exchange
    INE = "INE"  # Shanghai International Energy Exchange
    SSE = "SSE"  # Shanghai Stock Exchange
    SZSE = "SZSE"  # Shenzhen Stock Exchange
    BSE = "BSE"  # Beijing Stock Exchange
    SGE = "SGE"  # Shanghai Gold Exchange
    WXE = "WXE"  # Wuxi Steel Exchange
    CFETS = "CFETS"  # CFETS Bond Market Maker Trading System
    XBOND = "XBOND"  # CFETS X-Bond Anonymous Trading System

    # Global
    SMART = "SMART"  # Smart Router for US stocks
    NYSE = "NYSE"  # New York Stock Exchnage
    NASDAQ = "NASDAQ"  # Nasdaq Exchange
    ARCA = "ARCA"  # ARCA Exchange
    EDGEA = "EDGEA"  # Direct Edge Exchange
    ISLAND = "ISLAND"  # Nasdaq Island ECN
    BATS = "BATS"  # Bats Global Markets
    IEX = "IEX"  # The Investors Exchange
    NYMEX = "NYMEX"  # New York Mercantile Exchange
    COMEX = "COMEX"  # COMEX of CME
    GLOBEX = "GLOBEX"  # Globex of CME
    IDEALPRO = "IDEALPRO"  # Forex ECN of Interactive Brokers
    CME = "CME"  # Chicago Mercantile Exchange
    ICE = "ICE"  # Intercontinental Exchange
    SEHK = "SEHK"  # Stock Exchange of Hong Kong
    HKFE = "HKFE"  # Hong Kong Futures Exchange
    SGX = "SGX"  # Singapore Global Exchange
    CBOT = "CBT"  # Chicago Board of Trade
    CBOE = "CBOE"  # Chicago Board Options Exchange
    CFE = "CFE"  # CBOE Futures Exchange
    DME = "DME"  # Dubai Mercantile Exchange
    EUREX = "EUX"  # Eurex Exchange
    APEX = "APEX"  # Asia Pacific Exchange
    LME = "LME"  # London Metal Exchange
    BMD = "BMD"  # Bursa Malaysia Derivatives
    TOCOM = "TOCOM"  # Tokyo Commodity Exchange
    EUNX = "EUNX"  # Euronext Exchange
    KRX = "KRX"  # Korean Exchange
    OTC = "OTC"  # OTC Product (Forex/CFD/Pink Sheet Equity)
    IBKRATS = "IBKRATS"  # Paper Trading Exchange of IB


class Currency(Enum):
    """
    Currency.
    """
    USD = "USD"
    HKD = "HKD"
    CNY = "CNY"


class Interval(Enum):
    """
    Interval of bar data.

    """
    MINUTE = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"

    HOUR = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"

    DAILY = "1d"
    DAILY_3 = "3d"

    WEEKLY = "1w"
    MONTH = "1M"

    TICK = "tick"
