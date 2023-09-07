"""
Defines constants and objects used in CtaStrategy App.
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict
from decimal import Decimal

from howtrader.trader.constant import Direction, Offset, Interval

APP_NAME = "CtaStrategy"
STOPORDER_PREFIX = "STOP"


class StopOrderStatus(Enum):
    WAITING = "WAITING"
    CANCELLED = "CANCELLED"
    TRIGGERED = "TRIGGERED"


class EngineType(Enum):
    # 枚举成员通常用于表示一组有限地可能状态或选项，这里的 LIVE 可以表示引擎类型为实盘交易。
    LIVE = "LIVE"
    # 这行代码定义了另一个枚举成员 BACKTESTING，它的值被设置为字符串 "BACKTESTING"。这个枚举成员可以表示引擎类型为回测。
    BACKTESTING = "BACKTESTING"


class BacktestingMode(Enum):
    BAR = 1
    TICK = 2


@dataclass
class StopOrder:
    vt_symbol: str
    direction: Direction
    offset: Offset
    price: Decimal
    volume: Decimal
    stop_orderid: str
    strategy_name: str
    datetime: datetime
    lock: bool = False
    net: bool = False
    vt_orderids: list = field(default_factory=list)
    status: StopOrderStatus = StopOrderStatus.WAITING


EVENT_CTA_LOG = "eCtaLog"
EVENT_CTA_STRATEGY = "eCtaStrategy"
EVENT_CTA_STOPORDER = "eCtaStopOrder"

INTERVAL_DELTA_MAP: Dict[Interval, timedelta] = {
    Interval.TICK: timedelta(milliseconds=1),
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta(days=1),
}
