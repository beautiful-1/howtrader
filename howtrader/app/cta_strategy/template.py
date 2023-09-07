from abc import ABC
from copy import copy
from typing import Any, Callable, List, Optional

from howtrader.trader.constant import Interval, Direction, Offset
from howtrader.trader.object import BarData, TickData, OrderData, TradeData, PositionData
from howtrader.trader.utility import virtual

from .base import StopOrder, EngineType
from decimal import Decimal


class CtaTemplate(ABC):
    """
    这个类是一个名为 CtaTemplate 的策略模板，用于开发和实现CTA（Commodity Trading Advisor）策略。
    CTA策略通常应用于期货市场，它们根据市场数据生成交易信号，并执行买入和卖出操作。
    """

    author: str = ""
    # 这个属性通常用于存储策略类的参数名列表。策略类的参数是可以在实例化策略对象时传递的配置项，允许用户在不修改策略代码的情况下调整策略的行为。
    parameters: list = []
    # 这个属性通常用于存储策略类的状态变量名列表。状态变量是在策略运行过程中记录策略内部状态的变量，用于监视和调试策略的行为。
    variables: list = []

    def __init__(
            self,
            cta_engine: Any,
            strategy_name: str,
            vt_symbol: str,
            setting: dict,
    ) -> None:
        """"""
        self.cta_engine: "CtaEngine" = cta_engine
        self.strategy_name: str = strategy_name
        self.vt_symbol: str = vt_symbol
        # 表示策略是否已初始化，默认为 False。
        self.inited: bool = False
        # 表示策略是否正在交易，默认为 False。
        self.trading: bool = False
        # 表示策略当前的仓位，默认为0。
        self.pos: Decimal = Decimal("0")

        # Copy a new variables list here to avoid duplicate insert when multiple
        # strategy instances are created with the same strategy class.
        self.variables = copy(self.variables)

        # 包含了策略中的一些变量，如 inited、trading、pos
        self.variables.insert(0, "inited")
        self.variables.insert(1, "trading")
        self.variables.insert(2, "pos")

        self.update_setting(setting)

    """
    用于更新策略的设置参数。将设置参数中的值赋给策略对象的属性，以便在策略逻辑中使用。
    """

    def update_setting(self, setting: dict) -> None:
        """
        Update strategy parameter wtih value in setting dict.
        """
        for name in self.parameters:
            if name in setting:
                setattr(self, name, setting[name])

    """
    用于获取策略类的默认参数字典。
    这个类方法的作用是使得你可以在不创建策略实例的情况下获取策略类的默认参数，从而可以在需要的时候使用这些默认参数来初始化策略实例。
    例如，你可以通过 StrategyClass.get_class_parameters() 来获取某个策略类的默认参数。
    """

    # 这是一个装饰器，用于声明下面的函数是一个类方法，而不是实例方法。类方法是与类相关联的方法，而不是与类的实例相关联的方法。它可以通过类名调用，而无需创建类的实例。
    @classmethod
    def get_class_parameters(cls) -> dict:
        """
        Get default parameters dict of strategy class.
        """
        class_parameters: dict = {}
        for name in cls.parameters:
            class_parameters[name] = getattr(cls, name)
        return class_parameters

    """
    用于获取策略实例的当前参数字典。
    """

    def get_parameters(self) -> dict:
        """
        Get strategy parameters dict.
        """
        strategy_parameters: dict = {}
        for name in self.parameters:
            strategy_parameters[name] = getattr(self, name)
        return strategy_parameters

    def get_variables(self) -> dict:
        """
        Get strategy variables dict.
        """
        strategy_variables: dict = {}
        for name in self.variables:
            strategy_variables[name] = getattr(self, name)
        return strategy_variables

    def get_data(self) -> dict:
        """
        Get strategy data.
        """
        strategy_data: dict = {
            "strategy_name": self.strategy_name,
            "vt_symbol": self.vt_symbol,
            "class_name": self.__class__.__name__,
            "author": self.author,
            "parameters": self.get_parameters(),
            "variables": self.get_variables(),
        }
        return strategy_data

    @virtual
    def on_init(self) -> None:
        """
        Callback when strategy is inited.
        """
        pass

    @virtual
    def on_start(self) -> None:
        """
        Callback when strategy is started.
        """
        pass

    @virtual
    def on_stop(self) -> None:
        """
        Callback when strategy is stopped.
        """
        pass

    @virtual
    def on_tick(self, tick: TickData) -> None:
        """
        Callback of new tick data update.
        """
        pass

    @virtual
    def on_bar(self, bar: BarData) -> None:
        """
        Callback of new bar data update.
        """
        pass

    @virtual
    def on_trade(self, trade: TradeData) -> None:
        """
        Callback of new trade data update.
        """
        pass

    """
    @virtual
    这是一个装饰器，用于标记方法为虚拟方法。
    虚拟方法是一个在基类中定义但在派生类中需要重新实现的方法。
    在这里，on_order 被标记为虚拟方法，表示它必须在派生类中重新实现，以根据具体策略的需求处理新的交易数据。
    """

    @virtual
    def on_order(self, order: OrderData) -> None:
        """
        Callback of new order data update.
        """
        pass

    @virtual
    def on_stop_order(self, stop_order: StopOrder) -> None:
        """
        Callback of stop order update.
        """
        pass

    def buy(
            self,
            price: Decimal,
            volume: Decimal,
            stop: bool = False,
            lock: bool = False,
            net: bool = False,
            maker: bool = False
    ) -> list:
        """
        Send buy order to open a long position.
        """
        return self.send_order(
            Direction.LONG,
            Offset.OPEN,
            price,
            volume,
            stop,
            lock,
            net,
            maker
        )

    def sell(
            self,
            price: Decimal,
            volume: Decimal,
            stop: bool = False,
            lock: bool = False,
            net: bool = False,
            maker: bool = False
    ) -> list:
        """
        Send sell order to close a long position.
        """
        return self.send_order(
            Direction.SHORT,
            Offset.CLOSE,
            price,
            volume,
            stop,
            lock,
            net,
            maker
        )

    """
    方法名为 short，它是策略类中的一个操作方法，用于发送一个开空仓的订单，以建立一个空头（卖出）仓位。
    """

    def short(
            self,
            price: Decimal,
            volume: Decimal,
            stop: bool = False,
            lock: bool = False,
            net: bool = False,
            maker: bool = False
    ) -> list:
        """
        Send short order to open as short position.
        """
        return self.send_order(
            Direction.SHORT,
            Offset.OPEN,
            price,
            volume,
            stop,
            lock,
            net,
            maker
        )

    """
    策略类中的一个操作方法，用于发送一个平仓订单，以平掉（即关闭）一个已有的卖空仓位。
    """

    def cover(
            self,
            price: Decimal,
            volume: Decimal,
            stop: bool = False,
            lock: bool = False,
            net: bool = False,
            maker: bool = False
    ) -> list:
        """
        Send cover order to close a short position.
        """
        return self.send_order(
            Direction.LONG,
            Offset.CLOSE,
            price,
            volume,
            stop,
            lock,
            net,
            maker
        )

    def send_order(
            self,
            direction: Direction,
            # 这是方法的第三个参数，用于指定订单的开平方向。参数名为 offset，类型为 Offset，通常用于表示订单是开仓还是平仓
            offset: Offset,
            price: Decimal,
            volume: Decimal,
            stop: bool = False,
            lock: bool = False,
            net: bool = False,
            maker: bool = False
    ) -> list:
        """
        Send a new order.
        """
        if self.trading:
            vt_orderids: list = self.cta_engine.send_order(
                self, direction, offset, price, volume, stop, lock, net, maker
            )
            return vt_orderids
        else:
            return []

    def cancel_order(self, vt_orderid: str) -> None:
        """
        Cancel an existing order.
        """
        if self.trading:
            self.cta_engine.cancel_order(self, vt_orderid)

    def query_order(self, vt_orderid: str) -> None:
        """query order"""
        self.cta_engine.query_order(vt_orderid)

    def cancel_all(self) -> None:
        """
        Cancel all orders sent by strategy.
        """
        if self.trading:
            self.cta_engine.cancel_all(self)

    def write_log(self, msg: str) -> None:
        """
        Write a log message.
        """
        self.cta_engine.write_log(msg, self)

    def get_engine_type(self) -> EngineType:
        """
        Return whether the cta_engine is backtesting or live trading.
        """
        return self.cta_engine.get_engine_type()

    def get_pricetick(self) -> float:
        """
        Return pricetick data of trading contract.
        """
        return self.cta_engine.get_pricetick(self)

    def get_position(self, vt_positionid=None) -> Optional[PositionData]:
        """"""
        if vt_positionid is None:
            vt_positionid = self.vt_symbol + '.NET'
        return self.cta_engine.get_position(vt_positionid)

    def load_bar(
            self,
            days: int,
            interval: Interval = Interval.MINUTE,
            callback: Callable = None,
            use_database: bool = False
    ) -> None:
        """
        Load historical bar data for initializing strategy.
        """
        if not callback:
            callback: Callable = self.on_bar

        bars: List[BarData] = self.cta_engine.load_bar(
            self.vt_symbol,
            days,
            interval,
            callback,
            use_database
        )

        for bar in bars:
            callback(bar)

    def query_latest_kline(self, interval: Interval, limit=1000):
        """query the latest kline, to get the data you need to register event for getting the callback data
        the event_id == EVENT_ORIGINAL_KLINE + vt_symbol
        self.cta_engine.event_engine.register(EVENT_ORIGINAL_KLINE + vt_symbol, callback_function)
        """
        self.cta_engine.query_latest_kline(self.vt_symbol, interval, limit)

    def load_tick(self, days: int) -> None:
        """
        Load historical tick data for initializing strategy.
        """
        ticks: List[TickData] = self.cta_engine.load_tick(self.vt_symbol, days, self.on_tick)

        for tick in ticks:
            self.on_tick(tick)

    def put_event(self) -> None:
        """
        Put an strategy data event for ui update.
        """
        if self.inited:
            self.cta_engine.put_strategy_event(self)

    def send_email(self, msg) -> None:
        """
        Send email to default receiver.
        """
        if self.inited:
            self.cta_engine.send_email(msg, self)

    def sync_data(self) -> None:
        """
        Sync strategy variables value into disk storage.
        """
        if self.trading:
            self.cta_engine.sync_strategy_data(self)


class CtaSignal(ABC):
    """"""

    def __init__(self) -> None:
        """"""
        self.signal_pos = 0

    @virtual
    def on_tick(self, tick: TickData) -> None:
        """
        Callback of new tick data update.
        """
        pass

    @virtual
    def on_bar(self, bar: BarData) -> None:
        """
        Callback of new bar data update.
        """
        pass

    def set_signal_pos(self, pos) -> None:
        """"""
        self.signal_pos = pos

    def get_signal_pos(self) -> Any:
        """"""
        return self.signal_pos


class TargetPosTemplate(CtaTemplate):
    """"""
    tick_add = 1

    last_tick: TickData = None
    last_bar: BarData = None
    target_pos: Decimal = Decimal("0")

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting) -> None:
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.active_orderids: list = []
        self.cancel_orderids: list = []

        self.variables.append("target_pos")

    @virtual
    def on_tick(self, tick: TickData) -> None:
        """
        Callback of new tick data update.
        """
        self.last_tick = tick

        if self.trading:
            self.trade()

    @virtual
    def on_bar(self, bar: BarData) -> None:
        """
        Callback of new bar data update.
        """
        self.last_bar = bar

    @virtual
    def on_order(self, order: OrderData) -> None:
        """
        Callback of new order data update.
        """
        vt_orderid: str = order.vt_orderid

        if not order.is_active():
            if vt_orderid in self.active_orderids:
                self.active_orderids.remove(vt_orderid)

            if vt_orderid in self.cancel_orderids:
                self.cancel_orderids.remove(vt_orderid)

    def check_order_finished(self) -> bool:
        """"""
        if self.active_orderids:
            return False
        else:
            return True

    def set_target_pos(self, target_pos) -> None:
        """"""
        self.target_pos = target_pos
        self.trade()

    def trade(self) -> None:
        """"""
        if not self.check_order_finished():
            self.cancel_old_order()
        else:
            self.send_new_order()

    def cancel_old_order(self) -> None:
        """"""
        for vt_orderid in self.active_orderids:
            if vt_orderid not in self.cancel_orderids:
                self.cancel_order(vt_orderid)
                self.cancel_orderids.append(vt_orderid)

    def send_new_order(self) -> None:
        """"""
        pos_change = self.target_pos - self.pos
        if not pos_change:
            return

        long_price = 0
        short_price = 0

        if self.last_tick:
            if pos_change > 0:
                long_price = self.last_tick.ask_price_1 + self.tick_add
                if self.last_tick.limit_up:
                    long_price = min(long_price, self.last_tick.limit_up)
            else:
                short_price = self.last_tick.bid_price_1 - self.tick_add
                if self.last_tick.limit_down:
                    short_price = max(short_price, self.last_tick.limit_down)

        else:
            if pos_change > 0:
                long_price = self.last_bar.close_price + self.tick_add
            else:
                short_price = self.last_bar.close_price - self.tick_add

        if self.get_engine_type() == EngineType.BACKTESTING:
            if pos_change > 0:
                vt_orderids: list = self.buy(Decimal(long_price), abs(pos_change))
            else:
                vt_orderids: list = self.short(Decimal(short_price), abs(pos_change))
            self.active_orderids.extend(vt_orderids)

        else:
            if self.active_orderids:
                return

            if pos_change > 0:
                if self.pos < 0:
                    if pos_change < abs(self.pos):
                        vt_orderids: list = self.cover(Decimal(long_price), pos_change)
                    else:
                        vt_orderids: list = self.cover(Decimal(long_price), abs(self.pos))
                else:
                    vt_orderids: list = self.buy(Decimal(long_price), abs(pos_change))
            else:
                if self.pos > 0:
                    if abs(pos_change) < self.pos:
                        vt_orderids: list = self.sell(Decimal(short_price), abs(pos_change))
                    else:
                        vt_orderids: list = self.sell(Decimal(short_price), abs(self.pos))
                else:
                    vt_orderids: list = self.short(Decimal(short_price), abs(pos_change))
            self.active_orderids.extend(vt_orderids)
