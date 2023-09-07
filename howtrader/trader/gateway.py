from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from copy import copy

from howtrader.event import Event, EventEngine
from .event import (
    EVENT_TICK,
    EVENT_ORDER,
    EVENT_TRADE,
    EVENT_POSITION,
    EVENT_ACCOUNT,
    EVENT_CONTRACT,
    EVENT_LOG,
    EVENT_QUOTE,
    EVENT_ORIGINAL_KLINE,
    EVENT_PREMIUM_RATE
)
from .object import (
    TickData,
    OrderData,
    TradeData,
    OrderQueryRequest,
    PositionData,
    AccountData,
    ContractData,
    LogData,
    QuoteData,
    OrderRequest,
    CancelRequest,
    SubscribeRequest,
    HistoryRequest,
    QuoteRequest,
    Exchange,
    BarData,
    OriginalKlineData,
    PremiumRateData
)


class BaseGateway(ABC):
    """
    Abstract gateway class for creating gateways connection
    to different trading systems.

    # How to implement a gateway:

    ---
    ## Basics
    A gateway should satisfy:
    * this class should be thread-safe:
        * all methods should be thread-safe
        * no mutable shared properties between objects.
    * all methods should be non-blocked
    * satisfies all requirements written in docstring for every method and callbacks.
    * automatically reconnect if connection lost.

    ---
    ## methods must implements:
    all @abstractmethod

    ---
    ## callbacks must response manually:
    * on_tick
    * on_trade
    * on_order
    * on_position
    * on_account
    * on_contract

    All the XxxData passed to callback should be constant, which means that
        the object should not be modified after passing to on_xxxx.
    So if you use a cache to store reference of data, use copy.copy to create a new object
    before passing that data into on_xxxx
    """

    # Default name for the gateway.
    default_name: str = ""

    # Fields required in setting dict for connect function.
    default_setting: Dict[str, Any] = {}

    # Exchanges supported in the gateway.
    exchanges: List[Exchange] = []

    def __init__(self, event_engine: EventEngine, gateway_name: str) -> None:
        """"""
        self.event_engine: EventEngine = event_engine
        self.gateway_name: str = gateway_name

    def on_event(self, type: str, data: Any = None) -> None:
        """
        General event push.
        """
        event: Event = Event(type, data)
        self.event_engine.put(event)

    def on_tick(self, tick: TickData) -> None:
        """
        Tick event push.
        Tick event of a specific vt_symbol is also pushed.
        """
        # self.on_event(EVENT_TICK, tick)：这一行代码将一个名为 EVENT_TICK 的事件与接收到的 tick 数据关联起来，
        # 并通过事件引擎将该事件推送出去。这个事件通常用于通知系统中的其他部分，新的行情数据已经到达。
        self.on_event(EVENT_TICK, tick)
        # 行代码将特定交易品种的最新行情数据与一个特定的事件关联起来，并通过事件引擎将该事件推送出去。
        # 这个事件的名称是 EVENT_TICK 加上特定交易品种的 vt_symbol，通常用于通知系统中对特定品种感兴趣的部分，新的行情数据已经到达。
        self.on_event(EVENT_TICK + tick.vt_symbol, tick)

    def on_trade(self, trade: TradeData) -> None:
        """
        Trade event push.
        Trade event of a specific vt_symbol is also pushed.
        """
        self.on_event(EVENT_TRADE, trade)
        self.on_event(EVENT_TRADE + trade.vt_symbol, trade)

    def on_order(self, order: OrderData) -> None:
        """
        Order event push.
        Order event of a specific vt_orderid is also pushed.
        """
        self.on_event(EVENT_ORDER, order)
        self.on_event(EVENT_ORDER + order.vt_orderid, order)

    def on_position(self, position: PositionData) -> None:
        """
        Position event push.
        Position event of a specific vt_symbol is also pushed.
        """
        self.on_event(EVENT_POSITION, position)
        self.on_event(EVENT_POSITION + position.vt_symbol, position)

    def on_account(self, account: AccountData) -> None:
        """
        Account event push.
        Account event of a specific vt_accountid is also pushed.
        """
        self.on_event(EVENT_ACCOUNT, account)
        self.on_event(EVENT_ACCOUNT + account.vt_accountid, account)

    def on_kline(self, kline: OriginalKlineData):
        # self.on_event(EVENT_ORIGINAL_KLINE, kline)
        self.on_event(EVENT_ORIGINAL_KLINE + kline.vt_symbol, kline)

    def on_quote(self, quote: QuoteData) -> None:
        """
        Quote event push.
        Quote event of a specific vt_symbol is also pushed.
        """
        self.on_event(EVENT_QUOTE, quote)
        self.on_event(EVENT_QUOTE + quote.vt_symbol, quote)

    def on_log(self, log: LogData) -> None:
        """
        Log event push.
        """
        self.on_event(EVENT_LOG, log)

    def on_contract(self, contract: ContractData) -> None:
        """
        Contract event push.
        """
        self.on_event(EVENT_CONTRACT, contract)

    def on_premium_rate(self, premium_rate: PremiumRateData):
        self.on_event(EVENT_PREMIUM_RATE, premium_rate)
        self.on_event(EVENT_PREMIUM_RATE + premium_rate.vt_symbol, premium_rate)

    def write_log(self, msg: str) -> None:
        """
        Write a log event from gateway.
        """
        log: LogData = LogData(msg=msg, gateway_name=self.gateway_name)
        self.on_log(log)

    @abstractmethod
    def connect(self, setting: dict) -> None:
        """
        Start gateway connection.

        to implement this method, you must:
        * connect to server if necessary
        * log connected if all necessary connection is established
        * do the following query and response corresponding on_xxxx and write_log
            * contracts : on_contract
            * account asset : on_account
            * account holding: on_position
            * orders of account: on_order
            * trades of account: on_trade
        * if any of query above is failed,  write log.

        future plan:
        response callback/change status instead of write_log

        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close gateway connection.
        """
        pass

    @abstractmethod
    def subscribe(self, req: SubscribeRequest) -> None:
        """
        Subscribe tick data update.
        """
        pass

    @abstractmethod
    def send_order(self, req: OrderRequest) -> str:
        """
        Send a new order to server.

        implementation should finish the tasks blow:
        * create an OrderData from req using OrderRequest.create_order_data
        * assign a unique(gateway instance scope) id to OrderData.orderid
        * send request to server
            * if request is sent, OrderData.status should be set to Status.SUBMITTING
            * if request is failed to sent, OrderData.status should be set to Status.REJECTED
        * response on_order:
        * return vt_orderid

        :return str vt_orderid for created OrderData
        """
        pass

    @abstractmethod
    def cancel_order(self, req: CancelRequest) -> None:
        """
        Cancel an existing order.
        implementation should finish the tasks blow:
        * send request to server
        """
        pass

    def send_quote(self, req: QuoteRequest) -> str:
        """
        Send a new two-sided quote to server.

        implementation should finish the tasks blow:
        * create an QuoteData from req using QuoteRequest.create_quote_data
        * assign a unique(gateway instance scope) id to QuoteData.quoteid
        * send request to server
            * if request is sent, QuoteData.status should be set to Status.SUBMITTING
            * if request is failed to sent, QuoteData.status should be set to Status.REJECTED
        * response on_quote:
        * return vt_quoteid

        :return str vt_quoteid for created QuoteData
        """
        return ""

    def cancel_quote(self, req: CancelRequest) -> None:
        """
        Cancel an existing quote.
        implementation should finish the tasks blow:
        * send request to server
        """
        pass

    def query_order(self, req: OrderQueryRequest) -> None:
        """
        Cancel an existing quote.
        implementation should finish the tasks blow:
        * send request to server
        """
        pass

    @abstractmethod
    def query_account(self) -> None:
        """
        Query account balance.
        """
        pass

    def query_latest_kline(self, req: HistoryRequest) -> None:
        """
        Query account balance.
        """
        pass

    def query_position(self) -> None:
        """
        Query holding positions.
        """
        pass

    def query_premium_rate(self) -> None:
        """query premium rate/index of the perpetual product"""
        pass

    def query_history(self, req: HistoryRequest) -> List[BarData]:
        """
        Query bar history data.
        """
        pass

    def get_default_setting(self) -> Dict[str, Any]:
        """
        Return default setting dict.
        """
        return self.default_setting


class LocalOrderManager:
    """
    Management tool to support use local order id for trading.
    """

    def __init__(self, gateway: BaseGateway, order_prefix: str = "") -> None:
        """"""
        self.gateway: BaseGateway = gateway
        # 参数是一个用于生成本地订单标识的前缀，通常是一个字符串，以便将本地订单标识与系统订单标识区分开来。
        # For generating local orderid
        self.order_prefix: str = order_prefix
        # 用于生成本地订单标识的计数器
        self.order_count: int = 0
        # 存储订单信息的字典
        self.orders: Dict[str, OrderData] = {}  # local_orderid: order

        # Map between local and system orderid
        # 本地订单标识与系统订单标识之间的映射
        self.local_sys_orderid_map: Dict[str, str] = {}
        self.sys_local_orderid_map: Dict[str, str] = {}

        # Push order data buf
        # 存储推送数据的字典v
        self.push_data_buf: Dict[str, Dict] = {}  # sys_orderid: data

        # Callback for processing push order data
        # 用于处理推送数据的回调函数
        self.push_data_callback: Callable = None

        # Cancel request buf
        # 存储取消订单请求的字典
        self.cancel_request_buf: Dict[str, CancelRequest] = {}  # local_orderid: req

        # Hook cancel order function
        # 修改了 gateway 对象的 cancel_order 方法，将其替换为本地的 cancel_order 方法，以便进行订单的本地管理。
        self._cancel_order: Callable = gateway.cancel_order
        gateway.cancel_order = self.cancel_order
    """
    生成一个新的本地订单标识，将其返回。这个标识是通过在 order_prefix 前添加一个递增的计数器生成的。
    """
    def new_local_orderid(self) -> str:
        """
        Generate a new local orderid.
        """
        self.order_count += 1
        local_orderid: str = self.order_prefix + str(self.order_count).rjust(8, "0")
        return local_orderid
    """
    根据系统订单标识获取对应的本地订单标识，如果没有找到，则生成一个新的本地订单标识。
    这个方法用于将系统订单标识与本地订单标识进行映射，以便在本地管理订单时使用。
    """
    def get_local_orderid(self, sys_orderid: str) -> str:
        """
        Get local orderid with sys orderid.
        """
        local_orderid: str = self.sys_local_orderid_map.get(sys_orderid, "")

        if not local_orderid:
            local_orderid = self.new_local_orderid()
            self.update_orderid_map(local_orderid, sys_orderid)

        return local_orderid

    """
    根据本地订单标识获取对应的系统订单标识。
    这个方法用于将本地订单标识转换为系统订单标识，以便向交易所发送订单取消请求等。
    """

    def get_sys_orderid(self, local_orderid: str) -> str:
        """
        Get sys orderid with local orderid.
        """
        sys_orderid: str = self.local_sys_orderid_map.get(local_orderid, "")
        return sys_orderid

    """
    更新订单标识映射关系，将系统订单标识与本地订单标识互相映射。
    同时检查是否有等待中的取消订单请求或推送数据，如果有则执行相应的操作
    """

    def update_orderid_map(self, local_orderid: str, sys_orderid: str) -> None:
        """
        Update orderid map.
        """
        self.sys_local_orderid_map[sys_orderid] = local_orderid
        self.local_sys_orderid_map[local_orderid] = sys_orderid

        self.check_cancel_request(local_orderid)
        self.check_push_data(sys_orderid)

    """
    检查是否有等待中的推送数据，如果有则执行相应的回调函数。
    """

    def check_push_data(self, sys_orderid: str) -> None:
        """
        Check if any order push data waiting.
        """
        if sys_orderid not in self.push_data_buf:
            return

        data: dict = self.push_data_buf.pop(sys_orderid)
        if self.push_data_callback:
            self.push_data_callback(data)

    """
    将推送数据添加到推送数据缓冲区中，等待后续处理。
    """

    def add_push_data(self, sys_orderid: str, data: dict) -> None:
        """
        Add push data into buf.
        """
        self.push_data_buf[sys_orderid] = data

    def get_order_with_sys_orderid(self, sys_orderid: str) -> Optional[OrderData]:
        """"""
        local_orderid: str = self.sys_local_orderid_map.get(sys_orderid, None)
        if not local_orderid:
            return None
        else:
            return self.get_order_with_local_orderid(local_orderid)

    """
    根据系统订单标识或本地订单标识获取相应的订单数据对象。
    """

    def get_order_with_local_orderid(self, local_orderid: str) -> OrderData:
        """"""
        order: OrderData = self.orders[local_orderid]
        return copy(order)

    """
    用于处理订单事件，将订单信息存储在 orders 字典中，并通过 gateway 对象的 on_order 方法将订单事件推送出去。
    """

    def on_order(self, order: OrderData) -> None:
        """
        Keep an order buf before pushing it to gateway.
        """
        self.orders[order.orderid] = copy(order)
        self.gateway.on_order(order)

    """
    用于发送取消订单请求，但首先会根据本地订单标识查找相应的系统订单标识。
    如果找到系统订单标识，则发送取消订单请求；
    否则，将取消请求存储在 cancel_request_buf 中等待映射后再发送。
    """

    def cancel_order(self, req: CancelRequest) -> None:
        """"""
        sys_orderid: str = self.get_sys_orderid(req.orderid)
        if not sys_orderid:
            self.cancel_request_buf[req.orderid] = req
            return

        self._cancel_order(req)

    # 检查是否有等待中的取消订单请求，如果有，则发送这些请求。
    def check_cancel_request(self, local_orderid: str) -> None:
        """"""
        if local_orderid not in self.cancel_request_buf:
            return

        req: CancelRequest = self.cancel_request_buf.pop(local_orderid)
        self.gateway.cancel_order(req)
