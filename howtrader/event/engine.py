"""
Event-driven framework of Howtrader framework.
"""
# defaultdict类，它是一种特殊的字典，用于创建默认值为列表的字典
from collections import defaultdict
from queue import Empty, Queue
from threading import Thread
from time import sleep
# 导入用于类型提示的模块，包括Any、Callable和List。
from typing import Any, Callable, List

# 定义了一个名为EVENT_TIMER的常量，用于表示定时器事件的类型。
EVENT_TIMER = "eTimer"


class Event:
    """
    Event object consists of a type string which is used
    by event engine for distributing event, and a data
    object which contains the real data.
    """

    def __init__(self, type: str, data: Any = None) -> None:
        """"""
        self.type: str = type
        self.data: Any = data


"""
# Defines handler function to be used in event engine.
# 作用是为事件处理函数的类型进行类型提示和定义。他使用了类型提示的语法，
# 表示了事件处理函数应该是一个可以调用对象（callable)
# 他接收一个event类型的参数，并且不返回任何值
# callable 是一个类型提示，表示这个类型应该是可调用的，也就是说，它可以像函数一样被调用。
"""
HandlerType: callable = Callable[[Event], None]


class EventEngine:
    """
    Event engine distributes event object based on its type
    to those handlers registered.

    It also generates timer event by every interval seconds,
    which can be used for timing purpose.
    """

    def __init__(self, interval: int = 1) -> None:
        """
        Timer event is generated every 1 second by default, if
        interval not specified.
        """
        self._interval: int = interval
        self._queue: Queue = Queue()
        self._active: bool = False
        # 创建了一个线程对象 self._thread，并将 self._run 方法作为其目标函数（target）。
        # 这个线程用于事件处理，当启动后，会不断地从事件队列中获取事件并进行处理。
        self._thread: Thread = Thread(target=self._run)
        # 创建了一个线程对象self.timer，同样将『self._run_timer方法作为其目标函数
        # 这个线程用于生成定时器事件，他会每隔一段时间生成一个定时器事件，用于执行定时任务
        self._timer: Thread = Thread(target=self._run_timer)
        self._handlers: defaultdict = defaultdict(list)
        self._general_handlers: List = []

    def _run(self) -> None:
        """
        Get event from queue and then process it.
        """
        while self._active:
            try:
                event: Event = self._queue.get(block=True, timeout=1)
                self._process(event)
            except Empty:
                pass

    """
    用于处理事件，首先将事件分发给已注册的特定类型的事件处理函数，然后再分发给通用事件处理函数。
    """

    def _process(self, event: Event) -> None:
        """
        First distribute event to those handlers registered listening
        to this type.

        Then distribute event to those general handlers which listens
        to all types.
        """
        if event.type in self._handlers:
            # 这段代码是一个列表推导式（List Comprehension），
            # 用于在事件引擎中触发注册的特定事件类型的处理函数。
            [handler(event) for handler in self._handlers[event.type]]

        """
        这段代码不关心事件的类型，它只检查是否有一般性的事件处理函数注册了。
        如果有一般性的事件处理函数注册了（即 _general_handlers 不为空），
        它会使用列表推导式遍历所有这些一般性事件处理函数，并逐个执行它们。
        这段代码用于处理那些不特定于特定事件类型的操作，而是适用于所有事件类型的操作。
        通常，一般性事件处理函数用于执行一些全局性的操作，不需要考虑事件的具体类型。
        """
        if self._general_handlers:
            [handler(event) for handler in self._general_handlers]

    def _run_timer(self) -> None:
        """
        Sleep by interval second(s) and then generate a timer event.
        """
        while self._active:
            sleep(self._interval)
            event: Event = Event(EVENT_TIMER)
            self.put(event)

    def start(self) -> None:
        """
        Start event engine to process events and generate timer events.
        """
        self._active = True
        self._thread.start()
        self._timer.start()

    def stop(self) -> None:
        """
        Stop event engine.
        """
        self._active = False
        self._timer.join()
        self._thread.join()

    def put(self, event: Event) -> None:
        """
        Put an event object into event queue.
        """
        self._queue.put(event)

    def register(self, type: str, handler: HandlerType) -> None:
        """
        Register a new handler function for a specific event type. Every
        function can only be registered once for each event type.
        """
        # 首先，他创建一个名为handler_list 的变量，将其初始化为 self._handlers[type]
        # self._handlers 是一个字典，它以事件类型为键，以事件处理器列表为值存储注册的事件处理器。
        # 这行代码的目的是获取特定事件类型 type 对应的处理器列表。
        # 如果self._handlers 中如果没有对应的key，它将返回一个空的list
        handler_list: list = self._handlers[type]
        if handler not in handler_list:
            handler_list.append(handler)

    def unregister(self, type: str, handler: HandlerType) -> None:
        """
        Unregister an existing handler function from event engine.
        """
        handler_list: list = self._handlers[type]

        if handler in handler_list:
            handler_list.remove(handler)

        if not handler_list:
            self._handlers.pop(type)

    def register_general(self, handler: HandlerType) -> None:
        """
        Register a new handler function for all event types. Every
        function can only be registered once for each event type.
        """
        if handler not in self._general_handlers:
            self._general_handlers.append(handler)

    def unregister_general(self, handler: HandlerType) -> None:
        """
        Unregister an existing general handler function.
        """
        if handler in self._general_handlers:
            self._general_handlers.remove(handler)
