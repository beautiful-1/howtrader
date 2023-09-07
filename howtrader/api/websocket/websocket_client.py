import json
import sys
import traceback
from datetime import datetime
from types import coroutine
from threading import Thread
from asyncio import (
    get_running_loop,
    new_event_loop,
    set_event_loop,
    run_coroutine_threadsafe,
    AbstractEventLoop,
    TimeoutError
)
from typing import Optional
from aiohttp import ClientSession, ClientWebSocketResponse


class WebsocketClient:
    """
    WebsocketClient

    subclass requirements:

    * reload unpack_data method to implement the logic from server
    * reload on_connected: implement your logic when server connected
    * reload on_disconnected方
    * reload on_packet to subscribe data
    * reload on_error
    """

    def __init__(self):
        """Constructor"""
        # 标识WebSocket客户端是否处于活动状态，默认为False。
        self._active: bool = False
        # WebSocket客户端的会话对象，用于建立WebSocket连接。初始值为None。
        self._session: Optional[ClientSession] = None
        # 接收数据的超时时间，单位为秒，默认设置为5分钟。
        self.receive_timeout = 5 * 60  # 5 minutes for receiving timeout
        # WebSocket连接对象，用于发送和接收数据。初始值为None。
        self._ws: Optional[ClientWebSocketResponse] = None
        # 事件循环对象，用于异步处理WebSocket连接。初始值为None。
        self._loop: Optional[AbstractEventLoop] = None
        # WebSocket服务器的主机地址。初始值为空字符串。
        self._host: str = ""
        # 代理服务器的地址和端口，用于建立WebSocket连接。初始值为None。
        self._proxy: Optional[str] = None
        # 发送心跳包的时间间隔，单位为秒，默认为60秒。
        self._ping_interval: int = 60  # ping interval for 60 seconds
        # WebSocket请求头信息，用于在建立连接时发送给服务器。初始值为空字典。
        self._header: dict = {}
        # 用于记录最后一次发送和接收的文本数据，用于调试目的。初始值为空字符串。
        self._last_sent_text: str = ""
        self._last_received_text: str = ""

    def init(
            self,
            host: str,
            proxy_host: str = "",
            proxy_port: int = 0,
            ping_interval: int = 60,
            header: dict = None
    ):
        """
        init client, only support the http proxy.
        初始化WebSocket客户端的方法，用于设置WebSocket服务器地址、代理服务器信息、心跳包间隔和请求头信息。
        """
        self._host = host
        self._ping_interval = ping_interval

        if header:
            self._header = header

        if proxy_host and proxy_port:
            self._proxy = f"http://{proxy_host}:{proxy_port}"

    def start(self):
        """
        start client

        will call the on_connected callback when connected
        subscribe the data when call the on_connected callback

        启动WebSocket客户端的方法，开始建立连接并调用回调函数。
        """
        if self._active:
            return None

        self._active = True

        try:
            self._loop = get_running_loop()
        except RuntimeError:
            self._loop = new_event_loop()

        start_event_loop(self._loop)

        run_coroutine_threadsafe(self._run(), self._loop)

    """
    停止WebSocket客户端的方法，关闭连接。
    """

    def stop(self):
        """
        stop the client
        """
        self._active = False

        if self._ws:
            coro = self._ws.close()
            run_coroutine_threadsafe(coro, self._loop)

        if self._session:  # need to close the session.
            coro1 = self._session.close()
            run_coroutine_threadsafe(coro1, self._loop)

        if self._loop and self._loop.is_running():
            self._loop.stop()

    """
    等待线程完成的方法，这里不做任何处理。
    """

    def join(self):
        """
        wait for the thread to finish.
        """
        pass

    """
    发送数据包给服务器的方法，这里使用JSON格式发送数据。
    """

    def send_packet(self, packet: dict):
        """
        send data to server.
        if the data is not in json format, please reload this function.
        """
        if self._ws:
            text: str = json.dumps(packet)
            self._record_last_sent_text(text)

            coro: coroutine = self._ws.send_str(text)
            run_coroutine_threadsafe(coro, self._loop)

    """
    解析从服务器接收的数据包的方法，这里将接收到的JSON格式数据转换为字典。
    """

    def unpack_data(self, data: str):
        """
        unpack the data from server
        use json loads method to convert the str in to dict
        you may need to reload the unpack_data if server send the data not in str format
        """
        return json.loads(data)

    def on_connected(self):
        """on connected callback"""
        pass

    def on_disconnected(self):
        """on disconnected callback"""
        pass

    def on_packet(self, packet: dict):
        """on packed callback"""
        pass

    def on_error(
            self,
            exception_type: type,
            exception_value: Exception,
            tb
    ) -> None:
        """raise error"""
        try:
            print("WebsocketClient on error" + "-" * 10)
            print(self.exception_detail(exception_type, exception_value, tb))
        except Exception:
            traceback.print_exc()

    def exception_detail(
            self,
            exception_type: type,
            exception_value: Exception,
            tb
    ) -> str:
        """format the exception detail in str"""
        text = "[{}]: Unhandled WebSocket Error:{}\n".format(
            datetime.now().isoformat(), exception_type
        )
        text += "LastSentText:\n{}\n".format(self._last_sent_text)
        text += "LastReceivedText:\n{}\n".format(self._last_received_text)
        text += "Exception trace: \n"
        text += "".join(
            traceback.format_exception(exception_type, exception_value, tb)
        )
        return text

    """
    一个内部方法，用于在异步事件循环中运行WebSocket客户端。该方法循环处理连接、接收数据、执行回调等操作
    """

    async def _run(self):
        """
        run on the asyncio
        """
        while self._active:
            # try catch error/exception
            try:
                # connect ws server
                if not self._session:
                    self._session = ClientSession()

                if self._session.closed:
                    self._session = ClientSession()

                self._ws = await self._session.ws_connect(
                    self._host,
                    proxy=self._proxy,
                    verify_ssl=False,
                    heartbeat=self._ping_interval,  # send ping interval
                    receive_timeout=self.receive_timeout,
                )

                # call the on_connected function
                self.on_connected()

                # receive data from websocket
                async for msg in self._ws:
                    text: str = msg.data
                    self._record_last_received_text(text)

                    data: dict = self.unpack_data(text)
                    self.on_packet(data)

                # remove the _ws object
                self._ws = None

                # call the on_disconnected
                self.on_disconnected()
            # on exception
            except TimeoutError:
                pass
            except Exception:
                et, ev, tb = sys.exc_info()
                self.on_error(et, ev, tb)

    """
    用于记录最后一次发送和接收的文本数据，以便调试。
    """

    def _record_last_sent_text(self, text: str):
        """record the last send text for debugging"""
        self._last_sent_text = text[:1000]

    def _record_last_received_text(self, text: str):
        """record the last receive text for debugging"""
        self._last_received_text = text[:1000]


"""
用于启动和运行事件循环的辅助函数，确保事件循环在单独的线程中运行。
"""


def start_event_loop(loop: AbstractEventLoop) -> None:
    """start event loop"""
    # if the event loop is not running, then create the thread to run
    if not loop.is_running():
        thread = Thread(target=run_event_loop, args=(loop,))
        # 将新创建的线程标记为守护线程。守护线程是一种在主程序退出时会随之退出的线程。
        # 这是为了确保当主程序退出时，不会保持事件循环线程的运行。
        thread.daemon = True
        thread.start()


def run_event_loop(loop: AbstractEventLoop) -> None:
    """run event loop"""
    set_event_loop(loop)
    loop.run_forever()
