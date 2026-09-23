import socket
from typing import List, Dict

import requests
from requests import Response

from ocr_util.eval.dictionary_metrics.language_tool.common import (
    NoFreePortAvailableException,
    NoApiPortFoundException,
    ConnectionException,
    Constant,
)


class Util:
    @staticmethod
    def is_port_in_use(host: str, port: int):
        try:
            # Create a new socket object
            sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Set a timeout value for the connection attempt
            sock.settimeout(1)
            # Attempt to establish a connection to the specified host and port
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return True
            else:
                # The connection attempt was unsuccessful
                return False
        except socket.error:
            return False

    @staticmethod
    def find_free_port_in_range(
        host: str,
        from_port: int,
        to_port: int,
        exclude_ports: List[int] = None,
    ) -> int:
        if exclude_ports is None:
            exclude_ports = []
        for port in range(from_port, to_port + 1):  # include to_port
            if port in exclude_ports:
                continue
            if not Util.is_port_in_use(host, port):
                return port
        raise NoFreePortAvailableException

    @staticmethod
    def find_api_port(host: str, from_port: int, to_port: int, exclude_ports: List[int] = None) -> int:
        if exclude_ports is None:
            exclude_ports = []
        for port in range(from_port, to_port + 1):  # include to_port
            if port in exclude_ports:
                continue
            if Util.is_api_port(host, port):
                return port
        raise NoApiPortFoundException

    @staticmethod
    def is_api_port(host: str, port: int) -> bool:
        if not Util.is_port_in_use(host, port):
            return False
        test_url: str = f"{Constant.DEFAULT_PROTOCOL}{host}:{port}"
        return Util.is_api(test_url)

    @staticmethod
    def is_api(url) -> bool:
        try:
            # check whether a specific error is returned
            response: Response = Util.request(url)
            if response.ok:
                return False
            if response.status_code != 400:
                return False
            if not isinstance(response.text, str):
                return False
            if "LanguageTool API" in response.text:
                return True
        except ConnectionException:
            return False

    @staticmethod
    def request(url: str, data: Dict[str, str] = None, timeout: int = 10) -> Response:
        try:
            return requests.post(url=url, data=data, timeout=timeout)
        except requests.ConnectionError:
            raise ConnectionException(url)
