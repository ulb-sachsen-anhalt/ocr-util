from __future__ import annotations

import json
from copy import copy
from time import sleep
from typing import Final, Dict, Optional, Any

import docker
from docker import DockerClient
from docker.models.resource import Model
from requests import Response

from ocr_util.eval.dictionary_metrics.common import LANGUAGE_MAP, LANGUAGE_KEY_DEFAULT
from ocr_util.eval.dictionary_metrics.language_tool.Util import Util
from ocr_util.eval.dictionary_metrics.language_tool.common import (
    InvalidResponseException,
    NotInizializedException,
    Constant,
    ContainerException,
)

_REQ_DATA_TEMPLATE: Final[Dict[str, str]] = {
    "data": None,
    "language": None,
    "enableHiddenRules": "true",
    "level": "picky",
    "disabledRules": "WHITESPACE_RULE",
    "mode": "allButTextLevelOnly",
    "allowIncompleteResults": "false",
}


class LanguageTool:
    DEFAULT_URL: Final[str] = f"{Constant.DEFAULT_PROTOCOL}{Constant.LOCAL_HOST}:{Constant.DEFAULT_PORT}"

    __instance: LanguageTool = None

    def __init__(self):
        self.__url: Optional[str] = None
        self.__docker_client: Optional[DockerClient] = None
        self.__docker_container: Optional[DockerClient] = None

    @classmethod
    def instance(cls) -> LanguageTool:
        if cls.__instance is None:
            cls.__instance = LanguageTool()
        return cls.__instance

    @classmethod
    def check(cls, text: str, language: str = LANGUAGE_KEY_DEFAULT) -> Dict:
        return cls.instance().__check(text, language)

    @classmethod
    def initialize(cls, url: str = DEFAULT_URL) -> None:
        cls.instance().__initialize(url)

    @classmethod
    def deinitialize(cls) -> None:
        cls.instance().__deinitialize()

    def __initialize(self, url: str) -> None:
        self.__url = url if Util.is_api(url) else self.__run_container()

    def __run_container(self) -> str:
        free_port: int = Util.find_free_port_in_range(
            Constant.LOCAL_HOST,
            Constant.PORT_RANGE[0],
            Constant.PORT_RANGE[1],
            Constant.EXCLUDED_PORTS,
        )

        self.__docker_client = docker.from_env()
        container_name: str = f"digital_eval_languagetool_{free_port}"
        self.__docker_container = self.__docker_client.containers.run(
            Constant.DOCKER_IMAGE,
            name=container_name,
            detach=True,
            ports={"8010/tcp": free_port},
        )
        self.__docker_container.reload()
        while self.__docker_container.status != "running":
            self.__docker_container.reload()

        url: str = f"{Constant.DEFAULT_PROTOCOL}{Constant.LOCAL_HOST}:{free_port}"
        for i in range(10):
            if Util.is_api(url):
                return url
            sleep(1)
        raise ContainerException("container running failed")

    def __deinitialize(self) -> None:
        self.__url = None
        self.__docker_container.stop()
        self.__docker_container.remove()

    def __check(self, text: str, language: str) -> Dict:
        if self.__url is None:
            raise NotInizializedException
        return LanguageTool.__request_check_endpoint(base_url=self.__url, text=text, language=language)

    @classmethod
    def __request_check_endpoint(cls, base_url: str, text: str, language: str, timeout: int = 30) -> Dict:
        data: Dict[str, str] = copy(_REQ_DATA_TEMPLATE)
        data["data"] = json.dumps({"text": text})
        data["language"] = LANGUAGE_MAP[language].lt_variant
        url: str = f"{base_url}{Constant.ENDPOINT}"
        response: Response = Util.request(url, data, timeout)
        if not response.ok:
            raise InvalidResponseException(url, response)
        return response.json()
