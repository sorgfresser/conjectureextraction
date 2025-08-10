# import weave
from typing import List, Optional, TYPE_CHECKING, Dict, Callable
import subprocess
import time
from openai import OpenAI, APIConnectionError
from openai.types import Completion
from openai.types.chat import ChatCompletion
import os
import socket
import logging

logger = logging.getLogger(__name__)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        port = s.getsockname()[1]
        logger.info("Found port %s", port)
        return port


class vLLMWrapper:
    def __init__(self, model_path: str, dtype: str, vllm_bin: Optional[str] = None, max_retries: int = 3):
        self._model_path = model_path
        self._dtype = dtype
        self._port = find_free_port()
        self.client = OpenAI(
            base_url=f"http://localhost:{self._port}/v1",
            api_key="token-abc123",
        )
        self._vllm_proc = None
        self._vllm_bin = "vllm" if vllm_bin is None else vllm_bin
        self._start_proc()
        self._max_retries = max_retries

    def _start_proc(self):
        self._vllm_proc = subprocess.Popen(
            [self._vllm_bin, "serve", self._model_path, "--dtype", self._dtype, "--api-key", "token-abc123",
             "--gpu-memory-utilization", "0.95", "--seed", "42", "--port", str(self._port),
             "--host", "localhost"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=os.environ,
        )
        logger.info("Running vLLM with args %s", self._vllm_proc.args)
        os.set_blocking(self._vllm_proc.stderr.fileno(), False)
        time.sleep(60)
        early = False
        for i in range(10):
            try:
                self._generate("test", 1, 0.0, 5)
                early = True
                logger.info("Started successfully after %s attempts", i)
                break
            except APIConnectionError:
                time.sleep(60)
                if self._vllm_proc.poll() is not None:
                    msg = self._get_msg(self._vllm_proc.stderr)
                    raise RuntimeError(f"Process finished: {msg}")
                logger.debug("Attempt %s, retrying", i)
        if self._vllm_proc.poll() is not None:
            msg = self._get_msg(self._vllm_proc.stderr)
            raise RuntimeError(f"Process finished: {msg}")
        if not early:
            msg = self._get_msg(self._vllm_proc.stderr)
            logger.error(msg)
            raise RuntimeError("Did not manage to start within 660 seconds!")

    def _get_msg(self, fd) -> str:
        msg = ""
        if not fd:
            return msg
        for i in range(100):
            read = fd.read()
            if read is not None:
                msg += read.decode()
        return msg

    def _cleanup(self):
        if self._vllm_proc is not None:
            try:
                self._vllm_proc.terminate()
                self._vllm_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._vllm_proc.kill()
                time.sleep(1)
            except Exception:
                logger.exception(f"Exception when terminating vLLM")
                try:
                    self._vllm_proc.kill()
                except Exception:
                    logger.exception(f"Exception when killing vLLM")

    def __del__(self):
        self._cleanup()

    def _generate(self, prompt: str, n: int, temperature: float, max_tokens: int, timeout: float | None = None):
        completion = self.client.completions.create(
            model=self.model_path,
            prompt=prompt,
            temperature=temperature,
            logprobs=0,
            n=n,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return completion

    def _retry_loop(self, func: Callable):
        result = None
        for _ in range(self._max_retries):
            try:
                result = func()
                break
            except APIConnectionError:
                logger.exception("APIConnectionError, retrying!")
                if self._vllm_proc.poll() is not None:
                    msg = self._get_msg(self._vllm_proc.stderr)
                    logger.error("Process finished, stderr: %s", msg)
                msg = self._get_msg(self._vllm_proc.stderr)
                logger.info("Process stderr: %s", msg)
                # Restarting
                self._cleanup()
                self._start_proc()
                continue
        if result is None:
            raise RuntimeError("Failed to generate!")
        return result

    def generate(self, prompt: str, n: int, temperature: float, max_tokens: int) -> Completion:
        def func():
            return self._generate(prompt, n, temperature, max_tokens, 45.0)

        return self._retry_loop(func)

    def chat_completion(self, messages, n: int, temperature: float, max_tokens: int) -> ChatCompletion:
        def func():
            return self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=temperature,
                logprobs=True,
                n=n,
                max_tokens=max_tokens,
                timeout=45.0
            )

        return self._retry_loop(func)

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, value):
        self._cleanup()
        self._model_path = value
        self._start_proc()
