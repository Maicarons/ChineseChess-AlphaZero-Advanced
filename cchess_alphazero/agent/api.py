from multiprocessing import connection, Pipe
from threading import Thread
from time import sleep

import os
import numpy as np
import shutil

from cchess_alphazero.config import Config
from cchess_alphazero.lib.model_helper import load_best_model_weight, need_to_reload_best_model_weight
from cchess_alphazero.lib.web_helper import http_request, download_file
from time import time
from logging import getLogger

logger = getLogger(__name__)

class CChessModelAPI:

    def __init__(self, config: Config, agent_model):  
        self.agent_model = agent_model  # CChessModel
        self.pipes = []     # use for communication between processes/threads
        self.config = config
        self.need_reload = True
        self.done = False

    def start(self, need_reload=True):
        self.need_reload = need_reload
        prediction_worker = Thread(target=self.predict_batch_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def get_pipe(self, need_reload=True):
        me, you = Pipe()
        self.pipes.append(me)
        self.need_reload = need_reload
        return you

    def predict_batch_worker(self):
        if self.config.internet.distributed and self.need_reload:
            self.try_reload_model_from_internet()
        last_model_check_time = time()
        while not self.done:
            if last_model_check_time + 600 < time() and self.need_reload:
                self.try_reload_model()
                last_model_check_time = time()
            try:
                # Filter out closed pipes before waiting
                active_pipes = [p for p in self.pipes if hasattr(p, 'closed') and not p.closed or not hasattr(p, 'closed')]
                if not active_pipes:
                    sleep(0.1)  # No active pipes, wait a bit
                    continue
                ready = connection.wait(active_pipes, timeout=0.001)
            except (OSError, BrokenPipeError) as e:
                logger.warning(f"Pipe error during wait: {e}")
                sleep(0.1)
                continue
            if not ready:
                continue
            data, result_pipes, data_len = [], [], []
            for pipe in ready:
                try:
                    while pipe.poll():
                        try:
                            tmp = pipe.recv()
                        except EOFError as e:
                            logger.error(f"EOF error: {e}")
                            pipe.close()
                        except BrokenPipeError as e:
                            logger.warning(f"BrokenPipeError: {e}, closing pipe")
                            try:
                                pipe.close()
                            except:
                                pass
                        else:
                            data.extend(tmp)
                            data_len.append(len(tmp))
                            result_pipes.append(pipe)
                except (BrokenPipeError, OSError) as e:
                    logger.warning(f"Pipe error during wait: {e}")
                    try:
                        pipe.close()
                    except:
                        pass
            if not data:
                continue
            data = np.asarray(data, dtype=np.float32)
            # Keras 3.x with TF backend works in eager mode by default
            result = self.agent_model.model(data, training=False)
            policy_ary = result[0].numpy()
            value_ary = result[1].numpy()
            buf = []
            k, i = 0, 0
            for p, v in zip(policy_ary, value_ary):
                # Handle both scalar and 0-dimensional array values
                if hasattr(v, 'item'):
                    # v is a numpy array or tensorflow tensor
                    val = v.item() if v.ndim > 0 else float(v)
                else:
                    val = float(v)
                buf.append((p, val))
                k += 1
                if k >= data_len[i]:
                    try:
                        # Check if pipe is still available
                        if hasattr(result_pipes[i], 'closed') and not result_pipes[i].closed:
                            result_pipes[i].send(buf)
                        elif not hasattr(result_pipes[i], 'closed'):
                            # Fallback for older versions
                            result_pipes[i].send(buf)
                    except (BrokenPipeError, OSError) as e:
                        logger.warning(f"Failed to send result: {e}")
                    buf = []
                    k = 0
                    i += 1

    def try_reload_model(self, config_file=None):
        if config_file:
            config_path = os.path.join(self.config.resource.model_dir, config_file)
            shutil.copy(config_path, self.config.resource.model_best_config_path)
        try:
            if self.config.internet.distributed and not config_file:
                self.try_reload_model_from_internet()
            else:
                if self.need_reload and need_to_reload_best_model_weight(self.agent_model):
                    load_best_model_weight(self.agent_model)
        except Exception as e:
            logger.error(e)

    def try_reload_model_from_internet(self, config_file=None):
        response = http_request(self.config.internet.get_latest_digest)
        if response is None:
            logger.error(f"无法连接到远程服务器！请检查网络连接，并重新打开客户端")
            return
        digest = response['data']['digest']

        if digest != self.agent_model.fetch_digest(self.config.resource.model_best_weight_path):
            logger.info(f"正在下载最新权重，请稍后...")
            if download_file(self.config.internet.download_url, self.config.resource.model_best_weight_path):
                logger.info(f"权重下载完毕！开始训练...")
                try:
                    load_best_model_weight(self.agent_model)
                except ValueError as e:
                    logger.error(f"权重架构不匹配，自动重新加载 {e}")
                    self.try_reload_model(config_file='model_192x10_config.json')
                except Exception as e:
                    logger.error(f"加载权重发生错误：{e}，稍后重新下载")
                    os.remove(self.config.resource.model_best_weight_path)
                    self.try_reload_model_from_internet()
            else:
                logger.error(f"权重下载失败！请检查网络连接，并重新打开客户端")
        else:
            logger.info(f"检查完毕，权重未更新")

    def close(self):
        self.done = True
        # Close all pipes safely
        for pipe in self.pipes:
            try:
                # Check if pipe has 'closed' attribute or use alternative check
                if hasattr(pipe, 'closed') and not pipe.closed:
                    pipe.close()
                elif not hasattr(pipe, 'closed'):
                    # For older Python versions without 'closed' attribute
                    pipe.close()
            except Exception as e:
                logger.warning(f"Error closing pipe: {e}")
        self.pipes.clear()
