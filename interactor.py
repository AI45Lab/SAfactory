import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import httpx
from core.llm import LLM, BaseURLProvider
from core.data_manager.manager import DataManager

log = logging.getLogger("interactor")

@dataclass(slots=True, frozen=True)
class ActorHandle:
    """
    A lightweight reference to an env actor exposed via HTTP.

    base_url example:
      - remote: http://<ray-head-ip>:36663
      - local:  http://127.0.0.1:36663
    """
    base_url: str
    env_name: str
    env_id: str


class ActorPool(Protocol):
    pool_size: int

    async def start(self) -> None: ...
    async def acquire(self) -> Optional[ActorHandle]: ...
    async def done(self, actor: ActorHandle) -> None: ...
    async def aclose(self) -> None: ...


class Interactor:
    """
    Interactor drives episodes:
      get_task_prompt -> LLM -> step (repeat) until terminated/truncated or prompt empty.
    """

    def __init__(
        self,
        pool: ActorPool,
        base_url_provider: BaseURLProvider,
        api_key: str,
        model: str,
        data_manager: DataManager,
        temperature: float = 0.3,
        max_steps: int = 1000,
        message_cut: int = 3,
        env_http_timeout_s: float = 30.0,
        max_workers: Optional[int] = None,
        http_retries: int = 2,
        http_retry_backoff_s: float = 0.5,
        verbose: bool = True,
        log_actions_preview_chars: int = 120,
    ):
        self.pool = pool
        self.max_steps = int(max_steps)
        self.message_cut = int(message_cut)
        self.max_workers = int(max_workers) if max_workers is not None else None

        self.http_retries = max(0, int(http_retries))
        self.http_retry_backoff_s = float(http_retry_backoff_s)
        self.verbose = bool(verbose)

        self.log_actions_preview_chars = max(0, int(log_actions_preview_chars))
        
        self.model = model
        self.data_manager = data_manager

        base_url = self._get_base_url(base_url_provider)
        self.llm = LLM(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=float(temperature),
        )

        limits = httpx.Limits(
            max_connections=max(64, pool.pool_size * 8),
            max_keepalive_connections=max(32, pool.pool_size * 4),
        )
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(env_http_timeout_s),
            limits=limits,
            trust_env=True,
        )

        log.info(
            "Interactor initialized: model=%s temp=%.3f max_steps=%d message_cut=%d http_timeout=%.1fs retries=%d",
            model, float(temperature), self.max_steps, self.message_cut, float(env_http_timeout_s), self.http_retries
        )

    @staticmethod
    def _get_base_url(provider: BaseURLProvider) -> str:
        # Compatibility with both provider.get_base_url() and provider.get_base_url(None)
        try:
            return provider.get_base_url()
        except TypeError:
            return provider.get_base_url(None)

    def _trim_messages(self, prompt: Any) -> List[Dict[str, Any]]:
        """
        Keep system + last N turns for better LLM cost/control.
        """
        if not isinstance(prompt, list) or not prompt:
            return []
        if self.message_cut <= 0:
            return prompt

        out: List[Dict[str, Any]] = []
        start = 0
        if isinstance(prompt[0], dict) and prompt[0].get("role") == "system":
            out.append(prompt[0])
            start = 1

        tail = prompt[start:]
        out.extend(tail[-(self.message_cut * 2):])
        return out

    def _url(self, a: ActorHandle, suffix: str) -> str:
        return f"{a.base_url.rstrip('/')}/{a.env_name}/{a.env_id}/{suffix.lstrip('/')}"

    async def _llm_generate(self, messages: List[Dict[str, Any]]) -> str:
        # Compatibility with llm.generate(messages=...) and llm.generate(...)
        try:
            return await self.llm.generate(messages=messages)
        except TypeError:
            return await self.llm.generate(messages)

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[dict] = None,
        ctx: str = "",
    ) -> Any:
        """
        Request helper with light retry:
          - retries timeouts / transport errors
          - retries 5xx
          - does NOT retry most 4xx
        """
        attempts = 1 + self.http_retries
        last_err: Optional[Exception] = None

        for i in range(attempts):
            t0 = time.time()
            try:
                r = await self.http.request(method, url, json=json_body)
                dt = time.time() - t0

                if 500 <= r.status_code <= 599:
                    raise httpx.HTTPStatusError(
                        f"upstream 5xx: {r.status_code}",
                        request=r.request,
                        response=r,
                    )

                r.raise_for_status()
                log.debug("HTTP %s %s (%s) -> %d in %.3fs", method, url, ctx, r.status_code, dt)
                return r.json()

            except (httpx.TimeoutException, httpx.TransportError) as e:
                dt = time.time() - t0
                last_err = e
                log.warning(
                    "HTTP %s %s (%s) transport/timeout (attempt %d/%d) after %.3fs: %s",
                    method, url, ctx, i + 1, attempts, dt, e
                )

            except httpx.HTTPStatusError as e:
                dt = time.time() - t0
                resp = getattr(e, "response", None)
                status = getattr(resp, "status_code", None)
                body_preview = ""
                try:
                    if resp is not None:
                        body_preview = (resp.text or "")[:200]
                except Exception:
                    body_preview = ""

                if status is not None and 500 <= int(status) <= 599:
                    last_err = e
                    log.warning(
                        "HTTP %s %s (%s) -> %s (attempt %d/%d) after %.3fs body=%r",
                        method, url, ctx, status, i + 1, attempts, dt, body_preview
                    )
                else:
                    log.error(
                        "HTTP %s %s (%s) -> %s after %.3fs body=%r (not retrying)",
                        method, url, ctx, status, dt, body_preview
                    )
                    raise

            if i + 1 < attempts:
                await asyncio.sleep(self.http_retry_backoff_s * (2 ** i))

        assert last_err is not None
        raise last_err

    async def _run_one_environment(self, a: ActorHandle, worker_id: int) -> float:
        total_reward = 0.0
        env_key = f"{a.env_name}_{a.env_id}"
        trajectory = ""
        
        session = await self.data_manager.create_session(
            env_id=a.env_id,
            llm_model=self.model,
        )

        for step_i in range(1, self.max_steps + 1):
            prompt_raw = await self._request_json("GET", self._url(a, "get_task_prompt"), ctx=f"{env_key}/prompt")
            prompt = self._trim_messages(prompt_raw)
            if not prompt:
                log.info("worker=%d env=%s: empty prompt -> end episode", worker_id, env_key)
                break

            action = await self._llm_generate(prompt)

            if self.log_actions_preview_chars > 0:
                preview = action.replace("\n", "\\n")[: self.log_actions_preview_chars]
                log.debug("worker=%d env=%s step=%d action_preview=%r", worker_id, env_key, step_i, preview)

            out = await self._request_json(
                "POST",
                self._url(a, "step"),
                json_body={"action": action},
                ctx=f"{env_key}/step",
            )

            reward = float(out.get("reward", 0.0) or 0.0)
            total_reward += reward

            terminated = bool(out.get("terminated", False))
            truncated = bool(out.get("truncated", False))
            done = terminated or truncated

            log.debug(
                "worker=%d env=%s step=%d reward=%.4f total=%.4f terminated=%s truncated=%s",
                worker_id, env_key, step_i, reward, total_reward, terminated, truncated
            )

            if terminated or truncated:
                log.info("worker=%d env=%s done: terminated=%s truncated=%s", worker_id, env_key, terminated, truncated)
                break
            
            # 记录交互步骤
            await self.data_manager.record_step(
                session=session,
                step_id=step_i,
                prompt=prompt,
                response=action,
                reward=reward,
                done=done
            )
            
            trajectory += f"Step {step_i}:\nResponse: {action}\n\n"
            
        await self.data_manager.update_session(
                session=session,
                trajectory=trajectory,
                total_reward=total_reward,
                is_completed=True
            )

        return total_reward

    async def run_all_environments(self) -> Dict[str, float]:
        """
        Run workers until pool is exhausted (acquire() returns None).
        """
        await self.pool.start()

        results: Dict[str, float] = {}
        lock = asyncio.Lock()

        worker_count = self.pool.pool_size
        if self.max_workers is not None:
            worker_count = max(1, min(worker_count, int(self.max_workers)))

        log.info("run start: pool_size=%d workers=%d", self.pool.pool_size, worker_count)

        async def worker(worker_id: int) -> None:
            while True:
                actor = await self.pool.acquire()
                if actor is None:
                    log.info("worker=%d: no more actors -> exit", worker_id)
                    return

                env_key = f"{actor.env_name}_{actor.env_id}"
                t0 = time.time()

                try:
                    log.info("worker=%d acquired env=%s base_url=%s", worker_id, env_key, actor.base_url)
                    reward = await self._run_one_environment(actor, worker_id)
                except Exception as e:
                    reward = float("nan")
                    log.exception("worker=%d env=%s ERROR during episode: %s", worker_id, env_key, e)
                finally:
                    try:
                        await self.pool.done(actor)
                        log.info("worker=%d env=%s returned to pool (close+refill issued)", worker_id, env_key)
                    except Exception as e:
                        log.exception("worker=%d env=%s ERROR in pool.done(): %s", worker_id, env_key, e)

                dt = time.time() - t0
                log.info("worker=%d env=%s reward=%.6f time=%.2fs", worker_id, env_key, reward, dt)

                async with lock:
                    results[env_key] = reward

        tasks = [asyncio.create_task(worker(i), name=f"worker-{i}") for i in range(worker_count)]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            log.warning("run cancelled: cancelling workers...")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        except Exception:
            log.exception("run failed: cancelling workers...")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        log.info("run finished: episodes=%d", len(results))
        return results

    async def aclose(self) -> None:
        try:
            await self.http.aclose()
        except Exception:
            log.exception("failed to close http client (ignored)")
        try:
            await self.pool.aclose()
        except Exception:
            log.exception("failed to close pool (ignored)")
