"""The same AlfWorld class in an isolated worker; no environment semantics change."""
from __future__ import annotations

import multiprocessing


def worker(connection, game, seed, max_steps):
    from .environment import AlfWorld
    env = None
    try:
        env = AlfWorld(game, seed, max_steps)
        initial = env.reset()
        connection.send(("ok", initial))
        while True:
            command, argument = connection.recv()
            if command == "close":
                break
            if command != "step":
                raise ValueError("Unknown environment RPC")
            connection.send(("ok", env.step(argument)))
    except EOFError:
        pass
    except Exception as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        if env is not None:
            env.close()
        connection.close()


class RemoteAlfWorld:
    supports_async_step = True

    def __init__(self, game, seed=0, max_steps=50):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker, args=(child, str(game), seed, max_steps), daemon=True)
        self.process.start()
        child.close()
        self.closed = False

    def receive(self):
        if not self.connection.poll(120):
            raise TimeoutError("Environment worker response")
        status, value = self.connection.recv()
        if status != "ok":
            raise RuntimeError(f"Environment worker: {value}")
        return value

    def reset(self):
        return self.receive()

    def step_async(self, action):
        self.connection.send(("step", action))

    def step_wait(self):
        return self.receive()

    def step(self, action):
        self.step_async(action)
        return self.step_wait()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.connection.send(("close", None))
        except (BrokenPipeError, EOFError, OSError):
            pass
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.connection.close()
