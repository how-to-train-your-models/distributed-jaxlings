"""Fixtures expose each exercise function/class by name.

The same test function works in two contexts:
- pytest: receives the exercise from `exercises/chapter_02.py` via these fixtures
- notebook judge: receives the student's in-cell function via `judge.check(fn)`
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import pytest

from src.exercises import chapter_02


@pytest.fixture
def Block():
    return chapter_02.Block


@pytest.fixture
def TinyGPT():
    return chapter_02.TinyGPT


@pytest.fixture
def shard_model_and_batch():
    return chapter_02.shard_model_and_batch


@pytest.fixture
def loss_fn():
    return chapter_02.loss_fn


@pytest.fixture
def train_step():
    return chapter_02.train_step
