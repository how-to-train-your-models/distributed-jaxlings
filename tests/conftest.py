"""Fixtures expose each exercise function/class by name.

The same test function works in two contexts:
- pytest: receives the exercise from `src.solutions.chapter_02` (or, for the
  shared model components, from `src.common.models`) via these fixtures
- notebook judge: receives the student's in-cell function via `judge.check(fn)`
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import pytest

from src.common import models as common_models
from src.solutions import chapter_02 as solutions_ch02


@pytest.fixture
def Block():
    return common_models.Block


@pytest.fixture
def TinyGPT():
    return common_models.TinyGPT


@pytest.fixture
def shard_model_and_batch():
    return solutions_ch02.shard_model_and_batch


@pytest.fixture
def loss_fn():
    return solutions_ch02.loss_fn


@pytest.fixture
def train_step():
    return solutions_ch02.train_step
