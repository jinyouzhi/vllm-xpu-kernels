# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest tests/test_moe_sum.py`.
"""

import pytest
import torch

from tests.register_ops import moe_sum
from tests.utils import format_tc, opcheck

TOP_KS = [2, 6]

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "m": [1, 33],
        "k": [128, 256],
    },
}


@pytest.mark.parametrize("m", [1, 33, 64, 222])
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16],
                         ids=format_tc)
def test_moe_sum(m: int, topk: int, k: int, dtype: torch.dtype):
    input = torch.randn((m, topk, k), device="xpu", dtype=dtype)
    actual = torch.empty((m, k), device="xpu", dtype=dtype)

    expected = input.sum(dim=1)
    moe_sum(input, actual)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0)

    opcheck(torch.ops._moe_C.moe_sum, (input, actual, None, None))


@pytest.mark.parametrize("m", [1, 33])
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("k", [128, 511])
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16],
                         ids=format_tc)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64],
                         ids=format_tc)
@pytest.mark.parametrize("use_expert_map", [False, True])
def test_moe_sum_pad_aware(m: int, topk: int, k: int, dtype: torch.dtype,
                           index_dtype: torch.dtype,
                           use_expert_map: bool):
    num_experts = 8
    input = torch.randn((m, topk, k), device="xpu", dtype=dtype)
    actual = torch.empty((m, k), device="xpu", dtype=dtype)
    topk_ids = torch.randint(0,
                             num_experts, (m, topk),
                             device="xpu",
                             dtype=index_dtype)
    topk_ids[::2, -1] = -1

    expert_map = None
    valid = topk_ids >= 0
    if use_expert_map:
        expert_map = torch.arange(num_experts,
                                  device="xpu",
                                  dtype=torch.int32)
        expert_map[num_experts // 2:] = -1
        safe_ids = topk_ids.clamp_min(0).long()
        valid.logical_and_(expert_map[safe_ids] >= 0)

    expected = (input * valid.unsqueeze(-1).to(dtype)).sum(dim=1)
    moe_sum(input, actual, topk_ids, expert_map)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0)
    opcheck(torch.ops._moe_C.moe_sum,
            (input, actual, topk_ids, expert_map))
