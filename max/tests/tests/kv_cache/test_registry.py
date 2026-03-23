# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Tests for KV cache registry functions: load_kv_manager."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.kv_cache import load_kv_manager
from max.kv_cache.registry import load_multi_kv_managers
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams


def create_kv_params(
    num_layers: int = 32,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 128,
    dtype: DType = DType.bfloat16,
) -> KVCacheParams:
    """Helper to create KVCacheParams with common defaults."""
    return KVCacheParams(
        dtype=dtype,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        devices=[DeviceRef.GPU()],
        page_size=page_size,
    )


class TestLoadKvManager:
    """Tests for load_kv_manager function."""

    @patch("max.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_manager_creates_manager(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """load_kv_manager should create a PagedKVCacheManager."""
        mock_manager = MagicMock()
        mock_paged_manager_cls.return_value = mock_manager

        params = create_kv_params()
        mock_session = MagicMock()

        result = load_kv_manager(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=1024 * 1024 * 1024,  # 1 GB
        )

        assert result == mock_manager
        mock_paged_manager_cls.assert_called_once()

    @patch("max.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_manager_passes_correct_params(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """load_kv_manager should pass correct params to PagedKVCacheManager."""
        params = create_kv_params(num_layers=16)
        mock_session = MagicMock()

        load_kv_manager(
            params=params,
            max_batch_size=8,
            max_seq_len=1024,
            session=mock_session,
            available_cache_memory=512 * 1024 * 1024,  # 512 MB
        )

        call_kwargs = mock_paged_manager_cls.call_args.kwargs
        assert call_kwargs["params"] == params
        assert call_kwargs["session"] == mock_session
        assert call_kwargs["total_num_pages"] > 0

    def test_load_kv_manager_rejects_zero_batch_size(self) -> None:
        """load_kv_manager should raise ValueError for batch_size <= 0."""
        params = create_kv_params()
        mock_session = MagicMock()

        with pytest.raises(
            ValueError, match="max_batch_size must be greater than 0"
        ):
            load_kv_manager(
                params=params,
                max_batch_size=0,
                max_seq_len=2048,
                session=mock_session,
                available_cache_memory=1024 * 1024 * 1024,
            )

    def test_load_kv_manager_rejects_negative_batch_size(self) -> None:
        """load_kv_manager should raise ValueError for negative batch_size."""
        params = create_kv_params()
        mock_session = MagicMock()

        with pytest.raises(
            ValueError, match="max_batch_size must be greater than 0"
        ):
            load_kv_manager(
                params=params,
                max_batch_size=-1,
                max_seq_len=2048,
                session=mock_session,
                available_cache_memory=1024 * 1024 * 1024,
            )

    @patch("max.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_manager_rejects_invalid_page_size(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """load_kv_manager should reject page sizes that aren't multiples of 128."""
        # Create params with invalid page size (not multiple of 128)
        params = KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=32,
            devices=[DeviceRef.GPU()],
            page_size=64,  # Invalid: not a multiple of 128
        )
        mock_session = MagicMock()

        with pytest.raises(ValueError, match="multiple of 128"):
            load_kv_manager(
                params=params,
                max_batch_size=16,
                max_seq_len=2048,
                session=mock_session,
                available_cache_memory=1024 * 1024 * 1024,
            )


class TestLoadKvManagers:
    """Tests for load_kv_managers function (plural - supports MultiKVCacheParams)."""

    @patch("max.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_managers_single_params(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """load_kv_managers should return a list with one manager for KVCacheParams."""
        mock_manager = MagicMock()
        mock_paged_manager_cls.return_value = mock_manager

        params = create_kv_params()
        mock_session = MagicMock()

        result = load_kv_manager(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=1024 * 1024 * 1024,
        )

        assert result == mock_manager

    @patch("max.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_managers_multi_params(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """load_kv_managers should return multiple managers for MultiKVCacheParams."""
        mock_manager1 = MagicMock(name="manager1")
        mock_manager2 = MagicMock(name="manager2")
        mock_paged_manager_cls.side_effect = [mock_manager1, mock_manager2]

        params1 = create_kv_params(num_layers=16)
        params2 = create_kv_params(num_layers=16)
        multi_params = MultiKVCacheParams.from_params(params1, params2)
        mock_session = MagicMock()

        result = load_multi_kv_managers(
            params=multi_params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=1024 * 1024 * 1024,
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == mock_manager1
        assert result[1] == mock_manager2

    @patch("max.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_managers_shares_total_pages(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """All managers from MultiKVCacheParams should get the same total_num_pages."""
        params1 = create_kv_params(num_layers=16)
        params2 = create_kv_params(num_layers=16)
        multi_params = MultiKVCacheParams.from_params(params1, params2)
        mock_session = MagicMock()

        load_multi_kv_managers(
            params=multi_params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=1024 * 1024 * 1024,
        )

        # Both calls should have the same total_num_pages
        calls = mock_paged_manager_cls.call_args_list
        assert len(calls) == 2
        total_pages_1 = calls[0].kwargs["total_num_pages"]
        total_pages_2 = calls[1].kwargs["total_num_pages"]
        assert total_pages_1 == total_pages_2

    def test_load_kv_managers_rejects_zero_batch_size(self) -> None:
        """load_kv_managers should raise ValueError for batch_size <= 0."""
        params = create_kv_params()
        mock_session = MagicMock()

        with pytest.raises(
            ValueError, match="max_batch_size must be greater than 0"
        ):
            load_kv_manager(
                params=params,
                max_batch_size=0,
                max_seq_len=2048,
                session=mock_session,
                available_cache_memory=1024 * 1024 * 1024,
            )

    @patch("max.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_managers_nested_multi_params(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """load_kv_managers should handle nested MultiKVCacheParams (if supported)."""
        mock_managers = [MagicMock(name=f"manager{i}") for i in range(3)]
        mock_paged_manager_cls.side_effect = mock_managers

        params1 = create_kv_params(num_layers=16)
        params2 = create_kv_params(num_layers=16)
        params3 = create_kv_params(num_layers=16)
        multi_params = MultiKVCacheParams.from_params(params1, params2, params3)
        mock_session = MagicMock()

        result = load_multi_kv_managers(
            params=multi_params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=1024 * 1024 * 1024,
        )

        assert len(result) == 3


class TestLoadKvManagerPostWeightLoadMemory:
    """Tests for post-weight-load memory adjustment in load_kv_manager."""

    @patch("max.kv_cache.registry.PagedKVCacheManager")
    def test_no_adjustment_when_devices_not_provided(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """When devices is None, use estimated budget without adjustment."""
        params = create_kv_params()
        mock_session = MagicMock()
        budget = 1024 * 1024 * 1024  # 1 GiB

        load_kv_manager(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=budget,
            devices=None,
        )

        mock_paged_manager_cls.assert_called_once()

    @patch("max.kv_cache.registry.PagedKVCacheManager")
    @patch("max.kv_cache.registry._query_actual_free_memory")
    def test_adjusts_budget_when_actual_free_is_lower(
        self,
        mock_query_free: MagicMock,
        mock_paged_manager_cls: MagicMock,
    ) -> None:
        """When actual free memory is lower than estimate, use actual."""
        mock_query_free.return_value = 2 * 1024 * 1024 * 1024  # 2 GiB free
        utilization = 0.9
        # actual_available = 2 GiB * 0.9 = 1.8 GiB
        # estimated budget = 4 GiB (higher than actual)
        estimated_budget = 4 * 1024 * 1024 * 1024

        mock_devices = [MagicMock()]
        params = create_kv_params()
        mock_session = MagicMock()

        load_kv_manager(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=estimated_budget,
            devices=mock_devices,
            device_memory_utilization=utilization,
        )

        # compute_num_device_blocks should have been called with the
        # lower actual_available, not the estimated_budget.
        call_kwargs = mock_paged_manager_cls.call_args.kwargs
        # The total_num_pages should reflect the reduced budget.
        # With 4 GiB budget we'd get more pages than with 1.8 GiB.
        assert call_kwargs["total_num_pages"] > 0

    @patch("max.kv_cache.registry.PagedKVCacheManager")
    @patch("max.kv_cache.registry._query_actual_free_memory")
    def test_keeps_estimate_when_actual_free_is_higher(
        self,
        mock_query_free: MagicMock,
        mock_paged_manager_cls: MagicMock,
    ) -> None:
        """When actual free memory confirms estimate, keep estimated budget."""
        mock_query_free.return_value = 20 * 1024 * 1024 * 1024  # 20 GiB free
        utilization = 0.9
        # actual_available = 20 GiB * 0.9 = 18 GiB (higher than estimate)
        estimated_budget = 2 * 1024 * 1024 * 1024  # 2 GiB

        mock_devices = [MagicMock()]
        params = create_kv_params()
        mock_session = MagicMock()

        load_kv_manager(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=estimated_budget,
            devices=mock_devices,
            device_memory_utilization=utilization,
        )

        mock_paged_manager_cls.assert_called_once()

    @patch("max.kv_cache.registry.PagedKVCacheManager")
    @patch("max.kv_cache.registry._query_actual_free_memory")
    def test_graceful_fallback_when_device_stats_unavailable(
        self,
        mock_query_free: MagicMock,
        mock_paged_manager_cls: MagicMock,
    ) -> None:
        """When device stats query fails, fall back to estimated budget."""
        mock_query_free.return_value = None  # stats unavailable

        mock_devices = [MagicMock()]
        params = create_kv_params()
        mock_session = MagicMock()
        budget = 1024 * 1024 * 1024

        load_kv_manager(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=budget,
            devices=mock_devices,
            device_memory_utilization=0.9,
        )

        mock_paged_manager_cls.assert_called_once()

    @patch("max.kv_cache.registry.PagedKVCacheManager")
    @patch("max.kv_cache.registry._query_actual_free_memory")
    def test_multi_device_sums_free_memory(
        self,
        mock_query_free: MagicMock,
        mock_paged_manager_cls: MagicMock,
    ) -> None:
        """With multiple devices, _query_actual_free_memory sums across all."""
        # 4 devices with 5 GiB free each = 20 GiB total
        mock_query_free.return_value = 20 * 1024 * 1024 * 1024
        mock_devices = [MagicMock() for _ in range(4)]
        params = create_kv_params()
        mock_session = MagicMock()

        load_kv_manager(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=1024 * 1024 * 1024,
            devices=mock_devices,
            device_memory_utilization=0.9,
        )

        mock_query_free.assert_called_once_with(mock_devices)
        mock_paged_manager_cls.assert_called_once()


class TestQueryActualFreeMemory:
    """Tests for _query_actual_free_memory helper."""

    def test_returns_sum_of_device_free_memory(self) -> None:
        """Should sum free_memory stat across all devices."""
        from max.kv_cache.registry import _query_actual_free_memory

        dev1 = MagicMock()
        dev1.stats = {"free_memory": 10 * 1024 * 1024 * 1024}
        dev2 = MagicMock()
        dev2.stats = {"free_memory": 8 * 1024 * 1024 * 1024}

        result = _query_actual_free_memory([dev1, dev2])
        assert result == 18 * 1024 * 1024 * 1024

    def test_returns_none_on_exception(self) -> None:
        """Should return None if device stats are unavailable."""
        from max.kv_cache.registry import _query_actual_free_memory

        dev = MagicMock()
        dev.stats.__getitem__.side_effect = KeyError("free_memory")

        result = _query_actual_free_memory([dev])
        assert result is None

    def test_single_device(self) -> None:
        """Should work with a single device."""
        from max.kv_cache.registry import _query_actual_free_memory

        dev = MagicMock()
        dev.stats = {"free_memory": 24 * 1024 * 1024 * 1024}

        result = _query_actual_free_memory([dev])
        assert result == 24 * 1024 * 1024 * 1024


class TestLoadKvManagerVirtualDevice:
    """Tests for virtual device mode behavior."""

    @patch("max.kv_cache.registry.is_virtual_device_mode", return_value=True)
    def test_load_kv_manager_returns_mock_in_virtual_mode(
        self, mock_is_virtual: MagicMock
    ) -> None:
        """In virtual device mode, load_kv_manager should return a Mock."""
        params = create_kv_params()
        mock_session = MagicMock()

        result = load_kv_manager(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=1024 * 1024 * 1024,
        )

        assert isinstance(result, Mock)
