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

from __future__ import annotations

import logging
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from max.driver import CPU, load_devices
from max.pipelines.lib import MemoryEstimator
from max.pipelines.lib.interfaces import ArchConfigWithKVCache
from test_common.mocks import DummyPipelineConfig
from test_common.pipeline_model_dummy import (
    DUMMY_LLAMA_ARCH,
    DummyLlamaPipelineModel,
)


def test_memory_estimation__raise_oom_error_weights_size_exceeds_available_memory() -> (
    None
):
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "calculate_max_seq_len",
            return_value=100000,
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=50 * 1024 * 1024,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 5 * 1024 * 1024}
        with pytest.raises(
            RuntimeError, match="Model weights exceed available memory"
        ):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=None,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )

            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                DummyLlamaPipelineModel.estimate_weights_size(mock_config),
                DummyLlamaPipelineModel.estimate_activation_memory(
                    mock_config, mock_config.model.huggingface_config
                ),
            )


def test_memory_estimation__warns_when_activation_estimate_exceeds_but_weights_fit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When weights fit but weights + activations exceed free memory, warn and proceed."""
    GiB = 1024 * 1024 * 1024
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "calculate_max_seq_len",
            return_value=4096,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        # 20 GiB free, 15 GiB weights, 6 GiB activation = 21 GiB > 20 GiB
        # But weights alone (15 GiB) fit in 20 GiB.
        device_mock.return_value = {"free_memory": 20 * GiB}
        with caplog.at_level(logging.WARNING):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=1,
                max_length=128,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            # Should NOT raise — weights fit, activation estimate is suspect
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                model_weights_size=15 * GiB,
                activation_memory_size=6 * GiB,
            )

        assert "Activation estimate may be conservative" in caplog.text


def test_memory_estimation__still_rejects_when_weights_alone_exceed() -> None:
    """When weights alone exceed free memory, still raise RuntimeError."""
    GiB = 1024 * 1024 * 1024
    with (
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        # 10 GiB free, 15 GiB weights — weights alone don't fit
        device_mock.return_value = {"free_memory": 10 * GiB}
        with pytest.raises(
            RuntimeError, match="Model weights exceed available memory"
        ):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=None,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                model_weights_size=15 * GiB,
                activation_memory_size=0,
            )


def test_memory_estimation__kv_budget_fallback_when_activation_inflated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When activation estimate makes KV budget negative, fall back to weights-only."""
    GiB = 1024 * 1024 * 1024
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "calculate_max_seq_len",
            return_value=4096,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        # 24 GiB free × 0.9 util = 21.6 GiB usable
        # weights=15 GiB + activation=8 GiB = 23 GiB > 21.6 GiB → negative KV budget
        # But weights-only: 21.6 - 15 = 6.6 GiB → positive
        device_mock.return_value = {"free_memory": 24 * GiB}
        with caplog.at_level(logging.WARNING):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=1,
                max_length=128,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                model_weights_size=15 * GiB,
                activation_memory_size=8 * GiB,
            )

        assert "Using weights-only estimate for initial budget" in caplog.text


def test_memory_estimation__rejects_when_weights_only_budget_also_negative() -> None:
    """When even weights-only KV budget is negative, raise RuntimeError."""
    GiB = 1024 * 1024 * 1024
    with (
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        # 16 GiB free × 0.9 util = 14.4 GiB usable
        # weights = 15 GiB → weights-only budget = 14.4 - 15 = -0.6 GiB
        device_mock.return_value = {"free_memory": 16 * GiB}
        with pytest.raises(
            RuntimeError, match="don't leave room for KV cache"
        ):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=None,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                model_weights_size=15 * GiB,
                activation_memory_size=5 * GiB,
            )


def test_memory_estimation__infer_optimal_batch_size() -> None:
    # Max batch size on CPU is always 1.
    inferred_batch_size = MemoryEstimator._infer_optimal_batch_size(
        arch_config=MagicMock(spec=ArchConfigWithKVCache),
        devices=[CPU()],
    )
    assert inferred_batch_size == 1


@pytest.mark.skip("TODO: AITLIB-238")
def test_memory_estimation__raise_oom_error_all_defaults_no_valid_solution() -> (
    None
):
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=30000 * 1024 * 1024,
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_activation_memory",
            return_value=0,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 30641 * 1024 * 1024}
        with pytest.raises(
            RuntimeError,
        ):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=None,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                DummyLlamaPipelineModel.estimate_weights_size(mock_config),
                DummyLlamaPipelineModel.estimate_activation_memory(
                    mock_config, mock_config.model.huggingface_config
                ),
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_all_defaults(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=35000 * 1024 * 1024,
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_activation_memory",
            return_value=0,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with caplog.at_level(logging.WARNING):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=None,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                DummyLlamaPipelineModel.estimate_weights_size(mock_config),
                DummyLlamaPipelineModel.estimate_activation_memory(
                    mock_config, mock_config.model.huggingface_config
                ),
            )

        assert "Truncated model's default max_length from" in caplog.text


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_max_length_set() -> None:
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=35000 * 1024 * 1024,
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_activation_memory",
            return_value=0,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with pytest.raises(
            RuntimeError,
            match=r"Try reducing --max-length to \d+ .*supports batch size of",
        ):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=None,
                max_length=100000,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                DummyLlamaPipelineModel.estimate_weights_size(mock_config),
                DummyLlamaPipelineModel.estimate_activation_memory(
                    mock_config, mock_config.model.huggingface_config
                ),
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_max_batch_size_set() -> None:
    with (
        patch.object(
            DummyLlamaPipelineModel, "calculate_max_seq_len", return_value=4096
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=40000 * 1024 * 1024,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with pytest.raises(RuntimeError, match="reducing --max-batch-size to"):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=100000,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                DummyLlamaPipelineModel.estimate_weights_size(mock_config),
                DummyLlamaPipelineModel.estimate_activation_memory(
                    mock_config, mock_config.model.huggingface_config
                ),
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_max_batch_size_set_and_max_length_set() -> (
    None
):
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=40000 * 1024 * 1024,
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_activation_memory",
            return_value=0,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with pytest.raises(RuntimeError, match="reducing --max-batch-size to"):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=100000,
                max_length=4096,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                DummyLlamaPipelineModel.estimate_weights_size(mock_config),
                DummyLlamaPipelineModel.estimate_activation_memory(
                    mock_config, mock_config.model.huggingface_config
                ),
            )
