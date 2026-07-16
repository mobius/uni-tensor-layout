"""Multi-device layout partitioners + cost model + plan runner."""

from uni_cute_tensor.partition.cost_model import (
    CalibrationReport,
    CalibrationSample,
    DeviceModel,
    PlacementChoice,
    calibrate_from_samples,
    choose_best_placement,
    estimate_gemm_placement,
    get_device_model,
    load_calibration,
    prediction_error_report,
    save_calibration,
    try_autoload_calibration,
)
from uni_cute_tensor.partition.dispatch import (
    DispatchRecommendation,
    recommend_backend,
)
from uni_cute_tensor.partition.multi_device import (
    BACKEND_ATOMS,
    PlacementPlan,
    Shard,
    partition_matrix_cols,
    partition_matrix_k,
    partition_matrix_rows,
    partition_to_devices,
)
from uni_cute_tensor.partition.pcie_cost import estimate_h2d_seconds
from uni_cute_tensor.partition.runner import PlanRunResult, execute_auto, execute_plan

__all__ = [
    "PlacementPlan",
    "Shard",
    "BACKEND_ATOMS",
    "partition_matrix_rows",
    "partition_matrix_cols",
    "partition_matrix_k",
    "partition_to_devices",
    "estimate_h2d_seconds",
    "PlacementChoice",
    "DeviceModel",
    "CalibrationSample",
    "CalibrationReport",
    "choose_best_placement",
    "estimate_gemm_placement",
    "calibrate_from_samples",
    "save_calibration",
    "load_calibration",
    "try_autoload_calibration",
    "get_device_model",
    "prediction_error_report",
    "recommend_backend",
    "DispatchRecommendation",
    "PlanRunResult",
    "execute_plan",
    "execute_auto",
]
