#!/usr/bin/env bash
# PDM score evaluation for OpenpilotAgent (navsim.agents.openpilot.op_agent).
# Uses the same synthetic data layout as:
#   python -m navsim.planning.script.run_openpilot_agent_compute \
#     --scene-pkl .../synthetic_scene_pickles/<token>.pkl \
#     --sensor-blobs .../sensor_blobs
#
# Requires NUPLAN_MAPS_ROOT. Build or point metric_cache_path at a cache that
# covers the tokens in this train_test_split.
#
# OpenpilotAgent loads ONNX in initialize(); ensure agent.vision_model_path and
# agent.policy_model_path are set (Hydra overrides or openpilot_agent.yaml).
# Typical openpilot layout: .../openpilot/selfdrive/modeld/models/{driving_vision.onnx,driving_policy.onnx}
#
set -euo pipefail

# ONNX: default Hydra config uses driving_vision_fp32.onnx + driving_policy.onnx (see scripts/openpilot/convert_driving_vision_fp16_to_fp32.py).
export OPENPILOT_MODELD_MODELS="${OPENPILOT_MODELD_MODELS:-${HOME}/projects/navsim_workspace/checkpoints}"

TRAIN_TEST_SPLIT=navhard_two_stage

# Default: workspace dataset/navhard_two_stage (same layout as run_cv_pdm_score_evaluation.sh).
# Override DATASET_ROOT or SYNTHETIC_*_PATH if your data lives elsewhere.
DATASET_ROOT="${DATASET_ROOT:-${HOME}/projects/navsim_workspace/dataset/navhard_two_stage}"
SYNTHETIC_SENSOR_PATH="${SYNTHETIC_SENSOR_PATH:-${DATASET_ROOT}/sensor_blobs}"
SYNTHETIC_SCENES_PATH="${SYNTHETIC_SCENES_PATH:-${DATASET_ROOT}/synthetic_scene_pickles}"

CACHE_PATH="${CACHE_PATH:-${HOME}/projects/navsim_workspace/exp/metric_cache}"

if [[ -z "${NAVSIM_DEVKIT_ROOT:-}" ]]; then
  echo "Set NAVSIM_DEVKIT_ROOT to the navsim repo root (directory that contains navsim/)." >&2
  exit 1
fi

# Editable/source checkout: ensure the repo's ``navsim`` package is imported.
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score.py" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  agent=openpilot_agent \
  worker=sequential \
  experiment_name=op_agent \
  metric_cache_path="${CACHE_PATH}" \
  synthetic_sensor_path="${SYNTHETIC_SENSOR_PATH}" \
  synthetic_scenes_path="${SYNTHETIC_SCENES_PATH}"
