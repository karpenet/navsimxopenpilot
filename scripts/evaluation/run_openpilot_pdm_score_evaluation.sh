TRAIN_TEST_SPLIT=navhard_two_stage
CACHE_PATH="${CACHE_PATH:-$HOME/projects/navsim_workspace/exp/metric_cache}"
SYNTHETIC_SENSOR_PATH="${SYNTHETIC_SENSOR_PATH:-$HOME/projects/navsim_workspace/dataset/navhard_two_stage/sensor_blobs}"
SYNTHETIC_SCENES_PATH="${SYNTHETIC_SCENES_PATH:-$HOME/projects/navsim_workspace/dataset/navhard_two_stage/synthetic_scene_pickles}"

# OpenpilotAgent (navsim.agents.openpilot.op_agent): ONNX under OPENPILOT_MODELD_MODELS (see openpilot_agent.yaml).
# Default config uses driving_vision_fp32.onnx (see scripts/openpilot/convert_driving_vision_fp16_to_fp32.py).
export OPENPILOT_MODELD_MODELS="${OPENPILOT_MODELD_MODELS:-$HOME/projects/navsim_workspace/checkpoints}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"

# Sequential worker: one CUDA ONNX session at a time (thread pool + GPU causes OOM / stream capture errors).
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=openpilot_agent \
worker=sequential \
experiment_name=openpilot_agent \
metric_cache_path=$CACHE_PATH \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH