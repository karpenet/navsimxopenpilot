TRAIN_TEST_SPLIT=navhard_two_stage
CACHE_PATH=$HOME/projects/navsim_workspace/exp/metric_cache
SYNTHETIC_SENSOR_PATH=$HOME/projects/navsim_workspace/dataset/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$HOME/projects/navsim_workspace/dataset/navhard_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
experiment_name=cv_agent \
metric_cache_path=$CACHE_PATH \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
