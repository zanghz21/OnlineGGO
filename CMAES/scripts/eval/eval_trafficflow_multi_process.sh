PROJECT_DIR=""
SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0 --bind ${PROJECT_DIR}:${PROJECT_DIR}"
LOGDIR=""

singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_onlineGGO.sif \
    python env_search/traffic_mapf/multi_process_eval.py --logdir=$LOGDIR --n_workers=32 --n_evals=50 \
    --all_results_dir=../results_multi_process --is_runtime
