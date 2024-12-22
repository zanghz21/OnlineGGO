BASE_DIR=""
TIME_STR=""
N_EVALS=50

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi
echo "Singularity opts: ${SINGULARITY_OPTS}"

singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_onlineGGO.sif \
    python env_search/traffic_mapf/parse_multi_process_eval.py \
    --base_dir=$BASE_DIR --time_str=$TIME_STR --n_evals=$N_EVALS
  
BASE_DIR=$BASE_DIR/logs

singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_onlineGGO.sif \
    python env_search/traffic_mapf/parse_eval.py --base_dir=$BASE_DIR --time_str=$TIME_STR