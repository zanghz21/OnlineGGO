LOGDIR="$1"
N_WORKERS="$2"
N_EVALS="$3"
ALL_RESULTS_DIR="$4"

shift 4

while getopts "p:r:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      r) RELOAD_PATH=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done


SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi
echo "Singularity opts: ${SINGULARITY_OPTS}"

singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_onlineGGO.sif \
    python env_search/traffic_mapf/multi_process_eval.py --logdir=$LOGDIR \
    --n_workers=$N_WORKERS --n_evals=$N_EVALS --all_results_dir=$ALL_RESULTS_DIR