print_header() {
  echo
  echo "------------- $1 -------------"
}

LOAD_LOGDIR="$1"
NUM_WORKERS="$2"
N_EVALS="$3"
ALL_RESULTS_DIR="$4"
PARTITION="$5"
TOTAL_TIME="$6"

shift 6

DRY_RUN=""
RELOAD_ARG=""
while getopts "dr:" opt; do
  case $opt in
    d)
      echo "Using DRY RUN"
      DRY_RUN="1"
      ;;
    r)
      echo "Using RELOAD: $OPTARG"
      RELOAD_ARG="-r $OPTARG"
      ;;
  esac
done

DATE="$(date +'%Y-%m-%d_%H-%M-%S')"
LOGDIR="slurm_eval_logs/slurm_${DATE}"
echo "SLURM Log directory: ${LOGDIR}"
mkdir -p "$LOGDIR"
SEARCH_SCRIPT="$LOGDIR/search.slurm"
SEARCH_OUT="$LOGDIR/search.out"

echo "\
#!/bin/bash
#SBATCH --job-name=RM_search_${DATE}
#SBATCH -N 1
#SBATCH -p $PARTITION
#SBATCH -t $TOTAL_TIME
#SBATCH -n $NUM_WORKERS
#SBATCH --account=SOME_ACCOUNT
#SBATCH --output $SEARCH_OUT
#SBATCH --error $SEARCH_OUT

echo
echo \"========== Start ==========\"
date

bash scripts/eval/eval_trafficflow_local.sh $LOAD_LOGDIR $NUM_WORKERS $N_EVALS $ALL_RESULTS_DIR

echo
echo \"========== Done ==========\"

date" >"$SEARCH_SCRIPT"

if [ -z "$DRY_RUN" ]; then sbatch "$SEARCH_SCRIPT"; fi

print_header "Monitoring Instructions"
echo $SEARCH_OUT
echo "\
To view output from the search and main script, run:

  tail -f $SEARCH_OUT
"