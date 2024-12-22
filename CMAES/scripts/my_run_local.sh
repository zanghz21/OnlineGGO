#!/bin/bash
USAGE="Usage: bash scripts/run_local.sh CONFIG SEED NUM_WORKERS [-p PROJECT_DIR] [-r RELOAD_PATH]"

CONFIG=config/traffic_mapf/win_r.gin
# CONFIG=config/traffic_mapf/ggo_33x36.gin
SEED=11
NUM_WORKERS=32
PROJECT_DIR=""

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1
set -u  # Uninitialized vars are error.

#
# Run the experiment.
#

SCHEDULER_FILE=".$(date +'%Y-%m-%d_%H-%M-%S')_scheduler_info.json"
PIDS_TO_KILL=()

# Use different port number so multiple jobs can start on one node. 8786
# is the default port. Then add offset of 10 and then the seed.
SCHEDULER_PORT=$((8786 + 10 + $SEED))

print_header "Starting Dask scheduler on port $SCHEDULER_PORT"
# shellcheck disable=SC2086
dask-scheduler --port $SCHEDULER_PORT --scheduler-file $SCHEDULER_FILE &
PIDS_TO_KILL+=("$!")
sleep 2 # Wait for scheduler to start.

print_header "Starting Dask workers"
# shellcheck disable=SC2086
dask-worker --memory-limit="4 GiB" \
    --scheduler-file $SCHEDULER_FILE \
    --nprocs $NUM_WORKERS \
    --nthreads 1 &
PIDS_TO_KILL+=("$!")
sleep 5

print_header "Running experiment"
echo
print_thick_line
# shellcheck disable=SC2086
python env_search/main.py \
    --config "$CONFIG" \
    --address "127.0.0.1:$SCHEDULER_PORT" \
    --seed "$SEED"
print_thick_line

#
# Clean Up.
#

print_header "Cleanup"
for pid in ${PIDS_TO_KILL[*]}
do
  kill -9 "${pid}"
done

rm $SCHEDULER_FILE
