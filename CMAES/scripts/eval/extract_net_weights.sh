#!/bin/bash
PROJECT_DIR=""

USAGE="Usage: bash scripts/plot_tile_usage.sh LOGDIR MODE DOMAIN"

LOGDIR=""
LOGDIR_TYPE="qd"
MODE="extreme"
DOMAIN="trafficMAPF" # ["competition", "trafficMAPF"], competition for PIBT-based, trafficMAPF for GPIBT-based


export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
# shift 4
while getopts "p:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done


if [ -z "${LOGDIR}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${LOGDIR_TYPE}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${MODE}" ]
then
  echo "${USAGE}"
  exit 1
fi

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi

singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_onlineGGO.sif \
  python env_search/analysis/tile_usage.py \
    --logdir "$LOGDIR" \
    --logdir-type "$LOGDIR_TYPE" \
    --mode "$MODE" \
    --domain "$DOMAIN"