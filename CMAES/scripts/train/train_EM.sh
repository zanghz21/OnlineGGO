CONFIG=config/traffic_mapf/warehouse.gin
NUM_WORKERS=24
SEED=0
bash scripts/run_slurm_psc_EM_local.sh $CONFIG $NUM_WORKERS $SEED