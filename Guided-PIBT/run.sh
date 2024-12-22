set -ex

./compile.sh

OUTPUT_DIR="output/"
mkdir -p $OUTPUT_DIR

./guided-pibt-build/lifelong --inputFile guided-pibt/benchmark-lifelong/sortation_small_0_800.json --planTimeLimit 10 --output ${OUTPUT_DIR}output.json -l ${OUTPUT_DIR}event_log.txt