set -ex

LIB_TORCH_PATH=""
EIGEN_PATH=""
MINIDNN_PATH=""

mkdir -p simulators/base_lns simulators/sum_ovc simulators/net simulators/offline simulators/offline_lns
cmake -B guided-pibt-build ./guided-pibt -DDEV=OFF -DSWAP=ON \
    -DGUIDANCE=ON -DGUIDANCE_LNS=10 -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DRELAX=100 \
    -DOBJECTIVE=3 -DFOCAL_SEARCH=OFF -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_PREFIX_PATH=$LIB_TORCH_PATH \
    -DPYTHON_EXECUTABLE=$(which python3)

make -C guided-pibt-build -j
mv guided-pibt-build/py_driver.cpython-39-x86_64-linux-gnu.so simulators/base_lns

cmake -B guided-pibt-build ./guided-pibt -DDEV=OFF -DSWAP=ON \
    -DGUIDANCE=ON -DGUIDANCE_LNS=OFF -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DRELAX=100 \
    -DOBJECTIVE=3 -DFOCAL_SEARCH=OFF -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_PREFIX_PATH=$LIB_TORCH_PATH \
    -DEIGEN3_INCLUDE_DIR=$EIGEN_PATH \
    -DMINIDNN_DIR=$MINIDNN_PATH \
    -DPYTHON_EXECUTABLE=$(which python3)
make -C guided-pibt-build -j
mv guided-pibt-build/py_driver.cpython-39-x86_64-linux-gnu.so simulators/sum_ovc

cmake -B guided-pibt-build ./guided-pibt -DDEV=OFF -DSWAP=ON \
    -DGUIDANCE=ON -DGUIDANCE_LNS=OFF -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DRELAX=100 \
    -DOBJECTIVE=4 -DFOCAL_SEARCH=OFF -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_PREFIX_PATH=$LIB_TORCH_PATH \
    -DEIGEN3_INCLUDE_DIR=$EIGEN_PATH \
    -DMINIDNN_DIR=$MINIDNN_PATH \
    -DPYTHON_EXECUTABLE=$(which python3)
make -C guided-pibt-build -j
mv guided-pibt-build/py_driver.cpython-39-x86_64-linux-gnu.so simulators/net

cmake -B guided-pibt-build ./guided-pibt -DDEV=OFF -DSWAP=ON \
    -DGUIDANCE=ON -DGUIDANCE_LNS=OFF -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DRELAX=100 \
    -DOBJECTIVE=5 -DFOCAL_SEARCH=OFF -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_PREFIX_PATH=$LIB_TORCH_PATH \
    -DEIGEN3_INCLUDE_DIR=$EIGEN_PATH \
    -DMINIDNN_DIR=$MINIDNN_PATH \
    -DPYTHON_EXECUTABLE=$(which python3)
make -C guided-pibt-build -j
mv guided-pibt-build/py_driver.cpython-39-x86_64-linux-gnu.so simulators/offline
mv guided-pibt-build/period_on_sim.cpython-39-x86_64-linux-gnu.so simulators/offline

cmake -B guided-pibt-build ./guided-pibt -DDEV=OFF -DSWAP=ON \
    -DGUIDANCE=ON -DGUIDANCE_LNS=10 -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DRELAX=100 \
    -DOBJECTIVE=5 -DFOCAL_SEARCH=OFF -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_PREFIX_PATH=$LIB_TORCH_PATH \
    -DEIGEN3_INCLUDE_DIR=$EIGEN_PATH \
    -DMINIDNN_DIR=$MINIDNN_PATH \
    -DPYTHON_EXECUTABLE=$(which python3)
make -C guided-pibt-build -j
mv guided-pibt-build/py_driver.cpython-39-x86_64-linux-gnu.so simulators/offline_lns
mv guided-pibt-build/period_on_sim.cpython-39-x86_64-linux-gnu.so simulators/offline_lns