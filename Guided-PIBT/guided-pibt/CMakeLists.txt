cmake_minimum_required(VERSION 3.13)
project(lifelong)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-lto")

# Print the value of PYTHON
message(STATUS "PYTHON: ${PYTHON}")

if (DEV)
    message(STATUS "DEV is enabled")
    add_definitions(-DDEV)
endif()
# set(DEV OFF)

# set(NSWAP ON)
if (SWAP)
    message(STATUS "SWAP is enabled")
    add_definitions(-DSWAP)
endif()

if(GUIDANCE)
    message(STATUS "Guidance is enabled ")
    add_definitions(-DGUIDANCE)
endif()

if(RELAX)
    message(STATUS "RELAX is enabled: ${RELAX} traj/dist/heu table initialization per timestep")
    add_definitions(-DRELAX=${RELAX})
else()
    message(STATUS "RELAX is disabled default 100000 initialization per timestep")
    add_definitions(-DRELAX=100000)
endif()

if(FLOW_GUIDANCE)
    message(STATUS "FLOW_GUIDANCE is enabled, re init every: ${FLOW_GUIDANCE}")
    add_definitions(-DFLOW_GUIDANCE=${FLOW_GUIDANCE})
endif()

if(GUIDANCE_LNS)
    message(STATUS "Guidance LNS is enabled: ${GUIDANCE_LNS}")
    add_definitions(-DGUIDANCE_LNS=${GUIDANCE_LNS})
endif()

if(LNS_GROUP_SIZE)
    message(STATUS "LNS_GROUP_SIZE is set to: ${LNS_GROUP_SIZE}")
    add_definitions(-DLNS_GROUP_SIZE=${LNS_GROUP_SIZE})
else()
    message(STATUS "LNS_GROUP_SIZE is set to: 10")
    add_definitions(-DLNS_GROUP_SIZE=10)
endif()

if(LNS_EXPOUT)
    message(STATUS "Print lns exp data")
    add_definitions(-DLNS_EXPOUT)
endif()

if (LNS_DES_METHOD)
    message(STATUS "LNS_DES_METHOD is enabled: ${LNS_DES_METHOD}")
    add_definitions(-DLNS_DES_METHOD=${LNS_DES_METHOD})
endif()

if(INIT_PP)
    message(STATUS "INIT_PP is enabled")
    add_definitions(-DINIT_PP)
endif()

# if(MAPFT)
#     message(STATUS "MAPFT is enabled")
#     add_definitions(-DMAPFT)
# endif()
set(MAPFT OFF)

if(OBJECTIVE)
    message(STATUS "OBJECTIVE is enabled: ${OBJECTIVE}")
    add_definitions(-DOBJECTIVE=${OBJECTIVE})
else()
    message(STATUS "OBJECTIVE is default none")

    add_definitions(-DOBJECTIVE=0)
endif()


if(FOCAL_SEARCH)
    message(STATUS "FOCAL_SEARCH is enabled: ${FOCAL_SEARCH}")
    add_definitions(-DFOCAL_SEARCH=${FOCAL_SEARCH})
endif()

add_definitions(-DROBOT_RUNNERS)

IF(NOT CMAKE_BUILD_TYPE)
    # SET(CMAKE_BUILD_TYPE "DEBUG")
    SET(CMAKE_BUILD_TYPE "RELEASE")
    #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
    #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

include_directories("inc")
include_directories("traffic_mapf")

# Find Boost
ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)
#SET(Boost_USE_STATIC_LIBS ON)
find_package( Boost 1.71.0 REQUIRED COMPONENTS program_options system filesystem log log_setup)

include_directories( ${Boost_INCLUDE_DIRS} )


# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# include_directories(lifelong ${TORCH_INCLUDE_DIRS})

# Add Eigen directory
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${MINIDNN_DIR})

find_package(Python 3.9 COMPONENTS Interpreter Development REQUIRED)

add_subdirectory(pybind11)

message(STATUS "Python support is disabled")

file(GLOB SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX ".*src/py_driver\.cpp")
list(FILTER SOURCES EXCLUDE REGEX ".*src/period_on_sim\.cpp")
list(FILTER SOURCES EXCLUDE REGEX ".*inc/period_on_sim\.h")
file(GLOB TRAFFIC_SOURCES "traffic_mapf/*.cpp")

add_executable(lifelong ${SOURCES} ${TRAFFIC_SOURCES})
target_link_libraries(lifelong ${Boost_LIBRARIES} ${TORCH_LIBRARIES} pthread)

pybind11_add_module(py_driver "src/py_driver.cpp" ${SOURCES} ${TRAFFIC_SOURCES})
target_include_directories(py_driver PRIVATE ${pybind11_INCLUDE_DIRS})
target_link_libraries(py_driver PRIVATE ${Boost_LIBRARIES} ${TORCH_LIBRARIES} pthread)

pybind11_add_module(period_on_sim "src/period_on_sim.cpp" "inc/period_on_sim.h" ${SOURCES} ${TRAFFIC_SOURCES})
target_include_directories(period_on_sim PRIVATE ${pybind11_INCLUDE_DIRS})
target_link_libraries(period_on_sim PRIVATE ${Boost_LIBRARIES} ${TORCH_LIBRARIES} pthread)

add_custom_target(clean_all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/lifelong
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/build
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/*.so
)


set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES
    "${CMAKE_BINARY_DIR}/lifelong;
     ${CMAKE_BINARY_DIR}/build;
     ${CMAKE_BINARY_DIR}/*.so"
)



