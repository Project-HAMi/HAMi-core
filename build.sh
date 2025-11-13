#!/bin/bash
# Build script for SoftMig
# For Digital Research Alliance Canada / Compute Canada SLURM environments

root_dir=$(cd $(dirname $0); pwd)
cd $root_dir

mkdir -p build; cd build

# Number of parallel jobs (use half of available CPUs)
[[ -z "$J" ]] && J=`nproc | awk '{print int(($0 + 1)/ 2)}'`
CMAKE=${CMAKE-"cmake"}
CMAKE_OPTIONS=${CMAKE_OPTIONS-""}

# Core features (always enabled)
CMAKE_OPTIONS+=" -DDLSYM_HOOK_ENABLE=1"              # Enable dlsym hooking (required)
CMAKE_OPTIONS+=" -DMULTIPROCESS_LIMIT_ENABLE=1"       # Enable multiprocess memory limiting
CMAKE_OPTIONS+=" -DHOOK_MEMINFO_ENABLE=1"            # Hook memory info calls
CMAKE_OPTIONS+=" -DHOOK_NVML_ENABLE=1"               # Hook NVML calls

# Build type: Release for production (better performance), Debug for development
# For production deployments, use Release mode
CMAKE_OPTIONS+=" -DCMAKE_BUILD_TYPE=Release"
# For debugging/troubleshooting, uncomment the line below and comment the Release line above:
#CMAKE_OPTIONS+=" -DCMAKE_BUILD_TYPE=Debug"

# Debug options (uncomment for troubleshooting)
#CMAKE_OPTIONS+=" -DMEMORY_LIMIT_DEBUG=1"            # Enable memory limit debugging
#CMAKE_OPTIONS+=" -DDLSYM_HOOK_DEBUG=1"              # Enable dlsym hook debugging
# gitlab ci
CI_COMMIT_BRANCH=${CI_COMMIT_BRANCH-""}
CI_COMMIT_SHA=${CI_COMMIT_SHA-""}

# jenkins
if [ -z $CI_COMMIT_BRANCH ]; then
    CI_COMMIT_BRANCH=${BRANCH_NAME-"unknown"}
fi
if [ -z $CI_COMMIT_SHA ]; then
    CI_COMMIT_SHA=$(git describe --abbrev=100 --always)
    if [ $? -ne 0 ]; then
        CI_COMMIT_SHA="unknown"
    fi
fi

echo "CI_COMMIT_BRANCH:${CI_COMMIT_BRANCH}"
echo "CI_COMMIT_SHA:${CI_COMMIT_SHA}"

${CMAKE} .. ${CMAKE_OPTIONS} -DTEST_DEVICE_ID=0
make -j$J
