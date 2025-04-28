set -e
[[ -z ${SHORT_VERSION} ]] && SHORT_VERSION=$(git rev-parse --abbrev-ref HEAD)
[[ -z ${COMMIT_CODE} ]] && COMMIT_CODE=$(git describe --abbrev=100 --always)

export SHORT_VERSION
export COMMIT_CODE
export VERSION="${SHORT_VERSION}-${COMMIT_CODE}"
export LATEST_VERSION="latest"

IMAGE=${IMAGE-"4pdosc/k8s-vdevice"}

function go_build() {
  [[ -z "$J" ]] && J=$(nproc | awk '{print int(($0 + 1)/ 2)}')
  make -j$J
}

function docker_build() {
    docker build --build-arg VERSION="${VERSION}" -t "${IMAGE}:${VERSION}" -f docker/Dockerfile .
    docker tag "${IMAGE}:${VERSION}" "${IMAGE}:${SHORT_VERSION}"
    docker tag "${IMAGE}:${VERSION}" "${IMAGE}:${LATEST_VERSION}"
}

function docker_push() {
    #docker push "${IMAGE}:${VERSION}"
    docker push "${IMAGE}:${SHORT_VERSION}"
    docker push "${IMAGE}:${LATEST_VERSION}"
}

go_build
docker_build
docker_push