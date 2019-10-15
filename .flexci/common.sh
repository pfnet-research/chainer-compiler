systemctl stop docker.service
mount -t tmpfs tmpfs /var/lib/docker/ -o size=100%
systemctl start docker.service

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/ -o size=100%
cp -a . ${TEMP}/chainer-compiler
cd ${TEMP}/chainer-compiler

gcloud -q auth configure-docker
CI_IMAGE="asia.gcr.io/pfn-public-ci/chainer-compiler:ci-base-70323579"
