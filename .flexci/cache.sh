pull_chainer_whl() {
    pushd third_party/chainer
    CHAINER_COMMIT_ID=$(git rev-parse --short HEAD)
    popd
    set +e
    WHEEL_EXIST=$(gsutil -q stat gs://tmp-asia-pfn-public-ci/chainer-compiler/chainer-whl/$CHAINER_COMMIT_ID/*.whl; echo $?)
    set -e
    if [[ $WHEEL_EXIST = 0 ]]; then
        mkdir dist-chainer
        gsutil -q cp gs://tmp-asia-pfn-public-ci/chainer-compiler/chainer-whl/$CHAINER_COMMIT_ID/*.whl dist-chainer
        echo "Get cheched wheel, commit id: "$CHAINER_COMMIT_ID
    fi
}

push_chainer_whl() {
    pushd third_party/chainer
    if [[ -d dist ]]; then
        cd dist
        CHAINER_COMMIT_ID=$(git rev-parse --short HEAD)
        gsutil -q cp *.whl gs://tmp-asia-pfn-public-ci/chainer-compiler/chainer-whl/$CHAINER_COMMIT_ID/
        echo "Upload chainer wheel, commit id: "$CHAINER_COMMIT_ID
        cd ..
    fi
    popd
}
