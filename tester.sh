tests=(
    test_MLP.py
    test_LRN.py
    test_Alex.py
    test_GoogleNet.py
    test_Resnet.py
)


for f in ${tests[@]}; do
    echo $f
    if python3 $f > /dev/null; then
        echo "@test passed"
    else
        echo "@test failed"
        break
    fi
done

