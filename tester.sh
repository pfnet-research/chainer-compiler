nodes=(
    Concat.py
    SoftmaxClossEntropy.py
    AddMul.py
    NstepLSTM.py
    EmbedID.py
)


for f in ${nodes[@]}; do
    echo -n $f " "
    ./nikucheck.sh casetest/node/$f 2> result.txt > o.txt
    if cat result.txt | grep "OK!" > /dev/null; then
        echo "@ test passed"
    else
        echo "@ test failed"
        break
    fi
done

