node=(
    Linear.py
    Concat.py
    SoftmaxClossEntropy.py
    AddMul.py
    NstepLSTM.py
    EmbedID.py
    BatchNorm.py
)

syntax=(
    ListComp.py
)

syntax=()

model=(
    MLP_with_loss.py
    Alex_with_loss.py
    GoogleNet_with_loss.py
)

files=()

for f in ${node[@]}; do
    files+=(node/$f)
done
for f in ${syntax[@]}; do
    files+=(syntax/$f)
done
for f in ${model[@]}; do
    files+=(model/$f)
done

for f in ${files[@]}; do
    echo -n $f " "
    
    ./nikucheck.sh casetest/$f 2> result.txt > o.txt
    if cat result.txt | grep "OK!" > /dev/null; then
        echo "@ test passed"
    else
        echo "@ test failed"
        break
    fi
done

