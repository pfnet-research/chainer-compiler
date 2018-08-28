rm -r tmp
mkdir tmp
python3 $1 --test_data_dir tmp/test_data_set_0 --output tmp/model.onnx > /dev/null
./oniku/oniku/tools/run_onnx --test=tmp


