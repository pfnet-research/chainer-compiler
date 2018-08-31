rm -r tmp
mkdir tmp
cp $1 tmp.py
python3 tmp.py --test_data_dir tmp/test_data_set_0 --output tmp/model.onnx --quiet
./oniku/oniku/tools/run_onnx --test=tmp
