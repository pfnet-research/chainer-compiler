rm -r tmp
mkdir tmp
<<<<<<< HEAD

cp $1 tmp.py
python3 tmp.py --test_data_dir tmp/test_data_set_0 --output tmp/model.onnx
=======
python3 $1 --test_data_dir tmp/test_data_set_0 --output tmp/model.onnx --quiet
>>>>>>> Introduce --quiet flag to silence the compiler
./oniku/oniku/tools/run_onnx --test=tmp
