- chainer2onnx.py
    メイン。chainer2onnx(model, forward) で変換したonnxを返す
- test_mxnet.py
    ランダム入力に対する chainerでの実行結果と chainer -> onnx -> mxnet での実行結果を比較する
- test_MLP.py
    MLPを変換するテスト
