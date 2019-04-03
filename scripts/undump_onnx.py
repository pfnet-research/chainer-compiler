#!/usr/bin/env python3

import sys
from google.protobuf import text_format
from onnx import onnx_pb


with open(sys.argv[1]) as f:
    c = f.read()
m = onnx_pb.ModelProto()
text_format.Merge(c, m)
with open(sys.argv[2], 'wb') as f:
    f.write(m.SerializeToString())
