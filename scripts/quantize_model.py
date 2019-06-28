import onnx
import quantize
import sys

model_file = sys.argv[1]
output_file = sys.argv[2]

# Load the onnx model
model = onnx.load(model_file)

# Quantize
quantized_model = quantize.quantize(
    model, quantization_mode=quantize.QuantizationMode.IntegerOps)

# Save the quantized model
onnx.save(quantized_model, output_file)
