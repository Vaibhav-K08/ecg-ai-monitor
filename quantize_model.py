from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "cardiosentinel_v2.onnx",
    "cardiosentinel_v2_int8.onnx",
    weight_type=QuantType.QInt8
)

print("✅ Quantized model saved")
