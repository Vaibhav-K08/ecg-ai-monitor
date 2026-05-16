import tensorflow as tf
import tf2onnx

# Load your trained model
model = tf.keras.models.load_model("cardiosentinel_v2.keras")

# Convert to ONNX
spec = (tf.TensorSpec((None, 360, 1), tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

with open("cardiosentinel_v2.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

print("✅ ONNX model saved as cardiosentinel_v2.onnx")
