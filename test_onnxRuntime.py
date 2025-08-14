import onnxruntime as ort
print(ort.__version__)

import onnxruntime as ort
import numpy as np

# Create a dummy ONNX model (for testing purposes)
# In a real scenario, you would load an actual ONNX model
# For example: session = ort.InferenceSession("path/to/your/model.onnx")

# Example using a simple identity model (requires onnx package to create)
# If you don't have a model, you can skip this and load a known public model.
# For a quick test, you can assume a model exists and load it.

# Create a dummy ONNX model (requires 'onnx' package)
try:
    import onnx
    from onnx import helper, TensorProto

    # Define the graph
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 224, 224])
    node_def = helper.make_node('Identity', ['X'], ['Y'])
    graph_def = helper.make_graph([node_def], 'test-model', [X], [Y])
    model_def = helper.make_model(graph_def, producer_name='test-onnxruntime')
    onnx.save(model_def, 'dummy_model.onnx')
    model_path = 'dummy_model.onnx'
except ImportError:
    print("ONNX package not found. Skipping dummy model creation. Please provide an existing ONNX model for testing.")
    # As an alternative, you could download a small public ONNX model
    # For instance, a small MNIST model from ONNX Model Zoo
    model_path = "path/to/your/actual_model.onnx"  # Replace with a real model path

# Load the ONNX model
session = ort.InferenceSession(model_path, providers = ["CUDAExecutionProvider"])
# session = ort.InferenceSession(model_path, providers = ['CPUExecutionProvider'])

# Prepare dummy input data
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
dummy_input = np.random.rand(*input_shape).astype(np.float32)

# Run inference
output = session.run(None, {input_name: dummy_input})

print("ONNX Runtime inference successful!")
print("Output shape:", output[0].shape)