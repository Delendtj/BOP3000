
def build_engine(onnx_path, engine_path, use_fp16=False):

    print(f"\n{'=' * 60}")
    print(f"Building TensorRT engine from {onnx_path}")
    print(f"This may take a few minutes on first run...")
    print(f"{'=' * 60}\n")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse ONNX file')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 mode enabled")

    # Build engine
    print("Building engine... (this takes time)")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print('ERROR: Failed to build engine')
        return None

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"✓ Engine saved to {engine_path}\n")
    return serialized_engine


def load_engine(engine_path):
    """Load TensorRT engine from file"""
    with open(engine_path, 'rb') as f:
        return f.read()


# Build or load engine
import os

if os.path.exists(ENGINE_PATH):
    print(f"✓ Loading cached engine from {ENGINE_PATH}")
    serialized_engine = load_engine(ENGINE_PATH)
else:
    serialized_engine = build_engine(ONNX_MODEL_PATH, ENGINE_PATH, USE_FP16)

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

print(f"Engine info:")
print(f"  Input: {engine.get_tensor_name(0)}, shape: {engine.get_tensor_shape(engine.get_tensor_name(0))}")
print(f"  Output: {engine.get_tensor_name(1)}, shape: {engine.get_tensor_shape(engine.get_tensor_name(1))}")
print(f"  Device: {cuda.Device(0).name()}\n")

# Allocate buffers
input_binding = engine.get_tensor_name(0)
output_binding = engine.get_tensor_name(1)

input_shape = engine.get_tensor_shape(input_binding)
output_shape = engine.get_tensor_shape(output_binding)

input_size_bytes = trt.volume(input_shape) * np.dtype(np.float32).itemsize
output_size_bytes = trt.volume(output_shape) * np.dtype(np.float32).itemsize

# Allocate device memory
d_input = cuda.mem_alloc(input_size_bytes)
d_output = cuda.mem_alloc(output_size_bytes)

# Create CUDA stream
stream = cuda.Stream()
