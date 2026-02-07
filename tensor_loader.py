import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This initializes CUDA context automatically

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path, engine_path, use_fp16=False):
    print(f"Building TensorRT engine from {onnx_path}")
    print(f"This may take a few minutes on first run...")


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
        print("FP16 mode enabled")

    # Build engine
    print("Building engine... (this takes time)")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print('ERROR: Failed to build engine')
        return None

    # Save engine
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"Engine saved to {engine_path}")
    print(f"{'=' * 60}\n")
    return serialized_engine


def load_engine(engine_path):
    print(f"Loading cached engine from {engine_path}")
    with open(engine_path, 'rb') as f:
        return f.read()


def init_tensorrt(onnx_path, engine_path, use_fp16=False):

    # Build or load engine
    if os.path.exists(engine_path):
        serialized_engine = load_engine(engine_path)
    else:
        serialized_engine = build_engine(onnx_path, engine_path, use_fp16)

    if serialized_engine is None:
        raise RuntimeError("Failed to create TensorRT engine")

    # Deserialize engine
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    # Get tensor bindings
    input_binding = engine.get_tensor_name(0)
    output_binding = engine.get_tensor_name(1)

    # Get shapes
    input_shape = engine.get_tensor_shape(input_binding)
    output_shape = engine.get_tensor_shape(output_binding)

    # Calculate buffer sizes
    input_size_bytes = trt.volume(input_shape) * np.dtype(np.float32).itemsize
    output_size_bytes = trt.volume(output_shape) * np.dtype(np.float32).itemsize

    # Allocate device memory
    d_input = cuda.mem_alloc(input_size_bytes)
    d_output = cuda.mem_alloc(output_size_bytes)

    # Create CUDA stream
    stream = cuda.Stream()


    return {
        "engine": engine,
        "context": context,
        "input_binding": input_binding,
        "output_binding": output_binding,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "input_size_bytes": input_size_bytes,
        "output_size_bytes": output_size_bytes,
        "d_input": d_input,
        "d_output": d_output,
        "stream": stream,
    }

## Claude fikse dette, i dont get it - DL