import os

from ultralytics import YOLO


def _build_engine_from_onnx(
    onnx_path,
    engine_path,
    use_fp16=True,
    imgsz=640,
    batch=1,
    workspace_gb=8,
):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        model_bytes = f.read()
    if not parser.parse(model_bytes):
        errors = []
        for i in range(parser.num_errors):
            errors.append(str(parser.get_error(i)))
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    if network.num_inputs < 1:
        raise RuntimeError("ONNX network has no inputs.")

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gb) * (1 << 30)
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    target_shape = (int(batch), 3, int(imgsz), int(imgsz))
    input_shape = tuple(int(d) for d in input_tensor.shape)

    if any(d == -1 for d in input_shape):
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, target_shape, target_shape, target_shape)
        config.add_optimization_profile(profile)
    elif input_shape != target_shape:
        # Keep static engine shape aligned with runtime inference config.
        input_tensor.shape = target_shape

    serialized_engine = None
    if hasattr(builder, "build_serialized_network"):
        serialized_engine = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        if engine is not None:
            serialized_engine = engine.serialize()

    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build returned no serialized engine.")

    engine_dir = os.path.dirname(engine_path)
    if engine_dir:
        os.makedirs(engine_dir, exist_ok=True)

    with open(engine_path, "wb") as f:
        f.write(bytes(serialized_engine))


def init_tensorrt(config):
    pt_path = config.get("Model_PT_path", "models/best.pt")
    onnx_path = config.get("Model_ONNX_path", os.path.splitext(pt_path)[0] + ".onnx")
    use_fp16 = config.get("USE_FP16", True)
    imgsz = config.get("IMGSZ", 640)
    workspace_gb = config.get("TRT_WORKSPACE_GB", 8)
    batch = config.get("TRT_BATCH", 1)

    if "Tensor_engine_path" in config:
        engine_path = config["Tensor_engine_path"]
    else:
        engine_path = os.path.splitext(onnx_path)[0] + ".engine"

    if os.path.exists(engine_path):
        try:
            print(f"  Loading TensorRT engine: {engine_path}")
            model = YOLO(engine_path, task="detect")
            print("  TensorRT engine loaded successfully")
            return model
        except Exception as e:
            print(f"  Engine load failed ({e}), rebuilding...")

    if os.path.exists(onnx_path):
        try:
            print(f"  Building TensorRT engine from ONNX (TensorRT Python API): {onnx_path}")
            print("  This will take several minutes while TensorRT profiles your GPU...")
            _build_engine_from_onnx(
                onnx_path=onnx_path,
                engine_path=engine_path,
                use_fp16=use_fp16,
                imgsz=imgsz,
                batch=batch,
                workspace_gb=workspace_gb,
            )
            model = YOLO(engine_path, task="detect")
            print("  TensorRT engine built and loaded successfully")
            return model
        except Exception as e:
            print(f"  ONNX TensorRT build failed ({e}), falling back to PyTorch export...")

    if os.path.exists(pt_path):
        try:
            print(f"  Building TensorRT engine at imgsz={imgsz} from PyTorch: {pt_path}")
            print("  This will take several minutes while TensorRT profiles your GPU...")
            model = YOLO(pt_path, task="detect")
            # Keep optimized export values from the current implementation.
            exported_engine = model.export(
                format="engine",
                half=use_fp16,
                simplify=True,
                workspace=8,
                batch=1,
                dynamic=False,
                imgsz=imgsz,
            )

            load_engine_path = engine_path
            if exported_engine is not None and os.path.exists(str(exported_engine)):
                load_engine_path = str(exported_engine)

            model = YOLO(load_engine_path, task="detect")
            print("  TensorRT engine built and loaded successfully")
            return model
        except Exception as e:
            raise RuntimeError(f"TensorRT build failed from PyTorch fallback: {e}") from e

    raise FileNotFoundError(
        "No model files found. Checked:\n"
        f"  engine : {engine_path}\n"
        f"  onnx   : {onnx_path}\n"
        f"  pt     : {pt_path}"
    )
