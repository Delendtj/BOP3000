'''import numpy as np

#tror lowkey hele dette er garbage

class InferenceEngine:

    def __init__(self, model, hardware_type):

        self.model = model
        self.hardware_type = hardware_type

        if hardware_type == 'cuda':
            # Import CUDA for TensorRT inference
            try:
                import pycuda.driver as cuda
                self.cuda = cuda
            except ImportError:
                raise ImportError("pycuda not installed. Install with: pip install pycuda")

    def run(self, input_data):

        if self.hardware_type == 'cuda':
            return self._run_tensorrt(input_data)
        else:
            return self._run_openvino(input_data)

    def _run_tensorrt(self, input_data):

        context = self.model["context"]
        d_input = self.model["d_input"]
        d_output = self.model["d_output"]
        stream = self.model["stream"]
        input_binding = self.model["input_binding"]
        output_binding = self.model["output_binding"]
        output_shape = self.model["output_shape"]

        # Ensure input is contiguous and correct dtype
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)

        # Copy input to device
        self.cuda.memcpy_htod_async(d_input, input_data, stream)

        # Set tensor addresses
        context.set_tensor_address(input_binding, int(d_input))
        context.set_tensor_address(output_binding, int(d_output))

        # Run inference
        context.execute_async_v3(stream_handle=stream.handle)

        # Copy output back to host
        output_data = np.empty(output_shape, dtype=np.float32)
        self.cuda.memcpy_dtoh_async(output_data, d_output, stream)

        # Synchronize stream
        stream.synchronize()

        return output_data

    def _run_openvino(self, input_data):

        # Run inference
        result = self.model(input_data)

        # Extract output tensor (OpenVINO returns a dict)
        output_data = list(result.values())[0]

        return output_data

    def get_input_shape(self):

        if self.hardware_type == 'cuda':
            return tuple(self.model["input_shape"])
        else:
            # OpenVINO
            input_layer = self.model.input(0)
            return tuple(input_layer.shape)

    def get_output_shape(self):

        if self.hardware_type == 'cuda':
            return tuple(self.model["output_shape"])
        else:
            # OpenVINO
            output_layer = self.model.output(0)
            return tuple(output_layer.shape)