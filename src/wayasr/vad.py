from importlib.resources import as_file, files

import numpy as np
import onnxruntime as ort


class SileroVAD:
    def __init__(self):
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        with as_file(files("wayasr").joinpath("data/silero_vad.onnx")) as path:
            self.session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)
        self.sample_rate = np.array(16000, dtype=np.int64)

    def __call__(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.shape != (512,):
            raise ValueError("VAD requires 512 samples at 16 kHz.")
        audio = np.concatenate((self.context, samples[None, :]), axis=1)
        probability, self.state = self.session.run(
            None, {"input": audio, "state": self.state, "sr": self.sample_rate}
        )
        self.context = audio[:, -64:].copy()
        return float(probability.item())
