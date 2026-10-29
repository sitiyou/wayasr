Silero VAD 6.2.3
Source: https://github.com/snakers4/silero-vad
Model: silero_vad/data/silero_vad.onnx from the silero-vad 6.2.3 distribution.
License: MIT; see SILERO_LICENSE.

The application uses 512-sample frames at 16 kHz, a 64-sample context,
and recurrent state shaped (2, 1, 128), matching the upstream ONNX wrapper.
