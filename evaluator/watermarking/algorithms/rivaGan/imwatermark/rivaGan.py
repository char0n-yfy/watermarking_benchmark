import numpy as np
import torch
import cv2
import os
import time


class RivaWatermark(object):
    encoder = None
    decoder = None
    onnx_providers = None

    def __init__(self, watermarks=[], wmLen=32, threshold=0.52):
        self._watermarks = watermarks
        self._threshold = threshold
        if wmLen not in [32]:
            raise RuntimeError('rivaGan only supports 32 bits watermarks now.')
        self._data = torch.from_numpy(np.array([self._watermarks], dtype=np.float32))

    @classmethod
    def loadModel(cls):
        try:
            import onnxruntime
        except ImportError:
            raise ImportError(
                "The `RivaWatermark` class requires onnxruntime to be installed. "
                "You can install it with pip: `pip install onnxruntime`."
            )

        if RivaWatermark.encoder and RivaWatermark.decoder:
            return
        modelDir = os.environ.get('IMWATERMARK_RIVAGAN_MODEL_DIR')
        if not modelDir:
            modelDir = os.path.dirname(os.path.abspath(__file__))
        available_providers = onnxruntime.get_available_providers()
        requested_providers = os.environ.get(
            'IMWATERMARK_RIVAGAN_ONNX_PROVIDERS',
            os.environ.get('WM_BENCH_RIVAGAN_ONNX_PROVIDERS', ''),
        )
        if requested_providers:
            provider_preference = [
                provider.strip()
                for provider in requested_providers.replace(';', ',').split(',')
                if provider.strip()
            ]
        else:
            provider_preference = [
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        providers = [
            provider
            for provider in provider_preference
            if provider in available_providers
        ]
        if not providers:
            providers = (
                ['CPUExecutionProvider']
                if 'CPUExecutionProvider' in available_providers
                else available_providers
            )
        RivaWatermark.encoder = onnxruntime.InferenceSession(
            os.path.join(modelDir, 'rivagan_encoder.onnx'),
            providers=providers or None)
        RivaWatermark.decoder = onnxruntime.InferenceSession(
            os.path.join(modelDir, 'rivagan_decoder.onnx'),
            providers=providers or None)
        RivaWatermark.onnx_providers = {
            'available': available_providers,
            'selected': providers,
            'encoder': RivaWatermark.encoder.get_providers(),
            'decoder': RivaWatermark.decoder.get_providers(),
        }

    def encode(self, frame):
        if not RivaWatermark.encoder:
            raise RuntimeError('call loadModel method first')

        frame = torch.from_numpy(np.array([frame], dtype=np.float32)) / 127.5 - 1.0
        frame = frame.permute(3, 0, 1, 2).unsqueeze(0)

        inputs = {
            'frame': frame.detach().cpu().numpy(),
            'data': self._data.detach().cpu().numpy()
        }

        outputs = RivaWatermark.encoder.run(None, inputs)
        wm_frame = outputs[0]
        wm_frame = torch.clamp(torch.from_numpy(wm_frame), min=-1.0, max=1.0)
        wm_frame = (
            (wm_frame[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5
        ).detach().cpu().numpy().astype('uint8')

        return wm_frame

    def decode(self, frame):
        if not RivaWatermark.decoder:
            raise RuntimeError('you need load model first')

        frame = torch.from_numpy(np.array([frame], dtype=np.float32)) / 127.5 - 1.0
        frame = frame.permute(3, 0, 1, 2).unsqueeze(0)
        inputs = {
            'frame': frame.detach().cpu().numpy(),
        }
        outputs = RivaWatermark.decoder.run(None, inputs)
        data = outputs[0][0]
        return np.array(data > self._threshold, dtype=np.uint8)
