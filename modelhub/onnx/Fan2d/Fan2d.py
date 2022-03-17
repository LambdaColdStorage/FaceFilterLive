from pathlib import Path

from xlib.image import ImageProcessor
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)

import numpy as np

class Fan2d:
    """
    arguments

     device_info    ORTDeviceInfo

        use FaceMesh.get_available_devices()
        to determine a list of avaliable devices accepted by model

    raises
     Exception
    """

    @staticmethod
    def get_available_devices():
        return get_available_devices_info()

    def __init__(self, device_info : ORTDeviceInfo):
        if device_info not in Fan2d.get_available_devices():
            raise Exception(f'device_info {device_info} is not in available devices for 2DFAN')

        path = Path(__file__).parent / '2DFAN.onnx'
        self._sess = sess = InferenceSession_with_device(str(path), device_info)
        self._input_name = sess.get_inputs()[0].name
        self._input_width = 256
        self._input_height = 256

    def extract(self, img):
        """
        arguments

         img    np.ndarray      HW,HWC,NHWC uint8/float32

        returns (N,468,3)
        """
        ip = ImageProcessor(img)
        N,H,W,_ = ip.get_dims()

        h_scale = H / self._input_height
        w_scale = W / self._input_width

        feed_img = ip.resize( (self._input_width, self._input_height) ).to_ufloat32().ch(3).get_image('NCHW')

        hm = self._sess.run(None, {self._input_name: feed_img})[0]

        B, C, H, W = hm.shape
        hm_reshape = hm.reshape(B, C, H * W)
        idx = np.argmax(hm_reshape, axis=-1)
        # scores = np.take_along_axis(hm_reshape, np.expand_dims(idx, axis=-1), axis=-1).squeeze(-1)

        idx += 1
        preds = idx.repeat(2).reshape(B, C, 2).astype(np.float32)
        preds[:, :, 0] = (preds[:, :, 0] - 1) % W + 1
        preds[:, :, 1] = np.floor((preds[:, :, 1] - 1) / H) + 1

        for i in range(B):
            for j in range(C):
                hm_ = hm[i, j, :]
                pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
                if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                    diff = np.array([hm_[pY, pX + 1] - hm_[pY, pX - 1],
                                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                    preds[i, j] += np.sign(diff) * 0.25
        preds -= 0.5
        preds *= 4.0
        preds *= (w_scale, h_scale)

        return preds
