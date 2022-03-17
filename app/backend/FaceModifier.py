import time
import cv2
import onnxruntime as ort
import numpy as np
from xlib import os as lib_os
from xlib.mp import csw as lib_csw
from xlib.python import all_is_not_None
import sys
sys.path.append("psp")
import editor

from .BackendBase import (BackendConnection, BackendDB, BackendHost,
                          BackendSignal, BackendWeakHeap, BackendWorker,
                          BackendWorkerState)


class FaceModifier(BackendHost):
    def __init__(self, weak_heap :  BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection, backend_db : BackendDB = None):
        super().__init__(backend_db=backend_db,
                         sheet_cls=Sheet,
                         worker_cls=FaceModifierWorker,
                         worker_state_cls=WorkerState,
                         worker_start_args=[weak_heap, reemit_frame_signal, bc_in, bc_out])

    def get_control_sheet(self) -> 'Sheet.Host': return super().get_control_sheet()

class FaceModifierWorker(BackendWorker):
    def get_state(self) -> 'WorkerState': return super().get_state()
    def get_control_sheet(self) -> 'Sheet.Worker': return super().get_control_sheet()

    def on_start(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection):
        self.weak_heap = weak_heap
        self.reemit_frame_signal = reemit_frame_signal
        self.bc_in = bc_in
        self.bc_out = bc_out
        self.pending_bcd = None
        checkpoint_path = "psp_ffhq_encode.pt"

        encoder, decoder, latent_avg = editor.load_model(checkpoint_path)

        manipulator = editor.manipulate_model(decoder)
        manipulator.edits = {editor.idx_dict[v[0]]: {v[1]: 0} for k, v in editor.edits.items()}
        self.model = {
            "encoder": encoder,
            "decoder": decoder,
            "latent_avg": latent_avg,
            "manipulator": manipulator,
        }

        
        lib_os.set_timer_resolution(1)

        state, cs = self.get_state(), self.get_control_sheet()

        cs.beard.call_on_number(self.on_cs_beard)
        cs.beard.enable()
        cs.beard.set_config(lib_csw.Number.Config(min=-30, max=30, step=1, allow_instant_update=True))
        cs.beard.set_number(state.beard if state.beard is not None else 0)

        cs.smile.call_on_number(self.on_cs_smile)
        cs.smile.enable()
        cs.smile.set_config(lib_csw.Number.Config(min=-30, max=30, step=1, allow_instant_update=True))
        cs.smile.set_number(state.smile if state.smile is not None else 0)

    def on_cs_beard(self, val):
        state, cs = self.get_state(), self.get_control_sheet()
        state.beard = val
        cs.beard.set_number(val)
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_smile(self, val):
        state, cs = self.get_state(), self.get_control_sheet()
        state.smile = val
        cs.smile.set_number(val)
        self.save_state()
        self.reemit_frame_signal.send()

    def on_tick(self):
        state, cs = self.get_state(), self.get_control_sheet()

        if self.pending_bcd is None:
            self.start_profile_timing()

            bcd = self.bc_in.read(timeout=0.005)
            if bcd is not None:
                bcd.assign_weak_heap(self.weak_heap)
                for i, fsi in enumerate(bcd.get_face_swap_info_list()):
                    view_image = bcd.get_image(fsi.face_align_image_name)
                    
                    if all_is_not_None(view_image):
                        frame_image = cv2.resize(view_image, (256, 256))

                        mods = {
                            "goatee": state.beard if state.beard else 0,
                            "smile": state.smile if state.smile else 0,
                            }
                        manipulator = self.model["manipulator"]
                        for k, v in editor.edits.items():
                            layer_index, channel_index, sense = v
                            conv_name = editor.idx_dict[layer_index]
                            manipulator.edits[conv_name][channel_index] = mods.get(k, 0)*sense

                        inp = 2*frame_image[...,::-1].astype(np.float32).transpose(2,0,1)/255 - 1
                        output = editor.run(self.model["encoder"], self.model["decoder"], self.model["latent_avg"], inp, pil=False)
                        bcd.set_merged_image_name("modified_image")
                        bcd.set_image("modified_image", np.ascontiguousarray(output[...,::-1]).astype(np.uint8))


                self.stop_profile_timing()
                self.pending_bcd = bcd

        if self.pending_bcd is not None:
            if self.bc_out.is_full_read(1):
                self.bc_out.write(self.pending_bcd)
                self.pending_bcd = None
            else:
                time.sleep(0.001)

class Sheet:
    class Host(lib_csw.Sheet.Host):
        def __init__(self):
            super().__init__()
            self.face_coverage = lib_csw.Number.Client()
            self.resolution = lib_csw.Number.Client()
            self.exclude_moving_parts = lib_csw.Flag.Client()
            self.head_mode = lib_csw.Flag.Client()
            self.x_offset = lib_csw.Number.Client()
            self.y_offset = lib_csw.Number.Client()
            self.beard = lib_csw.Number.Client()
            self.smile = lib_csw.Number.Client()

    class Worker(lib_csw.Sheet.Worker):
        def __init__(self):
            super().__init__()
            self.face_coverage = lib_csw.Number.Host()
            self.resolution = lib_csw.Number.Host()
            self.exclude_moving_parts = lib_csw.Flag.Host()
            self.head_mode = lib_csw.Flag.Host()
            self.x_offset = lib_csw.Number.Host()
            self.y_offset = lib_csw.Number.Host()
            self.beard = lib_csw.Number.Host()
            self.smile = lib_csw.Number.Host()

class WorkerState(BackendWorkerState):
    face_coverage : float = None
    resolution    : int = None
    exclude_moving_parts : bool = None
    head_mode : bool = None
    x_offset : float = None
    y_offset : float = None
    beard : float = None
    smile : float = None
