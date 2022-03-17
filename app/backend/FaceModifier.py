import time
import cv2
import numpy as np
from modelhub.pytorch.psp import PspEditor
from xlib import os as lib_os
from xlib.mp import csw as lib_csw
from xlib.python import all_is_not_None


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
        
        self.model = PspEditor()

        
        lib_os.set_timer_resolution(1)

        state, cs = self.get_state(), self.get_control_sheet()

        cs.goatee.call_on_number(self.on_cs_beard)
        cs.goatee.enable()
        cs.goatee.set_config(lib_csw.Number.Config(min=-30, max=30, step=1, allow_instant_update=True))
        cs.goatee.set_number(state.goatee if state.goatee is not None else 0)

        cs.smile.call_on_number(self.on_cs_smile)
        cs.smile.enable()
        cs.smile.set_config(lib_csw.Number.Config(min=-30, max=30, step=1, allow_instant_update=True))
        cs.smile.set_number(state.smile if state.smile is not None else 0)

    def on_cs_beard(self, val):
        state, cs = self.get_state(), self.get_control_sheet()
        state.goatee = val
        cs.goatee.set_number(val)
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

                        edits = {
                            "goatee": state.goatee if state.goatee else 0,
                            "smile": state.smile if state.smile else 0,
                            }
                        output = self.model.run(frame_image, edits)
                        bcd.set_merged_image_name("modified_image")
                        bcd.set_image("modified_image", output)


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
            self.goatee = lib_csw.Number.Client()
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
            self.goatee = lib_csw.Number.Host()
            self.smile = lib_csw.Number.Host()

class WorkerState(BackendWorkerState):
    face_coverage : float = None
    resolution    : int = None
    exclude_moving_parts : bool = None
    head_mode : bool = None
    x_offset : float = None
    y_offset : float = None
    goatee : float = None
    smile : float = None
