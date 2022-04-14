from pathlib import Path
from typing import List

from localization import L, Localization
from resources.fonts import QXFontDB
from resources.gfx import QXImageDB
from xlib import qt as qtx
from xlib.qt.widgets.QXLabel import QXLabel

from . import backend
from .ui.QCameraSource import QCameraSource
from .ui.QFaceAligner import QFaceAligner
from .ui.QFaceDetector import QFaceDetector
from .ui.QFaceMarker import QFaceMarker
from .ui.QStreamOutput import QStreamOutput
from .ui.QFaceModifier import QFaceModifier
from .ui.widgets.QBCFaceAlignViewer import QBCFaceAlignViewer
from .ui.widgets.QBCMergedFrameViewer import QBCMergedFrameViewer
from .ui.widgets.QBCFrameViewer import QBCFrameViewer

class QLiveSwap(qtx.QXWidget):
    def __init__(self, userdata_path : Path,
                       settings_dirpath : Path):
        super().__init__()

        output_sequence_path = userdata_path / 'output_sequence'
        output_sequence_path.mkdir(parents=True, exist_ok=True)

        # Construct backend config
        backend_db          = self.backend_db          = backend.BackendDB( settings_dirpath / 'states.dat' )
        backed_weak_heap    = self.backed_weak_heap    = backend.BackendWeakHeap(size_mb=2048)
        reemit_frame_signal = self.reemit_frame_signal = backend.BackendSignal()

        multi_sources_bc_out  = backend.BackendConnection(multi_producer=True)
        face_detector_bc_out  = backend.BackendConnection()
        face_marker_bc_out    = backend.BackendConnection()
        face_aligner_bc_out   = backend.BackendConnection()
        face_modifier_bc_out = backend.BackendConnection()

        camera_source  = self.camera_source  = backend.CameraSource (weak_heap=backed_weak_heap, bc_out=multi_sources_bc_out, backend_db=backend_db)
        face_detector  = self.face_detector  = backend.FaceDetector (weak_heap=backed_weak_heap, reemit_frame_signal=reemit_frame_signal, bc_in=multi_sources_bc_out, bc_out=face_detector_bc_out, backend_db=backend_db )
        face_marker    = self.face_marker    = backend.FaceMarker   (weak_heap=backed_weak_heap, reemit_frame_signal=reemit_frame_signal, bc_in=face_detector_bc_out, bc_out=face_marker_bc_out, backend_db=backend_db)
        face_aligner   = self.face_aligner   = backend.FaceAligner  (weak_heap=backed_weak_heap, reemit_frame_signal=reemit_frame_signal, bc_in=face_marker_bc_out, bc_out=face_aligner_bc_out, backend_db=backend_db )
        face_modifier  = self.face_modifier  = backend.FaceModifier  (weak_heap=backed_weak_heap, reemit_frame_signal=reemit_frame_signal, bc_in=face_aligner_bc_out, bc_out=face_modifier_bc_out, backend_db=backend_db )
        stream_output  = self.stream_output  = backend.StreamOutput (weak_heap=backed_weak_heap, reemit_frame_signal=reemit_frame_signal, bc_in=face_modifier_bc_out, save_default_path=userdata_path, backend_db=backend_db)

        self.all_backends : List[backend.BackendHost] = [camera_source, face_detector, face_marker, face_aligner, face_modifier, stream_output]

        self.q_camera_source  = QCameraSource(self.camera_source)
        self.q_face_detector  = QFaceDetector(self.face_detector)
        self.q_face_marker    = QFaceMarker(self.face_marker)
        self.q_face_aligner   = QFaceAligner(self.face_aligner)
        self.q_face_modifier   = QFaceModifier(self.face_modifier)
        self.q_stream_output  = QStreamOutput(self.stream_output)

        self.q_ds_frame_viewer = QBCFrameViewer(backed_weak_heap, multi_sources_bc_out)
        self.q_ds_fa_viewer    = QBCFaceAlignViewer(backed_weak_heap, face_aligner_bc_out, preview_width=256)

        q_nodes = qtx.QXWidgetHBox([    qtx.QXWidgetVBox([self.q_camera_source], spacing=5, fixed_width=256),
                                        qtx.QXWidgetVBox([self.q_face_detector,  self.q_face_aligner,], spacing=5, fixed_width=256),
                                        qtx.QXWidgetVBox([self.q_face_marker, self.q_face_modifier, self.q_stream_output], spacing=5, fixed_width=256),
                                    ], spacing=5, size_policy=('fixed', 'fixed') )

        q_view_nodes = qtx.QXWidgetHBox([   (qtx.QXWidgetVBox([self.q_ds_frame_viewer], fixed_width=256), qtx.AlignTop),
                                            (qtx.QXWidgetVBox([self.q_ds_fa_viewer], fixed_width=256), qtx.AlignTop),
                                        ], spacing=5, size_policy=('fixed', 'fixed') )

        self.setLayout(qtx.QXVBoxLayout( [ (qtx.QXWidgetVBox([q_nodes, q_view_nodes], spacing=5), qtx.AlignCenter) ]))
                       
        self._timer = qtx.QXTimer(interval=5, timeout=self._on_timer_5ms, start=True)

    def _process_messages(self):
        self.backend_db.process_messages()
        for backend in self.all_backends:
            backend.process_messages()

    def _on_timer_5ms(self):
        self._process_messages()

    def clear_backend_db(self):
        self.backend_db.clear()

    def initialize(self):
        for backend in self.all_backends:
            backend.restore_on_off_state()

    def finalize(self):
        # Gracefully stop the backend
        for backend in self.all_backends:
            while backend.is_starting() or backend.is_stopping():
                self._process_messages()

            backend.save_on_off_state()
            backend.stop()

        while not all( x.is_stopped() for x in self.all_backends ):
            self._process_messages()

        self.backend_db.finish_pending_jobs()

        self.q_ds_frame_viewer.clear()
        self.q_ds_fa_viewer.clear()

class QDFLAppWindow(qtx.QXWindow):

    def __init__(self, userdata_path, settings_dirpath):
        super().__init__(save_load_state=True, size_policy=('minimum', 'minimum') )

        self._userdata_path = userdata_path
        self._settings_dirpath = settings_dirpath

        menu_bar = qtx.QXMenuBar( font=QXFontDB.get_default_font(size=10), size_policy=('fixed', 'minimumexpanding') )
        menu_file = menu_bar.addMenu( L('@QDFLAppWindow.file') )

        menu_file_action_reinitialize = menu_file.addAction( L('@QDFLAppWindow.reinitialize') )
        menu_file_action_reinitialize.triggered.connect(lambda: qtx.QXMainApplication.inst.reinitialize() )

        menu_file_action_reset_settings = menu_file.addAction( L('@QDFLAppWindow.reset_modules_settings') )
        menu_file_action_reset_settings.triggered.connect(self._on_reset_modules_settings)

        menu_file_action_quit = menu_file.addAction( L('@QDFLAppWindow.quit') )
        menu_file_action_quit.triggered.connect(lambda: qtx.QXMainApplication.quit() )

        self.q_live_swap = None
        self.q_live_swap_container = qtx.QXWidget()

        self.content_l = qtx.QXVBoxLayout()

        self.setLayout( qtx.QXVBoxLayout([  qtx.QXWidgetHBox([menu_bar, qtx.QXFrame() ], size_policy=('minimumexpanding', 'fixed')),
                                            5,
                                            qtx.QXWidget(layout=self.content_l)
                                         ]))

        self.call_on_closeEvent(self._on_closeEvent)

        q_live_swap = self.q_live_swap = QLiveSwap(userdata_path=self._userdata_path, settings_dirpath=self._settings_dirpath)
        q_live_swap.initialize()
        self.content_l.addWidget(q_live_swap)

    def _on_reset_modules_settings(self):
        if self.q_live_swap is not None:
            self.q_live_swap.clear_backend_db()
            qtx.QXMainApplication.inst.reinitialize()

    def finalize(self):
        self.q_live_swap.finalize()

    def _on_closeEvent(self):
        self.finalize()


class FaceFilterLiveApp(qtx.QXMainApplication):
    def __init__(self, userdata_path):
        self.userdata_path = userdata_path
        settings_dirpath = self.settings_dirpath =  userdata_path / 'settings'
        if not settings_dirpath.exists():
            settings_dirpath.mkdir(parents=True)
        super().__init__(app_name='FaceFilterLive', settings_dirpath=settings_dirpath)

        self.setFont( QXFontDB.get_default_font() )
        self.setWindowIcon( QXImageDB.app_icon().as_QIcon() )

        splash_wnd = self.splash_wnd =\
            qtx.QXSplashWindow(layout=qtx.QXVBoxLayout([ (qtx.QXLabel(image=QXImageDB.splash()), qtx.AlignCenter)
                                                       ], contents_margins=20))
        splash_wnd.show()
        splash_wnd.center_on_screen()

        self._dfl_wnd = None
        self._t = qtx.QXTimer(interval=1666, timeout=self._on_splash_wnd_expired, single_shot=True, start=True)
        self.initialize()

    def on_reinitialize(self):
        self.finalize()

        import gc
        gc.collect()
        gc.collect()

        self.initialize()
        self._dfl_wnd.show()

    def initialize(self):
        Localization.set_language( self.get_language() )

        if self._dfl_wnd is None:
            self._dfl_wnd = QDFLAppWindow(userdata_path=self.userdata_path, settings_dirpath=self.settings_dirpath)

    def finalize(self):
        if self._dfl_wnd is not None:
            self._dfl_wnd.close()
            self._dfl_wnd.deleteLater()
            self._dfl_wnd = None

    def _on_splash_wnd_expired(self):
        self._dfl_wnd.show()

        if self.splash_wnd is not None:
            self.splash_wnd.hide()
            self.splash_wnd.deleteLater()
            self.splash_wnd = None
