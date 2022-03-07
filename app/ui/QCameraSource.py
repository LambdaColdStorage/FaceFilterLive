from localization import L
from xlib import qt as qtx

from ..backend import CameraSource
from .widgets.QBackendPanel import QBackendPanel
from .widgets.QCheckBoxCSWFlag import QCheckBoxCSWFlag
from .widgets.QComboBoxCSWDynamicSingleSwitch import \
    QComboBoxCSWDynamicSingleSwitch
from .widgets.QLabelPopupInfo import QLabelPopupInfo
from .widgets.QSpinBoxCSWNumber import QSpinBoxCSWNumber
from .widgets.QXPushButtonCSWSignal import QXPushButtonCSWSignal


class QCameraSource(QBackendPanel):
    def __init__(self, backend : CameraSource):
        cs = backend.get_control_sheet()

        q_resolution_label = QLabelPopupInfo(label=L('@QCameraSource.resolution'), popup_info_text=L('@QCameraSource.help.resolution') )
        q_resolution       = QComboBoxCSWDynamicSingleSwitch(cs.resolution, reflect_state_widgets=[q_resolution_label])

        q_fps_label       = QLabelPopupInfo(label=L('@QCameraSource.fps'), popup_info_text=L('@QCameraSource.help.fps') )
        q_fps             = QSpinBoxCSWNumber(cs.fps, reflect_state_widgets=[q_fps_label])


        q_camera_settings_group_label = QLabelPopupInfo(label=L('@QCameraSource.camera_settings') )

        q_open_settings   = QXPushButtonCSWSignal(cs.open_settings, text=L('@QCameraSource.open_settings'))
        q_load_settings   = QXPushButtonCSWSignal(cs.load_settings, text=L('@QCameraSource.load_settings'), reflect_state_widgets=[q_camera_settings_group_label])
        q_save_settings   = QXPushButtonCSWSignal(cs.save_settings, text=L('@QCameraSource.save_settings'))

        grid_l = qtx.QXGridLayout(spacing=5)
        row = 0
        grid_l.addWidget(q_resolution_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_resolution, row, 1, alignment=qtx.AlignLeft )
        row += 1
        btn_height = 24
        grid_l.addWidget(q_camera_settings_group_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget( qtx.QXWidgetHBox([q_open_settings, q_load_settings, q_save_settings],
                                            contents_margins=(1,0,1,0), spacing=1, fixed_height=btn_height), row, 1, alignment=qtx.AlignLeft  )
        row += 1
        grid_l.addWidget(q_fps_label, row, 0, alignment=qtx.AlignRight | qtx.AlignVCenter  )
        grid_l.addWidget(q_fps, row, 1, alignment=qtx.AlignLeft )
        row += 1

        super().__init__(backend, L('@QCameraSource.module_title'),
                         layout=qtx.QXVBoxLayout([grid_l], spacing=5),
                         content_align_top=True)

