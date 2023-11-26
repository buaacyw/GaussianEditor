from typing import Any
import torch
import numpy as np
import time
import viser
import viser.transforms as tf
from omegaconf import OmegaConf
from collections import deque

from gaussiansplatting.scene.cameras import Simple_Camera, C2W_Camera
from gaussiansplatting.gaussian_renderer import render


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def get_c2w(camera):
    c2w = np.zeros([4, 4], dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz).T
    c2w[:3, 3] = camera.position
    c2w[3, 3] = 1.0

    c2w = torch.from_numpy(c2w).to("cuda")

    return c2w


class ViserViewer:
    def __init__(self, train_mode=False):
        self.device = "cuda:0"
        self.port = 8080

        self.train_mode = train_mode

        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)
        self.reset_view_button = self.server.add_gui_button("Reset View")

        self.toggle_axis = self.server.add_gui_checkbox(
            "Toggle Axis",
            initial_value=True,
        )

        self.need_update = False

        self.pause_training = False

        self.train_viewer_update_period_slider = self.server.add_gui_slider(
            "Update Period",
            min=1,
            max=100,
            step=1,
            initial_value=10,
            disabled=self.pause_training,
        )

        self.pause_training_button = self.server.add_gui_button("Pause Training")
        self.sh_order = self.server.add_gui_slider(
            "SH Order", min=1, max=4, step=1, initial_value=1
        )
        self.resolution_slider = self.server.add_gui_slider(
            "Resolution", min=384, max=4096, step=2, initial_value=1024
        )
        self.near_plane_slider = self.server.add_gui_slider(
            "Near", min=0.1, max=30, step=0.5, initial_value=0.1
        )
        self.far_plane_slider = self.server.add_gui_slider(
            "Far", min=30.0, max=1000.0, step=10.0, initial_value=1000.0
        )

        self.show_train_camera = self.server.add_gui_checkbox(
            "Show Train Camera", initial_value=False
        )

        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)

        self.axis = self.server.add_frame("Axis", show_axes=True, axes_length=1000)

        self.time_bar = self.server.add_gui_slider(
            "Timestep", min=0, max=1000, step=1, initial_value=0, visible=False
        )

        self.renderer_output = self.server.add_gui_dropdown(
            "Renderer Output",
            [
                "render",
            ],
        )

        @self.renderer_output.on_update
        def _(_):
            self.need_update = True

        @self.show_train_camera.on_update
        def _(_):
            self.need_update = True

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.near_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.far_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.pause_training_button.on_click
        def _(_):
            self.pause_training = not self.pause_training
            self.train_viewer_update_period_slider.disabled = not self.pause_training
            self.pause_training_button.name = (
                "Resume Training" if self.pause_training else "Pause Training"
            )

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                    [0.0, -1.0, 0.0]
                )

        @self.toggle_axis.on_update
        def _(_):
            self.need_update = True
            self.axis.show_axes = self.toggle_axis.value

        self.c2ws = []
        self.camera_infos = []

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

        self.debug_idx = 0

    def get_kwargs(self):
        out = {}
        if hasattr(self, "time_bar"):
            out["timestep"] = self.time_bar.value
        if hasattr(self, "mask_thresh"):
            out["mask_thresh"] = self.mask_thresh.value
        if hasattr(self, "invert_mask"):
            out["invert_mask"] = self.invert_mask.value

        return out

    def set_system(self, system):
        self.system = system

    @torch.no_grad()
    def update(self):
        if self.need_update:
            times = []
            for client in self.server.get_clients().values():
                camera = client.camera
                w = self.resolution_slider.value
                h = int(w / camera.aspect)
                # cam = Simple_Camera(0, )
                cam = C2W_Camera(get_c2w(camera), camera.fov, h, w)
                # c2w = torch.from_numpy(get_c2w(camera)).to(self.device)
                try:
                    start = time.time()
                    out = render(
                        cam,
                        self.system.gaussian,
                        self.system.pipe,
                        self.system.background_tensor,
                    )
                    self.renderer_output.options = list(out.keys())
                    out = (
                        out[self.renderer_output.value]
                        .detach()
                        .cpu()
                        .clamp(min=0.0, max=1.0)
                        .numpy()
                        * 255.0
                    ).astype(np.uint8)
                    end = time.time()
                    times.append(end - start)
                except RuntimeError as e:
                    print(e)
                    continue
                out = np.moveaxis(out.squeeze(), 0, -1)
                client.set_background_image(out, format="jpeg")
                del out

            self.render_times.append(np.mean(times))
            self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"

    def render_loop(self):
        while True:
            try:
                self.update()
                time.sleep(0.001)
            except KeyboardInterrupt:
                return
