import random
import viser.transforms as tf
import viser
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def remove_all(item_list):
    for item in item_list:
        item.remove()

def new_frustums(view_index, frame, cam, image, visible, server):
    H, W = cam.image_height, cam.image_width
    image = image[0].cpu().numpy()
    image = np.clip(image * 255, 0, 255).astype('uint8')

    frustum = server.add_camera_frustum(
        f"/train/frame_{view_index}/frustum",
        fov=cam.FoVy,
        aspect=W / H,
        scale=0.15,
        image=image, # 0-255 uint8 H W C
        visible=visible,
    )

    def attach_callback(
            frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
    ) -> None:
        @frustum.on_click
        def _(_) -> None:
            for client in server.get_clients().values():
                client.camera.wxyz = frame.wxyz
                client.camera.position = frame.position

    attach_callback(frustum, frame)
    return frustum

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 8.0)) * 8
    W = int(np.round(W / 8.0)) * 8
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def vis_depth(depth_map_tensor):
    normalized_depth_map = (depth_map_tensor - depth_map_tensor.min()) / (
                depth_map_tensor.max() - depth_map_tensor.min())

    depth_map_image = Image.fromarray((normalized_depth_map.numpy() * 255).astype('uint8'))

    plt.imshow(depth_map_image, cmap='gray')
    plt.show()

def resize_image_ctn(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def new_frustum_from_cam(cam, server, image):

    frame = server.add_frame(
        f"/inpaint/frame",
        wxyz=cam.wxyz,
        position=cam.position,
        axes_length=0.1,
        axes_radius=0.01,
    )
    H, W = image.shape[0], image.shape[1]
    frustum = server.add_camera_frustum(
        f"/inpaint/frame/frustum",
        fov=cam.fov,
        aspect=W / H,
        scale=0.15,
        image=image,
    )
    def attach_callback(
            frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
    ) -> None:
        @frustum.on_click
        def _(_) -> None:
            for client in server.get_clients().values():
                client.camera.wxyz = frame.wxyz
                client.camera.position = frame.position
    attach_callback(frustum, frame)

    return frame, frustum


def sample_train_camera(colmap_cameras, edit_cam_num, server):
    total_view_num = len(colmap_cameras)
    random.seed(0)  # make sure same views
    view_index = random.sample(
        range(0, total_view_num),
        min(total_view_num, edit_cam_num),
    )
    edit_cameras = [colmap_cameras[idx] for idx in view_index]
    train_frames = []
    train_frustums = []

    for idx, cam in enumerate(edit_cameras):
        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(cam.qvec), cam.T
        ).inverse()
        wxyz = T_world_camera.rotation().wxyz
        position = T_world_camera.translation()

        # breakpoint()
        frame = server.add_frame(
            f"/train/frame_{idx}",
            wxyz=wxyz,
            position=position,
            axes_length=0.1,
            axes_radius=0.01,
        )
        H, W = cam.image_height, cam.image_width
        frustum = server.add_camera_frustum(
            f"/train/frame_{idx}/frustum",
            fov=cam.FoVy,
            aspect=W / H,
            scale=0.15,
            image=None,
        )

        def attach_callback(
                frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
        ) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        attach_callback(frustum, frame)

        train_frames.append(frame)
        train_frustums.append(frustum)

    return edit_cameras, train_frames, train_frustums
