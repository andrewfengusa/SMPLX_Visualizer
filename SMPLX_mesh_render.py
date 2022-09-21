import pyrender
import numpy as np
import torch
import os
import pickle
import cv2
from human_body_prior.mesh import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c, colors
import trimesh
import datetime

from utils.rotation2xyz import SMPLXRotation2xyz


def prepare_mesh_viewer(img_shape, camera_pos=[0, 0, 3.75], yfov = np.pi / 3.0):
    mv = MeshViewer(
        width=img_shape[0], height=img_shape[1], use_offscreen=True
    )
    mv.scene = pyrender.Scene(
        bg_color=colors["white"], ambient_light=(0.3, 0.3, 0.3)
    )
    pc = pyrender.PerspectiveCamera(
        yfov=yfov, aspectRatio=float(img_shape[0]) / img_shape[1]
    )
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array(camera_pos)
    mv.camera_node = mv.scene.add(pc, pose=camera_pose, name="pc-camera")
    mv.viewer = pyrender.OffscreenRenderer(*mv.figsize)
    mv.use_raymond_lighting(5.0)
    #mv._add_raymond_light()
    return mv


class SMPLXVideoRenderer:
    def __init__(self, model_path, model_type='SMPLX', view_type='full-body', device = 'cpu'):
        self.device = device #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model_path = model_path
        if model_type == "SMPLX" and view_type == "full-body":
            self.pose_to_vertices = SMPLXRotation2xyz(self.device, model_path=self.model_path)
            camera_pos = [0, 0, 3.75]
            yfov = np.pi / 3.0
            self.image_shape = (1000, 1000)
        elif model_type == "SMPLX" and view_type == "upper-body":
            self.pose_to_vertices = SMPLXRotation2xyz(self.device, model_path = self.model_path)
            camera_pos = [0, 0, 3.75]
            yfov = np.pi / 7.0
            self.image_shape = (1600, 1000)
        self.mv = prepare_mesh_viewer(self.image_shape, camera_pos = camera_pos, yfov=yfov)

    def save_pose_to_video(self, poses, out_video_file, left_hand_pose = None, right_hand_pose = None, fps = 30, glob_rot=[0.0, 0.0, 0.0]):
        # assume it's a pytorch tensor of (N, J, 3, 3), we could handle the conversion here for list or numpy array
        # N = Num frames, J = Num joints, R = 3x3 rotation matrices
        all_recon_meshes = self.pose_to_vertices(
            poses, None, left_hand_pose = left_hand_pose, right_hand_pose = right_hand_pose,glob_rot = glob_rot)
        out_videos = cv2.VideoWriter(
            out_video_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, self.image_shape
        )
        faces = self.pose_to_vertices.model.faces
        for frame_idx in range(all_recon_meshes.shape[-1]):
            recon_mesh = all_recon_meshes[...,frame_idx]
            recon_body_mesh = trimesh.Trimesh(
                vertices=c2c(recon_mesh).squeeze() + np.array([0.0, 0.0, 0.0]),
                faces=faces,
                vertex_colors=np.tile(colors["yellow"][::-1], (10475, 1)),
            )
            # TODO: Add floor trimesh to the scene to display the ground plane
            self.mv.set_static_meshes([recon_body_mesh])
            body_image = self.mv.render()
            img = body_image.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out_videos.write(img)
        out_videos.release()


def to_tensors(poses, device):
    pose_t = torch.from_numpy(poses)
    final_poses = torch.unsqueeze(pose_t, dim = 0).to(device)
    return final_poses


def test_smplx_renderer():
    smplx_path = r'smplx_model/SMPLX_MALE.npz'
    gesture_data_path = r'data/2ZviHInGBJQ.pkl'
    out_dir = r'./output'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    smplx_renderer = SMPLXVideoRenderer(smplx_path, view_type='upper-body')
    motion_data = pickle.load(open(gesture_data_path, 'rb'))
    for segment_idx, gesture_segment in enumerate(motion_data):
        # print segment information
        start_time_str = str(datetime.timedelta(seconds=gesture_segment["start_time"]))
        end_time_str = str(datetime.timedelta(seconds=gesture_segment["end_time"]))
        print(f'segment {segment_idx}: {start_time_str}--{end_time_str}')
        print(' '.join([w[0] for w in gesture_segment['words']]))

        # run inference at each motion separately
        pose_params = gesture_segment['pose_params']
        body_poses = to_tensors(pose_params['body_pose'], smplx_renderer.device)
        left_hand_poses = to_tensors(pose_params['left_hand_pose'], smplx_renderer.device)
        right_hand_poses = to_tensors(pose_params['right_hand_pose'], smplx_renderer.device)
        global_poses = to_tensors(pose_params['global_pose'], smplx_renderer.device)

        out_video_file = os.path.join(out_dir, 'gesture_segment_%d.mp4' % (segment_idx))
        smplx_renderer.save_pose_to_video(body_poses, out_video_file, left_hand_poses, right_hand_poses, glob_rot = global_poses)


if __name__ == "__main__":
    test_smplx_renderer()

