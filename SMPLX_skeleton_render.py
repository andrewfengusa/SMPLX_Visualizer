import numpy as np
import torch
import os
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.rotation2xyz import SMPLXRotation2xyz


class SMPLXPoseVideoRenderer:
    def __init__(self, model_path, model_type='SMPLX', device='cpu'):
        self.device = device #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model_path = model_path
        self.rot_to_pos = SMPLXRotation2xyz(self.device, model_path=self.model_path)
        # see joint names here: https://github.com/vchoutas/smplx/blob/master/smplx/joint_names.py
        self.skeleton_links = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]

    def save_pose_to_video(self, poses, out_video_file, left_hand_pose=None, right_hand_pose=None, fps=30, glob_rot=[0.0, 0.0, 0.0]):
        joint_pos = self.rot_to_pos(
            poses, None, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, glob_rot=glob_rot, jointstype='all_joints')
        joint_pos = joint_pos[0].permute(2, 0, 1).contiguous()  # (frames, joints, 3)
        joint_pos = joint_pos.numpy()

        # make animation video
        fig = plt.figure(figsize=(10, 10), dpi=200)
        ax = fig.add_subplot(projection="3d")
        ax.view_init(10, 80)
        ax.set(xlim3d=(1, -1), xlabel='dim 0')
        ax.set(ylim3d=(-1, 1), ylabel='dim 2')
        ax.set(zlim3d=(-1, 1), zlabel='dim 1')

        title = ax.text2D(0.05, 0.95, "frame 0", transform=ax.transAxes)

        def update(t):
            data = joint_pos[t]

            # lines
            for line, link in zip(lines, self.skeleton_links):
                line.set_data(data[link, 0], data[link, 2])
                line.set_3d_properties(data[link, 1])

            # points
            pts.set_data(data[:, 0], data[:, 2])
            pts.set_3d_properties(data[:, 1])

            title.set_text(f"frame {t}")

            artists = [title, pts]
            artists.extend(lines)
            return artists

        pts, = ax.plot(joint_pos[0, :, 0], joint_pos[0, :, 1], joint_pos[0, :, 2], linestyle="", marker="o", markersize=4)
        lines = [ax.plot([], [], [], linewidth=4, color="blue", alpha=0.3)[0] for _ in self.skeleton_links]

        ani = animation.FuncAnimation(
            fig, update, joint_pos.shape[0], interval=30, blit=True)

        # write to video
        writervideo = animation.FFMpegWriter(fps=fps)
        ani.save(out_video_file, writer=writervideo)
        # plt.show()


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
    smplx_renderer = SMPLXPoseVideoRenderer(smplx_path)
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

        out_video_file = os.path.join(out_dir, 'gesture_segment_%d_skeleton.mp4' % (segment_idx))
        smplx_renderer.save_pose_to_video(body_poses, out_video_file, left_hand_poses, right_hand_poses, glob_rot=global_poses)


if __name__ == "__main__":
    test_smplx_renderer()

