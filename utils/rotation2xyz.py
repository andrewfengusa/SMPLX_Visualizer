import torch
from smplx import SMPLXLayer as _SMPLLayer
import contextlib
import numpy as np

from utils.rotation_conversions import axis_angle_to_matrix

JOINTSTYPE_ROOT = {
    "smpl": 0,
    "vibe": 8}  # 0 is the 8 position: OP MidHip below

JOINT_MAP = {
    'OP Nose': 24,
    'OP Neck': 12,
    'OP RShoulder': 17,
    'OP RElbow': 19,
    'OP RWrist': 21,
    'OP LShoulder': 16,
    'OP LElbow': 18,
    'OP LWrist': 20,
    'OP MidHip': 0,
    'OP RHip': 2,
    'OP RKnee': 5,
    'OP RAnkle': 8,
    'OP LHip': 1,
    'OP LKnee': 4,
    'OP LAnkle': 7,
    'OP REye': 25,
    'OP LEye': 26,
    'OP REar': 27,
    'OP LEar': 28,
    'OP LBigToe': 29,
    'OP LSmallToe': 30,
    'OP LHeel': 31,
    'OP RBigToe': 32,
    'OP RSmallToe': 33,
    'OP RHeel': 34,
    'Right Ankle': 8,
    'Right Knee': 5,
    'Right Hip': 45,
    'Left Hip': 46,
    'Left Knee': 4,
    'Left Ankle': 7,
    'Right Wrist': 21,
    'Right Elbow': 19,
    'Right Shoulder': 17,
    'Left Shoulder': 16,
    'Left Elbow': 18,
    'Left Wrist': 20,
    'Neck (LSP)': 47,
    'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49,
    'Thorax (MPII)': 50,
    'Spine (H36M)': 51,
    'Jaw (H36M)': 52,
    'Head (H36M)': 53,
    'Nose': 24,
    'Left Eye': 26,
    'Right Eye': 25,
    'Left Ear': 28,
    'Right Ear': 27
}

JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear'
]

DIM_FLIP = np.array([1, -1, -1], dtype=np.float32)
DIM_FLIP_TENSOR = torch.tensor([1, -1, -1], dtype=torch.float32)
def flip_pose(pose_vector, pose_format='rot-mat'):
    if pose_format == 'aa':
        if torch.is_tensor(pose_vector):
            dim_flip = DIM_FLIP_TENSOR
        else:
            dim_flip = DIM_FLIP
        return (pose_vector.reshape(-1, 3) * dim_flip).reshape(-1)
    elif pose_format == 'rot-mat':
        rot_mats = pose_vector.reshape(-1, 9).clone()

        rot_mats[:, [1, 2, 3, 6]] *= -1
        return rot_mats.view_as(pose_vector)
    else:
        raise ValueError(f'Unknown rotation format: {pose_format}')


# adapted from VIBE/SPIN to output smpl_joints, vibe joints and action2motion joints
class SMPL(_SMPLLayer):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, model_path, **kwargs):
        kwargs["model_path"] = model_path

        # remove the verbosity for the 10-shapes beta parameters
        with contextlib.redirect_stdout(None):
            super(SMPL, self).__init__(**kwargs)

        vibe_indexes = np.array([JOINT_MAP[i] for i in JOINT_NAMES])
        smpl_indexes = np.arange(24)

        self.maps = \
            {
                "vibe": vibe_indexes,
                "smpl": smpl_indexes}

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPL, self).forward(*args, **kwargs)

        # extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        all_joints = smpl_output.joints  # , extra_joints
        output = {
            "vertices": smpl_output.vertices,
            "all_joints": all_joints }
        for joinstype, indexes in self.maps.items():
            output[joinstype] = all_joints[:, indexes]
        return output

JOINTSTYPES = ["smpl", "vibe", "vertices"]
class SMPLXRotation2xyz:
    def __init__(self, device, model_path):
        self.device = device
        self.model = SMPL(model_path=model_path, num_betas=200).eval().to(device)

    def __call__(self, x, mask, left_hand_pose=None, right_hand_pose=None, translation=False, glob=False,
                 jointstype='vertices', vertstrans=False, betas=None, beta=0,
                 glob_rot=[0.0, 0.0, 0.0], **kwargs):

        if mask == None:
            mask = torch.ones((x.shape[0], x.shape[1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")

        if translation:
            x_translations = x[:, -1, :3]
            x_rotations = x[:, :-1]
        else:
            x_rotations = x

        #nsamples, time, njoints, feats = x_rotations.shape
        nsamples = x_rotations.shape[0]
        time = x_rotations.shape[1]
        # x_rotations = x_rotations.permute(0, 3, 1, 2)
        out = self.model_transform(x_rotations, mask, left_hand_pose, right_hand_pose, glob, betas, beta,
                                   glob_rot)

        # get the desirable joints
        joints = out[jointstype]

        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        # the first translation root at the origin on the prediction
        if jointstype != "vertices":
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

        if translation and vertstrans:
            # the first translation root at the origin
            x_translations = x_translations - x_translations[:, :, [0]]

            # add the translation to all the joints
            x_xyz = x_xyz + x_translations[:, None, :, :]

        return x_xyz

    def model_transform(self, x_rotations, mask, left_hand_pose=None, right_hand_pose=None, glob=False,
                        betas=None, beta=0, glob_rot=[0.0, 0.0, 0.0]):

        # Compute rotations (convert only masked sequences output)
        rotations = x_rotations[mask]
        if not glob:
            if len(glob_rot) == 3: # single global rotation as axis-angle
                global_orient = torch.tensor(glob_rot, device=x_rotations.device)
                global_orient = axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
                global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
            else: # rotation matrices
                global_orient = glob_rot
            #
        else:
            global_orient = rotations[:, 0]
            rotations = rotations[:, 1:]

        out = self.model(body_pose=rotations, global_orient=global_orient, betas=betas,
                         left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
        return out

