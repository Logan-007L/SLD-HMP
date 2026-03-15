import argparse
import os
import pickle
import random

import numpy as np
import torch

from motion_pred.utils.config import Config
from motion_pred.utils.dataset_h36m_multimodal import DatasetH36M
from motion_pred.utils.dataset_humaneva_multimodal import DatasetHumanEva
from models.motion_pred import get_model


def sanitize_name(name):
    return ''.join(ch if ch.isalnum() or ch in ['-', '_'] else '_' for ch in str(name))


def get_joint_name_map(dataset_name):
    if dataset_name == 'h36m':
        # original h36m indices -> names (aligned with kept_joints used by this project)
        return {
            0: 'Hip',
            1: 'RHip',
            2: 'RKnee',
            3: 'RFoot',
            6: 'LHip',
            7: 'LKnee',
            8: 'LFoot',
            12: 'Spine',
            13: 'Thorax',
            14: 'Neck',
            15: 'Head',
            17: 'LShoulder',
            18: 'LElbow',
            19: 'LWrist',
            25: 'RShoulder',
            26: 'RElbow',
            27: 'RWrist',
        }
    if dataset_name == 'humaneva':
        # original humaneva indices -> names
        return {
            0: 'Hip',
            1: 'Torso',
            2: 'LHip',
            3: 'LKnee',
            4: 'LFoot',
            5: 'RHip',
            6: 'RKnee',
            7: 'RFoot',
            8: 'LShoulder',
            9: 'LElbow',
            10: 'LWrist',
            11: 'RShoulder',
            12: 'RElbow',
            13: 'RWrist',
            14: 'Head',
        }
    return {}


def build_joint_labels(cfg, dataset, num_joints):
    # Model uses non-root joints: dataset.kept_joints[1:]
    if hasattr(dataset, 'kept_joints') and len(dataset.kept_joints) >= num_joints + 1:
        used_joint_ids = [int(x) for x in dataset.kept_joints[1:1 + num_joints]]
    else:
        used_joint_ids = list(range(1, num_joints + 1))
    name_map = get_joint_name_map(cfg.dataset)
    labels = [name_map.get(jid, f'J{jid}') for jid in used_joint_ids]
    if len(labels) != num_joints:
        labels = [f'J{i}' for i in range(num_joints)]
    return labels


def build_dataset(cfg, split):
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    kwargs = {}
    if 'multimodal_path' in cfg.specs:
        kwargs['multimodal_path'] = cfg.specs['multimodal_path']
    if 'data_candi_path' in cfg.specs:
        kwargs['data_candi_path'] = cfg.specs['data_candi_path']
    dataset = dataset_cls(
        split,
        cfg.t_his,
        cfg.t_pred,
        actions='all',
        use_vel=cfg.use_vel if cfg.dataset == 'h36m' else False,
        **kwargs,
    )
    if cfg.normalize_data:
        dataset.normalize_data()
    return dataset


def find_action_clips(dataset, action_keyword):
    clips = []
    keyword = action_keyword.lower().strip() if action_keyword else None
    for subject, data_s in dataset.data.items():
        for action, seq in data_s.items():
            if seq.shape[0] < dataset.t_total:
                continue
            if keyword is not None and keyword not in action.lower():
                continue
            clips.append((subject, action, seq))
    return clips


def pick_clip(dataset, action_keyword, sample_index=0, clip_start=None):
    clips = find_action_clips(dataset, action_keyword)
    if len(clips) == 0:
        raise ValueError(f'未找到包含关键字 "{action_keyword}" 的动作序列。')
    subject, action, seq = clips[sample_index % len(clips)]
    max_start = seq.shape[0] - dataset.t_total
    if clip_start is None:
        start = max_start // 2
    else:
        start = max(0, min(clip_start, max_start))
    traj = seq[None, start:start + dataset.t_total]
    return traj, subject, action, start


def build_model(cfg, dataset, checkpoint_path, device):
    model, _ = get_model(cfg, dataset, cfg.dataset)
    model_cp = pickle.load(open(checkpoint_path, 'rb'))
    model.load_state_dict(model_cp['model_dict'])
    model.to(device)
    model.eval()
    return model


def prepare_model_input(traj_np, device):
    # [1, T, J, 3] -> [T, 1, (J-1)*3], root joint is removed exactly as training/testing code.
    traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
    traj = torch.tensor(traj_np, dtype=torch.float32, device=device).permute(1, 0, 2).contiguous()
    return traj


def to_valid_time_index(index, total_steps):
    if index is None:
        return None
    if index < 0:
        index = total_steps + index
    if index < 0 or index >= total_steps:
        raise ValueError(f'time_index 超出范围: {index}, 合法范围是 [0, {total_steps - 1}]。')
    return index


def main():
    parser = argparse.ArgumentParser(description='可视化 ST_GAT 空间注意力热力图')
    parser.add_argument('--cfg', default='h36m', choices=['h36m', 'humaneva'])
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--iter', type=int, default=500, help='当未提供 --checkpoint 时使用该 epoch')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型权重路径，如 results/h36m/models/0500.p')
    parser.add_argument('--action_keyword', type=str, default='walking', help='动作名关键字（不区分大小写）')
    parser.add_argument('--sample_index', type=int, default=0, help='第几个匹配动作样本')
    parser.add_argument('--clip_start', type=int, default=None, help='序列起始帧；默认取中间片段')
    parser.add_argument('--time_index', type=int, default=None, help='从注意力的时间维中选某一帧；默认对时间平均')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_npy', type=str, default=None, help='保存 [B,H,V,V] 的注意力矩阵路径')
    parser.add_argument('--save_fig', type=str, default=None, help='保存热力图路径')
    parser.add_argument('--hide_joint_labels', action='store_true', help='隐藏坐标轴关节名称标签')
    args = parser.parse_args()
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('matplotlib 未安装，请先执行: pip install matplotlib') from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    cfg = Config(args.cfg, test=True)
    dataset = build_dataset(cfg, args.split)

    checkpoint_path = args.checkpoint if args.checkpoint else (cfg.model_path % args.iter)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'checkpoint 不存在: {checkpoint_path}')

    traj_np, subject, action, start = pick_clip(
        dataset,
        action_keyword=args.action_keyword,
        sample_index=args.sample_index,
        clip_start=args.clip_start,
    )
    model_input = prepare_model_input(traj_np, device)

    model = build_model(cfg, dataset, checkpoint_path, device)

    default_dir = os.path.join(cfg.result_dir, 'attention_maps')
    os.makedirs(default_dir, exist_ok=True)
    action_safe = sanitize_name(action)
    default_npy = os.path.join(default_dir, f'{args.cfg}_{args.split}_{action_safe}_idx{args.sample_index}_start{start}_attn.npy')
    default_fig = os.path.join(default_dir, f'{args.cfg}_{args.split}_{action_safe}_idx{args.sample_index}_start{start}_heatmap.png')
    save_npy = args.save_npy if args.save_npy else default_npy
    save_fig = args.save_fig if args.save_fig else default_fig

    with torch.no_grad():
        _, _, _, attention_info = model(
            model_input,
            return_attention=True,
            attention_save_path=save_npy,
        )

    attn_spatial = attention_info['last_encoder_st_gat_attn']
    attn_temporal = attention_info['last_encoder_st_gat_attn_temporal']
    if attn_spatial is None or attn_temporal is None:
        raise RuntimeError('模型未返回最后一层编码器 ST_GAT 注意力，请检查模型结构。')

    # attn_spatial: [B,H,V,V], attn_temporal: [B,H,T,V,V]
    if attn_temporal.dim() != 5:
        raise RuntimeError(f'注意力张量维度异常，期望 [B,H,T,V,V]，实际: {tuple(attn_temporal.shape)}')

    picked_t = to_valid_time_index(args.time_index, attn_temporal.shape[2])
    if picked_t is None:
        heatmap = attn_spatial[0].mean(dim=0).cpu().numpy()
        time_desc = 'temporal_mean'
    else:
        heatmap = attn_temporal[0, :, picked_t].mean(dim=0).cpu().numpy()
        time_desc = f't{picked_t}'

    fig = plt.figure(figsize=(7, 6))
    im = plt.imshow(heatmap, cmap='viridis')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    v = heatmap.shape[0]
    ticks = np.arange(v)
    if args.hide_joint_labels:
        plt.xticks(ticks)
        plt.yticks(ticks)
    else:
        joint_labels = build_joint_labels(cfg, dataset, v)
        plt.xticks(ticks, joint_labels, rotation=45, ha='right', fontsize=8)
        plt.yticks(ticks, joint_labels, fontsize=8)
    if args.hide_joint_labels:
        plt.xlabel('Key joint index')
        plt.ylabel('Query joint index')
    else:
        plt.xlabel('Key joint name')
        plt.ylabel('Query joint name')
    plt.title(f'Spatial Attention Heatmap ({action}, {time_desc}, V={v})')
    plt.tight_layout()
    plt.savefig(save_fig, dpi=220)
    plt.close(fig)

    print(f'[INFO] checkpoint: {checkpoint_path}')
    print(f'[INFO] subject/action: {subject} / {action}, clip_start={start}')
    print(f'[INFO] attention_spatial shape: {tuple(attn_spatial.shape)} (saved to {save_npy})')
    print(f'[INFO] attention_temporal shape: {tuple(attn_temporal.shape)}')
    print(f'[INFO] heatmap ({time_desc}) shape: {heatmap.shape}, saved to {save_fig}')


if __name__ == '__main__':
    main()
