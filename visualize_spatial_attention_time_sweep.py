import argparse
import csv
import json
import os
import random

import numpy as np
import torch

from motion_pred.utils.config import Config
from visualize_spatial_attention import (
    annotate_topk_edges,
    build_dataset,
    build_joint_labels,
    build_model,
    compute_attention_stats,
    get_topk_cross_joint_edges,
    pick_clip,
    prepare_model_input,
    sanitize_name,
)


def normalize_index(index, total_steps):
    if index < 0:
        index = total_steps + index
    if index < 0 or index >= total_steps:
        raise ValueError(f'时间索引超出范围: {index}, 合法范围是 [0, {total_steps - 1}]')
    return index


def main():
    parser = argparse.ArgumentParser(description='批量导出 time_index 范围内的空间注意力热力图')
    parser.add_argument('--cfg', default='h36m', choices=['h36m', 'humaneva'])
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--iter', type=int, default=500, help='当未提供 --checkpoint 时使用该 epoch')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型权重路径，如 results/h36m/models/0500.p')
    parser.add_argument('--action_keyword', type=str, default='walking', help='动作名关键字（不区分大小写）')
    parser.add_argument('--sample_index', type=int, default=0, help='第几个匹配动作样本')
    parser.add_argument('--clip_start', type=int, default=None, help='序列起始帧；默认取中间片段')
    parser.add_argument('--time_start', type=int, default=0, help='起始时间索引（可为负）')
    parser.add_argument('--time_end', type=int, default=19, help='结束时间索引（可为负，含端点）')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--topk_edges', type=int, default=5, help='每个时间步自动标注top-k跨关节连边')
    parser.add_argument('--hide_joint_labels', action='store_true', help='隐藏坐标轴关节名称标签')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录；默认 results/.../attention_maps/time_sweep')
    parser.add_argument('--prefix', type=str, default=None, help='输出文件前缀')
    parser.add_argument('--save_npy', type=str, default=None, help='保存 [B,H,V,V] 注意力矩阵路径')
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

    default_output_dir = os.path.join(cfg.result_dir, 'attention_maps', 'time_sweep')
    output_dir = args.output_dir if args.output_dir else default_output_dir
    os.makedirs(output_dir, exist_ok=True)

    action_safe = sanitize_name(action)
    default_prefix = f'{args.cfg}_{args.split}_{action_safe}_idx{args.sample_index}_start{start}'
    prefix = args.prefix if args.prefix else default_prefix
    save_npy = args.save_npy if args.save_npy else os.path.join(output_dir, f'{prefix}_attn.npy')

    with torch.no_grad():
        _, _, _, attention_info = model(
            model_input,
            return_attention=True,
            attention_save_path=save_npy,
        )

    attn_temporal = attention_info['last_encoder_st_gat_attn_temporal']
    if attn_temporal is None or attn_temporal.dim() != 5:
        raise RuntimeError('模型未返回期望的时序注意力 [B,H,T,V,V]。')

    total_steps = attn_temporal.shape[2]
    t_start = normalize_index(args.time_start, total_steps)
    t_end = normalize_index(args.time_end, total_steps)
    if t_start > t_end:
        t_start, t_end = t_end, t_start

    summary_rows = []
    for t_idx in range(t_start, t_end + 1):
        heatmap = attn_temporal[0, :, t_idx].mean(dim=0).cpu().numpy()
        v = heatmap.shape[0]
        joint_labels = build_joint_labels(cfg, dataset, v)
        stats = compute_attention_stats(heatmap)
        topk_edges = get_topk_cross_joint_edges(heatmap, joint_labels, args.topk_edges)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(heatmap, cmap='viridis')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        ticks = np.arange(v)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if args.hide_joint_labels:
            ax.set_xticklabels(ticks)
            ax.set_yticklabels(ticks)
            ax.set_xlabel('Key joint index')
            ax.set_ylabel('Query joint index')
        else:
            ax.set_xticklabels(joint_labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(joint_labels, fontsize=8)
            ax.set_xlabel('Key joint name')
            ax.set_ylabel('Query joint name')
        annotate_topk_edges(ax, topk_edges)
        ax.set_title(f'Spatial Attention Heatmap ({action}, t{t_idx}, V={v})')
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f'{prefix}_t{t_idx:02d}_heatmap.png')
        plt.savefig(fig_path, dpi=220)
        plt.close(fig)

        summary_rows.append({
            'time_index': int(t_idx),
            'diag_mean': stats['diag_mean'],
            'offdiag_mean': stats['offdiag_mean'],
            'diag_over_offdiag': stats['diag_over_offdiag'],
            'topk_cross_joint_edges': topk_edges,
            'figure_path': fig_path,
        })
        print(
            f"[INFO] t{t_idx:02d}: diag_mean={stats['diag_mean']:.6f}, "
            f"offdiag_mean={stats['offdiag_mean']:.6f}, "
            f"diag/offdiag={stats['diag_over_offdiag']:.6f}, fig={fig_path}"
        )

    csv_path = os.path.join(output_dir, f'{prefix}_stats.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['time_index', 'diag_mean', 'offdiag_mean', 'diag_over_offdiag', 'figure_path'],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                'time_index': row['time_index'],
                'diag_mean': row['diag_mean'],
                'offdiag_mean': row['offdiag_mean'],
                'diag_over_offdiag': row['diag_over_offdiag'],
                'figure_path': row['figure_path'],
            })

    json_path = os.path.join(output_dir, f'{prefix}_stats_topk.json')
    payload = {
        'checkpoint': checkpoint_path,
        'subject': subject,
        'action': action,
        'clip_start': int(start),
        'time_range': [int(t_start), int(t_end)],
        'attention_temporal_shape': [int(x) for x in attn_temporal.shape],
        'attention_npy_path': save_npy,
        'rows': summary_rows,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f'[INFO] checkpoint: {checkpoint_path}')
    print(f'[INFO] subject/action: {subject} / {action}, clip_start={start}')
    print(f'[INFO] attention_temporal shape: {tuple(attn_temporal.shape)}')
    print(f'[INFO] attention npy saved to: {save_npy}')
    print(f'[INFO] summary csv saved to: {csv_path}')
    print(f'[INFO] summary json saved to: {json_path}')


if __name__ == '__main__':
    main()
