"""
test.py - 加载训练好的 SNN 模型权重，在测试集上进行评估
用法：
    python test.py                              # 默认使用 best_agri_snn.pth
    python test.py --model best_agri_snn.pth    # 指定权重文件
    python test.py --show 10                    # 展示前 10 条预测详情
"""

import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from agri_snn import (
    AgriDatasetParser,
    MaskedPoissonEncoder,
    DualBranchAgriSNN,
    create_dataloaders,
)


def parse_args():
    parser = argparse.ArgumentParser(description='农业 SNN 模型测试')
    parser.add_argument('--model', type=str, default='best_agri_snn.pth',
                        help='模型权重文件路径')
    parser.add_argument('--data', type=str,
                        default='Agriculture_dataset_with_metadata.xlsx',
                        help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='测试批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备 (cuda / cpu)')
    parser.add_argument('--show', type=int, default=0,
                        help='展示前 N 条预测详情（0 表示不展示）')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ==================== 1. 加载数据 ====================
    print("\n[1/5] 加载数据...")
    parser = AgriDatasetParser(args.data)
    X_ms, X_rgb, y_action = parser.run_pipeline()

    # 划分测试集（与训练时保持一致）
    _, test_loader = create_dataloaders(
        X_ms, X_rgb, y_action,
        test_ratio=0.2, batch_size=args.batch_size, random_state=42
    )

    # ==================== 2. 构建编码器 ====================
    print("\n[2/5] 初始化编码器...")
    T = 16
    encoder = MaskedPoissonEncoder(T=T)

    # ==================== 3. 加载模型 ====================
    print(f"\n[3/5] 加载模型权重: {args.model}")
    model = DualBranchAgriSNN(
        ms_dim=parser.input_dims['ms'],
        rgb_dim=parser.input_dims['rgb']
    ).to(device)
    model.load_state_dict(
        torch.load(args.model, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ==================== 4. 测试 ====================
    print("\n[4/5] 开始测试...")
    all_preds = []
    all_labels = []
    all_probs = []  # softmax 概率

    for batch in test_loader:
        x_ms, x_rgb, ms_mask, rgb_mask, y_action_batch = batch
        x_ms = x_ms.to(device)
        x_rgb = x_rgb.to(device)
        ms_mask = ms_mask.to(device)
        rgb_mask = rgb_mask.to(device)

        # 编码
        ms_seq = encoder(x_ms, ms_mask)
        rgb_seq = encoder(x_rgb, rgb_mask)

        # 推理
        action_rate = model(ms_seq, rgb_seq)
        model.reset()

        probs = torch.softmax(action_rate, dim=1)
        preds = action_rate.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y_action_batch.tolist())
        all_probs.extend(probs.cpu().tolist())

    # ==================== 5. 评估指标 ====================
    print("\n[5/5] 计算指标...")
    class_names = parser.label_encoders['Action_Suggested'].classes_

    # 基础准确率
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n{'='*60}")
    print(f"  测试集准确率: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*60}")

    # 分类报告
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    )
    print("\n分类报告:")
    print(report)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("混淆矩阵 (行=真实, 列=预测):")
    print(f"{'':>20}", end='')
    for name in class_names:
        print(f"{name:>18}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>20}", end='')
        for val in row:
            print(f"{val:>18}", end='')
        print()

    # 各类别详细指标
    print("\n各类别指标:")
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)
    support = np.bincount(all_labels, minlength=len(class_names))

    header = f"{'类别':<20} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'样本数':<10}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        print(f"{name:<20} {precision[i]:<10.4f} {recall[i]:<10.4f} "
              f"{f1[i]:<10.4f} {support[i]:<10}")

    # ==================== 可选：展示预测详情 ====================
    if args.show > 0:
        print(f"\n{'='*60}")
        print(f"  展示前 {args.show} 条预测详情")
        print(f"{'='*60}")

        # 获取对应的原始数据行
        all_indices = list(range(len(test_loader.dataset)))
        # 由于 test_loader 是 DataLoader，我们直接从 dataset 拿数据
        test_dataset = test_loader.dataset

        count = 0
        for i in range(len(test_dataset)):
            if count >= args.show:
                break

            x_ms_i, x_rgb_i, _, _, label_i = test_dataset[i]

            # 推理单条
            model.reset()
            x_ms_i = x_ms_i.unsqueeze(0).to(device)
            x_rgb_i = x_rgb_i.unsqueeze(0).to(device)
            ms_mask_i = (x_ms_i != 0).float()
            rgb_mask_i = (x_rgb_i != 0).float()

            ms_seq_i = encoder(x_ms_i, ms_mask_i)
            rgb_seq_i = encoder(x_rgb_i, rgb_mask_i)
            rate_i = model(ms_seq_i, rgb_seq_i)
            prob_i = torch.softmax(rate_i, dim=1)

            pred_i = rate_i.argmax(dim=1).item()
            true_i = label_i.item()
            confidence_i = prob_i.max().item()

            pred_name = class_names[pred_i]
            true_name = class_names[true_i]

            # 获取原始语义标签（如果 parser.df 可用）
            semantic = ''
            if parser.df is not None:
                # 测试集索引需要映射回原始 DataFrame
                # 简化处理：直接从 test_dataset 拿不到原始索引，这里只显示序号
                pass

            print(f"\n样本 #{count+1}")
            print(f"  真实动作: {true_name}")
            print(f"  预测动作: {pred_name}")
            print(f"  置信度:   {confidence_i:.2%}")
            print(f"  各概率: ", end='')
            for j, name in enumerate(class_names):
                print(f"{name}={prob_i[0, j]:.3f}  ", end='')
            print()
            print(f"  {'✓' if pred_i == true_i else '✗'} 预测{'' if pred_i == true_i else '错'}")
            count += 1

    print(f"\n{'='*60}")
    print("  测试完成")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()