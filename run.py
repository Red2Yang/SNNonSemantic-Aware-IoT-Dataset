import torch
import numpy as np
from agri_snn import (
    AgriDatasetParser,
    MaskedPoissonEncoder,
    DualBranchAgriSNN,
    SingleTaskTrainer,
    AgriDecisionInference,
    create_dataloaders,
)


def main():
    # 1. 数据管道
    parser = AgriDatasetParser('Agriculture_dataset_with_metadata.xlsx')
    X_ms, X_rgb, y_action = parser.run_pipeline()

    # 2. 数据划分
    train_loader, test_loader = create_dataloaders(
        X_ms, X_rgb, y_action,
        test_ratio=0.2, batch_size=64
    )

    # 3. 编码器
    T = 16
    encoder = MaskedPoissonEncoder(T=T)

    # 4. 构建模型
    model = DualBranchAgriSNN(
        ms_dim=parser.input_dims['ms'],
        rgb_dim=parser.input_dims['rgb'],
        hidden_dim=64, tau=2.0
    )
    print(f"\n[模型参数] {sum(p.numel() for p in model.parameters()):,}")

    # 5. 训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = SingleTaskTrainer(
        model=model,
        encoder=encoder,
        device=device,
        lr=1e-3,
        class_weights=parser.action_class_weights,
        log_dir= 'runs'       # 可自定义
    )
    # 传入标签编码器，用于 TensorBoard 记录类别名
    trainer.set_label_encoders(parser.label_encoders)

    history = trainer.fit(train_loader, test_loader,
                          epochs=150, patience=10)

    # 6. 推理示例
    infer = AgriDecisionInference(
        'best_agri_snn.pth', parser, device=device, T=T
    )
    sample = 0
    sample_ms = torch.FloatTensor(X_ms[sample]).unsqueeze(0)
    sample_rgb = torch.FloatTensor(X_rgb[sample]).unsqueeze(0)
    decision = infer.predict(sample_ms, sample_rgb)
    report = infer.explain_decision(
        decision,
        semantic_tags=parser.df['Semantic_Tag'].iloc[sample]
    )
    print(report)


if __name__ == '__main__':
    main()