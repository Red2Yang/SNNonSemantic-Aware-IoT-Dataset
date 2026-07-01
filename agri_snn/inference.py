import torch
from .parser import AgriDatasetParser
from .encoder import MaskedPoissonEncoder
from .network import DualBranchAgriSNN
import logging

logging.getLogger('spikingjelly').setLevel(logging.ERROR)

class AgriDecisionInference:
    def __init__(self, model_path, parser: AgriDatasetParser,
                 device='cpu', T=16):
        self.device = torch.device(device)
        self.parser = parser
        self.encoder = MaskedPoissonEncoder(T=T)

        self.model = DualBranchAgriSNN(
            ms_dim=parser.input_dims['ms'],
            rgb_dim=parser.input_dims['rgb']
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, x_ms: torch.Tensor, x_rgb: torch.Tensor):
        x_ms = x_ms.to(self.device)
        x_rgb = x_rgb.to(self.device)

        ms_mask = (x_ms != 0).float()
        rgb_mask = (x_rgb != 0).float()

        ms_seq = self.encoder(x_ms, ms_mask)    # [T, N, ms_dim]
        rgb_seq = self.encoder(x_rgb, rgb_mask) # [T, N, rgb_dim]

        action_rate = self.model(ms_seq, rgb_seq)  # [N, 4]
        self.model.reset()

        action_idx = action_rate.argmax(dim=1).item()
        action_name = self.parser.label_encoders[
            'Action_Suggested'
        ].inverse_transform([action_idx])[0]
        confidence = torch.softmax(action_rate, dim=1).max().item()

        return {'action': action_name, 'confidence': confidence}

    def explain_decision(self, decision: dict, semantic_tags: str = ''):
        msg = (
            "╔═════════════════════════════╗\n"
            "║      农业 SNN 决策报告       ║\n"
            "╠═════════════════════════════╣\n"
            f"║ 语义标签: {semantic_tags:<30s}║\n"
            f"║ ───────────────────────────║\n"
            f"║ 动作决策: {decision['action']:<23s} ║\n"
            f"║ 置信度:   {decision['confidence']:.2%}               ║\n"
            "╚═════════════════════════════╝"
        )
        return msg