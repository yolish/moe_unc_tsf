import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..')))

import argparse
import math
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from data_provider.synthetic_regression import SyntheticDatasetA, SyntheticDatasetB
from models import SimpleMLP, RegressionMoE

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def set_seed(seed=42):
    """קיבוע אקראיות מוחלט כדי להבטיח הוגנות באתחול בין המודלים"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_and_get_weights(args, model_name, prob_expert, unc_gating):
    configs = Config(
        num_experts=args.num_experts,
        prob_expert=prob_expert,
        unc_gating=unc_gating,
        d_model=args.d_model
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DatasetClass = SyntheticDatasetA if args.dataset == 'A' else SyntheticDatasetB
    train_loader = DataLoader(DatasetClass(10000, 'train'), batch_size=args.batch_size, shuffle=True)
    
    # ---------------------------------------------------------
    # כאן אנחנו מקבעים את האקראיות רגע לפני בניית המודל!
    # ---------------------------------------------------------
    set_seed(42)
    
    model = RegressionMoE.Model(configs, SimpleMLP.Model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Training {model_name}...")
    model.train()
    for epoch in range(args.epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            expert_out, expert_unc, weights = model(x)
            
            if prob_expert and not unc_gating:
                # Classic MoG Loss
                log_probs = []
                for i in range(args.num_experts):
                    mean = expert_out[:, i, :]
                    var = expert_unc[:, i, :] + 1e-8
                    w = weights[:, i, :]
                    
                    log_w = torch.log(w + 1e-8)
                    log_norm = -0.5 * math.log(2 * math.pi) - 0.5 * torch.log(var) - 0.5 * ((y - mean)**2) / var
                    log_probs.append(log_w + log_norm)
                    
                log_probs_tensor = torch.stack(log_probs, dim=1)
                loss = -torch.logsumexp(log_probs_tensor, dim=1).mean()
                
            elif prob_expert and unc_gating:
                # MoGU Loss
                weighted_loss = 0.0
                for i in range(args.num_experts):
                    mean = expert_out[:, i, :]
                    var = expert_unc[:, i, :] + 1e-8
                    w = weights[:, i, :]
                    expert_loss = 0.5 * (torch.log(var) + ((y - mean)**2) / var)
                    weighted_loss += w * expert_loss
                loss = weighted_loss.mean()
                
            else:
                # Standard MoE Loss
                weighted_loss = 0.0
                for i in range(args.num_experts):
                    mean = expert_out[:, i, :]
                    w = weights[:, i, :]
                    expert_loss = (y - mean)**2
                    weighted_loss += w * expert_loss
                loss = weighted_loss.mean()
                
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        x_plot = torch.linspace(-1, 1, 1000).unsqueeze(1).to(device)
        _, _, weights = model(x_plot)
        weights = weights.squeeze(-1).cpu().numpy()
        
    return x_plot.cpu().numpy().flatten(), weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='A', help='Dataset A or B')
    parser.add_argument('--num_experts', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    
    os.makedirs('./synthetic_results', exist_ok=True)
    print(f"=== Starting Comparison for Dataset {args.dataset} ===")
    
    x_plot, weights_moe = train_and_get_weights(args, "Standard MoE", prob_expert=False, unc_gating=False)
    _, weights_mog = train_and_get_weights(args, "Probabilistic MoE (MoG)", prob_expert=True, unc_gating=False)
    _, weights_mogu = train_and_get_weights(args, "MoGU", prob_expert=True, unc_gating=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    titles = ["Standard MoE\n(Expert Collapse)", "Probabilistic MoE / MoG\n(Instability/Collapse)", "MoGU (Ours)\n(Perfect Specialization)"]
    weights_list = [weights_moe, weights_mog, weights_mogu]
    
    for idx, ax in enumerate(axes):
        for i in range(args.num_experts):
            ax.plot(x_plot, weights_list[idx][:, i], label=f'Expert {i+1}', linewidth=2.5)
        ax.set_title(titles[idx], fontsize=15, pad=15)
        ax.set_xlabel('x (Input Space)', fontsize=13)
        if idx == 0:
            ax.set_ylabel('Routing Weight (Probability)', fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.suptitle(f'Load Balancing & Expert Specialization - Dataset {args.dataset}', fontsize=18, y=1.05)
    plt.tight_layout()
    
    save_path = f'./synthetic_results/Final_Comparison_Dataset{args.dataset}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Finished! Comparison plot saved to {save_path}")

if __name__ == '__main__':
    main()