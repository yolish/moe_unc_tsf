import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..')))

import argparse
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from data_provider.synthetic_regression import SyntheticDatasetA, SyntheticDatasetB
from models import SimpleMLP, RegressionMoE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='A', help='Dataset A or B')
    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--prob_expert', action='store_true', help='Use probabilistic experts')
    parser.add_argument('--unc_gating', action='store_true', help='Use uncertainty gating (MoGU)')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='./synthetic_results')
    return parser.parse_args()

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

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DatasetClass = SyntheticDatasetA if args.dataset == 'A' else SyntheticDatasetB
    train_loader = DataLoader(DatasetClass(10000, 'train'), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(DatasetClass(2000, 'test'), batch_size=args.batch_size, shuffle=False)

    # ---------------------------------------------------------
    # קיבוע האקראיות רגע לפני יצירת המודל
    # ---------------------------------------------------------
    set_seed(42)

    if args.num_experts > 1:
        model = RegressionMoE.Model(args, SimpleMLP.Model).to(device)
    else:
        model = SimpleMLP.Model(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if args.num_experts > 1:
                expert_out, expert_unc, weights = model(x)
                
                if args.prob_expert and not args.unc_gating:
                    # 1. Classic MoG Loss
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
                    
                elif args.prob_expert and args.unc_gating:
                    # 2. MoGU Loss
                    weighted_loss = 0.0
                    for i in range(args.num_experts):
                        mean = expert_out[:, i, :]
                        var = expert_unc[:, i, :] + 1e-8
                        w = weights[:, i, :]
                        
                        expert_loss = 0.5 * (torch.log(var) + ((y - mean)**2) / var)
                        weighted_loss += w * expert_loss
                    loss = weighted_loss.mean()
                    
                else:
                    # 3. Standard MoE Loss
                    weighted_loss = 0.0
                    for i in range(args.num_experts):
                        mean = expert_out[:, i, :]
                        w = weights[:, i, :]
                        
                        expert_loss = (y - mean)**2
                        weighted_loss += w * expert_loss
                    loss = weighted_loss.mean()
            else:
                # Single Expert
                if args.prob_expert:
                    mean, var = model(x)
                    loss = 0.5 * (torch.log(var) + ((y - mean)**2) / var).mean()
                else:
                    mean = model(x)
                    loss = nn.MSELoss()(mean, y)
                    
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation and Visualization
    model.eval()
    with torch.no_grad():
        x_plot = torch.linspace(-1, 1, 1000).unsqueeze(1).to(device)
        if args.num_experts > 1:
            expert_out, expert_unc, weights = model(x_plot)
            agg_pred = torch.sum(expert_out * weights, dim=1).cpu().numpy()
            weights = weights.squeeze(-1).cpu().numpy()
        else:
            agg_pred = model(x_plot)[0].cpu().numpy() if args.prob_expert else model(x_plot).cpu().numpy()
            weights = np.ones((1000, 1))

        x_plot_np = x_plot.cpu().numpy().flatten()
        test_dataset = test_loader.dataset
        
    plt.figure(figsize=(10, 6))
    plt.scatter(test_dataset.x.numpy().flatten(), test_dataset.y.numpy().flatten(), s=2, alpha=0.2, label='True Data (Test)')
    plt.plot(x_plot_np, agg_pred.flatten(), color='red', label='Aggregated Mean', linewidth=2)
    
    if args.num_experts > 1:
        expert_out_np = expert_out.squeeze(-1).cpu().numpy()
        for i in range(args.num_experts):
            plt.plot(x_plot_np, expert_out_np[:, i], linestyle='--', alpha=0.7, label=f'Expert {i+1} Mean')
            
    plt.legend()
    plt.title(f"Dataset {args.dataset} | Experts: {args.num_experts} | Prob: {args.prob_expert} | MoGU: {args.unc_gating}")
    
    setting_name = f"Dataset{args.dataset}_NE{args.num_experts}_Prob{args.prob_expert}_MoGU{args.unc_gating}"
    plt.savefig(os.path.join(args.output_dir, f"{setting_name}_predictions.png"))
    
    if args.num_experts > 1:
        plt.figure(figsize=(10, 4))
        for i in range(args.num_experts):
            plt.plot(x_plot_np, weights[:, i], label=f'Expert {i+1}')
        plt.legend()
        plt.title(f"Routing Weights across input space X | MoGU: {args.unc_gating}")
        plt.xlabel("x")
        plt.ylabel("Routing Weight")
        plt.savefig(os.path.join(args.output_dir, f"{setting_name}_weights.png"))

    print(f"Finished {setting_name}. Plots saved to {args.output_dir}.")

if __name__ == '__main__':
    main()