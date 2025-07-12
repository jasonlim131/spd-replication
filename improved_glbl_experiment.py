#!/usr/bin/env python3
"""
Improved GLBL Experiment - Targeting 96% Accuracy
Based on previous successful results showing ~97.2% GLBL accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import entropy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# Import the classes from the main file
import sys
sys.path.append('.')
from mlp_routing import StandardMLP, GLBLPathwayMLP, GLBLTrainer, evaluate_model

def run_improved_glbl_experiment():
    """Run improved GLBL experiment targeting 96%+ accuracy"""
    print("🎯 IMPROVED GLBL EXPERIMENT - Targeting 96%+ Accuracy")
    print("=" * 70)
    
    # Step 1: Train standard MLP baseline with better config
    standard_config = {
        'epochs': 10,  # More epochs
        'learning_rate': 0.001,
        'batch_size': 256,
        'report_every': 2
    }
    
    print("🔧 Training Standard MLP with improved config...")
    standard_mlp, test_loader, standard_history = train_standard_mlp_improved(standard_config)
    
    # Step 2: Create GLBL pathway MLP with better config
    print(f"\n🛤️ Creating GLBL Pathway MLP with optimized config...")
    
    # Better GLBL configuration based on previous successful results
    glbl_config = {
        'num_input_regions': 4,
        'num_hidden_groups': 4,
        'num_output_groups': 4,
        'momentum': 0.9,
        'router_hidden_size': 512,  # Increased capacity
        'router_dropout': 0.1
    }
    
    pathway_mlp = GLBLPathwayMLP(standard_mlp, glbl_config).to(device)
    
    # Step 3: Train GLBL with better hyperparameters
    print(f"\n🔄 Training GLBL Pathway MLP with improved hyperparameters...")
    
    # Better training configuration
    trainer_config = {
        'learning_rate': 0.0005,  # Slightly higher than before
        'glbl_weight_start': 0.005,  # Lower start
        'glbl_weight_end': 0.03,     # Lower end
        'glbl_weight_schedule': 'linear',
        'top_k': 16,  # More pathways active
        'temperature': 0.8,  # Slightly sharper selection
        'batch_size': 128,
        'report_every': 2
    }
    
    # Create training data - use more samples
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST('./data', train=True, transform=transform)
    subset_indices = torch.randperm(len(train_dataset))[:25000]  # More training data
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    
    # Train with improved GLBL
    trainer = GLBLTrainer(pathway_mlp, trainer_config)
    glbl_history = trainer.train(train_loader, test_loader, epochs=10)  # More epochs
    
    # Step 4: Evaluate performance
    pathway_accuracy = evaluate_model(pathway_mlp, test_loader)
    
    # Step 5: Results comparison
    print(f"\n🎉 IMPROVED EXPERIMENT RESULTS:")
    print("=" * 50)
    print(f"✅ Standard MLP accuracy: {evaluate_model(standard_mlp, test_loader):.2f}%")
    print(f"✅ GLBL Pathway MLP accuracy: {pathway_accuracy:.2f}%")
    print(f"🎯 Target was 96%+ - {'✅ ACHIEVED' if pathway_accuracy >= 96 else '❌ MISSED'}")
    
    # Step 6: Analysis
    pathway_analysis = pathway_mlp.get_pathway_analysis()
    print(f"\n📈 Training Analysis:")
    print(f"   Final GLBL loss: {pathway_analysis.get('final_glbl_loss', 0):.4f}")
    print(f"   Pathway utilization: {pathway_analysis.get('pathway_utilization', 0):.1%}")
    print(f"   Avg active pathways: {pathway_analysis.get('avg_active_pathways', 0):.1f}")
    
    return {
        'standard_mlp': standard_mlp,
        'pathway_mlp': pathway_mlp,
        'standard_history': standard_history,
        'glbl_history': glbl_history,
        'pathway_accuracy': pathway_accuracy
    }

def train_standard_mlp_improved(config):
    """Train standard MLP with improved configuration"""
    print(f"📊 Training Standard MLP for {config['epochs']} epochs")
    print("=" * 60)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )
    
    # Model and training
    model = StandardMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    training_history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'test_accuracy': []
    }
    
    model.train()
    for epoch in range(config['epochs']):
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            _, predicted = torch.max(output.data, 1)
            epoch_total += target.size(0)
            epoch_correct += (predicted == target).sum().item()
            
            # Batch-level reporting
            if batch_idx % 100 == 0:
                batch_accuracy = 100 * (predicted == target).sum().item() / target.size(0)
                print(f'Epoch {epoch+1}/{config["epochs"]}, Batch {batch_idx:3d}: '
                      f'Loss={loss.item():.4f}, Acc={batch_accuracy:.1f}%')
        
        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses)
        train_accuracy = 100 * epoch_correct / epoch_total
        
        # Store training history
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(avg_loss)
        training_history['train_accuracy'].append(train_accuracy)
        
        # Regular reporting
        if (epoch + 1) % config['report_every'] == 0 or epoch == config['epochs'] - 1:
            test_accuracy = evaluate_model(model, test_loader)
            training_history['test_accuracy'].append(test_accuracy)
            
            print(f"\n📊 EPOCH {epoch+1}/{config['epochs']} REPORT:")
            print(f"   Training Loss: {avg_loss:.4f}")
            print(f"   Training Accuracy: {train_accuracy:.2f}%")
            print(f"   Test Accuracy: {test_accuracy:.2f}%")
            print("-" * 60)
        else:
            # Brief epoch summary
            print(f'Epoch {epoch+1}/{config["epochs"]}: '
                  f'Loss={avg_loss:.4f}, Acc={train_accuracy:.1f}%')
    
    # Final results
    final_accuracy = evaluate_model(model, test_loader)
    print(f'\n✅ Standard MLP Final Results:')
    print(f'📈 Final Training Accuracy: {training_history["train_accuracy"][-1]:.2f}%')
    print(f'📊 Final Test Accuracy: {final_accuracy:.2f}%')
    print(f'📉 Final Training Loss: {training_history["train_loss"][-1]:.4f}')
    
    return model, test_loader, training_history

if __name__ == "__main__":
    results = run_improved_glbl_experiment()
    print(f"\n🎯 Final GLBL Accuracy: {results['pathway_accuracy']:.2f}%")