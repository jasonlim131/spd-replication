#!/usr/bin/env python3
"""
🚀 ULTRATHINK: Pretrained LLM + GLBL MLP Expertization
Ultra-optimized 1-epoch fine-tuning with layerwise MLP expertization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import time
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🚀 Using device: {device}")

@dataclass
class UltraGLBLConfig:
    """Ultra-optimized GLBL configuration for MLP expertization"""
    # Base model
    model_name: str = "roneneldan/TinyStories-1M"
    
    # GLBL MLP Configuration - ULTRATHINK optimized
    num_mlp_experts: int = 8          # Focused MLP experts
    expert_layers: Optional[List[int]] = None   # Which layers to expertize
    top_k_experts: int = 2            # Ultra-sparse routing
    
    # Training Configuration - 1 epoch optimized
    batch_size: int = 4               # Larger batches for efficiency
    learning_rate: float = 5e-4       # Higher LR for 1 epoch
    max_seq_len: int = 256            # Reasonable sequence length
    num_samples: int = 1000           # Enough for 1 epoch
    
    # Expert routing parameters
    router_temperature: float = 0.5   # Sharp routing decisions
    load_balancing_weight: float = 0.1 # Strong load balancing
    
    # Efficiency parameters
    gradient_checkpointing: bool = False
    mixed_precision: bool = False
    
    def __post_init__(self):
        if self.expert_layers is None:
            # Focus on middle layers for MLP expertization
            self.expert_layers = [2, 4, 6]

class GLBLMLPExpert(nn.Module):
    """Individual MLP expert for GLBL routing"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Expert MLP layers
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with small weights for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize expert weights"""
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert"""
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x

class GLBLMLPRouter(nn.Module):
    """Ultra-efficient router for MLP experts"""
    
    def __init__(self, d_model: int, num_experts: int, temperature: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.temperature = temperature
        
        # Lightweight routing network
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_experts)
        )
        
        # Load balancing tracking
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts"""
        batch_size, seq_len, d_model = x.shape
        
        # Compute routing scores
        routing_scores = self.gate(x)  # [batch, seq, num_experts]
        routing_probs = F.softmax(routing_scores / self.temperature, dim=-1)
        
        # Update usage statistics
        if self.training:
            usage = routing_probs.mean(dim=(0, 1))
            self.expert_usage.data = 0.99 * self.expert_usage.data + 0.01 * usage
            self.total_tokens += batch_size * seq_len
        
        # Compute load balancing loss
        load_balance_loss = self._compute_load_balance_loss(routing_probs)
        
        return routing_probs, load_balance_loss
    
    def _compute_load_balance_loss(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss"""
        # Compute expert usage for this batch
        batch_usage = routing_probs.mean(dim=(0, 1))
        
        # Load balancing loss encourages uniform usage
        uniform_usage = torch.ones_like(batch_usage) / self.num_experts
        load_balance_loss = F.mse_loss(batch_usage, uniform_usage)
        
        return load_balance_loss

class GLBLMLPLayer(nn.Module):
    """GLBL MLP layer with expert routing"""
    
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int, 
                 temperature: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = GLBLMLPRouter(d_model, num_experts, temperature)
        
        # Expert MLPs
        self.experts = nn.ModuleList([
            GLBLMLPExpert(d_model, d_ff, dropout) 
            for _ in range(num_experts)
        ])
        
        # Load balancing weight
        self.load_balance_weight = 0.1
        
        # Expert activation tracking
        self.expert_activations = defaultdict(list)
    
    def forward(self, x: torch.Tensor, record_activations: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with expert routing"""
        batch_size, seq_len, d_model = x.shape
        
        # Route to experts
        routing_probs, load_balance_loss = self.router(x)
        
        # Select top-k experts
        top_k_values, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_values, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process through selected experts
        for expert_idx in range(self.num_experts):
            # Get expert probabilities for this expert
            expert_probs = torch.zeros(batch_size, seq_len, 1, device=x.device)
            
            for k in range(self.top_k):
                k_mask = (top_k_indices[:, :, k] == expert_idx)
                if k_mask.any():
                    # Use advanced indexing to properly assign values
                    batch_idx, seq_idx = torch.where(k_mask)
                    expert_probs[batch_idx, seq_idx, 0] = top_k_probs[batch_idx, seq_idx, k]
            
            # Skip if this expert isn't used
            if expert_probs.sum() == 0:
                continue
            
            # Apply expert
            expert_output = self.experts[expert_idx](x)
            output += expert_output * expert_probs
            
            # Record activations
            if record_activations:
                num_tokens = (expert_probs > 0).sum().item()
                self.expert_activations[f'expert_{expert_idx}'].append({
                    'usage': expert_probs.sum().item(),
                    'avg_prob': expert_probs.mean().item(),
                    'num_tokens': num_tokens
                })
        
        return {
            'output': output,
            'load_balance_loss': load_balance_loss,
            'routing_probs': routing_probs,
            'expert_usage': self.router.expert_usage.clone()
        }

class UltraGLBLModel(nn.Module):
    """Ultra-optimized GLBL model wrapping pretrained LLM"""
    
    def __init__(self, config: UltraGLBLConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained base model
        logger.info(f"📥 Loading pretrained model: {config.model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32  # Use float32 for CPU compatibility
        )
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        logger.info("❄️  Base model frozen - only training GLBL layers")
        
        # Get model dimensions
        self.d_model = self.base_model.config.hidden_size
        self.d_ff = getattr(self.base_model.config, 'intermediate_size', None)
        
        # If d_ff is None, use standard 4x multiplier
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
            logger.info(f"📐 Using default d_ff = 4 * d_model = {self.d_ff}")
        
        self.num_layers = self.base_model.config.num_hidden_layers
        
        logger.info(f"📐 Model dimensions: d_model={self.d_model}, d_ff={self.d_ff}, num_layers={self.num_layers}")
        
        # Create GLBL MLP layers
        self.glbl_mlp_layers = nn.ModuleDict()
        for layer_idx in config.expert_layers:
            if layer_idx < self.num_layers:
                self.glbl_mlp_layers[f'layer_{layer_idx}'] = GLBLMLPLayer(
                    self.d_model, 
                    self.d_ff,
                    config.num_mlp_experts,
                    config.top_k_experts,
                    config.router_temperature
                )
        
        logger.info(f"🧠 Created {len(self.glbl_mlp_layers)} GLBL MLP layers")
        
        # Training statistics
        self.training_stats = defaultdict(list)
        self.expert_activations = defaultdict(list)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                record_activations: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with GLBL MLP expertization"""
        
        # Get base model outputs with hidden states
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract hidden states and logits
        hidden_states = list(base_outputs.hidden_states)
        
        # Apply GLBL MLP layers
        total_load_balance_loss = 0.0
        expert_outputs = {}
        
        for layer_name, glbl_layer in self.glbl_mlp_layers.items():
            layer_idx = int(layer_name.split('_')[1])
            
            # Get hidden state for this layer
            layer_hidden = hidden_states[layer_idx]
            
            # Apply GLBL MLP
            glbl_output = glbl_layer(layer_hidden, record_activations)
            
            # Add residual connection
            modified_hidden = layer_hidden + glbl_output['output']
            hidden_states[layer_idx] = modified_hidden
            
            # Accumulate losses
            total_load_balance_loss += glbl_output['load_balance_loss']
            
            # Store expert outputs
            expert_outputs[layer_name] = glbl_output
            
            # Record activations
            if record_activations:
                self.expert_activations[layer_name].extend(
                    glbl_layer.expert_activations[f'expert_{i}']
                    for i in range(self.config.num_mlp_experts)
                )
        
        # Recompute final logits with modified hidden states
        final_hidden = hidden_states[-1]
        final_logits = self.base_model.lm_head(final_hidden)
        
        return {
            'logits': final_logits,
            'total_load_balance_loss': total_load_balance_loss,
            'expert_outputs': expert_outputs,
            'hidden_states': tuple(hidden_states)
        }
    
    def count_trainable_parameters(self) -> int:
        """Count only trainable parameters (GLBL layers)"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_expert_usage_stats(self) -> Dict[str, Any]:
        """Get expert usage statistics"""
        stats = {}
        for layer_name, glbl_layer in self.glbl_mlp_layers.items():
            usage = glbl_layer.router.expert_usage.cpu().numpy()
            stats[layer_name] = {
                'usage_distribution': usage.tolist(),
                'usage_entropy': -np.sum(usage * np.log(usage + 1e-8)),
                'max_usage': float(usage.max()),
                'min_usage': float(usage.min()),
                'usage_std': float(usage.std())
            }
        return stats

def create_ultrathink_dataset(config: UltraGLBLConfig, tokenizer) -> DataLoader:
    """Create ultra-optimized dataset for 1 epoch training"""
    
    logger.info("📚 Loading TinyStories dataset...")
    
    # Load dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    # Take samples
    stories = []
    for i, example in enumerate(dataset):
        if i >= config.num_samples:
            break
        stories.append(example['text'])
    
    logger.info(f"📖 Loaded {len(stories)} stories")
    
    # Tokenize
    def tokenize_function(text):
        return tokenizer(
            text,
            max_length=config.max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
    
    # Tokenize all stories
    tokenized_data = []
    for story in stories:
        tokens = tokenize_function(story)
        tokenized_data.append(tokens['input_ids'].squeeze())
    
    # Create dataset
    dataset = torch.stack(tokenized_data)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    return dataloader

def train_ultrathink_glbl(config: UltraGLBLConfig) -> Dict[str, Any]:
    """Ultra-optimized 1 epoch GLBL training"""
    
    logger.info("🚀 ULTRATHINK GLBL TRAINING STARTED")
    logger.info("=" * 70)
    
    # Load tokenizer
    logger.info("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    logger.info("🧠 Creating GLBL model...")
    model = UltraGLBLModel(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.count_trainable_parameters()
    
    logger.info(f"📊 Total parameters: {total_params:,}")
    logger.info(f"📊 Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # Create dataset
    dataloader = create_ultrathink_dataset(config, tokenizer)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Training setup
    model.train()
    total_steps = len(dataloader)
    
    logger.info(f"🎯 Training for 1 epoch ({total_steps} steps)")
    logger.info(f"📊 Batch size: {config.batch_size}")
    logger.info(f"📊 Learning rate: {config.learning_rate}")
    logger.info("")
    
    # Training loop
    start_time = time.time()
    epoch_losses = {'lm': [], 'load_balance': [], 'total': []}
    
    for step, batch in enumerate(dataloader):
        batch = batch.to(device)
        
        # Create attention mask
        attention_mask = (batch != tokenizer.pad_token_id)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch, attention_mask, record_activations=(step % 50 == 0))
        
        # Language modeling loss
        logits = outputs['logits']
        lm_loss = criterion(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            batch[:, 1:].contiguous().view(-1)
        )
        
        # Load balancing loss
        load_balance_loss = outputs['total_load_balance_loss']
        
        # Total loss
        total_loss = lm_loss + config.load_balancing_weight * load_balance_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Record losses
        epoch_losses['lm'].append(lm_loss.item())
        epoch_losses['load_balance'].append(load_balance_loss.item())
        epoch_losses['total'].append(total_loss.item())
        
        # Logging
        if step % 50 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            eta = (total_steps - step - 1) / steps_per_sec / 60
            
            logger.info(f"Step {step+1}/{total_steps} ({steps_per_sec:.1f} steps/s, ETA: {eta:.1f}m)")
            logger.info(f"  LM Loss: {lm_loss.item():.4f}")
            logger.info(f"  Load Balance Loss: {load_balance_loss.item():.4f}")
            logger.info(f"  Total Loss: {total_loss.item():.4f}")
    
    # Training complete
    total_time = time.time() - start_time
    logger.info(f"✅ Training complete in {total_time:.1f}s ({total_time/60:.1f}m)")
    
    # Final statistics
    avg_lm_loss = np.mean(epoch_losses['lm'])
    avg_load_balance_loss = np.mean(epoch_losses['load_balance'])
    avg_total_loss = np.mean(epoch_losses['total'])
    
    logger.info(f"📊 Final Statistics:")
    logger.info(f"  Average LM Loss: {avg_lm_loss:.4f}")
    logger.info(f"  Average Load Balance Loss: {avg_load_balance_loss:.4f}")
    logger.info(f"  Average Total Loss: {avg_total_loss:.4f}")
    
    # Get expert usage statistics
    expert_stats = model.get_expert_usage_stats()
    logger.info(f"🧠 Expert Usage Statistics:")
    for layer_name, stats in expert_stats.items():
        logger.info(f"  {layer_name}:")
        logger.info(f"    Usage Entropy: {stats['usage_entropy']:.3f}")
        logger.info(f"    Usage Range: {stats['min_usage']:.3f} - {stats['max_usage']:.3f}")
        logger.info(f"    Usage Std: {stats['usage_std']:.3f}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'config': config,
        'training_time': total_time,
        'final_losses': {
            'lm': avg_lm_loss,
            'load_balance': avg_load_balance_loss,
            'total': avg_total_loss
        },
        'expert_stats': expert_stats,
        'trainable_params': trainable_params
    }

def analyze_expert_specialization(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze expert specialization patterns"""
    
    logger.info("🔬 Analyzing expert specialization...")
    
    model = results['model']
    expert_stats = results['expert_stats']
    
    # Analyze specialization
    specialization_analysis = {}
    
    for layer_name, stats in expert_stats.items():
        usage_dist = np.array(stats['usage_distribution'])
        
        # Specialization metrics
        specialization_analysis[layer_name] = {
            'entropy': stats['usage_entropy'],
            'gini_coefficient': compute_gini_coefficient(usage_dist),
            'dominant_experts': np.argsort(usage_dist)[-3:].tolist(),
            'specialization_score': 1.0 - stats['usage_entropy'] / np.log(len(usage_dist))
        }
    
    logger.info("📊 Specialization Analysis:")
    for layer_name, analysis in specialization_analysis.items():
        logger.info(f"  {layer_name}:")
        logger.info(f"    Specialization Score: {analysis['specialization_score']:.3f}")
        logger.info(f"    Gini Coefficient: {analysis['gini_coefficient']:.3f}")
        logger.info(f"    Dominant Experts: {analysis['dominant_experts']}")
    
    return specialization_analysis

def compute_gini_coefficient(x):
    """Compute Gini coefficient for inequality measurement"""
    x = np.array(x)
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n

def run_ultrathink_experiment():
    """Run the complete ULTRATHINK experiment"""
    
    logger.info("🚀 ULTRATHINK EXPERIMENT STARTING")
    logger.info("=" * 70)
    
    # Create ultra-optimized config
    config = UltraGLBLConfig(
        model_name="roneneldan/TinyStories-1M",
        num_mlp_experts=8,
        expert_layers=[2, 4, 6],  # Focus on key layers
        top_k_experts=2,
        batch_size=4,
        learning_rate=5e-4,
        max_seq_len=256,
        num_samples=1000,
        router_temperature=0.5,
        load_balancing_weight=0.1,
        gradient_checkpointing=False,  # Disable for CPU
        mixed_precision=False  # Disable for CPU
    )
    
    # Run training
    results = train_ultrathink_glbl(config)
    
    # Analyze specialization
    specialization = analyze_expert_specialization(results)
    
    # Final summary
    logger.info("\n🎉 ULTRATHINK EXPERIMENT COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"⏱️  Total time: {results['training_time']/60:.1f} minutes")
    logger.info(f"📊 Trainable parameters: {results['trainable_params']:,}")
    logger.info(f"📈 Final LM loss: {results['final_losses']['lm']:.4f}")
    logger.info(f"⚖️  Final load balance loss: {results['final_losses']['load_balance']:.4f}")
    logger.info(f"🧠 Expert layers: {len(results['expert_stats'])}")
    
    # Save results
    results['specialization_analysis'] = specialization
    
    return results

if __name__ == "__main__":
    try:
        results = run_ultrathink_experiment()
        
        print("\n🎉 SUCCESS! ULTRATHINK GLBL experiment completed!")
        print(f"⏱️  Training time: {results['training_time']/60:.1f} minutes")
        print(f"🧠 Specialized {len(results['expert_stats'])} expert layers")
        print(f"📊 Final loss: {results['final_losses']['total']:.4f}")
        
        # Show top specializations
        print("\n🏆 Top Expert Specializations:")
        for layer_name, analysis in results['specialization_analysis'].items():
            print(f"  {layer_name}: {analysis['specialization_score']:.3f}")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()