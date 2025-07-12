"""
Complete Multiplicative GLBL Implementation
Pre × MLP × Post Expert Combinations with Exponential Pathway Scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import math

class PathwayDecomposer:
    """Handles pathway index decomposition: pathway_idx → (pre_idx, mlp_idx, post_idx)"""
    
    def __init__(self, num_pre: int, num_mlp: int, num_post: int):
        self.num_pre = num_pre
        self.num_mlp = num_mlp  
        self.num_post = num_post
        self.total_pathways = num_pre * num_mlp * num_post
        
    def decompose(self, pathway_idx: int) -> Tuple[int, int, int]:
        """
        Decompose pathway index into expert combination
        
        Example: pathway_idx = 2847, num_pre=16, num_mlp=16, num_post=16
        pre_idx = 2847 // (16*16) = 2847 // 256 = 11
        remaining = 2847 % 256 = 31
        mlp_idx = 31 // 16 = 1  
        post_idx = 31 % 16 = 15
        → Pathway = Pre[11] × MLP[1] × Post[15]
        """
        pre_idx = pathway_idx // (self.num_mlp * self.num_post)
        remaining = pathway_idx % (self.num_mlp * self.num_post)
        mlp_idx = remaining // self.num_post
        post_idx = remaining % self.num_post
        return pre_idx, mlp_idx, post_idx
    
    def compose(self, pre_idx: int, mlp_idx: int, post_idx: int) -> int:
        """Compose expert indices back into pathway index"""
        return pre_idx * (self.num_mlp * self.num_post) + mlp_idx * self.num_post + post_idx

class MultiplicativeRouter(nn.Module):
    """Routes input to multiplicative pathway combinations"""
    
    def __init__(self, d_model: int, total_pathways: int):
        super().__init__()
        self.d_model = d_model
        self.total_pathways = total_pathways
        
        # Deep router for complex pathway selection
        self.router = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, total_pathways)
        )
        
        # Learnable temperature for routing sharpness
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route input to pathway scores [batch, seq, total_pathways]"""
        return self.router(x)

class MultiplicativeMLPLayer(nn.Module):
    """
    CORE MULTIPLICATIVE GLBL LAYER
    
    Creates Pre × MLP × Post expert combinations:
    - Pre-experts: Different input transformations  
    - MLP-experts: Different processing functions
    - Post-experts: Different output transformations
    
    Total pathways = num_pre × num_mlp × num_post
    Active pathways = top_k (sparse computation)
    """
    
    def __init__(self, d_model: int, num_pre: int = 16, num_mlp: int = 16, 
                 num_post: int = 16, top_k: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_pre = num_pre
        self.num_mlp = num_mlp
        self.num_post = num_post
        self.top_k = top_k
        self.total_pathways = num_pre * num_mlp * num_post
        
        print(f"🧠 Multiplicative MLP: {num_pre}×{num_mlp}×{num_post} = {self.total_pathways:,} pathways")
        print(f"   Active per forward: {top_k} ({100*top_k/self.total_pathways:.3f}% sparsity)")
        
        # Pre-embedding experts: Input transformations
        self.pre_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                self._get_activation(i, 'pre')
            ) for i in range(num_pre)
        ])
        
        # MLP experts: Core processing with varying architectures
        self.mlp_experts = nn.ModuleList([
            self._create_mlp_expert(d_model, i) for i in range(num_mlp)
        ])
        
        # Post-embedding experts: Output transformations
        self.post_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model) if i % 2 == 0 else nn.Identity(),
                nn.Dropout(0.05 + 0.15 * (i / num_post))
            ) for i in range(num_post)
        ])
        
        # Pathway router and decomposer
        self.router = MultiplicativeRouter(d_model, self.total_pathways)
        self.decomposer = PathwayDecomposer(num_pre, num_mlp, num_post)
        
        # GLBL tracking
        self.register_buffer('global_pathway_freq', torch.zeros(self.total_pathways))
        self.register_buffer('pathway_usage_count', torch.zeros(1))
        
        # Pathway activation tracking
        self.pathway_activations = defaultdict(list)
        
    def _get_activation(self, idx: int, expert_type: str) -> nn.Module:
        """Get diverse activation functions for different experts"""
        activations = [nn.GELU(), nn.ReLU(), nn.Tanh(), nn.SiLU()]
        return activations[idx % len(activations)]
    
    def _create_mlp_expert(self, d_model: int, idx: int) -> nn.Module:
        """Create diverse MLP expert architectures"""
        hidden_size = d_model * (2 + idx // 4)  # Varying widths: 2x, 3x, 4x, 5x
        
        return nn.Sequential(
            nn.Linear(d_model, hidden_size),
            self._get_activation(idx, 'mlp'),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, d_model),
            nn.Dropout(0.05)
        )
    
    def compute_glbl_loss(self, pathway_scores: torch.Tensor) -> torch.Tensor:
        """
        Global Load Balancing Loss for multiplicative pathways
        Prevents pathway collapse by encouraging even usage
        """
        # Current batch pathway probabilities
        pathway_probs = F.softmax(pathway_scores, dim=-1)
        current_freq = pathway_probs.mean(dim=(0, 1))
        
        # Update global frequency tracking
        if self.training:
            momentum = 0.9
            with torch.no_grad():
                self.global_pathway_freq.data = (
                    momentum * self.global_pathway_freq.data +
                    (1 - momentum) * current_freq.data
                )
                self.pathway_usage_count += 1
        
        # GLBL loss: minimize frequency variance (encourage uniform usage)
        ideal_freq = 1.0 / self.total_pathways
        freq_variance = torch.var(current_freq)
        glbl_loss = self.total_pathways * freq_variance
        
        return glbl_loss
    
    def forward(self, x: torch.Tensor, record_activations: bool = False) -> Dict[str, torch.Tensor]:
        """
        MULTIPLICATIVE FORWARD PASS
        
        1. Route to top-K pathway combinations
        2. For each active pathway, decompose to (pre_idx, mlp_idx, post_idx)
        3. Compute Pre[i] → MLP[j] → Post[k] pipeline
        4. Weighted sum of pathway outputs
        """
        batch_size, seq_len, d_model = x.shape
        
        # Step 1: Route to pathways
        pathway_scores = self.router(x)  # [batch, seq, total_pathways]
        glbl_loss = self.compute_glbl_loss(pathway_scores)
        
        # Step 2: Select top-K pathways
        pathway_probs = F.softmax(pathway_scores / self.router.temperature, dim=-1)
        top_values, top_indices = torch.topk(pathway_probs, self.top_k, dim=-1)
        
        # Create sparse pathway weights
        pathway_weights = torch.zeros_like(pathway_probs)
        pathway_weights.scatter_(-1, top_indices, top_values)
        pathway_weights = pathway_weights / (pathway_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Step 3: Compute active pathways
        output = torch.zeros_like(x)
        active_pathways = []
        pathway_details = []
        
        # Process each batch position
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                # Get top pathways for this position
                pos_top_indices = top_indices[batch_idx, seq_idx]  # [top_k]
                pos_top_weights = top_values[batch_idx, seq_idx]   # [top_k]
                
                for k in range(self.top_k):
                    pathway_idx = pos_top_indices[k].item()
                    weight = pos_top_weights[k].item()
                    
                    if weight < 1e-6:  # Skip negligible weights
                        continue
                    
                    # Decompose pathway into expert indices
                    pre_idx, mlp_idx, post_idx = self.decomposer.decompose(pathway_idx)
                    
                    # Compute pathway: Pre → MLP → Post
                    x_input = x[batch_idx:batch_idx+1, seq_idx:seq_idx+1]  # [1, 1, d_model]
                    
                    x_pre = self.pre_experts[pre_idx](x_input)
                    x_mlp = self.mlp_experts[mlp_idx](x_pre)  
                    x_post = self.post_experts[post_idx](x_mlp)
                    
                    # Accumulate weighted output
                    output[batch_idx, seq_idx] += weight * x_post.squeeze()
                    
                    # Record for analysis
                    if record_activations:
                        pathway_details.append({
                            'pathway_idx': pathway_idx,
                            'pre_idx': pre_idx,
                            'mlp_idx': mlp_idx, 
                            'post_idx': post_idx,
                            'weight': weight,
                            'position': (batch_idx, seq_idx),
                            'output_norm': x_post.norm().item()
                        })
        
        # Count unique pathways used
        unique_pathways = len(set(p['pathway_idx'] for p in pathway_details))
        sparsity = 1.0 - (unique_pathways / self.total_pathways)
        
        if record_activations:
            self.pathway_activations['current_batch'] = pathway_details
        
        return {
            'output': output,
            'glbl_loss': glbl_loss,
            'pathway_weights': pathway_weights,
            'active_pathways': pathway_details,
            'unique_pathways': unique_pathways,
            'sparsity': sparsity,
            'total_pathways': self.total_pathways
        }

class MultiplicativeGLBLTransformerLayer(nn.Module):
    """Transformer layer with multiplicative GLBL MLP"""
    
    def __init__(self, d_model: int, n_heads: int, layer_idx: int,
                 num_pre: int = 16, num_mlp: int = 16, num_post: int = 16):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=0.1, batch_first=True
        )
        
        # Multiplicative GLBL MLP
        self.mlp = MultiplicativeMLPLayer(d_model, num_pre, num_mlp, num_post)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                record_activations: bool = False) -> Dict[str, torch.Tensor]:
        
        # Attention with residual
        normed_x = self.ln1(x)
        attn_output, attn_weights = self.attention(
            normed_x, normed_x, normed_x, key_padding_mask=attention_mask
        )
        x = x + attn_output
        
        # Multiplicative GLBL MLP with residual  
        normed_x = self.ln2(x)
        mlp_result = self.mlp(normed_x, record_activations)
        x = x + mlp_result['output']
        
        return {
            'output': x,
            'glbl_loss': mlp_result['glbl_loss'],
            'active_pathways': mlp_result['active_pathways'],
            'unique_pathways': mlp_result['unique_pathways'],
            'sparsity': mlp_result['sparsity'],
            'total_pathways': mlp_result['total_pathways'],
            'attention_weights': attn_weights
        }

class MultiplicativeGLBLModel(nn.Module):
    """Complete Multiplicative GLBL Transformer Model"""
    
    def __init__(self, vocab_size: int = 4096, d_model: int = 128, n_layers: int = 8,
                 n_heads: int = 8, max_seq_len: int = 512, 
                 glbl_layers: List[int] = [1, 3, 5, 7]):
        super().__init__()
        self.d_model = d_model
        self.glbl_layers = glbl_layers
        
        print("🏗️  MULTIPLICATIVE GLBL TRANSFORMER")
        print("=" * 60)
        print(f"Vocab: {vocab_size:,}, d_model: {d_model}, layers: {n_layers}")
        print(f"GLBL layers: {glbl_layers}")
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(0.1)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        total_pathways = 0
        
        for i in range(n_layers):
            if i in glbl_layers:
                # Multiplicative GLBL layer with scaling expert counts
                base_experts = 12 + (i // 2) * 2  # 12, 14, 16, 18 experts
                layer = MultiplicativeGLBLTransformerLayer(
                    d_model, n_heads, i, base_experts, base_experts, base_experts
                )
                layer_pathways = base_experts ** 3
                total_pathways += layer_pathways
                print(f"Layer {i}: {base_experts}³ = {layer_pathways:,} pathways")
            else:
                # Standard transformer layer
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                    dropout=0.1, batch_first=True, norm_first=True
                )
                print(f"Layer {i}: Standard (no GLBL)")
            
            self.layers.append(layer)
        
        # Output layers
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        print(f"\n📊 TOTAL MULTIPLICATIVE PATHWAYS: {total_pathways:,}")
        print(f"📊 GLBL LAYERS: {len(glbl_layers)}")
        print(f"📊 PARAMETERS: {sum(p.numel() for p in self.parameters()):,}")
        print("=" * 60)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                record_activations: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass through multiplicative GLBL transformer"""
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Pass through layers
        total_glbl_loss = torch.tensor(0.0, device=device)
        all_active_pathways = []
        total_pathways = 0
        total_unique_pathways = 0
        layer_results = []
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MultiplicativeGLBLTransformerLayer):
                # GLBL layer
                layer_result = layer(x, attention_mask, record_activations)
                x = layer_result['output']
                total_glbl_loss += layer_result['glbl_loss']
                all_active_pathways.extend(layer_result['active_pathways'])
                total_pathways += layer_result['total_pathways']
                total_unique_pathways += layer_result['unique_pathways']
                layer_results.append(layer_result)
            else:
                # Standard layer
                if attention_mask is not None:
                    # Convert attention mask for standard layer
                    attention_mask_bool = attention_mask.bool()
                    x = layer(x, src_key_padding_mask=~attention_mask_bool)
                else:
                    x = layer(x)
        
        # Final processing
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate overall sparsity
        overall_sparsity = 1.0 - (total_unique_pathways / max(total_pathways, 1))
        
        return {
            'logits': logits,
            'hidden_states': x,
            'total_glbl_loss': total_glbl_loss,
            'all_active_pathways': all_active_pathways,
            'total_pathways': total_pathways,
            'unique_pathways': total_unique_pathways,
            'overall_sparsity': overall_sparsity,
            'layer_results': layer_results
        }
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, 
                 temperature: float = 1.0, do_sample: bool = True) -> torch.Tensor:
        """Generate text using the multiplicative GLBL model"""
        self.eval()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self(generated)
                logits = outputs['logits']
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # Sample from distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Stop if max sequence length reached
                if generated.shape[1] >= 512:  # max_seq_len
                    break
        
        return generated

def analyze_multiplicative_pathways(model_results: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Analyze multiplicative pathway usage and specialization"""
    
    pathways = model_results['all_active_pathways']
    if not pathways:
        return {'error': 'No pathway activations recorded'}
    
    # Pathway usage statistics
    pathway_counts = defaultdict(int)
    expert_usage = {'pre': defaultdict(int), 'mlp': defaultdict(int), 'post': defaultdict(int)}
    pathway_weights = defaultdict(list)
    
    for pathway in pathways:
        pathway_idx = pathway['pathway_idx']
        pathway_counts[pathway_idx] += 1
        pathway_weights[pathway_idx].append(pathway['weight'])
        
        # Track expert usage
        expert_usage['pre'][pathway['pre_idx']] += 1
        expert_usage['mlp'][pathway['mlp_idx']] += 1  
        expert_usage['post'][pathway['post_idx']] += 1
    
    # Most used pathways
    top_pathways = sorted(pathway_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Expert utilization
    expert_stats = {}
    for expert_type in ['pre', 'mlp', 'post']:
        usage = expert_usage[expert_type]
        total_usage = sum(usage.values())
        expert_stats[expert_type] = {
            'total_experts': len(usage),
            'total_usage': total_usage,
            'avg_usage': total_usage / len(usage) if usage else 0,
            'most_used': max(usage.items(), key=lambda x: x[1]) if usage else None,
            'least_used': min(usage.items(), key=lambda x: x[1]) if usage else None
        }
    
    return {
        'total_active_pathways': len(pathways),
        'unique_pathways': len(pathway_counts),
        'total_pathways': model_results['total_pathways'],
        'sparsity': model_results['overall_sparsity'],
        'top_pathways': top_pathways,
        'expert_stats': expert_stats,
        'avg_pathway_weight': np.mean([p['weight'] for p in pathways]),
        'glbl_loss': model_results['total_glbl_loss'].item()
    }

def demonstrate_multiplicative_glbl():
    """Comprehensive demonstration of multiplicative GLBL"""
    
    print("\n🚀 MULTIPLICATIVE GLBL DEMONSTRATION")
    print("=" * 80)
    
    # Create model
    model = MultiplicativeGLBLModel(
        vocab_size=1000, d_model=128, n_layers=6, n_heads=8,
        glbl_layers=[1, 3, 5]
    )
    
    # Create sample input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"\n📝 Input: {batch_size} × {seq_len} tokens")
    
    # Forward pass with pathway recording
    model.eval()
    with torch.no_grad():
        results = model(input_ids, record_activations=True)
    
    # Analyze results
    analysis = analyze_multiplicative_pathways(results)
    
    print(f"\n📊 RESULTS ANALYSIS")
    print("-" * 40)
    print(f"Total pathways available: {analysis['total_pathways']:,}")
    print(f"Unique pathways used: {analysis['unique_pathways']:,}")
    print(f"Total pathway activations: {analysis['total_active_pathways']:,}")
    print(f"Computational sparsity: {analysis['sparsity']:.4f}")
    print(f"Average pathway weight: {analysis['avg_pathway_weight']:.4f}")
    print(f"GLBL loss: {analysis['glbl_loss']:.6f}")
    
    print(f"\n🏆 TOP PATHWAYS")
    print("-" * 40)
    for pathway_idx, count in analysis['top_pathways'][:5]:
        decomposer = PathwayDecomposer(12, 12, 12)  # Adjust based on layer
        pre_idx, mlp_idx, post_idx = decomposer.decompose(pathway_idx)
        print(f"Pathway {pathway_idx}: Pre[{pre_idx}] × MLP[{mlp_idx}] × Post[{post_idx}] (used {count} times)")
    
    print(f"\n🔧 EXPERT UTILIZATION")
    print("-" * 40)
    for expert_type, stats in analysis['expert_stats'].items():
        print(f"{expert_type.upper()} experts:")
        print(f"  Total: {stats['total_experts']}")
        print(f"  Avg usage: {stats['avg_usage']:.1f}")
        if stats['most_used']:
            print(f"  Most used: Expert[{stats['most_used'][0]}] ({stats['most_used'][1]} times)")
    
    # Test generation
    print(f"\n🎯 GENERATION TEST")
    print("-" * 40)
    prompt = torch.randint(0, 1000, (1, 5))
    generated = model.generate(prompt, max_length=10, temperature=0.8)
    print(f"Generated sequence: {generated[0].tolist()}")
    
    print(f"\n✅ Multiplicative GLBL demonstration complete!")
    print(f"🎉 Successfully created {analysis['total_pathways']:,} pathways from combinatorial experts!")
    
    return model, results, analysis

if __name__ == "__main__":
    # Run demonstration
    model, results, analysis = demonstrate_multiplicative_glbl()
    
    print(f"\n🎯 IMPLEMENTATION LOCATION: multiplicative_glbl_complete.py")
    print(f"🎯 KEY CLASSES:")
    print(f"   - MultiplicativeMLPLayer: Core Pre×MLP×Post implementation")
    print(f"   - MultiplicativeGLBLModel: Complete transformer model")
    print(f"   - PathwayDecomposer: Pathway index decomposition")
    print(f"   - MultiplicativeRouter: Pathway routing mechanism")