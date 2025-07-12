"""
Multiplicative GLBL Mechanism: Pre × MLP × Post Expert Combinations
Explains and implements the correct multiplicative expert architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class MultiplicativeGLBLMechanism:
    """
    MULTIPLICATIVE GLBL MECHANISM EXPLANATION
    
    The key insight: Instead of having separate pathways that are additive,
    we create MULTIPLICATIVE combinations of expert components:
    
    Pre-embedding Experts × MLP Experts × Post-embedding Experts
    
    This creates exponentially many pathways with only linear expert growth!
    """
    
    @staticmethod
    def explain_mechanism():
        print("🧠 MULTIPLICATIVE GLBL MECHANISM")
        print("=" * 60)
        print()
        
        print("🔢 MULTIPLICATIVE COMBINATIONS:")
        print("   Pre-experts (P) = 16   (different input transformations)")
        print("   MLP-experts (M) = 16   (different processing functions)")  
        print("   Post-experts (O) = 16  (different output transformations)")
        print("   Total pathways = P × M × O = 16 × 16 × 16 = 4,096")
        print()
        
        print("💡 SPARSE COMPUTATION:")
        print("   Route to top-K = 8 pathways (out of 4,096)")
        print("   Computational cost = 8 pathway computations")
        print("   Memory cost = Store 48 experts (16+16+16)")
        print("   Efficiency = 8/4,096 = 0.2% active pathways")
        print()
        
        print("🎯 PATHWAY DECOMPOSITION:")
        print("   Pathway 2,847 → Pre[7] × MLP[3] × Post[15]")
        print("   Pathway 156   → Pre[0] × MLP[9] × Post[12]") 
        print("   Each pathway = unique combination of 3 experts")
        print()
        
        print("🚀 LAYERWISE SCALING:")
        print("   Layer 1: 16³ = 4,096 pathways")
        print("   Layer 3: 20³ = 8,000 pathways") 
        print("   Layer 5: 24³ = 13,824 pathways")
        print("   Layer 7: 32³ = 32,768 pathways")
        print("   Total: 58,688 pathways across 4 layers!")
        print("=" * 60)

class PathwayDecomposer:
    """Handles decomposition of pathway indices into expert combinations"""
    
    def __init__(self, num_pre: int, num_mlp: int, num_post: int):
        self.num_pre = num_pre
        self.num_mlp = num_mlp  
        self.num_post = num_post
        self.total_pathways = num_pre * num_mlp * num_post
        
    def decompose_pathway(self, pathway_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose pathway index into (pre_idx, mlp_idx, post_idx)
        
        Example: pathway_idx = 2,847 with 16×16×16 = 4,096 total
        → pre_idx = 2847 // (16*16) = 2847 // 256 = 11
        → remaining = 2847 % 256 = 31  
        → mlp_idx = 31 // 16 = 1
        → post_idx = 31 % 16 = 15
        → Pathway = Pre[11] × MLP[1] × Post[15]
        """
        pre_idx = pathway_idx // (self.num_mlp * self.num_post)
        remaining = pathway_idx % (self.num_mlp * self.num_post)
        mlp_idx = remaining // self.num_post
        post_idx = remaining % self.num_post
        
        return pre_idx, mlp_idx, post_idx
    
    def compose_pathway(self, pre_idx: int, mlp_idx: int, post_idx: int) -> int:
        """Compose expert indices back into pathway index"""
        return pre_idx * (self.num_mlp * self.num_post) + mlp_idx * self.num_post + post_idx

class MultiplicativeExpertRouter(nn.Module):
    """Routes input to multiplicative expert combinations"""
    
    def __init__(self, d_model: int, total_pathways: int, hidden_size: int = 256):
        super().__init__()
        self.d_model = d_model
        self.total_pathways = total_pathways
        
        print(f"🧭 Router: {d_model} dims → {total_pathways:,} pathway combinations")
        
        # Multi-layer router for complex pathway selection
        self.router = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, total_pathways)
        )
        
        # Temperature parameter for pathway selection sharpness
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Route input to pathway combinations
        
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            pathway_scores: [batch_size, seq_len, total_pathways]
        """
        return self.router(x)

class MultiplicativeMLPLayer(nn.Module):
    """
    MULTIPLICATIVE MLP LAYER
    
    This is the core of the multiplicative mechanism:
    1. Pre-embedding experts transform input in different ways
    2. MLP experts process transformed input through different functions
    3. Post-embedding experts transform MLP output in different ways
    4. Route to top-K combinations of (pre, mlp, post) experts
    """
    
    def __init__(self, d_model: int, num_pre: int = 16, num_mlp: int = 16, 
                 num_post: int = 16, top_k: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_pre = num_pre
        self.num_mlp = num_mlp
        self.num_post = num_post
        self.top_k = top_k
        
        # Calculate total pathway combinations
        self.total_pathways = num_pre * num_mlp * num_post
        
        print(f"🏗️  Multiplicative MLP Layer:")
        print(f"     Pre-experts: {num_pre}")
        print(f"     MLP-experts: {num_mlp}")
        print(f"     Post-experts: {num_post}")
        print(f"     Total combinations: {self.total_pathways:,}")
        print(f"     Active per forward: {top_k}")
        print(f"     Sparsity: {100 * (1 - top_k/self.total_pathways):.2f}%")
        print()
        
        # Pre-embedding experts: Different input transformations
        self.pre_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU() if i % 3 == 0 else (nn.ReLU() if i % 3 == 1 else nn.Tanh())
            ) for i in range(num_pre)
        ])
        
        # MLP experts: Different processing functions
        self.mlp_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * (2 + i // 4)),  # Varying widths
                nn.GELU() if i % 2 == 0 else nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * (2 + i // 4), d_model)
            ) for i in range(num_mlp)
        ])
        
        # Post-embedding experts: Different output transformations
        self.post_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model) if i % 2 == 0 else nn.Identity(),
                nn.Dropout(0.05 + 0.1 * (i / num_post))  # Varying dropout
            ) for i in range(num_post)
        ])
        
        # Pathway router
        self.router = MultiplicativeExpertRouter(d_model, self.total_pathways)
        
        # Pathway decomposer for index management
        self.decomposer = PathwayDecomposer(num_pre, num_mlp, num_post)
        
        # Global Load Balancing tracking
        self.register_buffer('global_pathway_frequencies', 
                           torch.zeros(self.total_pathways))
        self.register_buffer('pathway_usage_count', torch.zeros(1))
        
    def compute_glbl_loss(self, pathway_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute Global Load Balancing Loss for multiplicative pathways
        
        GLBL Loss ensures pathways are used evenly to prevent expert collapse
        """
        batch_size, seq_len, num_pathways = pathway_scores.shape
        
        # Current batch pathway probabilities
        pathway_probs = F.softmax(pathway_scores, dim=-1)
        current_frequencies = pathway_probs.mean(dim=(0, 1))
        
        # Update global statistics
        if self.training:
            momentum = 0.9
            with torch.no_grad():
                self.global_pathway_frequencies.data = (
                    momentum * self.global_pathway_frequencies.data +
                    (1 - momentum) * current_frequencies.data
                )
                self.pathway_usage_count += 1
        
        # GLBL loss: minimize concentration of pathway usage
        # Higher loss when pathways are unevenly used
        ideal_frequency = 1.0 / num_pathways
        frequency_variance = torch.var(current_frequencies)
        
        glbl_loss = num_pathways * frequency_variance
        
        return glbl_loss
    
    def forward(self, x: torch.Tensor, 
                record_activations: bool = False) -> Dict[str, torch.Tensor]:
        """
        MULTIPLICATIVE FORWARD PASS
        
        Step 1: Route to top-K pathway combinations
        Step 2: For each active pathway, decompose into (pre, mlp, post) 
        Step 3: Compute Pre[i] → MLP[j] → Post[k] for active pathways only
        Step 4: Weighted sum of pathway outputs
        """
        
        batch_size, seq_len, d_model = x.shape
        
        print(f"🚀 Forward pass: {batch_size} × {seq_len} × {d_model}")
        
        # Step 1: Route to pathway combinations
        pathway_scores = self.router(x)  # [batch, seq, total_pathways]
        
        # Compute GLBL loss
        glbl_loss = self.compute_glbl_loss(pathway_scores)
        
        # Step 2: Select top-K pathways
        pathway_probs = F.softmax(pathway_scores / self.router.temperature, dim=-1)
        top_values, top_indices = torch.topk(pathway_probs, self.top_k, dim=-1)
        
        print(f"   Selected {self.top_k} pathways out of {self.total_pathways:,}")
        
        # Create sparse pathway weights
        pathway_weights = torch.zeros_like(pathway_probs)
        pathway_weights.scatter_(-1, top_indices, top_values)
        pathway_weights = pathway_weights / (pathway_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Step 3: Compute active pathways only
        output = torch.zeros_like(x)
        active_pathways = []
        
        for k in range(self.top_k):
            # Get pathway indices for this k
            pathway_indices = top_indices[:, :, k]  # [batch, seq]
            weights = top_values[:, :, k].unsqueeze(-1)  # [batch, seq, 1]
            
            # Process each unique pathway index in this batch
            unique_pathways = torch.unique(pathway_indices)
            
            for pathway_idx in unique_pathways:
                pathway_idx_val = pathway_idx.item()
                
                # Step 3a: Decompose pathway into expert indices
                pre_idx, mlp_idx, post_idx = self.decomposer.decompose_pathway(pathway_idx_val)
                
                print(f"   Pathway {pathway_idx_val}: Pre[{pre_idx}] × MLP[{mlp_idx}] × Post[{post_idx}]")
                
                # Create mask for this pathway
                pathway_mask = (pathway_indices == pathway_idx).float().unsqueeze(-1)
                pathway_weight = weights * pathway_mask
                
                # Skip if this pathway is not used in current positions
                if pathway_weight.sum() < 1e-8:
                    continue
                
                # Step 3b: Compute pathway: Pre → MLP → Post
                x_pre = self.pre_experts[pre_idx](x)      # Pre-embedding transformation
                x_mlp = self.mlp_experts[mlp_idx](x_pre)  # MLP processing  
                x_post = self.post_experts[post_idx](x_mlp)  # Post-embedding transformation
                
                # Step 3c: Weight and accumulate
                pathway_output = x_post * pathway_weight
                output += pathway_output
                
                # Record for analysis
                if record_activations:
                    active_pathways.append({
                        'pathway_idx': pathway_idx_val,
                        'pre_idx': pre_idx,
                        'mlp_idx': mlp_idx,
                        'post_idx': post_idx,
                        'weight': pathway_weight.sum().item(),
                        'output_norm': x_post.norm().item()
                    })
        
        print(f"   ✅ Computed {len(active_pathways)} unique pathway combinations")
        print()
        
        return {
            'output': output,
            'glbl_loss': glbl_loss,
            'pathway_weights': pathway_weights,
            'active_pathways': active_pathways,
            'total_pathways': self.total_pathways,
            'sparsity': 1.0 - (len(active_pathways) / self.total_pathways)
        }

class MultiplicativeGLBLTransformerLayer(nn.Module):
    """Transformer layer with multiplicative GLBL MLP"""
    
    def __init__(self, d_model: int, layer_idx: int, 
                 num_pre: int = 16, num_mlp: int = 16, num_post: int = 16):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Standard attention (could also be multiplicative)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # Multiplicative GLBL MLP
        self.mlp = MultiplicativeMLPLayer(d_model, num_pre, num_mlp, num_post)
        
        print(f"🏗️  Layer {layer_idx}: {self.mlp.total_pathways:,} MLP pathway combinations")
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                record_activations: bool = False) -> Dict[str, torch.Tensor]:
        
        print(f"\n🔄 Layer {self.layer_idx} Forward Pass")
        print("-" * 40)
        
        # Attention with residual
        normed_x = self.ln1(x) 
        attn_output, _ = self.attention(normed_x, normed_x, normed_x,
                                      key_padding_mask=attention_mask)
        x = x + attn_output
        
        # Multiplicative GLBL MLP with residual
        normed_x = self.ln2(x)
        mlp_result = self.mlp(normed_x, record_activations)
        x = x + mlp_result['output']
        
        return {
            'output': x,
            'glbl_loss': mlp_result['glbl_loss'],
            'active_pathways': mlp_result['active_pathways'],
            'total_pathways': mlp_result['total_pathways'],
            'sparsity': mlp_result['sparsity']
        }

class MultiplicativeGLBLModel(nn.Module):
    """Complete model with multiplicative GLBL layers"""
    
    def __init__(self, vocab_size: int = 4096, d_model: int = 64, n_layers: int = 8,
                 glbl_layers: List[int] = [1, 3, 5, 7]):
        super().__init__()
        self.d_model = d_model
        self.glbl_layers = glbl_layers
        
        print("\n🏗️  MULTIPLICATIVE GLBL MODEL ARCHITECTURE")
        print("=" * 60)
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        total_pathways = 0
        
        for i in range(n_layers):
            if i in glbl_layers:
                # Multiplicative GLBL layer with increasing expert counts
                base_experts = 12 + (i // 2) * 4  # 12, 16, 20, 24 experts per layer
                layer = MultiplicativeGLBLTransformerLayer(
                    d_model, i, base_experts, base_experts, base_experts
                )
                total_pathways += layer.mlp.total_pathways
            else:
                # Standard layer  
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=4, dim_feedforward=d_model*4,
                    dropout=0.1, batch_first=True
                )
                print(f"🏗️  Layer {i}: Standard (no multiplicative pathways)")
            
            self.layers.append(layer)
        
        # Output layers
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        print(f"\n📊 TOTAL MULTIPLICATIVE PATHWAYS: {total_pathways:,}")
        print(f"📊 GLBL LAYERS: {len(glbl_layers)}")
        print(f"📊 SPARSITY: ~99.8% (only top-8 active per layer)")
        print("=" * 60)
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                record_activations: bool = False) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = input_ids.shape
        
        print(f"\n🚀 MODEL FORWARD PASS")
        print(f"Input shape: {batch_size} × {seq_len}")
        print("=" * 40)
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Pass through layers
        total_glbl_loss = torch.tensor(0.0, device=x.device)
        all_active_pathways = []
        total_pathways = 0
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MultiplicativeGLBLTransformerLayer):
                layer_result = layer(x, attention_mask, record_activations)
                x = layer_result['output']
                total_glbl_loss += layer_result['glbl_loss']
                all_active_pathways.extend(layer_result['active_pathways'])
                total_pathways += layer_result['total_pathways']
            else:
                # Standard layer
                x = layer(x)
        
        # Final processing
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        print(f"\n📊 FORWARD PASS SUMMARY:")
        print(f"   Total pathways available: {total_pathways:,}")
        print(f"   Active pathways computed: {len(all_active_pathways)}")
        print(f"   Sparsity achieved: {100 * (1 - len(all_active_pathways)/total_pathways):.2f}%")
        print(f"   GLBL loss: {total_glbl_loss.item():.6f}")
        
        return {
            'logits': logits,
            'total_glbl_loss': total_glbl_loss,
            'active_pathways': all_active_pathways,
            'total_pathways': total_pathways,
            'sparsity': 1.0 - (len(all_active_pathways) / total_pathways)
        }

def demonstrate_multiplicative_mechanism():
    """Demonstrate the multiplicative GLBL mechanism"""
    
    # Explain the mechanism
    MultiplicativeGLBLMechanism.explain_mechanism()
    
    # Create a small model for demonstration
    model = MultiplicativeGLBLModel(
        vocab_size=1000, d_model=64, n_layers=4, glbl_layers=[1, 3]
    )
    
    # Create sample input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print("\n🧪 DEMONSTRATION")
    print("=" * 60)
    
    # Forward pass
    with torch.no_grad():
        result = model(input_ids, record_activations=True)
    
    print(f"\n🎯 RESULTS:")
    print(f"   Model output shape: {result['logits'].shape}")
    print(f"   Total pathways: {result['total_pathways']:,}")
    print(f"   Active pathways: {len(result['active_pathways'])}")
    print(f"   Computational sparsity: {result['sparsity']:.4f}")
    print(f"   GLBL loss: {result['total_glbl_loss'].item():.6f}")
    
    # Show some active pathways
    print(f"\n🔍 SAMPLE ACTIVE PATHWAYS:")
    for i, pathway in enumerate(result['active_pathways'][:5]):
        print(f"   Pathway {pathway['pathway_idx']}: "
              f"Pre[{pathway['pre_idx']}] × MLP[{pathway['mlp_idx']}] × Post[{pathway['post_idx']}]")
    
    print("\n✅ Multiplicative GLBL mechanism demonstration complete!")

if __name__ == "__main__":
    demonstrate_multiplicative_mechanism()