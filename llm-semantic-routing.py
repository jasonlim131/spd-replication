"""
Semantic/Conceptual Routing for LLMs with GLBL
This implementation routes based on semantic features rather than syntactic tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

@dataclass
class SemanticRoutingConfig:
    """Configuration for semantic pathway routing"""
    num_pathways: int = 16
    top_k_pathways: int = 4
    semantic_embedding_dim: int = 256  # For semantic feature extraction
    router_hidden_size: int = 256
    router_dropout: float = 0.1
    glbl_weight_start: float = 0.01
    glbl_weight_end: float = 0.1
    momentum: float = 0.9
    temperature: float = 1.0
    route_every_n_layers: int = 2
    
    # Semantic categories for interpretability
    semantic_categories: List[str] = None
    
    def __post_init__(self):
        if self.semantic_categories is None:
            self.semantic_categories = [
                "spatial_relations",      # above, below, inside, outside
                "temporal_concepts",      # past, future, now, always
                "emotional_states",       # happy, sad, angry, excited
                "causal_reasoning",       # because, therefore, causes
                "quantitative",          # numbers, amounts, comparisons
                "descriptive_qualities",  # colors, sizes, textures
                "actions_movements",      # run, jump, create, destroy
                "social_interactions",    # talk, meet, help, argue
                "logical_operations",     # if, then, not, and, or
                "abstract_concepts",      # justice, freedom, idea
                "physical_objects",       # table, car, tree, book
                "living_entities",        # person, animal, plant
                "questions_queries",      # who, what, where, why
                "instructions_commands",  # do, make, stop, go
                "comparisons_relations",  # more, less, similar, different
                "ownership_possession"    # have, own, belong, mine
            ]

class SemanticFeatureExtractor(nn.Module):
    """Extract semantic features from hidden states for routing"""
    def __init__(self, hidden_size: int, semantic_dim: int, num_categories: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.semantic_dim = semantic_dim
        self.num_categories = num_categories
        
        # Project hidden states to semantic space
        self.semantic_projection = nn.Sequential(
            nn.Linear(hidden_size, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim)
        )
        
        # Semantic category heads - each outputs a score for that semantic category
        self.category_heads = nn.ModuleList([
            nn.Linear(semantic_dim, 1) for _ in range(num_categories)
        ])
        
        # Context aggregation - looks at surrounding tokens
        self.context_attention = nn.MultiheadAttention(
            embed_dim=semantic_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Extract semantic features from hidden states
        
        Returns:
            semantic_features: (batch, seq_len, semantic_dim)
            category_scores: (batch, seq_len, num_categories)
        """
        # Project to semantic space
        semantic_features = self.semantic_projection(hidden_states)
        
        # Apply context attention to aggregate information from surrounding tokens
        if attention_mask is not None:
            # Convert attention mask to the right format for MHA
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, -1, hidden_states.size(1), -1)
            attn_mask = attn_mask.reshape(-1, hidden_states.size(1), hidden_states.size(1))
        else:
            attn_mask = None
            
        contextualized_features, _ = self.context_attention(
            semantic_features, semantic_features, semantic_features,
            attn_mask=attn_mask
        )
        
        # Get scores for each semantic category
        category_scores = []
        for head in self.category_heads:
            score = head(contextualized_features)  # (batch, seq_len, 1)
            category_scores.append(score)
        
        category_scores = torch.cat(category_scores, dim=-1)  # (batch, seq_len, num_categories)
        
        return contextualized_features, torch.sigmoid(category_scores)

class SemanticPathwayRouter(nn.Module):
    """Routes based on semantic features rather than raw tokens"""
    def __init__(self, hidden_size: int, config: SemanticRoutingConfig):
        super().__init__()
        self.config = config
        
        # Semantic feature extraction
        self.semantic_extractor = SemanticFeatureExtractor(
            hidden_size, 
            config.semantic_embedding_dim,
            len(config.semantic_categories)
        )
        
        # Route based on semantic features + categories
        input_dim = config.semantic_embedding_dim + len(config.semantic_categories)
        self.router = nn.Sequential(
            nn.Linear(input_dim, config.router_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.router_dropout),
            nn.Linear(config.router_hidden_size, config.num_pathways)
        )
        
        # Learnable pathway embeddings that can specialize on semantic concepts
        self.pathway_embeddings = nn.Parameter(
            torch.randn(config.num_pathways, config.semantic_embedding_dim)
        )
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Route based on semantic content
        
        Returns:
            pathway_scores: (batch, seq_len, num_pathways)
            semantic_info: Dict with semantic analysis
        """
        # Extract semantic features
        semantic_features, category_scores = self.semantic_extractor(hidden_states, attention_mask)
        
        # Combine semantic features with category scores
        routing_input = torch.cat([semantic_features, category_scores], dim=-1)
        
        # Get base routing scores
        pathway_scores = self.router(routing_input)
        
        # Add semantic similarity bonus
        # This encourages pathways to specialize on coherent semantic concepts
        semantic_similarity = torch.matmul(
            semantic_features, self.pathway_embeddings.T
        )  # (batch, seq_len, num_pathways)
        
        # Combine routing scores with semantic similarity
        pathway_scores = pathway_scores + 0.1 * semantic_similarity
        
        semantic_info = {
            'category_scores': category_scores,
            'semantic_features': semantic_features
        }
        
        return pathway_scores, semantic_info

class SemanticRoutedTransformerLayer(nn.Module):
    """Transformer layer with semantic pathway routing and GLBL"""
    def __init__(self, base_layer: nn.Module, config: SemanticRoutingConfig):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        
        # Get dimensions
        self.hidden_size = base_layer.mlp.c_fc.in_features
        self.intermediate_size = base_layer.mlp.c_fc.out_features
        
        # Semantic pathway router
        self.pathway_router = SemanticPathwayRouter(self.hidden_size, config)
        
        # Pathway decomposition
        self._create_pathway_decomposition()
        
        # GLBL statistics - CRUCIAL for preventing collapse!
        self.register_buffer('global_pathway_frequencies', 
                           torch.zeros(config.num_pathways, device=device))
        self.register_buffer('global_pathway_scores', 
                           torch.zeros(config.num_pathways, device=device))
        self.register_buffer('update_count', torch.zeros(1, device=device))
        
        # Semantic specialization tracking
        self.pathway_semantic_profiles = defaultdict(lambda: defaultdict(float))
        self.last_glbl_loss = None
        
    def _create_pathway_decomposition(self):
        """Create pathway decomposition for the MLP"""
        hidden_per_pathway = self.hidden_size // self.config.num_pathways
        intermediate_per_pathway = self.intermediate_size // self.config.num_pathways
        
        self.hidden_indices = []
        self.intermediate_indices = []
        
        for i in range(self.config.num_pathways):
            hidden_start = i * hidden_per_pathway
            hidden_end = (i + 1) * hidden_per_pathway if i < self.config.num_pathways - 1 else self.hidden_size
            
            intermediate_start = i * intermediate_per_pathway
            intermediate_end = (i + 1) * intermediate_per_pathway if i < self.config.num_pathways - 1 else self.intermediate_size
            
            self.hidden_indices.append(torch.arange(hidden_start, hidden_end))
            self.intermediate_indices.append(torch.arange(intermediate_start, intermediate_end))
    
    def compute_glbl_loss(self, pathway_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Global Load Balancing Loss - ESSENTIAL for preventing collapse!
        
        GLBL = N_pathways × Σ(frequency_i × score_i)
        """
        # Flatten batch and sequence dimensions
        pathway_scores_flat = pathway_scores.view(-1, self.config.num_pathways)
        
        # Current batch pathway probabilities
        pathway_probs = F.softmax(pathway_scores_flat, dim=-1)
        current_frequencies = pathway_probs.mean(dim=0)
        current_avg_scores = pathway_probs.mean(dim=0)
        
        # Update global statistics with momentum
        if self.training:
            momentum = self.config.momentum
            with torch.no_grad():
                self.global_pathway_frequencies.data = (
                    momentum * self.global_pathway_frequencies.data +
                    (1 - momentum) * current_frequencies.data
                )
                self.global_pathway_scores.data = (
                    momentum * self.global_pathway_scores.data +
                    (1 - momentum) * current_avg_scores.data
                )
                self.update_count += 1
        
        # Compute GLBL loss - this prevents collapse!
        glbl_loss = self.config.num_pathways * torch.sum(
            current_frequencies * current_avg_scores
        )
        
        return glbl_loss, current_frequencies, current_avg_scores
    
    def select_pathways(self, pathway_scores: torch.Tensor) -> torch.Tensor:
        """Select top-k pathways with load balancing in mind"""
        batch_size, seq_len, _ = pathway_scores.shape
        
        # Apply temperature
        pathway_probs = F.softmax(pathway_scores / self.config.temperature, dim=-1)
        
        if self.training:
            # Add noise during training to encourage exploration
            noise = torch.randn_like(pathway_scores) * 0.1
            noisy_scores = pathway_scores + noise
            
            # Apply GLBL-aware penalty to overused pathways
            if self.update_count > 0:
                usage_penalty = self.global_pathway_frequencies * 0.5
                noisy_scores = noisy_scores - usage_penalty.unsqueeze(0).unsqueeze(0)
            
            top_values, top_indices = torch.topk(
                F.softmax(noisy_scores, dim=-1), self.config.top_k_pathways, dim=-1
            )
        else:
            # Deterministic selection during inference
            top_values, top_indices = torch.topk(pathway_probs, self.config.top_k_pathways, dim=-1)
        
        # Create selection mask
        selection_mask = torch.zeros_like(pathway_probs)
        selection_mask.scatter_(-1, top_indices, top_values)
        
        # Normalize selected pathways
        pathway_weights = selection_mask / (selection_mask.sum(dim=-1, keepdim=True) + 1e-8)
        
        return pathway_weights
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                record_semantics: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with semantic routing and GLBL"""
        residual = hidden_states
        
        # Layer norm
        hidden_states = self.base_layer.ln_1(hidden_states)
        
        # Self-attention (no routing)
        attn_output = self.base_layer.attn(hidden_states)[0]
        hidden_states = residual + attn_output
        residual = hidden_states
        
        # Layer norm before MLP
        hidden_states = self.base_layer.ln_2(hidden_states)
        
        # Semantic routing
        pathway_scores, semantic_info = self.pathway_router(hidden_states, attention_mask)
        
        # GLBL loss computation - CRUCIAL!
        glbl_loss, frequencies, scores = self.compute_glbl_loss(pathway_scores)
        self.last_glbl_loss = glbl_loss
        
        # Select pathways with load balancing
        pathway_weights = self.select_pathways(pathway_scores)
        
        # Compute pathway outputs
        batch_size, seq_len, hidden_size = hidden_states.shape
        mlp_output = torch.zeros_like(hidden_states)
        
        # Track semantic specialization
        if record_semantics:
            self._record_semantic_specialization(
                pathway_weights, semantic_info['category_scores']
            )
        
        # Process through pathways
        for pathway_idx in range(self.config.num_pathways):
            weights = pathway_weights[:, :, pathway_idx].unsqueeze(-1)
            
            if weights.sum() < 1e-6:
                continue
            
            # Get pathway indices
            h_indices = self.hidden_indices[pathway_idx].to(hidden_states.device)
            i_indices = self.intermediate_indices[pathway_idx].to(hidden_states.device)
            
            # Pathway computation
            h_subset = hidden_states[:, :, h_indices]
            
            W1_subset = self.base_layer.mlp.c_fc.weight[i_indices][:, h_indices]
            b1_subset = self.base_layer.mlp.c_fc.bias[i_indices]
            intermediate = F.linear(h_subset, W1_subset, b1_subset)
            intermediate = self.base_layer.mlp.act(intermediate)
            
            W2_subset = self.base_layer.mlp.c_proj.weight[h_indices][:, i_indices]
            b2_subset = self.base_layer.mlp.c_proj.bias[h_indices]
            pathway_output = F.linear(intermediate, W2_subset, b2_subset)
            
            mlp_output[:, :, h_indices] += pathway_output * weights
        
        # Add residual
        output = residual + mlp_output
        
        # Return semantic info for analysis
        analysis_info = {
            'glbl_loss': glbl_loss,
            'pathway_frequencies': frequencies,
            'semantic_info': semantic_info,
            'pathway_weights': pathway_weights
        }
        
        return output, analysis_info
    
    def _record_semantic_specialization(self, pathway_weights: torch.Tensor, 
                                      category_scores: torch.Tensor):
        """Record which semantic categories activate which pathways"""
        batch_size, seq_len, num_pathways = pathway_weights.shape
        _, _, num_categories = category_scores.shape
        
        # For each pathway, accumulate semantic category activations
        for pathway_idx in range(num_pathways):
            pathway_activation = pathway_weights[:, :, pathway_idx]  # (batch, seq)
            
            # Weight category scores by pathway activation
            for cat_idx in range(num_categories):
                cat_score = category_scores[:, :, cat_idx]  # (batch, seq)
                weighted_score = (pathway_activation * cat_score).sum().item()
                
                category_name = self.config.semantic_categories[cat_idx]
                self.pathway_semantic_profiles[pathway_idx][category_name] += weighted_score

def measure_semantic_monosemanticity(model: nn.Module, config: SemanticRoutingConfig) -> Dict[str, Any]:
    """
    Measure semantic monosemanticity - how specialized each pathway is for semantic concepts
    """
    results = {}
    
    for layer_idx, layer in enumerate(model.routed_layers):
        layer_results = {}
        
        # Analyze each pathway's semantic profile
        for pathway_idx, semantic_profile in layer.pathway_semantic_profiles.items():
            if not semantic_profile:
                continue
            
            # Normalize scores
            total_activation = sum(semantic_profile.values())
            if total_activation == 0:
                continue
                
            normalized_profile = {
                cat: score / total_activation 
                for cat, score in semantic_profile.items()
            }
            
            # Calculate semantic purity (how focused on specific concepts)
            scores = list(normalized_profile.values())
            entropy = -sum(s * np.log(s + 1e-10) for s in scores if s > 0)
            max_entropy = np.log(len(scores))
            semantic_purity = 1 - (entropy / max_entropy if max_entropy > 0 else 0)
            
            # Find dominant semantic categories
            sorted_categories = sorted(
                normalized_profile.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            layer_results[f"pathway_{pathway_idx}"] = {
                'semantic_purity': semantic_purity,
                'dominant_categories': sorted_categories,
                'full_profile': normalized_profile,
                'total_activation': total_activation
            }
        
        results[f"layer_{layer_idx}"] = layer_results
    
    # Calculate overall metrics
    all_purities = []
    for layer_results in results.values():
        for pathway_info in layer_results.values():
            all_purities.append(pathway_info['semantic_purity'])
    
    return {
        'avg_semantic_purity': np.mean(all_purities) if all_purities else 0,
        'max_semantic_purity': np.max(all_purities) if all_purities else 0,
        'layer_results': results,
        'num_specialized_pathways': sum(
            1 for p in all_purities if p > 0.5
        )
    }

def demonstrate_semantic_routing():
    """Demonstrate semantic routing concepts"""
    print("🧠 SEMANTIC/CONCEPTUAL ROUTING DEMONSTRATION")
    print("=" * 60)
    
    print("\n1. SEMANTIC vs SYNTACTIC ROUTING")
    print("-" * 40)
    print("SYNTACTIC (Token-based):")
    print("  - Pathway 1: Activates for tokens ['the', 'a', 'an']")
    print("  - Pathway 2: Activates for tokens ['.', ',', '!']")
    print("  - Limited interpretability, missing deeper meaning")
    
    print("\nSEMANTIC (Concept-based):")
    print("  - Pathway 1: Spatial relations (above, below, inside)")
    print("  - Pathway 2: Emotional states (happy, sad, angry)")
    print("  - Pathway 3: Causal reasoning (because, therefore)")
    print("  - Much more interpretable and meaningful!")
    
    print("\n2. HOW SEMANTIC ROUTING WORKS")
    print("-" * 40)
    print("1. Extract semantic features from hidden states")
    print("2. Classify into semantic categories (16 categories)")
    print("3. Route based on semantic content, not tokens")
    print("4. Apply GLBL to ensure all pathways are used")
    print("5. Pathways specialize on coherent concepts")
    
    print("\n3. GLBL IN SEMANTIC ROUTING")
    print("-" * 40)
    print("GLBL is CRUCIAL because without it:")
    print("  ❌ All 'emotion' text → Pathway 2 only")
    print("  ❌ Other pathways never used (collapse)")
    print("  ❌ Lose capacity and interpretability")
    print("\nWith GLBL:")
    print("  ✅ Forces distribution across pathways")
    print("  ✅ Pathway 2: Happy emotions")
    print("  ✅ Pathway 5: Sad emotions")
    print("  ✅ Pathway 8: Complex emotional states")
    print("  ✅ Better capacity utilization")
    
    print("\n4. EXAMPLE PATHWAY SPECIALIZATIONS")
    print("-" * 40)
    example_profiles = {
        "Pathway 0": {
            "spatial_relations": 0.65,
            "physical_objects": 0.20,
            "descriptive_qualities": 0.15
        },
        "Pathway 3": {
            "causal_reasoning": 0.70,
            "logical_operations": 0.25,
            "temporal_concepts": 0.05
        },
        "Pathway 7": {
            "emotional_states": 0.80,
            "social_interactions": 0.15,
            "actions_movements": 0.05
        }
    }
    
    for pathway, profile in example_profiles.items():
        top_category = max(profile.items(), key=lambda x: x[1])
        purity = top_category[1]
        print(f"\n{pathway} (purity={purity:.2f}):")
        print(f"  Primary: {top_category[0]} ({top_category[1]:.2%})")
        for cat, score in sorted(profile.items(), key=lambda x: x[1], reverse=True)[1:]:
            print(f"  Secondary: {cat} ({score:.2%})")
    
    # Save demo results
    demo_results = {
        'semantic_routing_benefits': [
            'More interpretable pathway specializations',
            'Pathways learn conceptual relationships',
            'Better generalization to unseen examples',
            'Can control model behavior through pathway manipulation'
        ],
        'glbl_importance': [
            'Prevents pathway collapse',
            'Ensures balanced utilization',
            'Enables fine-grained specialization',
            'Maintains model capacity'
        ],
        'example_specializations': example_profiles
    }
    
    with open('semantic_routing_demo.json', 'w') as f:
        json.dump(demo_results, f, indent=2)
    
    print("\n✅ Demo results saved to 'semantic_routing_demo.json'")

if __name__ == "__main__":
    demonstrate_semantic_routing()