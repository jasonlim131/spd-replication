import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy.stats import entropy
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🚀 Using device: {device}")

@dataclass
class GLBLTransformerConfig:
    """Configuration for GLBL Transformer decomposition"""
    model_name: str = "roneneldan/TinyStories-1M"
    max_seq_len: int = 512
    vocab_size: int = 4096
    d_model: int = 64
    n_layers: int = 8
    n_heads: int = 2
    d_ff: int = 256
    
    # GLBL Pathway Configuration
    num_semantic_pathways: int = 32
    num_attention_pathways: int = 16
    num_mlp_pathways: int = 16
    glbl_layers: List[int] = None  # Every other layer by default
    
    # Training Configuration
    batch_size: int = 8
    learning_rate: float = 1e-4
    glbl_weight_start: float = 0.01
    glbl_weight_end: float = 0.1
    top_k_pathways: int = 8
    temperature: float = 1.0
    momentum: float = 0.9
    
    # Analysis Configuration
    semantic_categories: List[str] = None
    pathway_analysis_samples: int = 1000
    interpretability_threshold: float = 0.3
    
    def __post_init__(self):
        if self.glbl_layers is None:
            self.glbl_layers = list(range(1, self.n_layers, 2))  # Every other layer
        
        if self.semantic_categories is None:
            self.semantic_categories = [
                "Characters", "Actions", "Objects", "Locations", "Emotions",
                "Time", "Dialogue", "Descriptions", "Narrative", "Relationships"
            ]

class SemanticPathwayRouter(nn.Module):
    """Routes tokens to semantic pathways based on contextual understanding"""
    
    def __init__(self, d_model: int, num_pathways: int, hidden_size: int = 128):
        super().__init__()
        self.d_model = d_model
        self.num_pathways = num_pathways
        
        # Multi-layer router for complex semantic routing
        self.router = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_pathways)
        )
        
        # Contextual attention for sequence-aware routing
        self.context_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=2, 
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Route tokens to pathways based on semantic content
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask
            
        Returns:
            pathway_scores: [batch_size, seq_len, num_pathways]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply contextual attention for sequence understanding
        attended_x, _ = self.context_attention(x, x, x, key_padding_mask=attention_mask)
        
        # Route to pathways
        pathway_scores = self.router(attended_x)
        
        return pathway_scores

class GLBLAttentionPathway(nn.Module):
    """GLBL-decomposed attention mechanism with pathway specialization"""
    
    def __init__(self, config: GLBLTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.num_pathways = config.num_attention_pathways
        
        # Pathway router
        self.pathway_router = SemanticPathwayRouter(
            config.d_model, self.num_pathways
        )
        
        # Pathway-specific attention heads
        self.pathway_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.n_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(self.num_pathways)
        ])
        
        # Global Load Balancing tracking
        self.register_buffer('global_pathway_frequencies', 
                           torch.zeros(self.num_pathways, device=device))
        self.register_buffer('global_pathway_scores', 
                           torch.zeros(self.num_pathways, device=device))
        self.register_buffer('update_count', torch.zeros(1, device=device))
        
        # Pathway activation tracking
        self.pathway_activations = defaultdict(list)
        self.semantic_specializations = defaultdict(list)
        
    def compute_glbl_loss(self, pathway_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Global Load Balancing Loss for attention pathways"""
        batch_size, seq_len, num_pathways = pathway_scores.shape
        
        # Current batch pathway probabilities
        pathway_probs = F.softmax(pathway_scores, dim=-1)
        current_frequencies = pathway_probs.mean(dim=(0, 1))
        current_avg_scores = pathway_probs.mean(dim=(0, 1))
        
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
        
        # Compute GLBL loss
        glbl_loss = self.num_pathways * torch.sum(
            current_frequencies * current_avg_scores
        )
        
        return glbl_loss, current_frequencies, current_avg_scores
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                record_activations: bool = False, input_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with pathway-based attention"""
        batch_size, seq_len, d_model = x.shape
        
        # Route to pathways
        pathway_scores = self.pathway_router(x, attention_mask)
        
        # Compute GLBL loss
        glbl_loss, frequencies, scores = self.compute_glbl_loss(pathway_scores)
        
        # Select top-k pathways
        pathway_probs = F.softmax(pathway_scores / self.config.temperature, dim=-1)
        top_values, top_indices = torch.topk(pathway_probs, self.config.top_k_pathways, dim=-1)
        
        # Create pathway selection mask
        pathway_weights = torch.zeros_like(pathway_probs)
        pathway_weights.scatter_(-1, top_indices, top_values)
        pathway_weights = pathway_weights / (pathway_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply pathway-specific attention
        output = torch.zeros_like(x)
        
        for pathway_idx in range(self.num_pathways):
            # Get pathway weights for this pathway
            weights = pathway_weights[:, :, pathway_idx].unsqueeze(-1)  # [batch, seq, 1]
            
            # Skip if pathway not used
            if weights.sum() < 1e-6:
                continue
            
            # Apply pathway-specific attention
            attended_output, attention_weights = self.pathway_attentions[pathway_idx](
                x, x, x, key_padding_mask=attention_mask
            )
            
            # Weight and accumulate output
            output += attended_output * weights
            
            # Record activations for analysis
            if record_activations and input_ids is not None:
                self._record_pathway_activations(
                    pathway_idx, weights, attended_output, attention_weights, 
                    input_ids, batch_size, seq_len
                )
        
        return {
            'output': output,
            'glbl_loss': glbl_loss,
            'pathway_weights': pathway_weights,
            'pathway_frequencies': frequencies,
            'pathway_scores': scores
        }
    
    def _record_pathway_activations(self, pathway_idx: int, weights: torch.Tensor, 
                                  output: torch.Tensor, attention_weights: torch.Tensor,
                                  input_ids: torch.Tensor, batch_size: int, seq_len: int):
        """Record pathway activations for semantic analysis"""
        pathway_name = f"Layer{self.layer_idx}_Attention_Pathway{pathway_idx}"
        
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                if weights[batch_idx, seq_idx, 0] > self.config.interpretability_threshold:
                    self.pathway_activations[pathway_name].append({
                        'layer_idx': self.layer_idx,
                        'pathway_idx': pathway_idx,
                        'pathway_type': 'attention',
                        'token_id': input_ids[batch_idx, seq_idx].item(),
                        'position': seq_idx,
                        'pathway_weight': weights[batch_idx, seq_idx, 0].item(),
                        'activation_norm': output[batch_idx, seq_idx].norm().item(),
                        'attention_entropy': entropy(attention_weights[batch_idx, :, seq_idx].cpu().numpy() + 1e-8)
                    })

class GLBLMLPPathway(nn.Module):
    """GLBL-decomposed MLP with pathway specialization"""
    
    def __init__(self, config: GLBLTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.num_pathways = config.num_mlp_pathways
        
        # Pathway router
        self.pathway_router = SemanticPathwayRouter(
            config.d_model, self.num_pathways
        )
        
        # Pathway-specific MLPs
        self.pathway_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.d_ff, config.d_model),
                nn.Dropout(0.1)
            ) for _ in range(self.num_pathways)
        ])
        
        # Global Load Balancing tracking
        self.register_buffer('global_pathway_frequencies', 
                           torch.zeros(self.num_pathways, device=device))
        self.register_buffer('global_pathway_scores', 
                           torch.zeros(self.num_pathways, device=device))
        self.register_buffer('update_count', torch.zeros(1, device=device))
        
        # Pathway activation tracking
        self.pathway_activations = defaultdict(list)
        self.semantic_specializations = defaultdict(list)
    
    def compute_glbl_loss(self, pathway_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Global Load Balancing Loss for MLP pathways"""
        batch_size, seq_len, num_pathways = pathway_scores.shape
        
        # Current batch pathway probabilities
        pathway_probs = F.softmax(pathway_scores, dim=-1)
        current_frequencies = pathway_probs.mean(dim=(0, 1))
        current_avg_scores = pathway_probs.mean(dim=(0, 1))
        
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
        
        # Compute GLBL loss
        glbl_loss = self.num_pathways * torch.sum(
            current_frequencies * current_avg_scores
        )
        
        return glbl_loss, current_frequencies, current_avg_scores
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                record_activations: bool = False, input_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with pathway-based MLP"""
        batch_size, seq_len, d_model = x.shape
        
        # Route to pathways
        pathway_scores = self.pathway_router(x, attention_mask)
        
        # Compute GLBL loss
        glbl_loss, frequencies, scores = self.compute_glbl_loss(pathway_scores)
        
        # Select top-k pathways
        pathway_probs = F.softmax(pathway_scores / self.config.temperature, dim=-1)
        top_values, top_indices = torch.topk(pathway_probs, self.config.top_k_pathways, dim=-1)
        
        # Create pathway selection mask
        pathway_weights = torch.zeros_like(pathway_probs)
        pathway_weights.scatter_(-1, top_indices, top_values)
        pathway_weights = pathway_weights / (pathway_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply pathway-specific MLPs
        output = torch.zeros_like(x)
        
        for pathway_idx in range(self.num_pathways):
            # Get pathway weights for this pathway
            weights = pathway_weights[:, :, pathway_idx].unsqueeze(-1)  # [batch, seq, 1]
            
            # Skip if pathway not used
            if weights.sum() < 1e-6:
                continue
            
            # Apply pathway-specific MLP
            mlp_output = self.pathway_mlps[pathway_idx](x)
            
            # Weight and accumulate output
            output += mlp_output * weights
            
            # Record activations for analysis
            if record_activations and input_ids is not None:
                self._record_pathway_activations(
                    pathway_idx, weights, mlp_output, input_ids, batch_size, seq_len
                )
        
        return {
            'output': output,
            'glbl_loss': glbl_loss,
            'pathway_weights': pathway_weights,
            'pathway_frequencies': frequencies,
            'pathway_scores': scores
        }
    
    def _record_pathway_activations(self, pathway_idx: int, weights: torch.Tensor, 
                                  output: torch.Tensor, input_ids: torch.Tensor, 
                                  batch_size: int, seq_len: int):
        """Record pathway activations for semantic analysis"""
        pathway_name = f"Layer{self.layer_idx}_MLP_Pathway{pathway_idx}"
        
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                if weights[batch_idx, seq_idx, 0] > self.config.interpretability_threshold:
                    self.pathway_activations[pathway_name].append({
                        'layer_idx': self.layer_idx,
                        'pathway_idx': pathway_idx,
                        'pathway_type': 'mlp',
                        'token_id': input_ids[batch_idx, seq_idx].item(),
                        'position': seq_idx,
                        'pathway_weight': weights[batch_idx, seq_idx, 0].item(),
                        'activation_norm': output[batch_idx, seq_idx].norm().item(),
                        'hidden_activation': output[batch_idx, seq_idx].mean().item()
                    })

class GLBLTransformerLayer(nn.Module):
    """Transformer layer with GLBL pathway decomposition"""
    
    def __init__(self, config: GLBLTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.use_glbl = layer_idx in config.glbl_layers
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        if self.use_glbl:
            # GLBL pathway components
            self.attention = GLBLAttentionPathway(config, layer_idx)
            self.mlp = GLBLMLPPathway(config, layer_idx)
        else:
            # Standard transformer components
            self.attention = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.n_heads,
                dropout=0.1,
                batch_first=True
            )
            self.mlp = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.d_ff, config.d_model),
                nn.Dropout(0.1)
            )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                record_activations: bool = False, input_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through transformer layer"""
        
        if self.use_glbl:
            # GLBL pathway processing
            
            # Attention with residual connection
            normed_x = self.ln1(x)
            attn_result = self.attention(normed_x, attention_mask, record_activations, input_ids)
            x = x + attn_result['output']
            
            # MLP with residual connection
            normed_x = self.ln2(x)
            mlp_result = self.mlp(normed_x, attention_mask, record_activations, input_ids)
            x = x + mlp_result['output']
            
            # Combine GLBL losses
            total_glbl_loss = attn_result['glbl_loss'] + mlp_result['glbl_loss']
            
            return {
                'output': x,
                'glbl_loss': total_glbl_loss,
                'attention_pathway_weights': attn_result['pathway_weights'],
                'mlp_pathway_weights': mlp_result['pathway_weights'],
                'attention_frequencies': attn_result['pathway_frequencies'],
                'mlp_frequencies': mlp_result['pathway_frequencies']
            }
        else:
            # Standard transformer processing
            
            # Attention with residual connection
            normed_x = self.ln1(x)
            attn_output, _ = self.attention(normed_x, normed_x, normed_x, 
                                          key_padding_mask=attention_mask)
            x = x + attn_output
            
            # MLP with residual connection
            normed_x = self.ln2(x)
            mlp_output = self.mlp(normed_x)
            x = x + mlp_output
            
            return {
                'output': x,
                'glbl_loss': torch.tensor(0.0, device=x.device)
            }

class GLBLTinyStoriesModel(nn.Module):
    """TinyStories transformer with GLBL pathway decomposition"""
    
    def __init__(self, config: GLBLTransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GLBLTransformerLayer(config, i) for i in range(config.n_layers)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Training statistics
        self.glbl_stats = defaultdict(list)
        self.pathway_activations = defaultdict(list)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                record_activations: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass through GLBL transformer"""
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).bool()
        
        # Token and position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Track GLBL losses across layers
        total_glbl_loss = torch.tensor(0.0, device=x.device)
        layer_results = []
        
        # Pass through transformer layers
        for layer in self.layers:
            layer_result = layer(x, attention_mask, record_activations, input_ids)
            x = layer_result['output']
            total_glbl_loss += layer_result['glbl_loss']
            layer_results.append(layer_result)
        
        # Final layer norm and language modeling head
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Collect pathway activations from layers
        if record_activations:
            for layer_idx, layer in enumerate(self.layers):
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'pathway_activations'):
                    for pathway_name, activations in layer.attention.pathway_activations.items():
                        self.pathway_activations[pathway_name].extend(activations)
                
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'pathway_activations'):
                    for pathway_name, activations in layer.mlp.pathway_activations.items():
                        self.pathway_activations[pathway_name].extend(activations)
        
        return {
            'logits': logits,
            'total_glbl_loss': total_glbl_loss,
            'layer_results': layer_results,
            'hidden_states': x
        }
    
    def generate_text(self, prompt: str, tokenizer, max_length: int = 100, temperature: float = 0.8) -> str:
        """Generate text using the GLBL model"""
        self.eval()
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self(generated_ids)
                logits = outputs['logits']
                
                # Sample next token
                next_token_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                # Check for end token or max length
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                # Truncate if too long
                if generated_ids.shape[1] > self.config.max_seq_len:
                    generated_ids = generated_ids[:, -self.config.max_seq_len:]
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

class SemanticAnalyzer:
    """Analyzes semantic specialization of pathways"""
    
    def __init__(self, config: GLBLTransformerConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        # Define semantic categories and their keywords
        self.semantic_keywords = {
            "Characters": ["boy", "girl", "man", "woman", "mother", "father", "friend", "child", "baby", "person"],
            "Actions": ["run", "walk", "jump", "play", "eat", "sleep", "sing", "dance", "laugh", "cry"],
            "Objects": ["ball", "toy", "book", "car", "house", "tree", "flower", "food", "water", "sun"],
            "Locations": ["home", "school", "park", "forest", "garden", "room", "kitchen", "bedroom", "outside", "inside"],
            "Emotions": ["happy", "sad", "angry", "excited", "scared", "love", "like", "want", "feel", "smile"],
            "Time": ["day", "night", "morning", "evening", "today", "tomorrow", "yesterday", "time", "hour", "minute"],
            "Dialogue": ["said", "asked", "told", "called", "shouted", "whispered", "talked", "spoke", "voice", "words"],
            "Descriptions": ["big", "small", "beautiful", "pretty", "nice", "good", "bad", "fast", "slow", "loud"],
            "Narrative": ["once", "then", "after", "before", "when", "where", "because", "so", "but", "and"],
            "Relationships": ["family", "together", "with", "friend", "love", "help", "share", "give", "take", "hug"]
        }
        
        # Convert keywords to token IDs
        self.semantic_token_ids = {}
        for category, keywords in self.semantic_keywords.items():
            token_ids = []
            for keyword in keywords:
                # Try different tokenization approaches
                try:
                    tokens = tokenizer.encode(keyword, add_special_tokens=False)
                    token_ids.extend(tokens)
                    tokens = tokenizer.encode(" " + keyword, add_special_tokens=False)
                    token_ids.extend(tokens)
                except:
                    continue
            self.semantic_token_ids[category] = list(set(token_ids))
    
    def analyze_pathway_specialization(self, pathway_activations: Dict[str, List[Dict]], 
                                     min_activations: int = 10) -> Dict[str, Dict]:
        """Analyze semantic specialization of pathways"""
        
        specializations = {}
        
        for pathway_name, activations in pathway_activations.items():
            if len(activations) < min_activations:
                continue
            
            # Collect tokens and their weights
            token_weights = defaultdict(list)
            position_weights = defaultdict(list)
            layer_info = None
            pathway_type = None
            
            for activation in activations:
                token_id = activation['token_id']
                weight = activation['pathway_weight']
                position = activation['position']
                
                token_weights[token_id].append(weight)
                position_weights[position].append(weight)
                
                if layer_info is None:
                    layer_info = activation['layer_idx']
                    pathway_type = activation['pathway_type']
            
            # Calculate semantic category preferences
            category_scores = {}
            for category, token_ids in self.semantic_token_ids.items():
                category_weight = 0
                category_count = 0
                
                for token_id in token_ids:
                    if token_id in token_weights:
                        category_weight += sum(token_weights[token_id])
                        category_count += len(token_weights[token_id])
                
                if category_count > 0:
                    category_scores[category] = category_weight / category_count
                else:
                    category_scores[category] = 0.0
            
            # Find dominant category
            if category_scores:
                dominant_category = max(category_scores.keys(), key=lambda k: category_scores[k])
                specialization_strength = category_scores[dominant_category]
            else:
                dominant_category = "Unknown"
                specialization_strength = 0.0
            
            # Calculate other metrics
            total_tokens = len(set(token_weights.keys()))
            avg_weight = np.mean([w for weights in token_weights.values() for w in weights])
            position_entropy = entropy(list(Counter(position_weights.keys()).values()) + [1e-8])
            
            # Token diversity
            token_frequencies = [len(weights) for weights in token_weights.values()]
            token_entropy = entropy(token_frequencies + [1e-8])
            
            specializations[pathway_name] = {
                'layer_idx': layer_info,
                'pathway_type': pathway_type,
                'dominant_category': dominant_category,
                'specialization_strength': specialization_strength,
                'category_scores': category_scores,
                'total_activations': len(activations),
                'unique_tokens': total_tokens,
                'avg_pathway_weight': avg_weight,
                'position_entropy': position_entropy,
                'token_entropy': token_entropy,
                'interpretability_score': specialization_strength * (1 - token_entropy / np.log(total_tokens + 1))
            }
        
        return specializations

def create_tinystories_dataset(config: GLBLTransformerConfig, tokenizer, split: str = "train", 
                              num_samples: int = 1000):
    """Create TinyStories dataset for training/evaluation"""
    
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    
    # Take first num_samples
    stories = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        stories.append(example['text'])
    
    # Tokenize stories
    tokenized_stories = []
    for story in stories:
        tokens = tokenizer.encode(story, 
                                max_length=config.max_seq_len,
                                truncation=True,
                                padding='max_length',
                                return_tensors='pt')
        tokenized_stories.append(tokens.squeeze())
    
    return torch.stack(tokenized_stories)

def train_glbl_tinystories(config: GLBLTransformerConfig, num_epochs: int = 3, 
                          num_samples: int = 1000):
    """Train GLBL TinyStories model"""
    
    logger.info("🚀 Starting GLBL TinyStories Training")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Update config with actual vocab size
    config.vocab_size = len(tokenizer)
    
    # Create model
    model = GLBLTinyStoriesModel(config).to(device)
    
    # Create dataset
    logger.info("Loading TinyStories dataset...")
    train_data = create_tinystories_dataset(config, tokenizer, "train", num_samples)
    eval_data = create_tinystories_dataset(config, tokenizer, "validation", num_samples // 10)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=config.batch_size, shuffle=False
    )
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = {'language_modeling': [], 'glbl': [], 'total': []}
        
        # Calculate GLBL weight for this epoch
        glbl_weight = (config.glbl_weight_start + 
                      (epoch / num_epochs) * (config.glbl_weight_end - config.glbl_weight_start))
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            # Create attention mask
            attention_mask = (batch != tokenizer.pad_token_id)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch, attention_mask, record_activations=(batch_idx % 10 == 0))
            
            # Language modeling loss
            logits = outputs['logits']
            lm_loss = criterion(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                batch[:, 1:].contiguous().view(-1)
            )
            
            # GLBL loss
            glbl_loss = outputs['total_glbl_loss']
            
            # Total loss
            total_loss = lm_loss + glbl_weight * glbl_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Record losses
            epoch_losses['language_modeling'].append(lm_loss.item())
            epoch_losses['glbl'].append(glbl_loss.item())
            epoch_losses['total'].append(total_loss.item())
            
            # Logging
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}:')
                logger.info(f'  LM Loss: {lm_loss.item():.4f}')
                logger.info(f'  GLBL Loss (w={glbl_weight:.3f}): {glbl_loss.item():.4f}')
                logger.info(f'  Total Loss: {total_loss.item():.4f}')
        
        # Epoch summary
        avg_lm_loss = np.mean(epoch_losses['language_modeling'])
        avg_glbl_loss = np.mean(epoch_losses['glbl'])
        avg_total_loss = np.mean(epoch_losses['total'])
        
        logger.info(f'Epoch {epoch} Summary:')
        logger.info(f'  Avg LM Loss: {avg_lm_loss:.4f}')
        logger.info(f'  Avg GLBL Loss: {avg_glbl_loss:.4f}')
        logger.info(f'  Avg Total Loss: {avg_total_loss:.4f}')
        logger.info('')
        
        # Evaluation
        if epoch % 1 == 0:
            eval_loss = evaluate_model(model, eval_loader, criterion, tokenizer)
            logger.info(f'Evaluation Loss: {eval_loss:.4f}')
            
            # Generate sample text
            sample_text = model.generate_text("Once upon a time", tokenizer, max_length=50)
            logger.info(f'Sample Generation: {sample_text}')
    
    return model, tokenizer

def evaluate_model(model, eval_loader, criterion, tokenizer):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            attention_mask = (batch != tokenizer.pad_token_id)
            
            outputs = model(batch, attention_mask)
            logits = outputs['logits']
            
            loss = criterion(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                batch[:, 1:].contiguous().view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches

def run_complete_tinystories_experiment():
    """Run complete GLBL TinyStories experiment with analysis"""
    
    logger.info("🚀 COMPLETE GLBL TINYSTORIES EXPERIMENT")
    logger.info("=" * 70)
    
    # Configuration
    config = GLBLTransformerConfig(
        n_layers=8,
        glbl_layers=[1, 3, 5, 7],  # Every other layer
        num_attention_pathways=16,
        num_mlp_pathways=16,
        batch_size=4,
        max_seq_len=256
    )
    
    # Train model
    model, tokenizer = train_glbl_tinystories(config, num_epochs=2, num_samples=500)
    
    # Analyze pathways
    logger.info("🔬 Analyzing Pathway Specializations...")
    analyzer = SemanticAnalyzer(config, tokenizer)
    
    # Run analysis on evaluation data
    model.eval()
    eval_data = create_tinystories_dataset(config, tokenizer, "validation", 100)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=1, shuffle=False)
    
    # Collect pathway activations
    model.pathway_activations.clear()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx >= 50:  # Limit for analysis
                break
            
            batch = batch.to(device)
            attention_mask = (batch != tokenizer.pad_token_id)
            _ = model(batch, attention_mask, record_activations=True)
    
    # Analyze specializations
    specializations = analyzer.analyze_pathway_specialization(model.pathway_activations)
    
    # Create results summary
    return {
        'model': model,
        'tokenizer': tokenizer,
        'config': config,
        'specializations': specializations,
        'analyzer': analyzer
    }

if __name__ == "__main__":
    # Run the complete experiment
    results = run_complete_tinystories_experiment()
    
    logger.info("✅ GLBL TinyStories experiment completed!")
    logger.info(f"Found {len(results['specializations'])} specialized pathways")