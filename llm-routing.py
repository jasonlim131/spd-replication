import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy.stats import entropy
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import json
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

@dataclass
class RoutingConfig:
    """Configuration for layer-wise routing in LLMs"""
    num_pathways: int = 16
    top_k_pathways: int = 4
    router_hidden_size: int = 256
    router_dropout: float = 0.1
    glbl_weight_start: float = 0.01
    glbl_weight_end: float = 0.1
    momentum: float = 0.9
    temperature: float = 1.0
    route_every_n_layers: int = 2  # Route every other layer

class PathwayRouter(nn.Module):
    """Router module for selecting pathways in a layer"""
    def __init__(self, hidden_size: int, num_pathways: int, router_hidden_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.num_pathways = num_pathways
        
        self.router = nn.Sequential(
            nn.Linear(hidden_size, router_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_size, num_pathways)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
        Returns:
            Pathway scores of shape (batch_size, seq_len, num_pathways)
        """
        return self.router(x)

class RoutedTransformerLayer(nn.Module):
    """Transformer layer with pathway-based routing for MLPs"""
    def __init__(self, base_layer: nn.Module, config: RoutingConfig):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        
        # Get hidden size from the layer
        self.hidden_size = base_layer.mlp.c_fc.in_features
        self.intermediate_size = base_layer.mlp.c_fc.out_features
        
        # Create pathway router
        self.pathway_router = PathwayRouter(
            self.hidden_size, 
            config.num_pathways,
            config.router_hidden_size,
            config.router_dropout
        )
        
        # Decompose MLP into pathways
        self._create_pathway_decomposition()
        
        # Global load balancing statistics
        self.register_buffer('global_pathway_frequencies', 
                           torch.zeros(config.num_pathways, device=device))
        self.register_buffer('global_pathway_scores', 
                           torch.zeros(config.num_pathways, device=device))
        self.register_buffer('update_count', torch.zeros(1, device=device))
        
        # Tracking for analysis
        self.pathway_activations = defaultdict(list)
        self.last_glbl_loss = None
        
    def _create_pathway_decomposition(self):
        """Create pathway decomposition for the MLP"""
        # Decompose hidden dimensions into groups
        hidden_per_pathway = self.hidden_size // self.config.num_pathways
        intermediate_per_pathway = self.intermediate_size // self.config.num_pathways
        
        # Create indices for each pathway
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
        """Compute Global Load Balancing Loss"""
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
        
        # Compute GLBL loss
        glbl_loss = self.config.num_pathways * torch.sum(
            current_frequencies * current_avg_scores
        )
        
        return glbl_loss, current_frequencies, current_avg_scores
    
    def select_pathways(self, pathway_scores: torch.Tensor) -> torch.Tensor:
        """Select top-k pathways based on scores"""
        batch_size, seq_len, _ = pathway_scores.shape
        
        # Apply temperature
        pathway_probs = F.softmax(pathway_scores / self.config.temperature, dim=-1)
        
        if self.training:
            # Add noise during training
            noise = torch.randn_like(pathway_scores) * 0.1
            noisy_scores = pathway_scores + noise
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
                record_activations: bool = False, token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with pathway routing"""
        residual = hidden_states
        
        # Layer norm
        hidden_states = self.base_layer.ln_1(hidden_states)
        
        # Self-attention (no routing)
        attn_output = self.base_layer.attn(hidden_states)[0]
        hidden_states = residual + attn_output
        residual = hidden_states
        
        # Layer norm before MLP
        hidden_states = self.base_layer.ln_2(hidden_states)
        
        # Route through MLP pathways
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get pathway scores
        pathway_scores = self.pathway_router(hidden_states)
        
        # Compute GLBL loss
        glbl_loss, frequencies, scores = self.compute_glbl_loss(pathway_scores)
        self.last_glbl_loss = glbl_loss
        
        # Select active pathways
        pathway_weights = self.select_pathways(pathway_scores)
        
        # Compute pathway outputs
        mlp_output = torch.zeros_like(hidden_states)
        
        for pathway_idx in range(self.config.num_pathways):
            # Get pathway weight
            weights = pathway_weights[:, :, pathway_idx].unsqueeze(-1)  # (batch, seq, 1)
            
            # Skip if pathway not used
            if weights.sum() < 1e-6:
                continue
            
            # Get pathway indices
            h_indices = self.hidden_indices[pathway_idx].to(hidden_states.device)
            i_indices = self.intermediate_indices[pathway_idx].to(hidden_states.device)
            
            # Extract pathway-specific inputs and weights
            h_subset = hidden_states[:, :, h_indices]
            
            # First projection (fc_in)
            W1_subset = self.base_layer.mlp.c_fc.weight[i_indices][:, h_indices]
            b1_subset = self.base_layer.mlp.c_fc.bias[i_indices]
            intermediate = F.linear(h_subset, W1_subset, b1_subset)
            
            # Activation
            intermediate = self.base_layer.mlp.act(intermediate)
            
            # Second projection (fc_out)
            W2_subset = self.base_layer.mlp.c_proj.weight[h_indices][:, i_indices]
            b2_subset = self.base_layer.mlp.c_proj.bias[h_indices]
            pathway_output = F.linear(intermediate, W2_subset, b2_subset)
            
            # Accumulate weighted pathway output
            mlp_output[:, :, h_indices] += pathway_output * weights
            
            # Record activations for analysis
            if record_activations and token_ids is not None:
                self._record_pathway_activations(
                    pathway_idx, weights, intermediate, pathway_output, token_ids
                )
        
        # Add residual connection
        output = residual + mlp_output
        
        return output
    
    def _record_pathway_activations(self, pathway_idx: int, weights: torch.Tensor, 
                                  intermediate: torch.Tensor, output: torch.Tensor, 
                                  token_ids: torch.Tensor):
        """Record pathway activations for monosemanticity analysis"""
        batch_size, seq_len = weights.shape[:2]
        
        for b in range(batch_size):
            for s in range(seq_len):
                if weights[b, s] > 1e-3:  # Only record active pathways
                    self.pathway_activations[f"pathway_{pathway_idx}"].append({
                        'token_id': token_ids[b, s].item() if token_ids is not None else -1,
                        'pathway_weight': weights[b, s].item(),
                        'intermediate_activation': intermediate[b, s].mean().item(),
                        'output_activation': output[b, s].mean().item(),
                        'position': s
                    })

class RoutedLLM(nn.Module):
    """LLM with layer-wise routing on every other layer"""
    def __init__(self, base_model: nn.Module, config: RoutingConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Get model config
        self.model_config = base_model.config
        self.num_layers = self.model_config.n_layer
        
        # Replace every other layer with routed version
        self.routed_layers = nn.ModuleList()
        self.routed_layer_indices = []
        
        for i in range(self.num_layers):
            if i % config.route_every_n_layers == 0:
                # Create routed layer
                original_layer = base_model.transformer.h[i]
                routed_layer = RoutedTransformerLayer(original_layer, config)
                self.routed_layers.append(routed_layer)
                self.routed_layer_indices.append(i)
                # Replace in base model
                base_model.transformer.h[i] = routed_layer
        
        print(f"🧠 Created Routed LLM:")
        print(f"   Base model: {type(base_model).__name__}")
        print(f"   Total layers: {self.num_layers}")
        print(f"   Routed layers: {len(self.routed_layers)} at indices {self.routed_layer_indices}")
        print(f"   Pathways per layer: {config.num_pathways}")
        print(f"   Active pathways (top-k): {config.top_k_pathways}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                record_activations: bool = False, **kwargs) -> Any:
        """Forward pass through the routed LLM"""
        # For recording activations, we need to handle the forward pass manually
        if record_activations:
            # Get embeddings
            inputs_embeds = self.base_model.transformer.wte(input_ids)
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            position_embeds = self.base_model.transformer.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
            
            # Pass through each layer
            for i, layer in enumerate(self.base_model.transformer.h):
                if isinstance(layer, RoutedTransformerLayer):
                    hidden_states = layer(hidden_states, attention_mask, 
                                        record_activations=True, token_ids=input_ids)
                else:
                    # Standard layer forward
                    outputs = layer(hidden_states, attention_mask=attention_mask)
                    hidden_states = outputs[0]
            
            # Final layer norm
            hidden_states = self.base_model.transformer.ln_f(hidden_states)
            
            # LM head
            logits = self.base_model.lm_head(hidden_states)
            
            return logits
        else:
            # Standard forward pass
            return self.base_model(input_ids, attention_mask, **kwargs)
    
    def get_total_glbl_loss(self) -> torch.Tensor:
        """Get total GLBL loss across all routed layers"""
        total_loss = torch.tensor(0.0, device=device)
        for layer in self.routed_layers:
            if layer.last_glbl_loss is not None:
                total_loss += layer.last_glbl_loss
        return total_loss / len(self.routed_layers)
    
    def analyze_pathway_specialization(self, tokenizer: AutoTokenizer) -> Dict[str, Any]:
        """Analyze pathway specialization for monosemanticity"""
        results = {}
        
        for layer_idx, layer in zip(self.routed_layer_indices, self.routed_layers):
            layer_results = {}
            
            # Analyze each pathway
            for pathway_name, activations in layer.pathway_activations.items():
                if len(activations) < 10:  # Skip underused pathways
                    continue
                
                # Token frequency analysis
                token_counts = Counter([act['token_id'] for act in activations])
                total_tokens = len(activations)
                
                # Calculate entropy/purity
                token_probs = np.array(list(token_counts.values())) / total_tokens
                pathway_entropy = entropy(token_probs)
                max_entropy = np.log(len(token_counts))
                normalized_entropy = pathway_entropy / max_entropy if max_entropy > 0 else 0
                purity = 1 - normalized_entropy
                
                # Get top tokens
                top_tokens = token_counts.most_common(10)
                top_token_info = []
                for token_id, count in top_tokens:
                    if token_id >= 0:
                        token_str = tokenizer.decode([token_id])
                        top_token_info.append({
                            'token': token_str,
                            'token_id': token_id,
                            'count': count,
                            'frequency': count / total_tokens
                        })
                
                layer_results[pathway_name] = {
                    'purity': purity,
                    'entropy': pathway_entropy,
                    'normalized_entropy': normalized_entropy,
                    'total_activations': total_tokens,
                    'unique_tokens': len(token_counts),
                    'top_tokens': top_token_info,
                    'avg_weight': np.mean([act['pathway_weight'] for act in activations]),
                    'avg_activation': np.mean([act['intermediate_activation'] for act in activations])
                }
            
            results[f'layer_{layer_idx}'] = layer_results
        
        return results

def measure_monosemanticity(routed_model: RoutedLLM, tokenizer: AutoTokenizer, 
                          test_texts: List[str], max_length: int = 128) -> Dict[str, Any]:
    """Measure monosemanticity of pathways in the routed LLM"""
    print("🔬 Measuring pathway monosemanticity...")
    
    # Clear previous activations
    for layer in routed_model.routed_layers:
        layer.pathway_activations.clear()
    
    routed_model.eval()
    
    # Process test texts
    with torch.no_grad():
        for text in tqdm(test_texts, desc="Processing texts"):
            inputs = tokenizer(text, return_tensors='pt', max_length=max_length, 
                             truncation=True, padding=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Forward pass with activation recording
            _ = routed_model(input_ids, attention_mask, record_activations=True)
    
    # Analyze specialization
    specialization_results = routed_model.analyze_pathway_specialization(tokenizer)
    
    # Compute overall metrics
    all_purities = []
    all_entropies = []
    total_pathways = 0
    active_pathways = 0
    
    for layer_name, layer_results in specialization_results.items():
        for pathway_name, pathway_stats in layer_results.items():
            total_pathways += 1
            if pathway_stats['total_activations'] > 0:
                active_pathways += 1
                all_purities.append(pathway_stats['purity'])
                all_entropies.append(pathway_stats['normalized_entropy'])
    
    metrics = {
        'avg_purity': np.mean(all_purities) if all_purities else 0,
        'std_purity': np.std(all_purities) if all_purities else 0,
        'max_purity': np.max(all_purities) if all_purities else 0,
        'avg_normalized_entropy': np.mean(all_entropies) if all_entropies else 0,
        'pathway_utilization': active_pathways / total_pathways if total_pathways > 0 else 0,
        'active_pathways': active_pathways,
        'total_pathways': total_pathways,
        'layer_results': specialization_results
    }
    
    return metrics

def visualize_monosemanticity(metrics: Dict[str, Any], save_path: str = "llm_monosemanticity.png"):
    """Visualize monosemanticity analysis results"""
    layer_results = metrics['layer_results']
    num_layers = len(layer_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LLM Pathway Monosemanticity Analysis', fontsize=16)
    
    # 1. Purity by layer
    layer_indices = []
    layer_purities = []
    
    for layer_name, layer_data in layer_results.items():
        layer_idx = int(layer_name.split('_')[1])
        purities = [stats['purity'] for stats in layer_data.values()]
        if purities:
            layer_indices.extend([layer_idx] * len(purities))
            layer_purities.extend(purities)
    
    axes[0, 0].scatter(layer_indices, layer_purities, alpha=0.6)
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('Pathway Purity')
    axes[0, 0].set_title('Pathway Purity Across Layers')
    axes[0, 0].axhline(y=metrics['avg_purity'], color='r', linestyle='--', 
                       label=f'Average: {metrics["avg_purity"]:.3f}')
    axes[0, 0].legend()
    
    # 2. Purity distribution
    all_purities = []
    for layer_data in layer_results.values():
        all_purities.extend([stats['purity'] for stats in layer_data.values()])
    
    if all_purities:
        axes[0, 1].hist(all_purities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(metrics['avg_purity'], color='red', linestyle='--')
        axes[0, 1].set_xlabel('Purity')
        axes[0, 1].set_ylabel('Number of Pathways')
        axes[0, 1].set_title('Distribution of Pathway Purities')
    
    # 3. Top specialized pathways
    all_pathways = []
    for layer_name, layer_data in layer_results.items():
        for pathway_name, stats in layer_data.items():
            if stats['total_activations'] > 50:  # Minimum threshold
                all_pathways.append({
                    'name': f"{layer_name}_{pathway_name}",
                    'purity': stats['purity'],
                    'top_token': stats['top_tokens'][0]['token'] if stats['top_tokens'] else 'N/A',
                    'frequency': stats['top_tokens'][0]['frequency'] if stats['top_tokens'] else 0
                })
    
    # Sort by purity and show top 15
    all_pathways.sort(key=lambda x: x['purity'], reverse=True)
    top_pathways = all_pathways[:15]
    
    if top_pathways:
        pathway_names = [p['name'] for p in top_pathways]
        pathway_purities = [p['purity'] for p in top_pathways]
        pathway_tokens = [f"{p['top_token']} ({p['frequency']:.2f})" for p in top_pathways]
        
        y_pos = np.arange(len(pathway_names))
        axes[1, 0].barh(y_pos, pathway_purities)
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels([f"{name}\n{token}" for name, token in 
                                    zip(pathway_names, pathway_tokens)], fontsize=8)
        axes[1, 0].set_xlabel('Purity')
        axes[1, 0].set_title('Top 15 Most Specialized Pathways')
        axes[1, 0].set_xlim(0, 1)
    
    # 4. Pathway utilization by layer
    layer_utilization = {}
    for layer_name, layer_data in layer_results.items():
        layer_idx = int(layer_name.split('_')[1])
        total = len(layer_data)
        active = sum(1 for stats in layer_data.values() if stats['total_activations'] > 0)
        layer_utilization[layer_idx] = active / total if total > 0 else 0
    
    if layer_utilization:
        layers = sorted(layer_utilization.keys())
        utilizations = [layer_utilization[l] for l in layers]
        
        axes[1, 1].bar(layers, utilizations)
        axes[1, 1].set_xlabel('Layer Index')
        axes[1, 1].set_ylabel('Pathway Utilization')
        axes[1, 1].set_title('Pathway Utilization by Layer')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axhline(y=metrics['pathway_utilization'], color='r', linestyle='--',
                          label=f'Average: {metrics["pathway_utilization"]:.2f}')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved as '{save_path}'")

def create_test_dataset() -> List[str]:
    """Create a simple test dataset for monosemanticity analysis"""
    # Mix of different text types to test pathway specialization
    test_texts = [
        # Simple stories
        "Once upon a time, there was a little girl who loved to play in the garden.",
        "The brave knight rode his horse through the dark forest.",
        "A magical fairy granted three wishes to the kind boy.",
        
        # Factual statements
        "The sun rises in the east and sets in the west.",
        "Water freezes at zero degrees Celsius.",
        "Birds can fly because they have wings and hollow bones.",
        
        # Questions
        "What is your favorite color?",
        "How many apples are in the basket?",
        "Where do pandas live in the wild?",
        
        # Commands/Instructions
        "Please close the door behind you.",
        "Mix the flour and eggs together in a bowl.",
        "Turn left at the next intersection.",
        
        # Descriptive text
        "The beautiful sunset painted the sky in shades of orange and pink.",
        "The old oak tree stood tall in the middle of the meadow.",
        "Snowflakes danced gently through the cold winter air.",
        
        # Dialogue
        "Hello, how are you today?",
        "I'm fine, thank you for asking!",
        "Would you like some tea?",
        
        # Numbers and counting
        "One, two, three, four, five, six, seven, eight, nine, ten.",
        "There are twelve months in a year.",
        "She had five red balloons and three blue ones.",
    ]
    
    # Duplicate and add variations
    extended_texts = []
    for text in test_texts:
        extended_texts.append(text)
        # Add lowercase version
        extended_texts.append(text.lower())
        # Add with different punctuation
        extended_texts.append(text.replace(".", "!"))
        extended_texts.append(text.replace(".", "?"))
    
    return extended_texts * 5  # Repeat to get more data

def train_routed_llm(model: RoutedLLM, train_texts: List[str], tokenizer: AutoTokenizer,
                    num_epochs: int = 3, batch_size: int = 4, learning_rate: float = 5e-5):
    """Train the routed LLM with GLBL loss"""
    print("🔄 Training Routed LLM with GLBL...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Prepare data
    def prepare_batch(texts):
        inputs = tokenizer(texts, return_tensors='pt', max_length=128, 
                          truncation=True, padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        # Shift labels for language modeling
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last token
        return input_ids, attention_mask, labels
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_glbl_loss = 0
        num_batches = len(train_texts) // batch_size
        
        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i:i+batch_size]
            input_ids, attention_mask, labels = prepare_batch(batch_texts)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Language modeling loss
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), 
                                     ignore_index=-100)
            
            # GLBL loss
            glbl_loss = model.get_total_glbl_loss()
            
            # Combine losses
            glbl_weight = model.config.glbl_weight_start + (
                (model.config.glbl_weight_end - model.config.glbl_weight_start) * 
                (epoch / num_epochs)
            )
            total_batch_loss = lm_loss + glbl_weight * glbl_loss
            
            # Backward pass
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += lm_loss.item()
            total_glbl_loss += glbl_loss.item()
        
        avg_loss = total_loss / num_batches
        avg_glbl_loss = total_glbl_loss / num_batches
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Average LM Loss: {avg_loss:.4f}")
        print(f"  Average GLBL Loss: {avg_glbl_loss:.4f}")
        print(f"  GLBL Weight: {glbl_weight:.4f}")

def main():
    """Main function to run the experiment"""
    print("🚀 LLM LAYER-WISE ROUTING EXPERIMENT")
    print("=" * 70)
    
    # Load TinyStories model
    print("\n📚 Loading TinyStories-1M model...")
    model_name = "roneneldan/TinyStories-1M"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    print(f"✅ Loaded model: {model_name}")
    print(f"   Vocabulary size: {base_model.config.vocab_size}")
    print(f"   Hidden size: {base_model.config.n_embd}")
    print(f"   Number of layers: {base_model.config.n_layer}")
    
    # Create routed model
    print("\n🛤️ Creating Routed LLM...")
    routing_config = RoutingConfig(
        num_pathways=16,
        top_k_pathways=4,
        router_hidden_size=128,  # Smaller for tiny model
        glbl_weight_start=0.01,
        glbl_weight_end=0.05
    )
    
    routed_model = RoutedLLM(base_model, routing_config).to(device)
    
    # Prepare training data
    train_texts = create_test_dataset()
    
    # Train with GLBL
    train_routed_llm(routed_model, train_texts, tokenizer, num_epochs=3)
    
    # Measure monosemanticity
    print("\n🔬 Measuring monosemanticity...")
    test_texts = create_test_dataset()[:100]  # Use subset for testing
    
    metrics = measure_monosemanticity(routed_model, tokenizer, test_texts)
    
    # Print results
    print("\n📊 MONOSEMANTICITY RESULTS:")
    print("=" * 50)
    print(f"Average Purity: {metrics['avg_purity']:.4f} ± {metrics['std_purity']:.4f}")
    print(f"Maximum Purity: {metrics['max_purity']:.4f}")
    print(f"Average Normalized Entropy: {metrics['avg_normalized_entropy']:.4f}")
    print(f"Pathway Utilization: {metrics['pathway_utilization']:.2%}")
    print(f"Active Pathways: {metrics['active_pathways']} / {metrics['total_pathways']}")
    
    # Show top specialized pathways
    print("\n🏆 Top Specialized Pathways:")
    for layer_name, layer_results in metrics['layer_results'].items():
        # Sort pathways by purity
        sorted_pathways = sorted(layer_results.items(), 
                               key=lambda x: x[1]['purity'], reverse=True)[:3]
        
        if sorted_pathways:
            print(f"\n{layer_name}:")
            for pathway_name, stats in sorted_pathways:
                if stats['top_tokens']:
                    top_token = stats['top_tokens'][0]
                    print(f"  {pathway_name}: purity={stats['purity']:.3f}, "
                          f"top_token='{top_token['token']}' ({top_token['frequency']:.2f})")
    
    # Create visualization
    visualize_monosemanticity(metrics)
    
    # Save results
    with open('llm_routing_results.json', 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        json_metrics = {
            'avg_purity': float(metrics['avg_purity']),
            'std_purity': float(metrics['std_purity']),
            'max_purity': float(metrics['max_purity']),
            'avg_normalized_entropy': float(metrics['avg_normalized_entropy']),
            'pathway_utilization': float(metrics['pathway_utilization']),
            'active_pathways': metrics['active_pathways'],
            'total_pathways': metrics['total_pathways']
        }
        json.dump(json_metrics, f, indent=2)
    
    print("\n✅ Results saved to 'llm_routing_results.json'")
    
    return routed_model, metrics

if __name__ == "__main__":
    routed_model, metrics = main()