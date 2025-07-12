import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy.stats import entropy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

class StandardMLP_FiveLayer(nn.Module):
    """Baseline 5-layer MLP with GELU activation"""
    def __init__(self, input_size=784, hidden_size1=512, hidden_size2=256, hidden_size3=128, 
                 hidden_size4=64, hidden_size5=32, num_classes=10, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.fc5 = nn.Linear(hidden_size4, hidden_size5)
        self.fc6 = nn.Linear(hidden_size5, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        x = F.gelu(self.fc3(x))
        x = self.dropout(x)
        x = F.gelu(self.fc4(x))
        x = self.dropout(x)
        x = F.gelu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
        return x

class GlobalPathwayMLP_FiveLayer(nn.Module):
    """Global Pathway MLP with 5 layers - full end-to-end routing"""
    def __init__(self, pretrained_mlp, config=None):
        super().__init__()
        
        # Configuration for 5-layer global routing (same as before)
        default_config = {
            'num_input_regions': 4,        # Spatial image regions
            'num_hidden1_groups': 2,       # Hidden1 layer neuron groups  
            'num_hidden2_groups': 2,       # Hidden2 layer neuron groups
            'num_hidden3_groups': 2,       # Hidden3 layer neuron groups
            'num_hidden4_groups': 2,       # Hidden4 layer neuron groups
            'num_hidden5_groups': 2,       # Hidden5 layer neuron groups
            'num_output_groups': 4,        # Output class groups
            'momentum': 0.9,               # For global statistics
            'router_hidden_size': 256,     # Pathway router capacity
            'router_dropout': 0.1          # Pathway router dropout
        }
        self.config = {**default_config, **(config or {})}
        
        # Network dimensions
        self.input_dim = pretrained_mlp.fc1.in_features
        self.hidden1_dim = pretrained_mlp.fc1.out_features
        self.hidden2_dim = pretrained_mlp.fc2.out_features
        self.hidden3_dim = pretrained_mlp.fc3.out_features
        self.hidden4_dim = pretrained_mlp.fc4.out_features
        self.hidden5_dim = pretrained_mlp.fc5.out_features
        self.output_dim = pretrained_mlp.fc6.out_features
        
        # Pathway configuration
        self.num_input_regions = self.config['num_input_regions']
        self.num_hidden1_groups = self.config['num_hidden1_groups']
        self.num_hidden2_groups = self.config['num_hidden2_groups']
        self.num_hidden3_groups = self.config['num_hidden3_groups']
        self.num_hidden4_groups = self.config['num_hidden4_groups']
        self.num_hidden5_groups = self.config['num_hidden5_groups']
        self.num_output_groups = self.config['num_output_groups']
        self.num_pathways = (self.num_input_regions * 
                           self.num_hidden1_groups * 
                           self.num_hidden2_groups *
                           self.num_hidden3_groups *
                           self.num_hidden4_groups *
                           self.num_hidden5_groups *
                           self.num_output_groups)
        
        print(f"🧠 Global 5-Layer Configuration:")
        print(f"   Input: {self.input_dim} → H1: {self.hidden1_dim} → H2: {self.hidden2_dim} → H3: {self.hidden3_dim} → H4: {self.hidden4_dim} → H5: {self.hidden5_dim} → Output: {self.output_dim}")
        print(f"   Pathways: {self.num_input_regions}×{self.num_hidden1_groups}×{self.num_hidden2_groups}×{self.num_hidden3_groups}×{self.num_hidden4_groups}×{self.num_hidden5_groups}×{self.num_output_groups} = {self.num_pathways}")
        
        # Copy pretrained weights
        self.fc1 = nn.Linear(self.input_dim, self.hidden1_dim)
        self.fc2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.fc3 = nn.Linear(self.hidden2_dim, self.hidden3_dim)
        self.fc4 = nn.Linear(self.hidden3_dim, self.hidden4_dim)
        self.fc5 = nn.Linear(self.hidden4_dim, self.hidden5_dim)
        self.fc6 = nn.Linear(self.hidden5_dim, self.output_dim)
        
        with torch.no_grad():
            self.fc1.weight.copy_(pretrained_mlp.fc1.weight)
            self.fc1.bias.copy_(pretrained_mlp.fc1.bias)
            self.fc2.weight.copy_(pretrained_mlp.fc2.weight)
            self.fc2.bias.copy_(pretrained_mlp.fc2.bias)
            self.fc3.weight.copy_(pretrained_mlp.fc3.weight)
            self.fc3.bias.copy_(pretrained_mlp.fc3.bias)
            self.fc4.weight.copy_(pretrained_mlp.fc4.weight)
            self.fc4.bias.copy_(pretrained_mlp.fc4.bias)
            self.fc5.weight.copy_(pretrained_mlp.fc5.weight)
            self.fc5.bias.copy_(pretrained_mlp.fc5.bias)
            self.fc6.weight.copy_(pretrained_mlp.fc6.weight)
            self.fc6.bias.copy_(pretrained_mlp.fc6.bias)
        
        # Create pathway decomposition
        self._create_pathway_structure()
        
        # Pathway router network
        self.pathway_router = nn.Sequential(
            nn.Linear(self.input_dim, self.config['router_hidden_size']),
            nn.ReLU(),
            nn.Dropout(self.config['router_dropout']),
            nn.Linear(self.config['router_hidden_size'], self.num_pathways)
        )
        
        # Global Load Balancing statistics
        self.register_buffer('global_pathway_frequencies', 
                           torch.zeros(self.num_pathways).to(device))
        self.register_buffer('global_pathway_scores', 
                           torch.zeros(self.num_pathways).to(device))
        self.register_buffer('update_count', torch.zeros(1).to(device))
        
        # Tracking for analysis
        self.pathway_activations = defaultdict(list)
        self.glbl_stats = defaultdict(list)
        self.last_glbl_loss = None
        
    def _create_pathway_structure(self):
        """Create pathway decomposition indices"""
        # Input regions (spatial decomposition for 28x28 images)
        input_groups = self._create_spatial_regions(28, 28, self.num_input_regions)
        self.register_buffer('input_groups_indices', input_groups)
        
        # Hidden layer neuron groups
        hidden1_groups = self._create_neuron_groups(self.hidden1_dim, self.num_hidden1_groups)
        self.register_buffer('hidden1_groups_indices', hidden1_groups)
        
        hidden2_groups = self._create_neuron_groups(self.hidden2_dim, self.num_hidden2_groups)
        self.register_buffer('hidden2_groups_indices', hidden2_groups)
        
        hidden3_groups = self._create_neuron_groups(self.hidden3_dim, self.num_hidden3_groups)
        self.register_buffer('hidden3_groups_indices', hidden3_groups)
        
        hidden4_groups = self._create_neuron_groups(self.hidden4_dim, self.num_hidden4_groups)
        self.register_buffer('hidden4_groups_indices', hidden4_groups)
        
        hidden5_groups = self._create_neuron_groups(self.hidden5_dim, self.num_hidden5_groups)
        self.register_buffer('hidden5_groups_indices', hidden5_groups)
        
        # Output class groups
        output_groups = self._create_neuron_groups(self.output_dim, self.num_output_groups)
        self.register_buffer('output_groups_indices', output_groups)
        
        print(f"✅ Created global pathway structure:")
        print(f"   Input regions: {input_groups.shape}")
        print(f"   Hidden1 groups: {hidden1_groups.shape}")
        print(f"   Hidden2 groups: {hidden2_groups.shape}")
        print(f"   Hidden3 groups: {hidden3_groups.shape}")
        print(f"   Hidden4 groups: {hidden4_groups.shape}")
        print(f"   Hidden5 groups: {hidden5_groups.shape}")
        print(f"   Output groups: {output_groups.shape}")
    
    def _create_spatial_regions(self, height, width, num_regions):
        """Create spatial regions for image input"""
        if num_regions == 4:
            # Quadrant decomposition
            regions = [
                (0, height//2, 0, width//2),      # Top-left
                (0, height//2, width//2, width),  # Top-right  
                (height//2, height, 0, width//2), # Bottom-left
                (height//2, height, width//2, width) # Bottom-right
            ]
        else:
            raise NotImplementedError(f"Only 4 regions supported, got {num_regions}")
        
        max_pixels_per_region = (height * width) // num_regions
        groups = torch.zeros(num_regions, max_pixels_per_region, dtype=torch.long)
        
        for region_idx, (h_start, h_end, w_start, w_end) in enumerate(regions):
            indices = []
            for h in range(h_start, h_end):
                for w in range(w_start, w_end):
                    indices.append(h * width + w)
            
            # Pad or truncate to max_pixels_per_region
            indices = indices[:max_pixels_per_region]
            while len(indices) < max_pixels_per_region:
                indices.append(indices[0])  # Repeat first index if needed
                
            groups[region_idx] = torch.tensor(indices, dtype=torch.long)
        
        return groups.to(device)
    
    def _create_neuron_groups(self, total_neurons, num_groups):
        """Create neuron groups for hidden/output layers"""
        neurons_per_group = total_neurons // num_groups
        groups = torch.zeros(num_groups, neurons_per_group, dtype=torch.long)
        
        for group_idx in range(num_groups):
            start_idx = group_idx * neurons_per_group
            end_idx = start_idx + neurons_per_group
            groups[group_idx] = torch.arange(start_idx, end_idx, dtype=torch.long)
        
        return groups.to(device)
    
    def compute_glbl_loss(self, pathway_scores):
        """Compute Global Load Balancing Loss"""
        batch_size = pathway_scores.shape[0]
        
        # Current batch pathway probabilities
        pathway_probs = F.softmax(pathway_scores, dim=-1)
        current_frequencies = pathway_probs.mean(dim=0)
        current_avg_scores = pathway_probs.mean(dim=0)
        
        # Update global statistics with momentum
        if self.training:
            momentum = self.config['momentum']
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
        
        # Compute GLBL loss using current batch statistics
        glbl_loss = self.num_pathways * torch.sum(
            current_frequencies * current_avg_scores
        )
        
        return glbl_loss, current_frequencies, current_avg_scores
    
    def select_pathways(self, pathway_scores, top_k=12, temperature=1.0):
        """Smart pathway selection with load balancing awareness"""
        batch_size = pathway_scores.shape[0]
        pathway_probs = F.softmax(pathway_scores / temperature, dim=-1)
        
        if self.training:
            # During training: Use differentiable top-k with some randomness
            noise = torch.randn_like(pathway_scores) * 0.1
            noisy_scores = pathway_scores + noise
            top_values, top_indices = torch.topk(
                F.softmax(noisy_scores, dim=-1), top_k, dim=-1
            )
        else:
            # During inference: Use deterministic top-k
            top_values, top_indices = torch.topk(pathway_probs, top_k, dim=-1)
        
        # Create selection mask
        selection_mask = torch.zeros_like(pathway_probs)
        selection_mask.scatter_(1, top_indices, top_values)
        
        # Normalize selected pathways
        pathway_weights = selection_mask / (selection_mask.sum(dim=-1, keepdim=True) + 1e-8)
        
        return pathway_weights
    
    def forward(self, x, top_k=12, temperature=1.0, record_activations=False, true_labels=None):
        """Forward pass with global pathway-based computation"""
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Step 1: Route inputs to pathways
        pathway_scores = self.pathway_router(x_flat)
        
        # Step 2: Compute GLBL loss
        glbl_loss, frequencies, scores = self.compute_glbl_loss(pathway_scores)
        self.last_glbl_loss = glbl_loss
        
        # Step 3: Select active pathways
        pathway_weights = self.select_pathways(pathway_scores, top_k, temperature)
        
        # Step 4: Compute pathway outputs (full 5-layer end-to-end)
        output = torch.zeros(batch_size, self.output_dim, device=device)
        
        pathway_idx = 0
        for i in range(self.num_input_regions):
            for j in range(self.num_hidden1_groups):
                for k in range(self.num_hidden2_groups):
                    for l in range(self.num_hidden3_groups):
                        for m in range(self.num_hidden4_groups):
                            for n in range(self.num_hidden5_groups):
                                for o in range(self.num_output_groups):
                                    
                                    # Get pathway activation weights
                                    weights = pathway_weights[:, pathway_idx].unsqueeze(1)
                                    
                                    # Skip computation if pathway not used
                                    if weights.sum() < 1e-6:
                                        pathway_idx += 1
                                        continue
                                    
                                    # Get pathway indices
                                    input_indices = self.input_groups_indices[i]
                                    hidden1_indices = self.hidden1_groups_indices[j]
                                    hidden2_indices = self.hidden2_groups_indices[k]
                                    hidden3_indices = self.hidden3_groups_indices[l]
                                    hidden4_indices = self.hidden4_groups_indices[m]
                                    hidden5_indices = self.hidden5_groups_indices[n]
                                    output_indices = self.output_groups_indices[o]
                                    
                                    # Extract pathway-specific weights and inputs
                                    input_subset = x_flat[:, input_indices]
                                    
                                    # Layer 1: input → hidden1
                                    W1_subset = self.fc1.weight[hidden1_indices][:, input_indices]
                                    b1_subset = self.fc1.bias[hidden1_indices]
                                    hidden1_output = F.gelu(F.linear(input_subset, W1_subset, b1_subset))
                                    
                                    # Layer 2: hidden1 → hidden2
                                    W2_subset = self.fc2.weight[hidden2_indices][:, hidden1_indices]
                                    b2_subset = self.fc2.bias[hidden2_indices]
                                    hidden2_output = F.gelu(F.linear(hidden1_output, W2_subset, b2_subset))
                                    
                                    # Layer 3: hidden2 → hidden3
                                    W3_subset = self.fc3.weight[hidden3_indices][:, hidden2_indices]
                                    b3_subset = self.fc3.bias[hidden3_indices]
                                    hidden3_output = F.gelu(F.linear(hidden2_output, W3_subset, b3_subset))
                                    
                                    # Layer 4: hidden3 → hidden4
                                    W4_subset = self.fc4.weight[hidden4_indices][:, hidden3_indices]
                                    b4_subset = self.fc4.bias[hidden4_indices]
                                    hidden4_output = F.gelu(F.linear(hidden3_output, W4_subset, b4_subset))
                                    
                                    # Layer 5: hidden4 → hidden5
                                    W5_subset = self.fc5.weight[hidden5_indices][:, hidden4_indices]
                                    b5_subset = self.fc5.bias[hidden5_indices]
                                    hidden5_output = F.gelu(F.linear(hidden4_output, W5_subset, b5_subset))
                                    
                                    # Layer 6: hidden5 → output
                                    W6_subset = self.fc6.weight[output_indices][:, hidden5_indices]
                                    b6_subset = self.fc6.bias[output_indices]
                                    pathway_output = F.linear(hidden5_output, W6_subset, b6_subset)
                                    
                                    # Accumulate weighted pathway output
                                    output[:, output_indices] += pathway_output * weights
                                    
                                    # Record activations for analysis
                                    if record_activations and true_labels is not None:
                                        self._record_pathway_activations(
                                            pathway_idx, i, j, k, l, m, n, o, weights, 
                                            hidden1_output, hidden2_output, hidden3_output, 
                                            hidden4_output, hidden5_output, pathway_output, 
                                            true_labels, batch_size
                                        )
                                    
                                    pathway_idx += 1
        
        # Update statistics
        if self.training:
            self.glbl_stats['glbl_loss'].append(glbl_loss.item())
            self.glbl_stats['pathway_usage_entropy'].append(
                entropy(frequencies.detach().cpu().numpy() + 1e-8)
            )
            self.glbl_stats['active_pathways_per_batch'].append(
                (pathway_weights > 1e-3).sum().item()
            )
        
        return output
    
    def _record_pathway_activations(self, pathway_idx, i, j, k, l, m, n, o, weights, 
                                  hidden1_output, hidden2_output, hidden3_output, 
                                  hidden4_output, hidden5_output, pathway_output, true_labels, batch_size):
        """Record pathway activations for analysis"""
        pathway_name = f"Input{i}_H1{j}_H2{k}_H3{l}_H4{m}_H5{n}_Output{o}"
        
        for sample_idx in range(batch_size):
            if weights[sample_idx] > 1e-3:  # Only record active pathways
                self.pathway_activations[pathway_name].append({
                    'pathway_idx': pathway_idx,
                    'true_class': true_labels[sample_idx].item(),
                    'pathway_weight': weights[sample_idx].item(),
                    'hidden1_activation': hidden1_output[sample_idx].mean().item(),
                    'hidden2_activation': hidden2_output[sample_idx].mean().item(),
                    'hidden3_activation': hidden3_output[sample_idx].mean().item(),
                    'hidden4_activation': hidden4_output[sample_idx].mean().item(),
                    'hidden5_activation': hidden5_output[sample_idx].mean().item(),
                    'output_activation': pathway_output[sample_idx].max().item()
                })
    
    def get_pathway_analysis(self):
        """Comprehensive pathway analysis"""
        if not self.glbl_stats['glbl_loss']:
            return {}
        
        analysis = {
            # GLBL metrics
            'avg_glbl_loss': np.mean(self.glbl_stats['glbl_loss']),
            'final_glbl_loss': self.glbl_stats['glbl_loss'][-1],
            'glbl_loss_trend': np.polyfit(range(len(self.glbl_stats['glbl_loss'])), 
                                        self.glbl_stats['glbl_loss'], 1)[0],
            
            # Pathway usage metrics
            'avg_pathway_entropy': np.mean(self.glbl_stats['pathway_usage_entropy']),
            'avg_active_pathways': np.mean(self.glbl_stats['active_pathways_per_batch']),
            'pathway_utilization': len(self.pathway_activations) / self.num_pathways,
            
            # Global statistics
            'global_frequencies': self.global_pathway_frequencies.cpu().numpy(),
            'global_scores': self.global_pathway_scores.cpu().numpy(),
            'update_count': self.update_count.item()
        }
        
        return analysis

class MiddleLayerPathwayMLP_FiveLayer(nn.Module):
    """5-layer MLP with layerwise routing applied to ONE middle layer"""
    def __init__(self, pretrained_mlp, middle_layer=3, config=None):
        super().__init__()
        
        # Configuration for middle layer routing
        default_config = {
            'input_groups': 4,     # Number of input groups for the middle layer
            'output_groups': 4,    # Number of output groups for the middle layer
            'temperature': 1.0
        }
        self.config = {**default_config, **(config or {})}
        self.middle_layer = middle_layer  # Which layer to apply routing to (1-based)
        
        print(f"🧠 Middle Layer Routing Configuration:")
        print(f"   Architecture: 784→512→256→128→64→32→10 (5 layers)")
        print(f"   Routing applied to: Layer {middle_layer}")
        print(f"   Middle layer pathways: {self.config['input_groups']}×{self.config['output_groups']} = {self.config['input_groups'] * self.config['output_groups']}")
        
        # Copy pretrained weights
        self.fc1 = nn.Linear(pretrained_mlp.fc1.in_features, pretrained_mlp.fc1.out_features)
        self.fc2 = nn.Linear(pretrained_mlp.fc2.in_features, pretrained_mlp.fc2.out_features)
        self.fc3 = nn.Linear(pretrained_mlp.fc3.in_features, pretrained_mlp.fc3.out_features)
        self.fc4 = nn.Linear(pretrained_mlp.fc4.in_features, pretrained_mlp.fc4.out_features)
        self.fc5 = nn.Linear(pretrained_mlp.fc5.in_features, pretrained_mlp.fc5.out_features)
        self.fc6 = nn.Linear(pretrained_mlp.fc6.in_features, pretrained_mlp.fc6.out_features)
        
        with torch.no_grad():
            self.fc1.weight.copy_(pretrained_mlp.fc1.weight)
            self.fc1.bias.copy_(pretrained_mlp.fc1.bias)
            self.fc2.weight.copy_(pretrained_mlp.fc2.weight)
            self.fc2.bias.copy_(pretrained_mlp.fc2.bias)
            self.fc3.weight.copy_(pretrained_mlp.fc3.weight)
            self.fc3.bias.copy_(pretrained_mlp.fc3.bias)
            self.fc4.weight.copy_(pretrained_mlp.fc4.weight)
            self.fc4.bias.copy_(pretrained_mlp.fc4.bias)
            self.fc5.weight.copy_(pretrained_mlp.fc5.weight)
            self.fc5.bias.copy_(pretrained_mlp.fc5.bias)
            self.fc6.weight.copy_(pretrained_mlp.fc6.weight)
            self.fc6.bias.copy_(pretrained_mlp.fc6.bias)
        
        # Create router for the middle layer only
        # Layer input dimensions: layer 1 gets 784, layer 2 gets 512, layer 3 gets 256, etc.
        layer_input_dims = [784, 512, 256, 128, 64, 32]  # Input dimensions for each layer
        router_input_dim = layer_input_dims[middle_layer - 1]  # Input to the middle layer
        num_pathways = self.config['input_groups'] * self.config['output_groups']
        
        self.middle_router = nn.Linear(router_input_dim, num_pathways)
        
        # Create pathway decompositions for the middle layer
        self._create_middle_layer_pathways()
        
        # Pathway activation tracking
        self.pathway_activations = defaultdict(list)
        
        print(f"✅ Created middle layer routing for Layer {middle_layer}")
    
    def _create_middle_layer_pathways(self):
        """Create pathway decompositions for the middle layer"""
        # Get dimensions for the middle layer
        layer_dims = [
            (784, 512),   # Layer 1: fc1
            (512, 256),   # Layer 2: fc2  
            (256, 128),   # Layer 3: fc3
            (128, 64),    # Layer 4: fc4
            (64, 32),     # Layer 5: fc5
            (32, 10)      # Layer 6: fc6
        ]
        
        input_dim, output_dim = layer_dims[self.middle_layer - 1]
        
        # Create input and output groups for the middle layer
        input_groups = self._create_neuron_groups(input_dim, self.config['input_groups'])
        output_groups = self._create_neuron_groups(output_dim, self.config['output_groups'])
        
        self.pathway_indices = {
            'input_groups': input_groups,
            'output_groups': output_groups
        }
        
        print(f"   Middle layer decomposition: {input_groups.shape} → {output_groups.shape}")
    
    def _create_neuron_groups(self, dim, num_groups):
        """Create neuron groups for the middle layer"""
        neurons_per_group = dim // num_groups
        groups = torch.zeros(num_groups, neurons_per_group, dtype=torch.long)
        
        for group_idx in range(num_groups):
            start_idx = group_idx * neurons_per_group
            end_idx = start_idx + neurons_per_group
            groups[group_idx] = torch.arange(start_idx, end_idx, dtype=torch.long)
        
        return groups.to(device)
    
    def forward(self, x, record_activations=False, true_labels=None):
        """Forward pass with middle layer routing"""
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        # Forward through layers before the middle layer
        if self.middle_layer == 1:
            middle_input = x
        elif self.middle_layer == 2:
            x = F.gelu(self.fc1(x))
            middle_input = x
        elif self.middle_layer == 3:
            x = F.gelu(self.fc1(x))
            x = F.gelu(self.fc2(x))
            middle_input = x
        elif self.middle_layer == 4:
            x = F.gelu(self.fc1(x))
            x = F.gelu(self.fc2(x))
            x = F.gelu(self.fc3(x))
            middle_input = x
        elif self.middle_layer == 5:
            x = F.gelu(self.fc1(x))
            x = F.gelu(self.fc2(x))
            x = F.gelu(self.fc3(x))
            x = F.gelu(self.fc4(x))
            middle_input = x
        
        # Apply routing to the middle layer
        router_scores = self.middle_router(middle_input)
        pathway_weights = F.softmax(router_scores, dim=-1)
        
        # Get the middle layer
        if self.middle_layer == 1:
            middle_layer = self.fc1
        elif self.middle_layer == 2:
            middle_layer = self.fc2
        elif self.middle_layer == 3:
            middle_layer = self.fc3
        elif self.middle_layer == 4:
            middle_layer = self.fc4
        elif self.middle_layer == 5:
            middle_layer = self.fc5
        
        # Compute routed middle layer output
        middle_output = self._compute_middle_layer_pathways(
            middle_input, pathway_weights, middle_layer, 
            record_activations, true_labels, batch_size
        )
        middle_output = F.gelu(middle_output)
        
        # Forward through remaining layers after the middle layer
        if self.middle_layer == 1:
            x = F.gelu(self.fc2(middle_output))
            x = F.gelu(self.fc3(x))
            x = F.gelu(self.fc4(x))
            x = F.gelu(self.fc5(x))
            x = self.fc6(x)
        elif self.middle_layer == 2:
            x = F.gelu(self.fc3(middle_output))
            x = F.gelu(self.fc4(x))
            x = F.gelu(self.fc5(x))
            x = self.fc6(x)
        elif self.middle_layer == 3:
            x = F.gelu(self.fc4(middle_output))
            x = F.gelu(self.fc5(x))
            x = self.fc6(x)
        elif self.middle_layer == 4:
            x = F.gelu(self.fc5(middle_output))
            x = self.fc6(x)
        elif self.middle_layer == 5:
            x = self.fc6(middle_output)
        
        return x
    
    def _compute_middle_layer_pathways(self, input_tensor, pathway_weights, layer, 
                                     record_activations, true_labels, batch_size):
        """Compute pathways for the middle layer"""
        input_groups = self.pathway_indices['input_groups']
        output_groups = self.pathway_indices['output_groups']
        
        output_dim = layer.out_features
        output = torch.zeros(batch_size, output_dim, device=device)
        
        pathway_idx = 0
        for i in range(input_groups.shape[0]):
            for j in range(output_groups.shape[0]):
                # Get pathway weight
                weights = pathway_weights[:, pathway_idx].unsqueeze(1)
                
                if weights.sum() > 1e-6:  # Only process active pathways
                    # Get indices
                    input_indices = input_groups[i]
                    output_indices = output_groups[j]
                    
                    # Extract pathway-specific computation
                    input_subset = input_tensor[:, input_indices]
                    W_subset = layer.weight[output_indices][:, input_indices]
                    b_subset = layer.bias[output_indices]
                    
                    # Compute pathway output
                    pathway_output = F.linear(input_subset, W_subset, b_subset)
                    
                    # Accumulate weighted output
                    output[:, output_indices] += pathway_output * weights
                    
                    # Record activations
                    if record_activations and true_labels is not None:
                        self._record_middle_layer_activations(
                            pathway_idx, i, j, weights, pathway_output, 
                            true_labels, batch_size
                        )
                
                pathway_idx += 1
        
        return output
    
    def _record_middle_layer_activations(self, pathway_idx, i, j, weights, 
                                       pathway_output, true_labels, batch_size):
        """Record middle layer pathway activations"""
        pathway_name = f"MiddleLayer{self.middle_layer}_Input{i}_Output{j}"
        
        for sample_idx in range(batch_size):
            if weights[sample_idx] > 1e-3:
                self.pathway_activations[pathway_name].append({
                    'pathway_idx': pathway_idx,
                    'true_class': true_labels[sample_idx].item(),
                    'pathway_weight': weights[sample_idx].item(),
                    'output_activation': pathway_output[sample_idx].max().item()
                })

class GlobalTrainer_FiveLayer:
    """Trainer for Global 5-Layer model"""
    def __init__(self, model, config=None):
        self.model = model
        default_config = {
            'lr': 0.001,
            'glbl_weight_start': 0.001,
            'glbl_weight_end': 0.01,
            'patience': 5
        }
        self.config = {**default_config, **(config or {})}
        
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], 
            lr=self.config['lr']
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def _get_glbl_weight(self, epoch, max_epochs):
        """Get GLBL weight for current epoch"""
        progress = epoch / max_epochs
        return self.config['glbl_weight_start'] + progress * (
            self.config['glbl_weight_end'] - self.config['glbl_weight_start']
        )
    
    def train(self, train_loader, epochs=3, verbose=True):
        """Train the model"""
        self.model.train()
        training_stats = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_glbl_loss = 0
            correct = 0
            total = 0
            
            glbl_weight = self._get_glbl_weight(epoch, epochs)
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data, record_activations=True, true_labels=target)
                
                # Compute losses
                classification_loss = self.criterion(output, target)
                glbl_loss = self.model.last_glbl_loss or 0
                
                total_loss = classification_loss + glbl_weight * glbl_loss
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                # Statistics
                epoch_loss += total_loss.item()
                epoch_glbl_loss += glbl_loss.item() if isinstance(glbl_loss, torch.Tensor) else glbl_loss
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                if verbose and batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}: '
                          f'Loss: {total_loss.item():.4f}, '
                          f'GLBL: {glbl_loss:.4f}, '
                          f'Acc: {100.*correct/total:.2f}%')
            
            avg_loss = epoch_loss / len(train_loader)
            avg_glbl_loss = epoch_glbl_loss / len(train_loader)
            accuracy = 100. * correct / total
            
            training_stats.append({
                'epoch': epoch,
                'loss': avg_loss,
                'glbl_loss': avg_glbl_loss,
                'accuracy': accuracy,
                'glbl_weight': glbl_weight
            })
            
            if verbose:
                print(f'Epoch {epoch}: Loss: {avg_loss:.4f}, '
                      f'GLBL: {avg_glbl_loss:.4f}, '
                      f'Acc: {accuracy:.2f}%, '
                      f'GLBL Weight: {glbl_weight:.4f}')
        
        return training_stats

class MiddleLayerTrainer_FiveLayer:
    """Trainer for Middle Layer 5-Layer model"""
    def __init__(self, model, config=None):
        self.model = model
        default_config = {
            'lr': 0.001,
            'patience': 5
        }
        self.config = {**default_config, **(config or {})}
        
        # Separate optimizers for main layers and router
        main_params = (
            list(self.model.fc1.parameters()) + 
            list(self.model.fc2.parameters()) + 
            list(self.model.fc3.parameters()) + 
            list(self.model.fc4.parameters()) + 
            list(self.model.fc5.parameters()) + 
            list(self.model.fc6.parameters())
        )
        router_params = list(self.model.middle_router.parameters())
        
        self.main_optimizer = torch.optim.Adam(main_params, lr=self.config['lr'])
        self.router_optimizer = torch.optim.Adam(router_params, lr=self.config['lr'] * 0.1)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, train_loader, epochs=3):
        """Train the model"""
        self.model.train()
        training_stats = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                self.main_optimizer.zero_grad()
                self.router_optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data, record_activations=True, true_labels=target)
                
                # Compute loss
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.main_optimizer.step()
                self.router_optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}: '
                          f'Loss: {loss.item():.4f}, '
                          f'Acc: {100.*correct/total:.2f}%')
            
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100. * correct / total
            
            training_stats.append({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy
            })
            
            print(f'Epoch {epoch}: Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')
        
        return training_stats

def train_standard_mlp_five_layer(config=None):
    """Train a standard 5-layer MLP"""
    print("🏋️ Training Standard 5-Layer MLP...")
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    
    # Use subset for faster training
    subset_indices = torch.randperm(len(train_dataset))[:15000]
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Create and train model
    model = StandardMLP_FiveLayer().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}: '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        print(f'Epoch {epoch}: Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')
    
    return model, test_loader

def evaluate_model(model, test_loader):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy

def analyze_pathway_specialization(pathway_mlp, test_loader, max_samples=2000):
    """Analyze pathway specialization patterns"""
    print(f"\n🎯 Analyzing Pathway Specialization...")
    
    pathway_mlp.eval()
    pathway_mlp.pathway_activations.clear()
    
    # Record activations
    samples_processed = 0
    with torch.no_grad():
        for data, target in test_loader:
            if samples_processed >= max_samples:
                break
            data, target = data.to(device), target.to(device)
            _ = pathway_mlp(data, record_activations=True, true_labels=target)
            samples_processed += data.size(0)
    
    # Analyze specialization
    specializations = {}
    
    for pathway_name, activations in pathway_mlp.pathway_activations.items():
        if len(activations) >= 5:  # Minimum usage threshold
            class_counts = defaultdict(int)
            total_usage = len(activations)
            
            for activation in activations:
                class_counts[activation['true_class']] += 1
            
            # Calculate purity (specialization to dominant class)
            if class_counts:
                dominant_class = max(class_counts.keys(), key=lambda k: class_counts[k])
                purity = class_counts[dominant_class] / total_usage
                
                specializations[pathway_name] = {
                    'dominant_class': dominant_class,
                    'purity': purity,
                    'usage': total_usage,
                    'class_distribution': dict(class_counts)
                }
    
    # Print analysis
    if specializations:
        avg_purity = np.mean([s['purity'] for s in specializations.values()])
        high_purity = sum(1 for s in specializations.values() if s['purity'] > 0.4)
        
        print(f"📊 Pathway Specialization Results:")
        print(f"   Samples analyzed: {samples_processed}")
        print(f"   Active pathways: {len(specializations)}")
        print(f"   Average purity: {avg_purity:.3f}")
        print(f"   High purity pathways (>40%): {high_purity}/{len(specializations)}")
        
        # Top specialized pathways
        sorted_pathways = sorted(
            specializations.items(), 
            key=lambda x: x[1]['purity'], 
            reverse=True
        )
        
        print(f"\n🏆 Top Specialized Pathways:")
        for pathway, stats in sorted_pathways[:5]:
            print(f"   {pathway}: {stats['purity']:.3f} purity → "
                  f"digit {stats['dominant_class']} ({stats['usage']} uses)")
    
    return specializations

def create_five_layer_visualization(global_specs, middle_specs, middle_layer=3, title="5-Layer: Global vs Middle Layer Routing"):
    """Create visualization comparing 5-layer results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16)
    
    # Global pathway data
    if global_specs:
        global_purities = [spec['purity'] for spec in global_specs.values()]
        global_usages = [spec['usage'] for spec in global_specs.values()]
        global_classes = [spec['dominant_class'] for spec in global_specs.values()]
    else:
        global_purities = []
        global_usages = []
        global_classes = []
    
    # Middle layer pathway data
    if middle_specs:
        middle_purities = [spec['purity'] for spec in middle_specs.values()]
        middle_usages = [spec['usage'] for spec in middle_specs.values()]
        middle_classes = [spec['dominant_class'] for spec in middle_specs.values()]
    else:
        middle_purities = []
        middle_usages = []
        middle_classes = []
    
    # 1. Purity comparison
    if global_purities:
        axes[0, 0].hist(global_purities, bins=15, alpha=0.7, label='Global', color='blue')
    if middle_purities:
        axes[0, 0].hist(middle_purities, bins=15, alpha=0.7, label=f'Middle Layer {middle_layer}', color='orange')
    axes[0, 0].set_xlabel('Pathway Purity')
    axes[0, 0].set_ylabel('Number of Pathways')
    axes[0, 0].set_title('Pathway Purity Distribution')
    axes[0, 0].legend()
    
    # 2. Usage comparison
    if global_usages:
        axes[0, 1].hist(global_usages, bins=15, alpha=0.7, label='Global', color='blue')
    if middle_usages:
        axes[0, 1].hist(middle_usages, bins=15, alpha=0.7, label=f'Middle Layer {middle_layer}', color='orange')
    axes[0, 1].set_xlabel('Usage Count')
    axes[0, 1].set_ylabel('Number of Pathways')
    axes[0, 1].set_title('Pathway Usage Distribution')
    axes[0, 1].legend()
    
    # 3. Average purity by approach
    global_avg = np.mean(global_purities) if global_purities else 0
    middle_avg = np.mean(middle_purities) if middle_purities else 0
    
    approaches = ['Global\n(All Layers)', f'Middle Layer\n(Layer {middle_layer})']
    avg_purities = [global_avg, middle_avg]
    
    axes[0, 2].bar(approaches, avg_purities, color=['blue', 'orange'], alpha=0.7)
    axes[0, 2].set_ylabel('Average Purity')
    axes[0, 2].set_title('Average Pathway Purity by Approach')
    
    # 4. High purity pathway count
    global_high = sum(1 for p in global_purities if p > 0.4)
    middle_high = sum(1 for p in middle_purities if p > 0.4)
    
    high_counts = [global_high, middle_high]
    axes[1, 0].bar(approaches, high_counts, color=['blue', 'orange'], alpha=0.7)
    axes[1, 0].set_ylabel('High Purity Pathways (>40%)')
    axes[1, 0].set_title('High Specialization Pathways')
    
    # 5. Pathway count comparison
    global_count = len(global_purities)
    middle_count = len(middle_purities)
    
    pathway_counts = [global_count, middle_count]
    axes[1, 1].bar(approaches, pathway_counts, color=['blue', 'orange'], alpha=0.7)
    axes[1, 1].set_ylabel('Active Pathways')
    axes[1, 1].set_title('Active Pathway Count')
    
    # 6. Usage vs Purity scatter
    if global_usages and global_purities:
        axes[1, 2].scatter(global_usages, global_purities, alpha=0.7, 
                          label='Global', color='blue', s=50)
    if middle_usages and middle_purities:
        axes[1, 2].scatter(middle_usages, middle_purities, alpha=0.7, 
                          label=f'Middle Layer {middle_layer}', color='orange', s=50)
    axes[1, 2].set_xlabel('Usage Count')
    axes[1, 2].set_ylabel('Purity')
    axes[1, 2].set_title('Usage vs Specialization')
    axes[1, 2].legend()
    
    plt.tight_layout()
    filename = f'five_layer_middle{middle_layer}_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved as '{filename}'")

def run_five_layer_middle_comparison(middle_layer=3):
    """Run comprehensive comparison: Global vs Middle Layer routing in 5-layer network"""
    print("🚀 5-LAYER COMPARISON: Global vs Middle Layer Routing")
    print("="*70)
    print(f"Testing Global (all layers) vs Middle Layer {middle_layer} routing")
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    
    # Use subset for faster training
    subset_indices = torch.randperm(len(train_dataset))[:15000]
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Step 1: Train baseline standard MLP
    print("\n🎯 Training Standard 5-Layer MLP...")
    standard_mlp, test_loader_baseline = train_standard_mlp_five_layer()
    standard_accuracy = evaluate_model(standard_mlp, test_loader_baseline)
    
    # Step 2: Create and train Global Pathway MLP
    print("\n🛤️ Creating Global Pathway MLP (5-Layer)...")
    global_pathway_mlp = GlobalPathwayMLP_FiveLayer(standard_mlp).to(device)
    
    print("\n🔄 Training Global Pathway MLP...")
    global_trainer = GlobalTrainer_FiveLayer(global_pathway_mlp)
    global_trainer.train(train_loader, epochs=3)
    
    global_accuracy = evaluate_model(global_pathway_mlp, test_loader)
    global_specializations = analyze_pathway_specialization(global_pathway_mlp, test_loader)
    
    # Step 3: Create and train Middle Layer Pathway MLP
    print(f"\n🔗 Creating Middle Layer Pathway MLP (Layer {middle_layer})...")
    middle_pathway_mlp = MiddleLayerPathwayMLP_FiveLayer(standard_mlp, middle_layer=middle_layer).to(device)
    
    print(f"\n🔄 Training Middle Layer Pathway MLP...")
    middle_trainer = MiddleLayerTrainer_FiveLayer(middle_pathway_mlp)
    middle_trainer.train(train_loader, epochs=3)
    
    middle_accuracy = evaluate_model(middle_pathway_mlp, test_loader)
    middle_specializations = analyze_pathway_specialization(middle_pathway_mlp, test_loader)
    
    # Step 4: Create visualization
    print("\n📊 Creating Visualization...")
    create_five_layer_visualization(global_specializations, middle_specializations, 
                                  middle_layer, f"5-Layer: Global vs Middle Layer {middle_layer} Comparison")
    
    # Step 5: Comprehensive comparison analysis
    print("\n🔬 COMPREHENSIVE 5-LAYER COMPARISON")
    print("="*70)
    
    # Calculate global pathway metrics
    if global_specializations:
        global_purities = [spec['purity'] for spec in global_specializations.values()]
        global_avg_purity = np.mean(global_purities)
        global_high_purity = sum(1 for p in global_purities if p > 0.4)
        global_perfect_specs = sum(1 for p in global_purities if p >= 0.99)
        global_active_pathways = len(global_specializations)
        global_total_pathways = global_pathway_mlp.num_pathways
    else:
        global_avg_purity = 0
        global_high_purity = 0
        global_perfect_specs = 0
        global_active_pathways = 0
        global_total_pathways = global_pathway_mlp.num_pathways
    
    # Calculate middle layer pathway metrics
    if middle_specializations:
        middle_purities = [spec['purity'] for spec in middle_specializations.values()]
        middle_avg_purity = np.mean(middle_purities)
        middle_high_purity = sum(1 for p in middle_purities if p > 0.4)
        middle_perfect_specs = sum(1 for p in middle_purities if p >= 0.99)
        middle_active_pathways = len(middle_specializations)
        middle_total_pathways = middle_pathway_mlp.config['input_groups'] * middle_pathway_mlp.config['output_groups']
    else:
        middle_avg_purity = 0
        middle_high_purity = 0
        middle_perfect_specs = 0
        middle_active_pathways = 0
        middle_total_pathways = middle_pathway_mlp.config['input_groups'] * middle_pathway_mlp.config['output_groups']
    
    # Print detailed comparison
    print(f"\n📊 5-LAYER ARCHITECTURAL COMPARISON:")
    print(f"{'Metric':<30} {'Global':<15} {f'Middle L{middle_layer}':<15} {'Winner':<10}")
    print("-" * 70)
    print(f"{'Total Pathways':<30} {global_total_pathways:<15} {middle_total_pathways:<15} {'Middle' if middle_total_pathways < global_total_pathways else 'Global':<10}")
    print(f"{'Active Pathways':<30} {global_active_pathways:<15} {middle_active_pathways:<15} {'Global' if global_active_pathways > middle_active_pathways else 'Middle':<10}")
    
    print(f"\n🎯 PERFORMANCE COMPARISON:")
    print(f"{'Metric':<30} {'Global':<15} {f'Middle L{middle_layer}':<15} {'Winner':<10}")
    print("-" * 70)
    print(f"{'Standard MLP Accuracy':<30} {standard_accuracy:.2f}%<15 {standard_accuracy:.2f}%<15 {'Tie':<10}")
    print(f"{'Pathway MLP Accuracy':<30} {global_accuracy:.2f}%<15 {middle_accuracy:.2f}%<15 {'Global' if global_accuracy > middle_accuracy else 'Middle':<10}")
    
    print(f"\n🔬 SPECIALIZATION COMPARISON:")
    print(f"{'Metric':<30} {'Global':<15} {f'Middle L{middle_layer}':<15} {'Winner':<10}")
    print("-" * 70)
    print(f"{'Average Purity':<30} {global_avg_purity:.3f}<15 {middle_avg_purity:.3f}<15 {'Global' if global_avg_purity > middle_avg_purity else 'Middle':<10}")
    print(f"{'High Purity (>40%)':<30} {global_high_purity}<15 {middle_high_purity}<15 {'Global' if global_high_purity > middle_high_purity else 'Middle':<10}")
    print(f"{'Perfect Specialists':<30} {global_perfect_specs}<15 {middle_perfect_specs}<15 {'Global' if global_perfect_specs > middle_perfect_specs else 'Middle':<10}")
    
    print(f"\n💡 5-LAYER INSIGHTS:")
    print(f"="*70)
    print(f"🏆 Architecture: Global ({global_total_pathways} pathways) vs Middle Layer {middle_layer} ({middle_total_pathways} pathways)")
    print(f"🎯 Specialization: Global ({global_avg_purity:.3f}) vs Middle Layer {middle_layer} ({middle_avg_purity:.3f})")
    print(f"📊 Performance: Global ({global_accuracy:.2f}%) vs Middle Layer {middle_layer} ({middle_accuracy:.2f}%)")
    
    if global_avg_purity > middle_avg_purity:
        print(f"🏅 GLOBAL APPROACH WINS on 5-layer specialization")
        advantage = global_avg_purity / middle_avg_purity if middle_avg_purity > 0 else float('inf')
        print(f"   Global advantage: {advantage:.1f}x better specialization")
    else:
        print(f"🏅 MIDDLE LAYER APPROACH WINS on 5-layer specialization")
        advantage = middle_avg_purity / global_avg_purity if global_avg_purity > 0 else float('inf')
        print(f"   Middle layer advantage: {advantage:.1f}x better specialization")
    
    efficiency_global = global_avg_purity / global_total_pathways
    efficiency_middle = middle_avg_purity / middle_total_pathways
    print(f"⚡ Efficiency: Global ({efficiency_global:.6f}) vs Middle ({efficiency_middle:.6f})")
    
    # Create summary
    results = {
        'standard_accuracy': standard_accuracy,
        'middle_layer': middle_layer,
        'global': {
            'accuracy': global_accuracy,
            'total_pathways': global_total_pathways,
            'active_pathways': global_active_pathways,
            'avg_purity': global_avg_purity,
            'high_purity_count': global_high_purity,
            'perfect_specialists': global_perfect_specs,
            'efficiency': efficiency_global
        },
        'middle': {
            'accuracy': middle_accuracy,
            'total_pathways': middle_total_pathways,
            'active_pathways': middle_active_pathways,
            'avg_purity': middle_avg_purity,
            'high_purity_count': middle_high_purity,
            'perfect_specialists': middle_perfect_specs,
            'efficiency': efficiency_middle
        }
    }
    
    return results

if __name__ == "__main__":
    # Run the 5-layer middle layer comparison experiment
    # Test with middle layer 3 (the center of the 5-layer network)
    results = run_five_layer_middle_comparison(middle_layer=3)