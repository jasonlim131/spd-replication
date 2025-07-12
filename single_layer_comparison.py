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

class StandardMLP_SingleLayer(nn.Module):
    """Baseline MLP with one hidden layer and GELU activation"""
    def __init__(self, input_size=784, hidden_size=512, num_classes=10, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GLBLPathwayMLP_SingleLayer(nn.Module):
    """Global Pathway MLP with one hidden layer"""
    def __init__(self, pretrained_mlp, config=None):
        super().__init__()
        
        # Configuration for single layer
        default_config = {
            'num_input_regions': 4,        # Spatial image regions
            'num_hidden_groups': 4,        # Hidden layer neuron groups  
            'num_output_groups': 4,        # Output class groups
            'momentum': 0.9,               # For global statistics
            'router_hidden_size': 128,     # Pathway router capacity
            'router_dropout': 0.1          # Pathway router dropout
        }
        self.config = {**default_config, **(config or {})}
        
        # Network dimensions
        self.input_dim = pretrained_mlp.fc1.in_features
        self.hidden_dim = pretrained_mlp.fc1.out_features
        self.output_dim = pretrained_mlp.fc2.out_features
        
        # Pathway configuration
        self.num_input_regions = self.config['num_input_regions']
        self.num_hidden_groups = self.config['num_hidden_groups']
        self.num_output_groups = self.config['num_output_groups']
        self.num_pathways = (self.num_input_regions * 
                           self.num_hidden_groups * 
                           self.num_output_groups)
        
        print(f"🧠 Global GLBL Single Layer Configuration:")
        print(f"   Input: {self.input_dim} → Hidden: {self.hidden_dim} → Output: {self.output_dim}")
        print(f"   Pathways: {self.num_input_regions}×{self.num_hidden_groups}×{self.num_output_groups} = {self.num_pathways}")
        
        # Copy pretrained weights
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        
        with torch.no_grad():
            self.fc1.weight.copy_(pretrained_mlp.fc1.weight)
            self.fc1.bias.copy_(pretrained_mlp.fc1.bias)
            self.fc2.weight.copy_(pretrained_mlp.fc2.weight)
            self.fc2.bias.copy_(pretrained_mlp.fc2.bias)
        
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
        self.register_buffer('update_count', torch.zeros(1, device=device))
        
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
        hidden_groups = self._create_neuron_groups(self.hidden_dim, self.num_hidden_groups)
        self.register_buffer('hidden_groups_indices', hidden_groups)
        
        # Output class groups
        output_groups = self._create_neuron_groups(self.output_dim, self.num_output_groups)
        self.register_buffer('output_groups_indices', output_groups)
        
        print(f"✅ Created pathway structure:")
        print(f"   Input regions: {input_groups.shape}")
        print(f"   Hidden groups: {hidden_groups.shape}")
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
    
    def select_pathways(self, pathway_scores, top_k=8, temperature=1.0):
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
    
    def forward(self, x, top_k=8, temperature=1.0, record_activations=False, true_labels=None):
        """Forward pass with pathway-based computation"""
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Step 1: Route inputs to pathways
        pathway_scores = self.pathway_router(x_flat)
        
        # Step 2: Compute GLBL loss
        glbl_loss, frequencies, scores = self.compute_glbl_loss(pathway_scores)
        self.last_glbl_loss = glbl_loss
        
        # Step 3: Select active pathways
        pathway_weights = self.select_pathways(pathway_scores, top_k, temperature)
        
        # Step 4: Compute pathway outputs
        output = torch.zeros(batch_size, self.output_dim, device=device)
        
        pathway_idx = 0
        for i in range(self.num_input_regions):
            for j in range(self.num_hidden_groups):
                for k in range(self.num_output_groups):
                    
                    # Get pathway activation weights
                    weights = pathway_weights[:, pathway_idx].unsqueeze(1)
                    
                    # Skip computation if pathway not used
                    if weights.sum() < 1e-6:
                        pathway_idx += 1
                        continue
                    
                    # Get pathway indices
                    input_indices = self.input_groups_indices[i]
                    hidden_indices = self.hidden_groups_indices[j]
                    output_indices = self.output_groups_indices[k]
                    
                    # Extract pathway-specific weights and inputs
                    input_subset = x_flat[:, input_indices]
                    
                    # Layer 1: input → hidden
                    W1_subset = self.fc1.weight[hidden_indices][:, input_indices]
                    b1_subset = self.fc1.bias[hidden_indices]
                    hidden_output = F.gelu(F.linear(input_subset, W1_subset, b1_subset))
                    
                    # Layer 2: hidden → output
                    W2_subset = self.fc2.weight[output_indices][:, hidden_indices]
                    b2_subset = self.fc2.bias[output_indices]
                    pathway_output = F.linear(hidden_output, W2_subset, b2_subset)
                    
                    # Accumulate weighted pathway output
                    output[:, output_indices] += pathway_output * weights
                    
                    # Record activations for analysis
                    if record_activations and true_labels is not None:
                        self._record_pathway_activations(
                            pathway_idx, i, j, k, weights, 
                            hidden_output, pathway_output, 
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
    
    def _record_pathway_activations(self, pathway_idx, i, j, k, weights, 
                                  hidden_output, pathway_output, true_labels, batch_size):
        """Record pathway activations for analysis"""
        pathway_name = f"Input{i}_Hidden{j}_Output{k}"
        
        for sample_idx in range(batch_size):
            if weights[sample_idx] > 1e-3:  # Only record active pathways
                self.pathway_activations[pathway_name].append({
                    'pathway_idx': pathway_idx,
                    'true_class': true_labels[sample_idx].item(),
                    'pathway_weight': weights[sample_idx].item(),
                    'hidden_activation': hidden_output[sample_idx].mean().item(),
                    'output_activation': pathway_output[sample_idx].max().item()
                })
    
    def get_pathway_analysis(self):
        """Get pathway analysis statistics"""
        if not self.glbl_stats['glbl_loss']:
            return {}
        
        analysis = {
            'avg_glbl_loss': np.mean(self.glbl_stats['glbl_loss']),
            'final_glbl_loss': self.glbl_stats['glbl_loss'][-1],
            'avg_pathway_entropy': np.mean(self.glbl_stats['pathway_usage_entropy']),
            'avg_active_pathways': np.mean(self.glbl_stats['active_pathways_per_batch']),
            'pathway_utilization': len(self.pathway_activations) / self.num_pathways,
            'global_frequencies': self.global_pathway_frequencies.cpu().numpy(),
            'global_scores': self.global_pathway_scores.cpu().numpy(),
            'update_count': self.update_count.item()
        }
        
        return analysis

class LayerwisePathwayMLP_SingleLayer(nn.Module):
    """Layerwise Pathway MLP with one hidden layer - separate routing per layer"""
    def __init__(self, pretrained_mlp, config=None):
        super().__init__()
        
        # Configuration for single layer layerwise routing
        default_config = {
            'layer_configs': [
                {'input_groups': 4, 'output_groups': 4},  # Layer 1: 16 pathways
                {'input_groups': 4, 'output_groups': 4},  # Layer 2: 16 pathways
            ],
            'momentum': 0.9,
            'temperature': 1.0
        }
        self.config = {**default_config, **(config or {})}
        
        # Copy pretrained weights
        self.fc1 = nn.Linear(pretrained_mlp.fc1.in_features, pretrained_mlp.fc1.out_features)
        self.fc2 = nn.Linear(pretrained_mlp.fc2.in_features, pretrained_mlp.fc2.out_features)
        
        with torch.no_grad():
            self.fc1.weight.copy_(pretrained_mlp.fc1.weight)
            self.fc1.bias.copy_(pretrained_mlp.fc1.bias)
            self.fc2.weight.copy_(pretrained_mlp.fc2.weight)
            self.fc2.bias.copy_(pretrained_mlp.fc2.bias)
        
        # Create separate routers for each layer
        self.routers = nn.ModuleList([
            nn.Linear(784, 16),   # Layer 1 router: input → hidden
            nn.Linear(512, 16),   # Layer 2 router: hidden → output
        ])
        
        # Total pathways: 16+16 = 32
        self.total_pathways = sum(cfg['input_groups'] * cfg['output_groups'] 
                                 for cfg in self.config['layer_configs'])
        
        # Create pathway decompositions for each layer
        self._create_layerwise_pathways()
        
        # Pathway activation tracking
        self.pathway_activations = defaultdict(list)
        
        print(f"🧠 Layerwise Single Layer Configuration:")
        print(f"   Input: {784} → Hidden: {512} → Output: {10}")
        print(f"   Layer 1 pathways: {self.config['layer_configs'][0]['input_groups']}×{self.config['layer_configs'][0]['output_groups']} = 16")
        print(f"   Layer 2 pathways: {self.config['layer_configs'][1]['input_groups']}×{self.config['layer_configs'][1]['output_groups']} = 16")
        print(f"   Total pathways: {self.total_pathways}")
    
    def _create_layerwise_pathways(self):
        """Create pathway decompositions for each layer"""
        self.pathway_indices = []
        
        # Layer 1: Input → Hidden
        input_groups = self._create_spatial_regions(28, 28, 4)
        hidden_groups = self._create_neuron_groups(512, 4)
        self.pathway_indices.append({
            'input_groups': input_groups,
            'output_groups': hidden_groups
        })
        
        # Layer 2: Hidden → Output
        hidden_groups_2 = self._create_neuron_groups(512, 4)
        output_groups = self._create_output_class_groups(10, 4)
        self.pathway_indices.append({
            'input_groups': hidden_groups_2,
            'output_groups': output_groups
        })
        
        print(f"✅ Created layerwise pathways:")
        print(f"   Layer 1: {input_groups.shape} → {hidden_groups.shape}")
        print(f"   Layer 2: {hidden_groups_2.shape} → {output_groups.shape}")
    
    def _create_spatial_regions(self, height, width, num_regions):
        """Create spatial regions for image input"""
        if num_regions == 4:
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
            
            indices = indices[:max_pixels_per_region]
            while len(indices) < max_pixels_per_region:
                indices.append(indices[0])
                
            groups[region_idx] = torch.tensor(indices, dtype=torch.long)
        
        return groups.to(device)
    
    def _create_neuron_groups(self, dim, num_groups):
        """Create neuron groups for hidden layers"""
        neurons_per_group = dim // num_groups
        groups = torch.zeros(num_groups, neurons_per_group, dtype=torch.long)
        
        for group_idx in range(num_groups):
            start_idx = group_idx * neurons_per_group
            end_idx = start_idx + neurons_per_group
            groups[group_idx] = torch.arange(start_idx, end_idx, dtype=torch.long)
        
        return groups.to(device)
    
    def _create_output_class_groups(self, num_classes, num_groups):
        """Create output class groups"""
        classes_per_group = max(1, num_classes // num_groups)
        groups = []
        
        for group_idx in range(num_groups):
            start_idx = group_idx * classes_per_group
            end_idx = min(start_idx + classes_per_group, num_classes)
            if start_idx < num_classes:
                group_classes = list(range(start_idx, end_idx))
                # Pad if needed
                while len(group_classes) < classes_per_group and len(group_classes) < num_classes:
                    group_classes.append(group_classes[-1])
                groups.append(torch.tensor(group_classes, dtype=torch.long))
        
        # Ensure we have exactly num_groups
        while len(groups) < num_groups:
            groups.append(torch.tensor([num_classes-1], dtype=torch.long))
        
        max_size = max(len(g) for g in groups)
        padded_groups = torch.zeros(num_groups, max_size, dtype=torch.long)
        
        for i, group in enumerate(groups):
            padded_groups[i, :len(group)] = group
            if len(group) < max_size:
                padded_groups[i, len(group):] = group[-1]  # Pad with last element
        
        return padded_groups.to(device)
    
    def forward(self, x, record_activations=False, true_labels=None):
        """Forward pass with layerwise routing"""
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Layer 1: Input → Hidden with routing
        layer1_scores = self.routers[0](x_flat)
        layer1_weights = F.softmax(layer1_scores, dim=-1)
        
        hidden_output = self._compute_layer_pathways(
            x_flat, layer1_weights, self.fc1, self.pathway_indices[0], 
            0, record_activations, true_labels, batch_size
        )
        hidden_output = F.gelu(hidden_output)
        
        # Layer 2: Hidden → Output with routing
        layer2_scores = self.routers[1](hidden_output)
        layer2_weights = F.softmax(layer2_scores, dim=-1)
        
        output = self._compute_layer_pathways(
            hidden_output, layer2_weights, self.fc2, self.pathway_indices[1], 
            1, record_activations, true_labels, batch_size
        )
        
        return output
    
    def _compute_layer_pathways(self, input_tensor, pathway_weights, layer, 
                               pathway_indices, layer_idx, record_activations, 
                               true_labels, batch_size):
        """Compute pathways for a single layer"""
        input_groups = pathway_indices['input_groups']
        output_groups = pathway_indices['output_groups']
        
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
                        self._record_layerwise_activations(
                            layer_idx, pathway_idx, weights, pathway_output, 
                            true_labels, batch_size
                        )
                
                pathway_idx += 1
        
        return output
    
    def _record_layerwise_activations(self, layer_idx, pathway_idx, weights, 
                                    pathway_output, true_labels, batch_size):
        """Record layerwise pathway activations"""
        pathway_name = f"Layer{layer_idx+1}_Pathway{pathway_idx}"
        
        for sample_idx in range(batch_size):
            if weights[sample_idx] > 1e-3:
                self.pathway_activations[pathway_name].append({
                    'layer_idx': layer_idx,
                    'pathway_idx': pathway_idx,
                    'true_class': true_labels[sample_idx].item(),
                    'pathway_weight': weights[sample_idx].item(),
                    'output_activation': pathway_output[sample_idx].max().item()
                })

class GLBLTrainer_SingleLayer:
    """Trainer for Global GLBL Single Layer model"""
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
    
    def train(self, train_loader, epochs=5, verbose=True):
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

class LayerwiseTrainer_SingleLayer:
    """Trainer for Layerwise Single Layer model"""
    def __init__(self, model, config=None):
        self.model = model
        default_config = {
            'lr': 0.001,
            'patience': 5
        }
        self.config = {**default_config, **(config or {})}
        
        # Separate optimizers for layers and routers
        layer_params = list(self.model.fc1.parameters()) + list(self.model.fc2.parameters())
        router_params = list(self.model.routers.parameters())
        
        self.layer_optimizer = torch.optim.Adam(layer_params, lr=self.config['lr'])
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
                
                self.layer_optimizer.zero_grad()
                self.router_optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data, record_activations=True, true_labels=target)
                
                # Compute loss
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.layer_optimizer.step()
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

def train_standard_mlp_single_layer(config=None):
    """Train a standard single layer MLP"""
    print("🏋️ Training Standard Single Layer MLP...")
    
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
    model = StandardMLP_SingleLayer().to(device)
    
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

def analyze_layerwise_specialization(model, test_loader):
    """Analyze specialization patterns in layerwise pathway model"""
    print(f"\n🎯 Analyzing Layerwise Pathway Specialization...")
    
    model.eval()
    model.pathway_activations.clear()
    
    # Record activations
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _ = model(data, record_activations=True, true_labels=target)
    
    # Analyze each layer's pathways
    layer_analyses = {}
    
    for layer_idx in range(len(model.routers)):
        layer_name = f"Layer{layer_idx+1}"
        layer_pathways = {}
        
        # Collect pathways for this layer
        for pathway_name, activations in model.pathway_activations.items():
            if pathway_name.startswith(layer_name):
                if len(activations) > 0:
                    layer_pathways[pathway_name] = activations
        
        # Analyze specialization for this layer
        layer_specializations = {}
        for pathway_name, activations in layer_pathways.items():
            if len(activations) >= 5:  # Minimum usage threshold
                class_counts = defaultdict(int)
                total_usage = len(activations)
                
                for activation in activations:
                    class_counts[activation['true_class']] += 1
                
                # Calculate purity (specialization to dominant class)
                if class_counts:
                    dominant_class = max(class_counts.keys(), key=lambda k: class_counts[k])
                    purity = class_counts[dominant_class] / total_usage
                    
                    layer_specializations[pathway_name] = {
                        'dominant_class': dominant_class,
                        'purity': purity,
                        'usage': total_usage,
                        'class_distribution': dict(class_counts)
                    }
        
        if layer_specializations:
            avg_purity = np.mean([spec['purity'] for spec in layer_specializations.values()])
            high_purity = sum(1 for spec in layer_specializations.values() if spec['purity'] > 0.4)
            
            layer_analyses[layer_name] = {
                'specializations': layer_specializations,
                'avg_purity': avg_purity,
                'high_purity_count': high_purity,
                'total_active': len(layer_specializations)
            }
            
            print(f"📊 {layer_name} Analysis:")
            print(f"   Active pathways: {len(layer_specializations)}")
            print(f"   Average purity: {avg_purity:.3f}")
            print(f"   High purity (>40%): {high_purity}/{len(layer_specializations)}")
    
    return layer_analyses

def create_single_layer_visualization(global_specs, layerwise_analyses, title="Single Layer Comparison"):
    """Create visualization comparing single layer results"""
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
    
    # Layerwise pathway data
    layerwise_purities = []
    layerwise_usages = []
    layerwise_classes = []
    
    if layerwise_analyses:
        for layer_data in layerwise_analyses.values():
            for spec in layer_data['specializations'].values():
                layerwise_purities.append(spec['purity'])
                layerwise_usages.append(spec['usage'])
                layerwise_classes.append(spec['dominant_class'])
    
    # 1. Purity comparison
    if global_purities:
        axes[0, 0].hist(global_purities, bins=15, alpha=0.7, label='Global', color='blue')
    if layerwise_purities:
        axes[0, 0].hist(layerwise_purities, bins=15, alpha=0.7, label='Layerwise', color='red')
    axes[0, 0].set_xlabel('Pathway Purity')
    axes[0, 0].set_ylabel('Number of Pathways')
    axes[0, 0].set_title('Pathway Purity Distribution')
    axes[0, 0].legend()
    
    # 2. Usage comparison
    if global_usages:
        axes[0, 1].hist(global_usages, bins=15, alpha=0.7, label='Global', color='blue')
    if layerwise_usages:
        axes[0, 1].hist(layerwise_usages, bins=15, alpha=0.7, label='Layerwise', color='red')
    axes[0, 1].set_xlabel('Usage Count')
    axes[0, 1].set_ylabel('Number of Pathways')
    axes[0, 1].set_title('Pathway Usage Distribution')
    axes[0, 1].legend()
    
    # 3. Average purity by approach
    global_avg = np.mean(global_purities) if global_purities else 0
    layerwise_avg = np.mean(layerwise_purities) if layerwise_purities else 0
    
    approaches = ['Global', 'Layerwise']
    avg_purities = [global_avg, layerwise_avg]
    
    axes[0, 2].bar(approaches, avg_purities, color=['blue', 'red'], alpha=0.7)
    axes[0, 2].set_ylabel('Average Purity')
    axes[0, 2].set_title('Average Pathway Purity by Approach')
    
    # 4. High purity pathway count
    global_high = sum(1 for p in global_purities if p > 0.4)
    layerwise_high = sum(1 for p in layerwise_purities if p > 0.4)
    
    high_counts = [global_high, layerwise_high]
    axes[1, 0].bar(approaches, high_counts, color=['blue', 'red'], alpha=0.7)
    axes[1, 0].set_ylabel('High Purity Pathways (>40%)')
    axes[1, 0].set_title('High Specialization Pathways')
    
    # 5. Pathway count comparison
    global_count = len(global_purities)
    layerwise_count = len(layerwise_purities)
    
    pathway_counts = [global_count, layerwise_count]
    axes[1, 1].bar(approaches, pathway_counts, color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_ylabel('Active Pathways')
    axes[1, 1].set_title('Active Pathway Count')
    
    # 6. Usage vs Purity scatter
    if global_usages and global_purities:
        axes[1, 2].scatter(global_usages, global_purities, alpha=0.7, 
                          label='Global', color='blue', s=50)
    if layerwise_usages and layerwise_purities:
        axes[1, 2].scatter(layerwise_usages, layerwise_purities, alpha=0.7, 
                          label='Layerwise', color='red', s=50)
    axes[1, 2].set_xlabel('Usage Count')
    axes[1, 2].set_ylabel('Purity')
    axes[1, 2].set_title('Usage vs Specialization')
    axes[1, 2].legend()
    
    plt.tight_layout()
    filename = 'single_layer_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved as '{filename}'")

def run_single_layer_comparison():
    """Run comprehensive comparison: Global vs Layerwise single layer"""
    print("🚀 SINGLE LAYER COMPARISON: Global vs Layerwise")
    print("="*70)
    
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
    print("\n🎯 Training Standard Single Layer MLP...")
    standard_mlp, test_loader_baseline = train_standard_mlp_single_layer()
    standard_accuracy = evaluate_model(standard_mlp, test_loader_baseline)
    
    # Step 2: Create and train Global Pathway MLP
    print("\n🛤️ Creating Global Pathway MLP...")
    global_pathway_mlp = GLBLPathwayMLP_SingleLayer(standard_mlp).to(device)
    
    print("\n🔄 Training Global Pathway MLP...")
    global_trainer = GLBLTrainer_SingleLayer(global_pathway_mlp)
    global_trainer.train(train_loader, epochs=3)
    
    global_accuracy = evaluate_model(global_pathway_mlp, test_loader)
    global_specializations = analyze_pathway_specialization(global_pathway_mlp, test_loader)
    
    # Step 3: Create and train Layerwise Pathway MLP
    print("\n🔗 Creating Layerwise Pathway MLP...")
    layerwise_pathway_mlp = LayerwisePathwayMLP_SingleLayer(standard_mlp).to(device)
    
    print("\n🔄 Training Layerwise Pathway MLP...")
    layerwise_trainer = LayerwiseTrainer_SingleLayer(layerwise_pathway_mlp)
    layerwise_trainer.train(train_loader, epochs=3)
    
    layerwise_accuracy = evaluate_model(layerwise_pathway_mlp, test_loader)
    layerwise_analyses = analyze_layerwise_specialization(layerwise_pathway_mlp, test_loader)
    
    # Step 4: Create visualization
    print("\n📊 Creating Visualization...")
    create_single_layer_visualization(global_specializations, layerwise_analyses, 
                                    "Single Layer: Global vs Layerwise Comparison")
    
    # Step 5: Comprehensive comparison analysis
    print("\n🔬 COMPREHENSIVE SINGLE LAYER COMPARISON")
    print("="*70)
    
    # Calculate global pathway metrics
    if global_specializations:
        global_purities = [spec['purity'] for spec in global_specializations.values()]
        global_avg_purity = np.mean(global_purities)
        global_high_purity = sum(1 for p in global_purities if p > 0.4)
        global_active_pathways = len(global_specializations)
        global_total_pathways = global_pathway_mlp.num_pathways
    else:
        global_avg_purity = 0
        global_high_purity = 0
        global_active_pathways = 0
        global_total_pathways = global_pathway_mlp.num_pathways
    
    # Calculate layerwise pathway metrics
    if layerwise_analyses:
        layerwise_all_purities = []
        layerwise_high_purity = 0
        layerwise_active_pathways = 0
        
        for layer_data in layerwise_analyses.values():
            layer_purities = [spec['purity'] for spec in layer_data['specializations'].values()]
            layerwise_all_purities.extend(layer_purities)
            layerwise_high_purity += layer_data['high_purity_count']
            layerwise_active_pathways += layer_data['total_active']
        
        layerwise_avg_purity = np.mean(layerwise_all_purities) if layerwise_all_purities else 0
        layerwise_total_pathways = layerwise_pathway_mlp.total_pathways
    else:
        layerwise_avg_purity = 0
        layerwise_high_purity = 0
        layerwise_active_pathways = 0
        layerwise_total_pathways = layerwise_pathway_mlp.total_pathways
    
    # Print detailed comparison
    print(f"\n📊 SINGLE LAYER ARCHITECTURAL COMPARISON:")
    print(f"{'Metric':<30} {'Global':<15} {'Layerwise':<15} {'Winner':<10}")
    print("-" * 70)
    print(f"{'Total Pathways':<30} {global_total_pathways:<15} {layerwise_total_pathways:<15} {'Layerwise' if layerwise_total_pathways < global_total_pathways else 'Global':<10}")
    print(f"{'Active Pathways':<30} {global_active_pathways:<15} {layerwise_active_pathways:<15} {'Global' if global_active_pathways > layerwise_active_pathways else 'Layerwise':<10}")
    
    print(f"\n🎯 PERFORMANCE COMPARISON:")
    print(f"{'Metric':<30} {'Global':<15} {'Layerwise':<15} {'Winner':<10}")
    print("-" * 70)
    print(f"{'Standard MLP Accuracy':<30} {standard_accuracy:.2f}%<15 {standard_accuracy:.2f}%<15 {'Tie':<10}")
    print(f"{'Pathway MLP Accuracy':<30} {global_accuracy:.2f}%<15 {layerwise_accuracy:.2f}%<15 {'Global' if global_accuracy > layerwise_accuracy else 'Layerwise':<10}")
    
    print(f"\n🔬 SPECIALIZATION COMPARISON:")
    print(f"{'Metric':<30} {'Global':<15} {'Layerwise':<15} {'Winner':<10}")
    print("-" * 70)
    print(f"{'Average Purity':<30} {global_avg_purity:.3f}<15 {layerwise_avg_purity:.3f}<15 {'Global' if global_avg_purity > layerwise_avg_purity else 'Layerwise':<10}")
    print(f"{'High Purity (>40%)':<30} {global_high_purity}<15 {layerwise_high_purity}<15 {'Global' if global_high_purity > layerwise_high_purity else 'Layerwise':<10}")
    
    print(f"\n💡 SINGLE LAYER INSIGHTS:")
    print(f"="*70)
    print(f"🏆 Architecture: Global ({global_total_pathways} pathways) vs Layerwise ({layerwise_total_pathways} pathways)")
    print(f"🎯 Specialization: Global ({global_avg_purity:.3f}) vs Layerwise ({layerwise_avg_purity:.3f})")
    print(f"📊 Performance: Global ({global_accuracy:.2f}%) vs Layerwise ({layerwise_accuracy:.2f}%)")
    
    if global_avg_purity > layerwise_avg_purity:
        print("🏅 GLOBAL APPROACH WINS on single layer specialization")
    else:
        print("🏅 LAYERWISE APPROACH WINS on single layer specialization")
    
    # Create summary
    results = {
        'standard_accuracy': standard_accuracy,
        'global': {
            'accuracy': global_accuracy,
            'total_pathways': global_total_pathways,
            'active_pathways': global_active_pathways,
            'avg_purity': global_avg_purity,
            'high_purity_count': global_high_purity,
        },
        'layerwise': {
            'accuracy': layerwise_accuracy,
            'total_pathways': layerwise_total_pathways,
            'active_pathways': layerwise_active_pathways,
            'avg_purity': layerwise_avg_purity,
            'high_purity_count': layerwise_high_purity,
        }
    }
    
    return results

if __name__ == "__main__":
    # Run the single layer comparison experiment
    results = run_single_layer_comparison()