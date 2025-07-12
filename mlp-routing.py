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

class StandardMLP(nn.Module):
    """Baseline MLP for comparison with five hidden layers and GELU activation"""
    def __init__(self, input_size=784, hidden_size1=512, hidden_size2=256, hidden_size3=128, hidden_size4=64, hidden_size5=32, num_classes=10, dropout=0.2):
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

class GLBLPathwayMLP(nn.Module):
    """
    Global Load Balancing (GLBL) Pathway MLP for Computational Monosemanticity
    
    Key Features:
    - Decomposes MLP into semantic pathways
    - Uses GLBL loss to prevent pathway collapse
    - Enables interpretable information flow control
    """
    
    def __init__(self, pretrained_mlp, config=None):
        super().__init__()
        
        # Default configuration  
        default_config = {
            'num_input_regions': 4,        # Spatial image regions
            'num_hidden1_groups': 2,       # First hidden layer neuron groups  
            'num_hidden2_groups': 2,       # Second hidden layer neuron groups
            'num_hidden3_groups': 2,       # Third hidden layer neuron groups
            'num_hidden4_groups': 2,       # Fourth hidden layer neuron groups
            'num_hidden5_groups': 2,       # Fifth hidden layer neuron groups
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
        
        print(f"🧠 GLBL Pathway MLP Configuration:")
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
                           torch.zeros(self.num_pathways, device=device))
        self.register_buffer('global_pathway_scores', 
                           torch.zeros(self.num_pathways, device=device))
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
        
        print(f"✅ Created pathway structure:")
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
        """
        Compute Global Load Balancing Loss
        
        GLBL = N_E * Σ(f̄_i * P̄_i)
        where:
        - f̄_i = global frequency of pathway i being selected
        - P̄_i = average routing score for pathway i
        - N_E = number of pathways (normalization factor)
        """
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
        """
        Smart pathway selection with load balancing awareness
        """
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
        """
        Forward pass with pathway-based computation
        """
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

class GLBLTrainer:
    """Trainer class for GLBL Pathway MLP"""
    
    def __init__(self, model, config=None):
        self.model = model
        
        default_config = {
            'learning_rate': 0.0005,
            'glbl_weight_start': 0.01,
            'glbl_weight_end': 0.05,
            'glbl_weight_schedule': 'linear',
            'top_k': 16,  # Optimized for 256 pathways (4×2×2×2×2×2×4)
            'temperature': 1.0,
            'batch_size': 128,
            'subset_size': 15000
        }
        self.config = {**default_config, **(config or {})}
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate']
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def _get_glbl_weight(self, epoch, max_epochs):
        """Get GLBL weight based on schedule"""
        if self.config['glbl_weight_schedule'] == 'linear':
            progress = epoch / max_epochs
            return (self.config['glbl_weight_start'] + 
                   progress * (self.config['glbl_weight_end'] - 
                              self.config['glbl_weight_start']))
        else:
            return self.config['glbl_weight_start']
    
    def train(self, train_loader, epochs=5, verbose=True):
        """Train the GLBL pathway MLP"""
        self.model.train()
        
        for epoch in range(epochs):
            glbl_weight = self._get_glbl_weight(epoch, epochs)
            epoch_losses = {'classification': [], 'glbl': [], 'total': []}
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(
                    data, 
                    top_k=self.config['top_k'],
                    temperature=self.config['temperature'],
                    record_activations=True,
                    true_labels=target
                )
                
                # Compute losses
                classification_loss = self.criterion(output, target)
                glbl_loss = self.model.last_glbl_loss
                total_loss = classification_loss + glbl_weight * glbl_loss
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                # Record losses
                epoch_losses['classification'].append(classification_loss.item())
                epoch_losses['glbl'].append(glbl_loss.item())
                epoch_losses['total'].append(total_loss.item())
                
                # Verbose logging
                if verbose and batch_idx % 20 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}:')
                    print(f'  Classification: {classification_loss.item():.4f}')
                    print(f'  GLBL (w={glbl_weight:.3f}): {glbl_loss.item():.4f}')
                    print(f'  Total: {total_loss.item():.4f}')
            
            # Epoch summary
            if verbose:
                avg_class_loss = np.mean(epoch_losses['classification'])
                avg_glbl_loss = np.mean(epoch_losses['glbl'])
                avg_total_loss = np.mean(epoch_losses['total'])
                print(f'Epoch {epoch} Summary:')
                print(f'  Avg Classification: {avg_class_loss:.4f}')
                print(f'  Avg GLBL: {avg_glbl_loss:.4f}')
                print(f'  Avg Total: {avg_total_loss:.4f}')
                print()

def train_standard_mlp(config=None):
    """Train baseline standard MLP"""
    default_config = {
        'epochs': 6,
        'learning_rate': 0.001,
        'batch_size': 256
    }
    config = {**default_config, **(config or {})}
    
    print("🎯 Training Standard MLP Baseline...")
    
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
    
    model.train()
    for epoch in range(config['epochs']):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    # Test accuracy
    accuracy = evaluate_model(model, test_loader)
    print(f'✅ Standard MLP Accuracy: {accuracy:.2f}%')
    
    return model, test_loader

def evaluate_model(model, test_loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total

def measure_neuron_selectivity(model, test_loader, model_name="Standard"):
    """Measure neuron selectivity for baseline comparison"""
    print(f"\n🔬 Measuring {model_name} Neuron Selectivity...")
    
    model.eval()
    hidden_layers_by_class = [[] for _ in range(10)]
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x_flat = data.view(data.size(0), -1)
            
            # Forward pass through all hidden layers
            h1 = F.gelu(model.fc1(x_flat))
            h2 = F.gelu(model.fc2(h1))
            h3 = F.gelu(model.fc3(h2))
            h4 = F.gelu(model.fc4(h3))
            h5 = F.gelu(model.fc5(h4))
            
            # Concatenate all hidden activations
            all_hidden = torch.cat([h1, h2, h3, h4, h5], dim=1)
            all_hidden_cpu = all_hidden.cpu().numpy()
            target_cpu = target.cpu().numpy()
            
            for i in range(len(target_cpu)):
                class_label = target_cpu[i]
                hidden_layers_by_class[class_label].append(all_hidden_cpu[i])
    
    # Calculate selectivity scores for all neurons across all hidden layers
    total_neurons = model.fc1.out_features + model.fc2.out_features + model.fc3.out_features + model.fc4.out_features + model.fc5.out_features
    neuron_scores = []
    
    for neuron_idx in range(total_neurons):
        class_means = []
        for class_idx in range(10):
            if hidden_layers_by_class[class_idx]:
                activations = [h[neuron_idx] for h in hidden_layers_by_class[class_idx]]
                class_means.append(np.mean(activations))
            else:
                class_means.append(0.0)
        
        class_means = np.array(class_means)
        probs = class_means + 1e-8
        probs = probs / probs.sum()
        
        neuron_entropy = entropy(probs)
        max_entropy = entropy(np.ones(10) / 10)
        selectivity = 1 - (neuron_entropy / max_entropy)
        neuron_scores.append(selectivity)
    
    avg_selectivity = np.mean(neuron_scores)
    highly_selective = np.sum(np.array(neuron_scores) > 0.5)
    
    print(f"📊 {model_name} Neuron Selectivity:")
    print(f"   Average selectivity: {avg_selectivity:.3f}")
    print(f"   Highly selective neurons (>0.5): {highly_selective}/{len(neuron_scores)}")
    print(f"   Total neurons: {total_neurons} (512+256+128+64+32)")
    
    return neuron_scores

def analyze_pathway_specialization(pathway_mlp, test_loader, max_samples=2000):
    """Comprehensive pathway specialization analysis"""
    print(f"\n🎯 Analyzing Pathway Specialization...")
    
    # Clear previous activations
    pathway_mlp.pathway_activations.clear()
    
    # Collect pathway activations
    pathway_mlp.eval()
    samples_processed = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            if samples_processed >= max_samples:
                break
                
            data, target = data.to(device), target.to(device)
            remaining_samples = min(len(data), max_samples - samples_processed)
            
            if remaining_samples < len(data):
                data = data[:remaining_samples]
                target = target[:remaining_samples]
            
            _ = pathway_mlp(data, record_activations=True, true_labels=target)
            samples_processed += len(data)
    
    # Analyze specialization
    specializations = {}
    for pathway_name, activations in pathway_mlp.pathway_activations.items():
        if len(activations) > 5:  # Minimum usage threshold
            classes = [act['true_class'] for act in activations]
            class_counts = np.bincount(classes, minlength=10)
            class_dist = class_counts / class_counts.sum()
            
            purity = np.max(class_dist)
            dominant_class = np.argmax(class_dist)
            
            specializations[pathway_name] = {
                'purity': purity,
                'dominant_class': dominant_class,
                'usage': len(activations),
                'avg_weight': np.mean([act['pathway_weight'] for act in activations]),
                'class_distribution': dict(enumerate(class_counts))
            }
    
    # Results summary
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

def create_visualization(pathway_specializations, model_name="GLBL_5Layer"):
    """Create comprehensive pathway analysis visualization"""
    if not pathway_specializations:
        print("⚠️ No specialization data to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} Pathway Analysis (5-Layer GELU)', fontsize=16)
    
    # Extract data
    purities = [stats['purity'] for stats in pathway_specializations.values()]
    usages = [stats['usage'] for stats in pathway_specializations.values()]
    dominant_classes = [stats['dominant_class'] for stats in pathway_specializations.values()]
    
    # 1. Pathway Purity Distribution
    axes[0, 0].hist(purities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(purities), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(purities):.3f}')
    axes[0, 0].set_xlabel('Pathway Purity')
    axes[0, 0].set_ylabel('Number of Pathways')
    axes[0, 0].set_title('Distribution of Pathway Purities')
    axes[0, 0].legend()
    
    # 2. Pathway Usage Distribution
    axes[0, 1].hist(usages, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(np.mean(usages), color='red', linestyle='--',
                       label=f'Mean: {np.mean(usages):.1f}')
    axes[0, 1].set_xlabel('Usage Count')
    axes[0, 1].set_ylabel('Number of Pathways')
    axes[0, 1].set_title('Distribution of Pathway Usage')
    axes[0, 1].legend()
    
    # 3. Input Region Specialization Heatmap
    region_class_matrix = np.zeros((4, 10))
    
    for pathway_name, stats in pathway_specializations.items():
        if 'Input' in pathway_name:
            # Handle new format: Input0_H1j_H2k_H3l_H4m_H5n_Output o
            region_idx = int(pathway_name.split('_')[0].replace('Input', ''))
            for class_idx, count in stats['class_distribution'].items():
                region_class_matrix[region_idx, class_idx] += count * stats['purity']
    
    # Normalize by row
    for i in range(4):
        row_sum = region_class_matrix[i].sum()
        if row_sum > 0:
            region_class_matrix[i] /= row_sum
    
    im = axes[1, 0].imshow(region_class_matrix, cmap='YlOrRd', aspect='auto')
    axes[1, 0].set_xlabel('Digit Class')
    axes[1, 0].set_ylabel('Input Region')
    axes[1, 0].set_title('Input Region → Digit Specialization (5-Layer)')
    axes[1, 0].set_xticks(range(10))
    axes[1, 0].set_yticks(range(4))
    axes[1, 0].set_yticklabels(['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right'])
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. Usage vs Specialization Scatter
    scatter = axes[1, 1].scatter(usages, purities, c=dominant_classes, 
                                cmap='tab10', alpha=0.7, s=50)
    axes[1, 1].set_xlabel('Usage Count')
    axes[1, 1].set_ylabel('Purity')
    axes[1, 1].set_title('Pathway Usage vs Specialization (5-Layer)')
    plt.colorbar(scatter, ax=axes[1, 1], label='Dominant Class')
    
    plt.tight_layout()
    # Unique filename for 5-layer results
    filename = f'{model_name.lower()}_5layer_gelu_pathway_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved as '{filename}'")

def run_complete_experiment():
    """Run the complete GLBL pathway experiment"""
    print("🚀 COMPLETE GLBL PATHWAY EXPERIMENT")
    print("=" * 70)
    
    # Step 1: Train standard MLP baseline
    standard_mlp, test_loader = train_standard_mlp()
    standard_selectivity = measure_neuron_selectivity(standard_mlp, test_loader)
    
    # Step 2: Create GLBL pathway MLP
    print(f"\n🛤️ Creating GLBL Pathway MLP...")
    pathway_mlp = GLBLPathwayMLP(standard_mlp).to(device)
    
    # Step 3: Train GLBL pathway MLP
    print(f"\n🔄 Training GLBL Pathway MLP...")
    
    # Create training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST('./data', train=True, transform=transform)
    subset_indices = torch.randperm(len(train_dataset))[:15000]
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    
    # Train with GLBL
    trainer = GLBLTrainer(pathway_mlp)
    trainer.train(train_loader, epochs=3)  # Reduced for faster 5-layer execution
    
    # Step 4: Evaluate performance
    pathway_accuracy = evaluate_model(pathway_mlp, test_loader)
    
    # Step 5: Analyze specialization
    pathway_specializations = analyze_pathway_specialization(pathway_mlp, test_loader)
    
    # Step 6: Create visualizations
    create_visualization(pathway_specializations, "GLBL_5Layer")
    
    # Step 7: Final results
    print(f"\n🎉 EXPERIMENT RESULTS:")
    print("=" * 50)
    print(f"✅ Standard MLP accuracy: {evaluate_model(standard_mlp, test_loader):.2f}%")
    print(f"✅ GLBL Pathway MLP accuracy: {pathway_accuracy:.2f}%")
    print(f"📊 Standard neuron selectivity: {np.mean(standard_selectivity):.3f}")
    
    if pathway_specializations:
        avg_pathway_purity = np.mean([s['purity'] for s in pathway_specializations.values()])
        improvement = avg_pathway_purity / np.mean(standard_selectivity)
        print(f"🎯 GLBL pathway purity: {avg_pathway_purity:.3f}")
        print(f"🚀 Improvement: {improvement:.1f}x better specialization!")
    
    # Step 8: Analysis summary
    pathway_analysis = pathway_mlp.get_pathway_analysis()
    print(f"\n📈 Training Analysis:")
    print(f"   Final GLBL loss: {pathway_analysis.get('final_glbl_loss', 0):.4f}")
    print(f"   Pathway utilization: {pathway_analysis.get('pathway_utilization', 0):.1%}")
    print(f"   Avg active pathways: {pathway_analysis.get('avg_active_pathways', 0):.1f}")
    
    return {
        'standard_mlp': standard_mlp,
        'pathway_mlp': pathway_mlp,
        'standard_selectivity': standard_selectivity,
        'pathway_specializations': pathway_specializations,
        'pathway_analysis': pathway_analysis
    }

if __name__ == "__main__":
    # Run the complete experiment
    results = run_complete_experiment()