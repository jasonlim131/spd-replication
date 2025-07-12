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
    """Baseline MLP for comparison"""
    def __init__(self, input_size=784, hidden_size=512, num_classes=10, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
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
            'num_input_regions': 4,      # Spatial image regions
            'num_hidden_groups': 4,      # Hidden neuron groups  
            'num_output_groups': 4,      # Output class groups
            'momentum': 0.9,             # For global statistics
            'router_hidden_size': 256,   # Pathway router capacity
            'router_dropout': 0.1        # Pathway router dropout
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
        
        print(f"🧠 GLBL Pathway MLP Configuration:")
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
        
        # Hidden neuron groups
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
                    hidden_output = F.relu(F.linear(input_subset, W1_subset, b1_subset))
                    
                    # Layer 2: hidden → output
                    W2_subset = self.fc2.weight[output_indices][:, hidden_indices]
                    b2_subset = self.fc2.bias[output_indices]
                    pathway_output = F.linear(hidden_output, W2_subset, b2_subset)
                    
                    # Accumulate weighted pathway output
                    output[:, output_indices] += pathway_output * weights
                    
                    # Record activations for analysis
                    if record_activations and true_labels is not None:
                        self._record_pathway_activations(
                            pathway_idx, i, j, k, weights, hidden_output, 
                            pathway_output, true_labels, batch_size
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
            'top_k': 12,
            'temperature': 1.0,
            'batch_size': 128,
            'subset_size': 15000,
            'report_every': 2  # Report every N epochs
        }
        self.config = {**default_config, **(config or {})}
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate']
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'classification_loss': [],
            'glbl_loss': [],
            'total_loss': []
        }
        
    def _get_glbl_weight(self, epoch, max_epochs):
        """Get GLBL weight based on schedule"""
        if self.config['glbl_weight_schedule'] == 'linear':
            progress = epoch / max_epochs
            return (self.config['glbl_weight_start'] + 
                   progress * (self.config['glbl_weight_end'] - 
                              self.config['glbl_weight_start']))
        else:
            return self.config['glbl_weight_start']
    
    def _evaluate_accuracy(self, data_loader, max_batches=None):
        """Evaluate model accuracy on given data loader"""
        self.model.eval()
        correct = 0
        total = 0
        batch_count = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                if max_batches and batch_count >= max_batches:
                    break
                    
                data, target = data.to(device), target.to(device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                batch_count += 1
        
        self.model.train()
        return 100 * correct / total if total > 0 else 0.0
    
    def train(self, train_loader, test_loader=None, epochs=5, verbose=True):
        """Train the GLBL pathway MLP with enhanced reporting"""
        self.model.train()
        
        print(f"🚀 Starting GLBL Training for {epochs} epochs")
        print(f"📊 Reporting every {self.config['report_every']} epochs")
        print("=" * 60)
        
        for epoch in range(epochs):
            glbl_weight = self._get_glbl_weight(epoch, epochs)
            epoch_losses = {'classification': [], 'glbl': [], 'total': []}
            epoch_correct = 0
            epoch_total = 0
            
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
                
                # Track accuracy
                _, predicted = torch.max(output.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()
                
                # Batch-level verbose logging
                if verbose and batch_idx % 50 == 0:
                    batch_accuracy = 100 * (predicted == target).sum().item() / target.size(0)
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx:3d}: '
                          f'Loss={total_loss.item():.4f}, '
                          f'Acc={batch_accuracy:.1f}%')
            
            # Calculate epoch metrics
            avg_class_loss = np.mean(epoch_losses['classification'])
            avg_glbl_loss = np.mean(epoch_losses['glbl'])
            avg_total_loss = np.mean(epoch_losses['total'])
            train_accuracy = 100 * epoch_correct / epoch_total
            
            # Store training history
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['train_loss'].append(avg_total_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['classification_loss'].append(avg_class_loss)
            self.training_history['glbl_loss'].append(avg_glbl_loss)
            self.training_history['total_loss'].append(avg_total_loss)
            
            # Regular reporting
            if (epoch + 1) % self.config['report_every'] == 0 or epoch == epochs - 1:
                print(f"\n📊 EPOCH {epoch+1}/{epochs} REPORT:")
                print(f"   Training Loss: {avg_total_loss:.4f}")
                print(f"   Training Accuracy: {train_accuracy:.2f}%")
                print(f"   Classification Loss: {avg_class_loss:.4f}")
                print(f"   GLBL Loss (w={glbl_weight:.3f}): {avg_glbl_loss:.4f}")
                
                # Test accuracy if test loader provided
                if test_loader is not None:
                    test_accuracy = self._evaluate_accuracy(test_loader, max_batches=20)
                    print(f"   Test Accuracy: {test_accuracy:.2f}%")
                
                # Pathway statistics
                pathway_analysis = self.model.get_pathway_analysis()
                if pathway_analysis:
                    print(f"   Active Pathways: {pathway_analysis.get('avg_active_pathways', 0):.1f}")
                    print(f"   Pathway Entropy: {pathway_analysis.get('avg_pathway_entropy', 0):.3f}")
                
                print("-" * 60)
            
            # Brief epoch summary for non-reporting epochs
            elif verbose:
                print(f'Epoch {epoch+1}/{epochs}: '
                      f'Loss={avg_total_loss:.4f}, '
                      f'Acc={train_accuracy:.1f}%')
        
        print(f"\n✅ Training completed!")
        print(f"📈 Final Training Accuracy: {self.training_history['train_accuracy'][-1]:.2f}%")
        print(f"📉 Final Training Loss: {self.training_history['train_loss'][-1]:.4f}")
        
        return self.training_history

def train_standard_mlp(config=None):
    """Train baseline standard MLP with enhanced reporting"""
    default_config = {
        'epochs': 6,
        'learning_rate': 0.001,
        'batch_size': 256,
        'report_every': 2  # Report every N epochs
    }
    config = {**default_config, **(config or {})}
    
    print("🎯 Training Standard MLP Baseline...")
    print(f"📊 Reporting every {config['report_every']} epochs")
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
            # Brief epoch summary for non-reporting epochs
            print(f'Epoch {epoch+1}/{config["epochs"]}: '
                  f'Loss={avg_loss:.4f}, Acc={train_accuracy:.1f}%')
    
    # Final results
    final_accuracy = evaluate_model(model, test_loader)
    print(f'\n✅ Standard MLP Final Results:')
    print(f'📈 Final Training Accuracy: {training_history["train_accuracy"][-1]:.2f}%')
    print(f'📊 Final Test Accuracy: {final_accuracy:.2f}%')
    print(f'📉 Final Training Loss: {training_history["train_loss"][-1]:.4f}')
    
    return model, test_loader, training_history

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
    hidden_by_class = [[] for _ in range(10)]
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x_flat = data.view(data.size(0), -1)
            hidden = F.relu(model.fc1(x_flat))
            
            hidden_cpu = hidden.cpu().numpy()
            target_cpu = target.cpu().numpy()
            
            for i in range(len(target_cpu)):
                class_label = target_cpu[i]
                hidden_by_class[class_label].append(hidden_cpu[i])
    
    # Calculate selectivity scores
    neuron_scores = []
    for neuron_idx in range(model.fc1.out_features):
        class_means = []
        for class_idx in range(10):
            if hidden_by_class[class_idx]:
                activations = [h[neuron_idx] for h in hidden_by_class[class_idx]]
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

def create_training_plots(standard_history, glbl_history):
    """Create training loss and accuracy plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Comparison: Standard MLP vs GLBL Pathway MLP', fontsize=16)
    
    # Plot 1: Training Loss Comparison
    axes[0, 0].plot(standard_history['epoch'], standard_history['train_loss'], 
                    'b-', label='Standard MLP', linewidth=2)
    axes[0, 0].plot(glbl_history['epoch'], glbl_history['train_loss'], 
                    'r-', label='GLBL MLP', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy Comparison
    axes[0, 1].plot(standard_history['epoch'], standard_history['train_accuracy'], 
                    'b-', label='Standard MLP', linewidth=2)
    axes[0, 1].plot(glbl_history['epoch'], glbl_history['train_accuracy'], 
                    'r-', label='GLBL MLP', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Training Accuracy (%)')
    axes[0, 1].set_title('Training Accuracy Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: GLBL Loss Components
    axes[1, 0].plot(glbl_history['epoch'], glbl_history['classification_loss'], 
                    'g-', label='Classification Loss', linewidth=2)
    axes[1, 0].plot(glbl_history['epoch'], glbl_history['glbl_loss'], 
                    'orange', label='GLBL Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('GLBL Loss Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Test Accuracy (if available)
    if 'test_accuracy' in standard_history and len(standard_history['test_accuracy']) > 0:
        # Create x-axis for test accuracy (reported every few epochs)
        test_epochs_std = [standard_history['epoch'][i] for i in range(0, len(standard_history['epoch']), 2)]
        test_epochs_std = test_epochs_std[:len(standard_history['test_accuracy'])]
        
        axes[1, 1].plot(test_epochs_std, standard_history['test_accuracy'], 
                        'b-o', label='Standard MLP', linewidth=2, markersize=6)
        
        # For GLBL, we might have test accuracy from the reporting
        if hasattr(glbl_history, 'test_accuracy') and len(glbl_history.get('test_accuracy', [])) > 0:
            test_epochs_glbl = [glbl_history['epoch'][i] for i in range(0, len(glbl_history['epoch']), 2)]
            test_epochs_glbl = test_epochs_glbl[:len(glbl_history['test_accuracy'])]
            axes[1, 1].plot(test_epochs_glbl, glbl_history['test_accuracy'], 
                            'r-o', label='GLBL MLP', linewidth=2, markersize=6)
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Test Accuracy (%)')
        axes[1, 1].set_title('Test Accuracy Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # If no test accuracy, show training vs total loss for GLBL
        axes[1, 1].plot(glbl_history['epoch'], glbl_history['train_loss'], 
                        'r-', label='Training Loss', linewidth=2)
        axes[1, 1].plot(glbl_history['epoch'], glbl_history['total_loss'], 
                        'purple', label='Total Loss (with GLBL)', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('GLBL Total vs Training Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Training plots saved as 'training_comparison.png'")

def create_visualization(pathway_specializations, model_name="GLBL"):
    """Create comprehensive pathway analysis visualization"""
    if not pathway_specializations:
        print("⚠️ No specialization data to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} Pathway Analysis', fontsize=16)
    
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
    axes[1, 0].set_title('Input Region → Digit Specialization')
    axes[1, 0].set_xticks(range(10))
    axes[1, 0].set_yticks(range(4))
    axes[1, 0].set_yticklabels(['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right'])
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. Usage vs Specialization Scatter
    scatter = axes[1, 1].scatter(usages, purities, c=dominant_classes, 
                                cmap='tab10', alpha=0.7, s=50)
    axes[1, 1].set_xlabel('Usage Count')
    axes[1, 1].set_ylabel('Purity')
    axes[1, 1].set_title('Pathway Usage vs Specialization')
    plt.colorbar(scatter, ax=axes[1, 1], label='Dominant Class')
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_pathway_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved as '{model_name.lower()}_pathway_analysis.png'")

def run_complete_experiment():
    """Run the complete GLBL pathway experiment"""
    print("🚀 COMPLETE GLBL PATHWAY EXPERIMENT")
    print("=" * 70)
    
    # Step 1: Train standard MLP baseline
    standard_mlp, test_loader, standard_history = train_standard_mlp()
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
    
    # Train with GLBL (now with test loader for evaluation)
    trainer = GLBLTrainer(pathway_mlp)
    glbl_history = trainer.train(train_loader, test_loader, epochs=5)
    
    # Step 4: Evaluate performance
    pathway_accuracy = evaluate_model(pathway_mlp, test_loader)
    
    # Step 5: Analyze specialization
    pathway_specializations = analyze_pathway_specialization(pathway_mlp, test_loader)
    
    # Step 6: Create visualizations
    create_training_plots(standard_history, glbl_history)
    create_visualization(pathway_specializations, "GLBL")
    
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
        'pathway_analysis': pathway_analysis,
        'standard_history': standard_history,
        'glbl_history': glbl_history
    }

if __name__ == "__main__":
    # Run the complete experiment
    results = run_complete_experiment()