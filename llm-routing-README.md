# LLM Layer-wise Routing with Monosemanticity Measurement

## Overview

This implementation (`llm-routing.py`) adapts the pathway-based routing mechanism from the original MLP routing code to work with Language Models (LLMs). It implements layer-wise routing on every other transformer layer and includes comprehensive monosemanticity measurement tools.

## Key Features

### 1. Layer-wise Routing Architecture

The implementation introduces routing at every other transformer layer (configurable via `route_every_n_layers`):

- **PathwayRouter**: A small neural network that assigns scores to different pathways based on the input hidden states
- **RoutedTransformerLayer**: Replaces standard transformer layers with pathway-routed versions
- **Pathway Decomposition**: Each MLP in the transformer is decomposed into multiple pathways, where each pathway processes a subset of hidden dimensions

### 2. Global Load Balancing (GLBL)

The implementation uses GLBL loss to ensure balanced pathway usage:
- Prevents pathway collapse where only a few pathways are used
- Encourages diverse pathway utilization
- Maintains global statistics with momentum updates

### 3. Monosemanticity Measurement

Monosemanticity is measured by analyzing how specialized each pathway becomes:

- **Token Specialization**: Tracks which tokens activate each pathway
- **Purity Metric**: Measures how concentrated a pathway's activations are on specific tokens
- **Entropy Analysis**: Uses Shannon entropy to quantify specialization
- **Visualization**: Creates comprehensive plots showing pathway specialization patterns

## Implementation Details

### Model Choice
- Uses **TinyStories-1M** model (roneneldan/TinyStories-1M) as it's one of the smallest available LLMs
- 1M parameters with 8 transformer layers
- Perfect for experimentation with limited computational resources

### Routing Configuration
```python
RoutingConfig(
    num_pathways=16,          # Number of pathways per routed layer
    top_k_pathways=4,         # Active pathways selected per forward pass
    router_hidden_size=128,   # Router network capacity
    route_every_n_layers=2    # Route every other layer
)
```

### How Routing Works

1. **Input Processing**: Hidden states from transformer layers are passed to the PathwayRouter
2. **Pathway Selection**: Router outputs scores for each pathway, top-k are selected
3. **Decomposed Computation**: Each selected pathway processes a subset of the MLP computation
4. **Weighted Aggregation**: Pathway outputs are weighted by their selection scores and combined

### Monosemanticity Metrics

1. **Purity**: `purity = 1 - normalized_entropy`
   - Measures how specialized a pathway is for specific tokens
   - Higher purity indicates stronger specialization

2. **Pathway Utilization**: Percentage of pathways that are actively used

3. **Token Frequency Analysis**: Tracks which tokens most frequently activate each pathway

## Usage

### Installation Requirements
```bash
pip install torch transformers numpy matplotlib scipy seaborn tqdm
```

### Running the Experiment
```python
python llm-routing.py
```

### What the Script Does

1. **Loads TinyStories-1M model**: A small GPT-2 style model trained on children's stories
2. **Creates Routed Version**: Replaces every other transformer layer with routed versions
3. **Trains with GLBL**: Fine-tunes the model with combined language modeling and GLBL losses
4. **Measures Monosemanticity**: Analyzes pathway specialization patterns
5. **Visualizes Results**: Creates plots showing:
   - Pathway purity across layers
   - Distribution of pathway specializations
   - Top specialized pathways and their preferred tokens
   - Pathway utilization rates

### Output Files

- `llm_monosemanticity.png`: Visualization of pathway specialization analysis
- `llm_routing_results.json`: Numerical results including average purity, utilization metrics

## Key Insights

### Advantages of Layer-wise Routing

1. **Interpretability**: Each pathway can specialize on specific linguistic features
2. **Efficiency**: Only top-k pathways are active, reducing computation
3. **Modularity**: Pathways can be analyzed and modified independently

### Monosemanticity Benefits

1. **Feature Discovery**: Pathways may specialize on:
   - Specific tokens or token types
   - Grammatical structures
   - Semantic categories

2. **Debugging**: Specialized pathways are easier to understand and debug

3. **Control**: Pathway activation can potentially be controlled for specific behaviors

## Extending the Implementation

### Possible Improvements

1. **Different Routing Strategies**:
   - Route attention layers as well as MLPs
   - Implement hierarchical routing across layers
   - Use learned routing patterns instead of top-k

2. **Enhanced Monosemanticity Measures**:
   - Analyze pathway specialization on syntactic features
   - Measure semantic coherence of pathway activations
   - Track pathway evolution during training

3. **Scaling to Larger Models**:
   - Adapt for models like Pythia-70M or Pythia-160M
   - Implement efficient routing for larger pathway counts
   - Optimize memory usage for bigger models

## Theoretical Background

The implementation is based on the hypothesis that neural networks can be decomposed into specialized pathways that handle different aspects of the computation. By encouraging this decomposition through routing and load balancing, we can:

1. Improve interpretability by having pathways that specialize on specific features
2. Potentially improve efficiency by only activating relevant pathways
3. Enable better analysis of how language models process information

The monosemanticity measurement helps quantify how well this decomposition works in practice, showing whether pathways indeed specialize on coherent linguistic features.