# Single Layer Comparison: Global vs Layerwise Pathway Routing

## Executive Summary

Successfully completed comparison experiment between Global and Layerwise pathway routing approaches using a single hidden layer MLP architecture (784→512→10). The results demonstrate clear advantages of the Global approach for single layer specialization.

## Architecture Comparison

### Standard MLP Baseline
- **Architecture**: Input: 784 → Hidden: 512 → Output: 10
- **Activation**: GELU
- **Training**: 5 epochs on 15,000 MNIST samples
- **Performance**: **96.45% accuracy**

### Global Pathway MLP
- **Architecture**: 4×4×4 = **64 pathways**
- **Routing**: Single global router with end-to-end pathways
- **GLBL Loss**: Global Load Balancing for pathway diversity
- **Performance**: **84.66% accuracy** (11.8% drop from baseline)

### Layerwise Pathway MLP
- **Architecture**: 16+16 = **32 pathways** (2 separate layer routers)
- **Routing**: Separate routing decisions per layer
- **Layer 1**: 4×4 = 16 pathways (Input → Hidden)
- **Layer 2**: 4×4 = 16 pathways (Hidden → Output)
- **Performance**: **82.94% accuracy** (13.5% drop from baseline)

## Key Findings

### 1. Pathway Specialization Quality
- **Global**: Average purity **0.459** (45.9% specialization)
- **Layerwise**: Average purity **0.168** (16.8% specialization)
- **Winner**: **Global approach** (2.7x better specialization)

### 2. High Purity Pathways
- **Global**: **31 out of 57** active pathways >40% purity
- **Layerwise**: **0 out of 32** active pathways >40% purity
- **Winner**: **Global approach** (complete dominance)

### 3. Pathway Utilization
- **Global**: 57/64 pathways active (89.1% utilization)
- **Layerwise**: 32/32 pathways active (100% utilization)
- **Winner**: **Layerwise approach** (better resource utilization)

### 4. Computational Efficiency
- **Global**: 64 total pathways
- **Layerwise**: 32 total pathways (50% fewer)
- **Winner**: **Layerwise approach** (2x more efficient)

## Top Specialized Pathways (Global)

1. **Input3_Hidden0_Output1**: 90.2% purity → digit 0 (92 uses)
2. **Input1_Hidden2_Output3**: 85.7% purity → digit 2 (7 uses)
3. **Input3_Hidden0_Output0**: 85.3% purity → digit 3 (34 uses)
4. **Input0_Hidden1_Output2**: 83.1% purity → digit 0 (59 uses)
5. **Input3_Hidden1_Output0**: 82.4% purity → digit 1 (187 uses)

## Layer-wise Analysis (Layerwise)

### Layer 1 (Input → Hidden)
- **Active pathways**: 16
- **Average purity**: 0.141 (14.1%)
- **High purity (>40%)**: 0/16

### Layer 2 (Hidden → Output)
- **Active pathways**: 16
- **Average purity**: 0.194 (19.4%)
- **High purity (>40%)**: 0/16

## Performance Trade-offs

### Single Layer Context
- **Global**: Better specialization but more complex (64 pathways)
- **Layerwise**: More efficient but weaker specialization (32 pathways)
- **Performance gap**: Global outperforms Layerwise by 1.7% accuracy

### Key Insights
1. **Global dominance**: For single layer MLPs, global routing achieves significantly better pathway specialization
2. **Efficiency trade-off**: Layerwise uses 50% fewer pathways but achieves much lower specialization
3. **Semantic richness**: End-to-end global pathways capture more meaningful digit-specific patterns
4. **Layer granularity**: Single layer may be too simple for layerwise routing to show advantages

## Comparison with Multi-layer Results

### Historical Context
- **5-layer Global**: 0.540 average purity (512 pathways)
- **5-layer Layerwise**: Not previously tested
- **1-layer Global**: 0.459 average purity (64 pathways)
- **1-layer Layerwise**: 0.168 average purity (32 pathways)

### Scaling Pattern
- Global approach maintains strong specialization across architectures
- Layerwise approach may require deeper architectures to show advantages
- Single layer provides insufficient complexity for layerwise benefits

## Recommendations

### For Single Layer MLPs
- **Use Global approach** when specialization quality is priority
- **Use Layerwise approach** when computational efficiency is critical
- **Consider hybrid** approaches for balanced performance

### For Future Research
1. Test layerwise approach on deeper architectures (3-5 layers)
2. Investigate optimal pathway allocation per layer
3. Explore adaptive routing mechanisms
4. Study specialization emergence patterns

## Conclusion

The single layer comparison experiment clearly demonstrates that **Global pathway routing outperforms Layerwise routing for single hidden layer architectures**. While the Layerwise approach offers computational efficiency (50% fewer pathways), the Global approach achieves superior pathway specialization (2.7x better) and slightly better accuracy performance.

This suggests that the benefits of layerwise routing may emerge more prominently in deeper architectures where layer-specific specialization patterns can develop more naturally.

## Files Generated
- `single_layer_comparison.py` - Complete experimental implementation
- `single_layer_comparison.png` - Comprehensive visualization of results
- `single_layer_comparison_results.md` - This summary document

---
**Experiment Date**: December 2024  
**Architecture**: 784→512→10 Single Layer MLP  
**Dataset**: MNIST (15,000 training samples)  
**Paradigm**: Global vs Layerwise Pathway Routing