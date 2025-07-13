"""
Simplified demonstration of LLM routing concepts
This script shows the key ideas without requiring GPU or large dependencies
"""

from collections import Counter
import json
import math
import random

def calculate_pathway_purity(token_activations):
    """
    Calculate the purity (monosemanticity) of a pathway based on token activations
    
    Args:
        token_activations: List of token IDs that activated this pathway
    
    Returns:
        purity: Float between 0 and 1, where 1 means perfectly monosemantic
    """
    if not token_activations:
        return 0.0
    
    # Count token frequencies
    token_counts = Counter(token_activations)
    total_tokens = len(token_activations)
    
    # Calculate probability distribution
    token_probs = [count / total_tokens for count in token_counts.values()]
    
    # Calculate entropy
    # Add small epsilon to avoid log(0)
    entropy = -sum(p * math.log(p + 1e-10) for p in token_probs)
    
    # Maximum possible entropy (uniform distribution)
    max_entropy = math.log(len(token_counts)) if len(token_counts) > 1 else 0
    
    # Normalize entropy and convert to purity
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    purity = 1 - normalized_entropy
    
    return purity

def simulate_pathway_activations(num_pathways=16, num_tokens=1000, specialization_strength=0.8):
    """
    Simulate pathway activations to demonstrate monosemanticity measurement
    
    Args:
        num_pathways: Number of pathways in the system
        num_tokens: Number of token activations to simulate
        specialization_strength: How specialized pathways are (0=random, 1=perfect)
    
    Returns:
        pathway_stats: Dictionary with pathway statistics
    """
    # Create a simple token vocabulary
    vocab_size = 100
    
    # Simulate different pathway specializations
    pathway_activations = {}
    
    for pathway_id in range(num_pathways):
        activations = []
        
        if pathway_id < num_pathways // 2:
            # Make half the pathways specialized
            # Each pathway specializes on a different set of tokens
            preferred_tokens = random.sample(range(vocab_size), 5)
            
            for _ in range(num_tokens):
                if random.random() < specialization_strength:
                    # Activate on preferred tokens
                    token = random.choice(preferred_tokens)
                else:
                    # Random activation
                    token = random.randint(0, vocab_size - 1)
                activations.append(token)
        else:
            # Make other half less specialized
            for _ in range(num_tokens):
                token = random.randint(0, vocab_size - 1)
                activations.append(token)
        
        pathway_activations[f"pathway_{pathway_id}"] = activations
    
    # Calculate statistics
    pathway_stats = {}
    for pathway_name, activations in pathway_activations.items():
        purity = calculate_pathway_purity(activations)
        token_counts = Counter(activations)
        top_tokens = token_counts.most_common(5)
        
        pathway_stats[pathway_name] = {
            'purity': purity,
            'total_activations': len(activations),
            'unique_tokens': len(token_counts),
            'top_tokens': [
                {'token_id': token_id, 'count': count, 'frequency': count/len(activations)}
                for token_id, count in top_tokens
            ]
        }
    
    return pathway_stats

def demonstrate_routing_concept():
    """Demonstrate the key concepts of pathway routing and monosemanticity"""
    
    print("🚀 LLM PATHWAY ROUTING CONCEPT DEMONSTRATION")
    print("=" * 60)
    
    # 1. Show pathway decomposition concept
    print("\n1. PATHWAY DECOMPOSITION")
    print("-" * 30)
    print("Original MLP: 768 hidden units → 3072 intermediate → 768 output")
    print("With 16 pathways:")
    print("  Each pathway: 48 hidden → 192 intermediate → 48 output")
    print("  Only 4 pathways active at a time (75% sparsity)")
    
    # 2. Simulate pathway activations
    print("\n2. SIMULATING PATHWAY ACTIVATIONS")
    print("-" * 30)
    
    # High specialization
    print("\nHigh Specialization (strength=0.9):")
    high_spec_stats = simulate_pathway_activations(
        num_pathways=8, num_tokens=500, specialization_strength=0.9
    )
    
    purities = [stats['purity'] for stats in high_spec_stats.values()]
    print(f"  Average purity: {sum(purities) / len(purities):.3f}")
    print(f"  Max purity: {max(purities):.3f}")
    
    # Low specialization
    print("\nLow Specialization (strength=0.3):")
    low_spec_stats = simulate_pathway_activations(
        num_pathways=8, num_tokens=500, specialization_strength=0.3
    )
    
    purities = [stats['purity'] for stats in low_spec_stats.values()]
    print(f"  Average purity: {sum(purities) / len(purities):.3f}")
    print(f"  Max purity: {max(purities):.3f}")
    
    # 3. Show example of specialized pathways
    print("\n3. EXAMPLE SPECIALIZED PATHWAYS")
    print("-" * 30)
    
    # Find most specialized pathways
    sorted_pathways = sorted(
        high_spec_stats.items(), 
        key=lambda x: x[1]['purity'], 
        reverse=True
    )[:3]
    
    for pathway_name, stats in sorted_pathways:
        print(f"\n{pathway_name} (purity={stats['purity']:.3f}):")
        print(f"  Total activations: {stats['total_activations']}")
        print(f"  Unique tokens: {stats['unique_tokens']}")
        print("  Top tokens:")
        for token_info in stats['top_tokens'][:3]:
            print(f"    Token {token_info['token_id']}: "
                  f"{token_info['count']} times ({token_info['frequency']:.2%})")
    
    # 4. Global Load Balancing concept
    print("\n4. GLOBAL LOAD BALANCING (GLBL)")
    print("-" * 30)
    print("GLBL Loss = N_pathways × Σ(frequency_i × score_i)")
    print("This encourages:")
    print("  - Balanced pathway usage")
    print("  - Prevents collapse to few pathways")
    print("  - Maintains diversity in routing")
    
    # 5. Benefits summary
    print("\n5. KEY BENEFITS")
    print("-" * 30)
    print("✓ Interpretability: Specialized pathways are easier to understand")
    print("✓ Efficiency: Only 25% of pathways active (4/16)")
    print("✓ Modularity: Can analyze/modify pathways independently")
    print("✓ Control: Can potentially steer behavior via pathway selection")
    
    # Save example results
    high_purities = [s['purity'] for s in high_spec_stats.values()]
    low_purities = [s['purity'] for s in low_spec_stats.values()]
    
    example_results = {
        'high_specialization': {
            'avg_purity': float(sum(high_purities) / len(high_purities)),
            'pathway_count': len(high_spec_stats)
        },
        'low_specialization': {
            'avg_purity': float(sum(low_purities) / len(low_purities)),
            'pathway_count': len(low_spec_stats)
        },
        'example_pathway': {
            'name': sorted_pathways[0][0],
            'purity': float(sorted_pathways[0][1]['purity']),
            'top_tokens': sorted_pathways[0][1]['top_tokens'][:3]
        }
    }
    
    with open('routing_demo_results.json', 'w') as f:
        json.dump(example_results, f, indent=2)
    
    print("\n✅ Demo results saved to 'routing_demo_results.json'")

if __name__ == "__main__":
    demonstrate_routing_concept()