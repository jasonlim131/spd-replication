"""
Pathway Monosemanticity Analysis for Multiplicative GLBL
Analyzes sample pathways using purity metrics and auto-interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import random
import json

# Import our multiplicative GLBL implementation
from multiplicative_glbl_complete import (
    MultiplicativeGLBLModel, 
    PathwayDecomposer,
    analyze_multiplicative_pathways
)

class PathwayMonosemanicityAnalyzer:
    """Analyzes pathway monosemanticity using multiple metrics"""
    
    def __init__(self, model: MultiplicativeGLBLModel):
        self.model = model
        self.pathway_activations = defaultdict(list)
        
        # Define semantic token categories for analysis
        self.semantic_categories = {
            'numbers': list(range(100, 200)),  # Mock number tokens
            'verbs': list(range(200, 300)),    # Mock verb tokens
            'nouns': list(range(300, 400)),    # Mock noun tokens
            'adjectives': list(range(400, 500)), # Mock adjective tokens
            'functions': list(range(500, 600)), # Mock function tokens
            'symbols': list(range(600, 700)),   # Mock symbol tokens
            'operators': list(range(700, 800)), # Mock operator tokens
            'connectors': list(range(800, 900)), # Mock connector tokens
        }
        
        # Create reverse mapping
        self.token_to_category = {}
        for category, tokens in self.semantic_categories.items():
            for token in tokens:
                self.token_to_category[token] = category
    
    def generate_diverse_inputs(self, num_samples: int = 100, seq_len: int = 20) -> torch.Tensor:
        """Generate diverse input sequences to activate different pathways"""
        
        samples = []
        
        for _ in range(num_samples):
            # Create sequences biased toward different semantic categories
            category = random.choice(list(self.semantic_categories.keys()))
            category_tokens = self.semantic_categories[category]
            
            # Mix of category-specific and random tokens
            sequence = []
            for pos in range(seq_len):
                if random.random() < 0.7:  # 70% category-specific tokens
                    sequence.append(random.choice(category_tokens))
                else:  # 30% random tokens
                    sequence.append(random.randint(0, 999))
            
            samples.append(sequence)
        
        return torch.tensor(samples, dtype=torch.long)
    
    def collect_pathway_activations(self, num_samples: int = 200) -> Dict[str, List[Dict]]:
        """Collect pathway activations across diverse inputs"""
        
        print("🔍 Collecting pathway activations...")
        
        # Generate diverse inputs
        inputs = self.generate_diverse_inputs(num_samples, seq_len=16)
        
        self.model.eval()
        all_pathways = []
        
        with torch.no_grad():
            # Process in batches
            batch_size = 8
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i+batch_size]
                
                # Forward pass with activation recording
                results = self.model(batch, record_activations=True)
                
                # Collect pathway activations
                for pathway in results['all_active_pathways']:
                    # Add input context
                    batch_idx, seq_idx = pathway['position']
                    token_id = batch[batch_idx, seq_idx].item()
                    
                    pathway_data = {
                        **pathway,
                        'token_id': token_id,
                        'token_category': self.token_to_category.get(token_id, 'other'),
                        'batch_id': i // batch_size,
                        'input_sequence': batch[batch_idx].tolist()
                    }
                    
                    all_pathways.append(pathway_data)
        
        # Group by pathway index
        pathway_groups = defaultdict(list)
        for pathway in all_pathways:
            pathway_groups[pathway['pathway_idx']].append(pathway)
        
        print(f"   Collected {len(all_pathways)} activations across {len(pathway_groups)} unique pathways")
        
        return pathway_groups
    
    def calculate_pathway_purity(self, pathway_activations: List[Dict]) -> Dict[str, float]:
        """Calculate multiple purity metrics for a pathway"""
        
        if not pathway_activations:
            return {'error': 'No activations'}
        
        # Token-level purity
        tokens = [act['token_id'] for act in pathway_activations]
        token_counts = Counter(tokens)
        token_probs = np.array(list(token_counts.values())) / len(tokens)
        token_entropy = -np.sum(token_probs * np.log(token_probs + 1e-8))
        token_purity = 1 - (token_entropy / np.log(len(token_counts)))
        
        # Category-level purity
        categories = [act['token_category'] for act in pathway_activations]
        category_counts = Counter(categories)
        category_probs = np.array(list(category_counts.values())) / len(categories)
        category_entropy = -np.sum(category_probs * np.log(category_probs + 1e-8))
        category_purity = 1 - (category_entropy / np.log(len(category_counts)))
        
        # Position-level purity
        positions = [act['position'][1] for act in pathway_activations]  # seq_idx
        position_counts = Counter(positions)
        position_probs = np.array(list(position_counts.values())) / len(positions)
        position_entropy = -np.sum(position_probs * np.log(position_probs + 1e-8))
        position_purity = 1 - (position_entropy / np.log(len(position_counts)))
        
        # Weight consistency (higher = more consistent activation strength)
        weights = [act['weight'] for act in pathway_activations]
        weight_std = np.std(weights)
        weight_consistency = 1 / (1 + weight_std)  # Higher consistency = lower std
        
        # Dominant category analysis
        dominant_category = max(category_counts.items(), key=lambda x: x[1])
        dominant_token = max(token_counts.items(), key=lambda x: x[1])
        
        return {
            'token_purity': token_purity,
            'category_purity': category_purity,
            'position_purity': position_purity,
            'weight_consistency': weight_consistency,
            'overall_purity': (token_purity + category_purity + weight_consistency) / 3,
            'activation_count': len(pathway_activations),
            'unique_tokens': len(token_counts),
            'unique_categories': len(category_counts),
            'dominant_category': dominant_category,
            'dominant_token': dominant_token,
            'avg_weight': np.mean(weights),
            'weight_std': weight_std
        }
    
    def auto_interpret_pathway(self, pathway_idx: int, pathway_activations: List[Dict], 
                              decomposer: PathwayDecomposer) -> Dict[str, Any]:
        """Auto-interpret pathway semantic specialization"""
        
        if not pathway_activations:
            return {'interpretation': 'No activations to analyze'}
        
        # Decompose pathway into expert indices
        pre_idx, mlp_idx, post_idx = decomposer.decompose(pathway_idx)
        
        # Analyze semantic patterns
        categories = [act['token_category'] for act in pathway_activations]
        category_counts = Counter(categories)
        total_activations = len(pathway_activations)
        
        # Analyze positional patterns
        positions = [act['position'][1] for act in pathway_activations]
        position_counts = Counter(positions)
        
        # Analyze token patterns
        tokens = [act['token_id'] for act in pathway_activations]
        token_counts = Counter(tokens)
        
        # Determine specialization type
        dominant_category = max(category_counts.items(), key=lambda x: x[1])
        category_dominance = dominant_category[1] / total_activations
        
        # Position preference
        preferred_positions = sorted(position_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Generate interpretation
        interpretation = {
            'pathway_composition': f"Pre[{pre_idx}] × MLP[{mlp_idx}] × Post[{post_idx}]",
            'specialization_type': self._determine_specialization_type(category_counts, position_counts),
            'dominant_category': dominant_category[0],
            'category_dominance': category_dominance,
            'preferred_positions': preferred_positions,
            'semantic_description': self._generate_semantic_description(
                dominant_category, category_dominance, preferred_positions, pre_idx, mlp_idx, post_idx
            ),
            'monosemanticity_level': self._assess_monosemanticity(category_dominance, len(category_counts)),
            'functional_role': self._infer_functional_role(pre_idx, mlp_idx, post_idx, dominant_category[0])
        }
        
        return interpretation
    
    def _determine_specialization_type(self, category_counts: Counter, position_counts: Counter) -> str:
        """Determine the type of specialization"""
        
        # Check if specialized by category
        total = sum(category_counts.values())
        max_category_ratio = max(category_counts.values()) / total
        
        # Check if specialized by position
        max_position_ratio = max(position_counts.values()) / total
        
        if max_category_ratio > 0.7:
            return "semantic_specialist"
        elif max_position_ratio > 0.7:
            return "positional_specialist"
        elif max_category_ratio > 0.5 and max_position_ratio > 0.5:
            return "mixed_specialist"
        else:
            return "generalist"
    
    def _generate_semantic_description(self, dominant_category: Tuple, category_dominance: float,
                                     preferred_positions: List[Tuple], pre_idx: int, 
                                     mlp_idx: int, post_idx: int) -> str:
        """Generate human-readable semantic description"""
        
        category_name = dominant_category[0]
        dominance_pct = category_dominance * 100
        
        # Position description
        if preferred_positions:
            pos_desc = f"positions {[p[0] for p in preferred_positions[:2]]}"
        else:
            pos_desc = "various positions"
        
        # Expert combination description
        expert_desc = self._describe_expert_combination(pre_idx, mlp_idx, post_idx)
        
        # Strength assessment
        if dominance_pct > 80:
            strength = "extremely specialized"
        elif dominance_pct > 60:
            strength = "highly specialized"
        elif dominance_pct > 40:
            strength = "moderately specialized"
        else:
            strength = "weakly specialized"
        
        return (f"This pathway is {strength} for {category_name} tokens ({dominance_pct:.1f}% dominance), "
               f"preferring {pos_desc}. {expert_desc}")
    
    def _describe_expert_combination(self, pre_idx: int, mlp_idx: int, post_idx: int) -> str:
        """Describe the expert combination"""
        
        # Simple heuristic descriptions based on indices
        pre_type = ["input_normalization", "feature_extraction", "pattern_detection", "context_integration"][pre_idx % 4]
        mlp_type = ["linear_transformation", "nonlinear_processing", "feature_combination", "abstraction"][mlp_idx % 4]
        post_type = ["output_shaping", "feature_refinement", "normalization", "activation_control"][post_idx % 4]
        
        return f"Uses {pre_type} → {mlp_type} → {post_type} processing pipeline."
    
    def _assess_monosemanticity(self, dominance: float, num_categories: int) -> str:
        """Assess monosemanticity level"""
        
        if dominance > 0.8 and num_categories <= 2:
            return "very_high"
        elif dominance > 0.6 and num_categories <= 3:
            return "high"
        elif dominance > 0.4 and num_categories <= 5:
            return "medium"
        else:
            return "low"
    
    def _infer_functional_role(self, pre_idx: int, mlp_idx: int, post_idx: int, category: str) -> str:
        """Infer functional role based on expert combination and specialization"""
        
        role_map = {
            ('numbers', 0, 0): "numerical_input_processor",
            ('verbs', 1, 1): "action_analyzer",
            ('nouns', 2, 2): "entity_identifier",
            ('adjectives', 3, 3): "attribute_processor",
            ('functions', 0, 1): "function_classifier",
            ('symbols', 1, 2): "symbol_interpreter",
            ('operators', 2, 3): "operation_handler",
            ('connectors', 3, 0): "relationship_mapper"
        }
        
        key = (category, pre_idx % 4, mlp_idx % 4)
        return role_map.get(key, f"{category}_handler")

def run_pathway_monosemanticity_analysis():
    """Run comprehensive pathway monosemanticity analysis"""
    
    print("🧠 MULTIPLICATIVE GLBL PATHWAY MONOSEMANTICITY ANALYSIS")
    print("=" * 80)
    
    # Create model
    print("\n🏗️  Creating Multiplicative GLBL Model...")
    model = MultiplicativeGLBLModel(
        vocab_size=1000, d_model=64, n_layers=4, n_heads=4,
        glbl_layers=[1, 2, 3]  # Dense GLBL for more pathway activations
    )
    
    # Create analyzer
    analyzer = PathwayMonosemanicityAnalyzer(model)
    
    # Collect pathway activations
    print("\n🔍 Collecting Pathway Activations...")
    pathway_groups = analyzer.collect_pathway_activations(num_samples=150)
    
    # Select 20 most active pathways for analysis
    pathway_usage = [(pathway_idx, len(activations)) 
                     for pathway_idx, activations in pathway_groups.items()]
    pathway_usage.sort(key=lambda x: x[1], reverse=True)
    
    top_20_pathways = pathway_usage[:20]
    
    print(f"\n📊 Analyzing Top 20 Most Active Pathways...")
    print(f"   Total unique pathways observed: {len(pathway_groups)}")
    print(f"   Total pathway activations: {sum(len(acts) for acts in pathway_groups.values())}")
    
    # Analyze each pathway
    pathway_analyses = []
    decomposer = PathwayDecomposer(12, 12, 12)  # Match model architecture
    
    for i, (pathway_idx, usage_count) in enumerate(top_20_pathways):
        print(f"\n🔬 Analyzing Pathway {i+1}/20: #{pathway_idx} ({usage_count} activations)")
        
        activations = pathway_groups[pathway_idx]
        
        # Calculate purity metrics
        purity_metrics = analyzer.calculate_pathway_purity(activations)
        
        # Auto-interpret pathway
        interpretation = analyzer.auto_interpret_pathway(pathway_idx, activations, decomposer)
        
        # Combine analysis
        analysis = {
            'pathway_idx': pathway_idx,
            'rank': i + 1,
            'usage_count': usage_count,
            'purity_metrics': purity_metrics,
            'interpretation': interpretation
        }
        
        pathway_analyses.append(analysis)
        
        # Print summary
        print(f"   Composition: {interpretation['pathway_composition']}")
        print(f"   Overall Purity: {purity_metrics['overall_purity']:.3f}")
        print(f"   Monosemanticity: {interpretation['monosemanticity_level']}")
        print(f"   Specialization: {interpretation['semantic_description']}")
    
    # Generate comprehensive report
    return generate_monosemanticity_report(pathway_analyses)

def generate_monosemanticity_report(pathway_analyses: List[Dict]) -> Dict[str, Any]:
    """Generate comprehensive monosemanticity analysis report"""
    
    print("\n📋 GENERATING MONOSEMANTICITY REPORT")
    print("=" * 60)
    
    # Overall statistics
    overall_purities = [p['purity_metrics']['overall_purity'] for p in pathway_analyses]
    category_purities = [p['purity_metrics']['category_purity'] for p in pathway_analyses]
    monosemanticity_levels = [p['interpretation']['monosemanticity_level'] for p in pathway_analyses]
    
    # Count monosemanticity levels
    mono_counts = Counter(monosemanticity_levels)
    
    # Count specialization types
    specialization_types = [p['interpretation']['specialization_type'] for p in pathway_analyses]
    spec_counts = Counter(specialization_types)
    
    # Count dominant categories
    dominant_categories = [p['interpretation']['dominant_category'] for p in pathway_analyses]
    category_counts = Counter(dominant_categories)
    
    # Generate report
    report = {
        'summary_statistics': {
            'total_pathways_analyzed': len(pathway_analyses),
            'avg_overall_purity': np.mean(overall_purities),
            'avg_category_purity': np.mean(category_purities),
            'std_overall_purity': np.std(overall_purities),
            'monosemanticity_distribution': dict(mono_counts),
            'specialization_distribution': dict(spec_counts),
            'category_distribution': dict(category_counts)
        },
        'pathway_details': pathway_analyses,
        'insights': generate_insights(pathway_analyses)
    }
    
    # Print summary
    print(f"📊 SUMMARY STATISTICS")
    print(f"   Average Overall Purity: {report['summary_statistics']['avg_overall_purity']:.3f}")
    print(f"   Average Category Purity: {report['summary_statistics']['avg_category_purity']:.3f}")
    print(f"   Purity Std Dev: {report['summary_statistics']['std_overall_purity']:.3f}")
    
    print(f"\n🎯 MONOSEMANTICITY DISTRIBUTION")
    for level, count in mono_counts.items():
        pct = 100 * count / len(pathway_analyses)
        print(f"   {level}: {count} pathways ({pct:.1f}%)")
    
    print(f"\n🔍 SPECIALIZATION TYPES")
    for spec_type, count in spec_counts.items():
        pct = 100 * count / len(pathway_analyses)
        print(f"   {spec_type}: {count} pathways ({pct:.1f}%)")
    
    print(f"\n📂 CATEGORY SPECIALIZATIONS")
    for category, count in category_counts.items():
        pct = 100 * count / len(pathway_analyses)
        print(f"   {category}: {count} pathways ({pct:.1f}%)")
    
    return report

def generate_insights(pathway_analyses: List[Dict]) -> List[str]:
    """Generate insights from pathway analysis"""
    
    insights = []
    
    # Purity insights
    purities = [p['purity_metrics']['overall_purity'] for p in pathway_analyses]
    avg_purity = np.mean(purities)
    
    if avg_purity > 0.7:
        insights.append("High average purity indicates strong monosemantic specialization")
    elif avg_purity > 0.5:
        insights.append("Moderate purity suggests emerging specialization patterns")
    else:
        insights.append("Low purity indicates distributed, polysemantic representations")
    
    # Specialization insights
    specializations = [p['interpretation']['specialization_type'] for p in pathway_analyses]
    spec_counts = Counter(specializations)
    
    if spec_counts['semantic_specialist'] > len(pathway_analyses) * 0.5:
        insights.append("Majority of pathways show semantic specialization")
    
    if spec_counts['positional_specialist'] > 3:
        insights.append("Strong positional specialization suggests sequence modeling")
    
    # Category insights
    categories = [p['interpretation']['dominant_category'] for p in pathway_analyses]
    category_counts = Counter(categories)
    
    if len(category_counts) > 6:
        insights.append("Wide category distribution shows diverse semantic coverage")
    
    # Expert combination insights
    expert_combos = [p['interpretation']['pathway_composition'] for p in pathway_analyses]
    if len(set(expert_combos)) == len(expert_combos):
        insights.append("All pathways use unique expert combinations - no redundancy")
    
    return insights

def create_detailed_pathway_table(pathway_analyses: List[Dict]) -> str:
    """Create detailed pathway analysis table"""
    
    table = "# Multiplicative GLBL Pathway Monosemanticity Analysis Table\n\n"
    table += "| Rank | Pathway ID | Expert Combination | Overall Purity | Category Purity | Monosemanticity | Dominant Category | Usage Count | Semantic Description |\n"
    table += "|------|------------|-------------------|----------------|-----------------|-----------------|-------------------|-------------|----------------------|\n"
    
    for analysis in pathway_analyses:
        rank = analysis['rank']
        pathway_idx = analysis['pathway_idx']
        composition = analysis['interpretation']['pathway_composition']
        overall_purity = analysis['purity_metrics']['overall_purity']
        category_purity = analysis['purity_metrics']['category_purity']
        monosemanticity = analysis['interpretation']['monosemanticity_level']
        dominant_cat = analysis['interpretation']['dominant_category']
        usage = analysis['usage_count']
        description = analysis['interpretation']['semantic_description'][:100] + "..."
        
        table += f"| {rank} | {pathway_idx} | {composition} | {overall_purity:.3f} | {category_purity:.3f} | {monosemanticity} | {dominant_cat} | {usage} | {description} |\n"
    
    return table

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run analysis
    report = run_pathway_monosemanticity_analysis()
    
    # Create detailed table
    table = create_detailed_pathway_table(report['pathway_details'])
    
    print("\n" + "="*80)
    print("📋 DETAILED PATHWAY ANALYSIS TABLE")
    print("="*80)
    print(table)
    
    print("\n🔍 KEY INSIGHTS:")
    for insight in report['insights']:
        print(f"   • {insight}")
    
    print(f"\n✅ Monosemanticity analysis complete!")
    print(f"📊 {len(report['pathway_details'])} pathways analyzed")
    print(f"🎯 Average purity: {report['summary_statistics']['avg_overall_purity']:.3f}")
    
    # Save results
    with open('pathway_monosemanticity_results.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"💾 Results saved to: pathway_monosemanticity_results.json")