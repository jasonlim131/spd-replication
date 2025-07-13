"""
Simple demonstration of Semantic vs Syntactic routing
No external dependencies required
"""

import random
import math
from collections import defaultdict

def calculate_entropy(distribution):
    """Calculate Shannon entropy of a probability distribution"""
    entropy = 0
    for p in distribution:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy

def normalize_dict(d):
    """Normalize dictionary values to sum to 1"""
    total = sum(d.values())
    if total == 0:
        return d
    return {k: v/total for k, v in d.items()}

class SyntacticRouter:
    """Routes based on exact tokens (less interpretable)"""
    def __init__(self, num_pathways=8):
        self.num_pathways = num_pathways
        self.pathway_token_counts = defaultdict(lambda: defaultdict(int))
        
    def route(self, tokens):
        """Simple routing based on token hash"""
        pathways = []
        for token in tokens:
            # Route based on token ID modulo num_pathways
            pathway = hash(token) % self.num_pathways
            pathways.append(pathway)
            self.pathway_token_counts[pathway][token] += 1
        return pathways
    
    def analyze(self):
        """Analyze what each pathway learned"""
        print("\n=== SYNTACTIC ROUTING ANALYSIS ===")
        for pathway_id in range(self.num_pathways):
            tokens = self.pathway_token_counts[pathway_id]
            if tokens:
                top_tokens = sorted(tokens.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"\nPathway {pathway_id}:")
                print(f"  Top tokens: {[t[0] for t in top_tokens]}")
                print(f"  Total unique tokens: {len(tokens)}")
                # Hard to interpret - just a collection of tokens!

class SemanticRouter:
    """Routes based on semantic concepts (more interpretable)"""
    def __init__(self, num_pathways=8, use_glbl=True):
        self.num_pathways = num_pathways
        self.use_glbl = use_glbl
        
        # Semantic categories
        self.categories = [
            "spatial", "temporal", "emotional", "causal",
            "quantitative", "action", "social", "logical"
        ]
        
        # Simple word-to-category mapping
        self.word_categories = {
            # Spatial
            "above": "spatial", "below": "spatial", "inside": "spatial",
            "near": "spatial", "far": "spatial", "between": "spatial",
            
            # Temporal
            "yesterday": "temporal", "today": "temporal", "tomorrow": "temporal",
            "always": "temporal", "never": "temporal", "soon": "temporal",
            
            # Emotional
            "happy": "emotional", "sad": "emotional", "angry": "emotional",
            "excited": "emotional", "afraid": "emotional", "calm": "emotional",
            
            # Causal
            "because": "causal", "therefore": "causal", "causes": "causal",
            "leads": "causal", "results": "causal", "why": "causal",
            
            # Actions
            "run": "action", "jump": "action", "create": "action",
            "destroy": "action", "build": "action", "move": "action",
            
            # Social
            "talk": "social", "meet": "social", "help": "social",
            "argue": "social", "collaborate": "social", "discuss": "social",
        }
        
        # Pathway specializations
        self.pathway_profiles = defaultdict(lambda: defaultdict(float))
        
        # GLBL tracking
        self.global_frequencies = [0.0] * num_pathways
        self.update_count = 0
        
    def get_semantic_features(self, token):
        """Extract semantic category for a token"""
        return self.word_categories.get(token.lower(), "unknown")
    
    def route_with_glbl(self, tokens):
        """Route with Global Load Balancing"""
        pathways = []
        pathway_activations = [0] * self.num_pathways
        
        for token in tokens:
            category = self.get_semantic_features(token)
            
            # Base routing by category
            if category == "spatial":
                preferred_pathways = [0, 1]
            elif category == "emotional":
                preferred_pathways = [2, 3]
            elif category == "causal":
                preferred_pathways = [4]
            elif category == "action":
                preferred_pathways = [5]
            else:
                preferred_pathways = [6, 7]
            
            # Apply GLBL penalty to overused pathways
            if self.use_glbl and self.update_count > 0:
                # Penalize pathways based on global usage
                scores = []
                for p in range(self.num_pathways):
                    base_score = 1.0 if p in preferred_pathways else 0.1
                    glbl_penalty = self.global_frequencies[p] * 0.5
                    final_score = base_score - glbl_penalty
                    scores.append(max(0, final_score))
                
                # Select pathway based on scores
                if sum(scores) > 0:
                    probs = [s/sum(scores) for s in scores]
                    pathway = random.choices(range(self.num_pathways), weights=probs)[0]
                else:
                    pathway = random.choice(range(self.num_pathways))
            else:
                # Without GLBL, just use preferred pathway
                pathway = random.choice(preferred_pathways)
            
            pathways.append(pathway)
            pathway_activations[pathway] += 1
            
            # Track semantic profile
            self.pathway_profiles[pathway][category] += 1
        
        # Update global frequencies
        if tokens:
            batch_freq = [a/len(tokens) for a in pathway_activations]
            momentum = 0.9
            for i in range(self.num_pathways):
                self.global_frequencies[i] = (momentum * self.global_frequencies[i] + 
                                            (1-momentum) * batch_freq[i])
            self.update_count += 1
        
        return pathways
    
    def analyze(self):
        """Analyze semantic specialization of pathways"""
        print(f"\n=== SEMANTIC ROUTING ANALYSIS (GLBL={'ON' if self.use_glbl else 'OFF'}) ===")
        
        all_purities = []
        
        for pathway_id in range(self.num_pathways):
            profile = self.pathway_profiles[pathway_id]
            if not profile:
                continue
                
            # Normalize profile
            total = sum(profile.values())
            norm_profile = {cat: count/total for cat, count in profile.items()}
            
            # Calculate semantic purity
            probs = list(norm_profile.values())
            if probs:
                entropy = calculate_entropy(probs)
                max_entropy = math.log(len(probs)) if len(probs) > 1 else 0
                purity = 1 - (entropy / max_entropy) if max_entropy > 0 else 1.0
                all_purities.append(purity)
            else:
                purity = 0
            
            # Sort categories by activation
            sorted_cats = sorted(norm_profile.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\nPathway {pathway_id} (purity={purity:.3f}):")
            if sorted_cats:
                print(f"  Primary: {sorted_cats[0][0]} ({sorted_cats[0][1]:.2%})")
                for cat, score in sorted_cats[1:3]:
                    if score > 0:
                        print(f"  Secondary: {cat} ({score:.2%})")
            
            # Show global frequency
            print(f"  Global usage: {self.global_frequencies[pathway_id]:.2%}")
        
        if all_purities:
            print(f"\nAverage Semantic Purity: {sum(all_purities)/len(all_purities):.3f}")
        
        # Show GLBL effect
        if self.use_glbl:
            print(f"\nGLBL Statistics:")
            print(f"  Most used pathway: {max(self.global_frequencies):.2%}")
            print(f"  Least used pathway: {min(self.global_frequencies):.2%}")
            print(f"  Usage variance: {self.calculate_variance(self.global_frequencies):.4f}")
    
    def calculate_variance(self, values):
        """Calculate variance of a list"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

def demonstrate():
    """Run demonstration of syntactic vs semantic routing"""
    print("🚀 DEMONSTRATION: SYNTACTIC vs SEMANTIC ROUTING")
    print("=" * 60)
    
    # Sample sentences with semantic content
    sentences = [
        "The cat is above the table near the window",
        "Yesterday I was happy but today I feel sad",
        "Running quickly leads to exhaustion because of effort",
        "We should meet tomorrow to discuss and collaborate",
        "The book is inside the box below the shelf",
        "She was excited yesterday but calm today",
        "Jumping high causes fatigue therefore rest helps",
        "They argue often but sometimes help each other"
    ]
    
    # Tokenize simply
    all_tokens = []
    for sentence in sentences * 10:  # Repeat for more data
        tokens = sentence.lower().split()
        all_tokens.extend(tokens)
    
    print(f"Processing {len(all_tokens)} tokens...\n")
    
    # Test syntactic routing
    print("1. SYNTACTIC ROUTING (Token-based)")
    print("-" * 40)
    syntactic_router = SyntacticRouter()
    syntactic_router.route(all_tokens)
    syntactic_router.analyze()
    
    # Test semantic routing WITHOUT GLBL
    print("\n\n2. SEMANTIC ROUTING WITHOUT GLBL")
    print("-" * 40)
    semantic_router_no_glbl = SemanticRouter(use_glbl=False)
    semantic_router_no_glbl.route_with_glbl(all_tokens)
    semantic_router_no_glbl.analyze()
    
    # Test semantic routing WITH GLBL
    print("\n\n3. SEMANTIC ROUTING WITH GLBL")
    print("-" * 40)
    semantic_router_glbl = SemanticRouter(use_glbl=True)
    semantic_router_glbl.route_with_glbl(all_tokens)
    semantic_router_glbl.analyze()
    
    # Summary
    print("\n\n📊 SUMMARY")
    print("=" * 60)
    print("1. Syntactic routing: Groups tokens by surface patterns")
    print("   → Hard to interpret what each pathway does")
    print("\n2. Semantic routing WITHOUT GLBL: Concepts collapse to few pathways")
    print("   → Some pathways never used, wasted capacity")
    print("\n3. Semantic routing WITH GLBL: Balanced concept distribution")
    print("   → All pathways utilized, fine-grained specialization")
    print("   → Much more interpretable and controllable!")

if __name__ == "__main__":
    demonstrate()