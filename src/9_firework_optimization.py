import json
import os
import numpy as np
import pickle
from datetime import datetime
import random

# Directories
RANKING_DIR = "data/ranking"
ONTOLOGY_DIR = "data/ontologies"
FIREWORK_DIR = "data/firework"

# Files
SIMRANK_FILE = os.path.join(RANKING_DIR, "simrank_matrix.npy")
SIMRANK_MAPPING_FILE = os.path.join(RANKING_DIR, "simrank_mapping.json")
RANKING_STATS_FILE = os.path.join(RANKING_DIR, "ranking_statistics.json")
ONTOLOGY_FILE = os.path.join(ONTOLOGY_DIR, "heritage_ontologies.json")
OPTIMIZED_RANKINGS_FILE = os.path.join(FIREWORK_DIR, "optimized_rankings.json")



def compute_ontology_similarity(query_ontology, doc_ontology):
    """Compute similarity between query and document ontologies"""
    
    score = 0.0
    max_score = 0.0
    
    # Match Heritage Types
    if 'HeritageType' in query_ontology.get('concepts', {}) and 'HeritageType' in doc_ontology.get('concepts', {}):
        query_types = query_ontology['concepts']['HeritageType']
        if isinstance(query_types, list) and query_types:
            if isinstance(query_types[0], dict):
                query_types = [c['value'] for c in query_types]
        
        doc_types = doc_ontology['concepts']['HeritageType']
        
        if query_types and doc_types:
            matches = len(set(query_types) & set(doc_types))
            score += matches * 2.0
            max_score += len(query_types) * 2.0
    
    # Match Domains
    if 'Domain' in query_ontology.get('concepts', {}) and 'Domain' in doc_ontology.get('concepts', {}):
        query_domains = query_ontology['concepts']['Domain']
        if isinstance(query_domains, list) and query_domains:
            if isinstance(query_domains[0], dict):
                query_domains = [c['value'] for c in query_domains]
        
        doc_domains = doc_ontology['concepts']['Domain']
        
        if query_domains and doc_domains:
            matches = len(set(query_domains) & set(doc_domains))
            score += matches * 1.5
            max_score += len(query_domains) * 1.5
    
    # Match Time Periods
    if 'TimePeriod' in query_ontology.get('concepts', {}) and 'TimePeriod' in doc_ontology.get('concepts', {}):
        query_periods = query_ontology['concepts']['TimePeriod']
        if isinstance(query_periods, list) and query_periods:
            if isinstance(query_periods[0], dict):
                query_periods = [c['value'] for c in query_periods]
        
        doc_periods = doc_ontology['concepts']['TimePeriod']
        
        if query_periods and doc_periods:
            matches = len(set(query_periods) & set(doc_periods))
            score += matches * 1.0
            max_score += len(query_periods) * 1.0
    
    # Normalize score
    if max_score > 0:
        return score / max_score
    else:
        return 0.0


# Firework Algorithm Parameters
NUM_FIREWORKS = 10  # Population size
MAX_ITERATIONS = 20  # Number of optimization iterations
EXPLOSION_AMPLITUDE = 5  # How far fireworks can "explode"
NUM_SPARKS = 5  # Sparks per firework
MUTATION_RATE = 0.2  # Gaussian mutation rate

# ========== FIREWORK ALGORITHM ==========

class Firework:
    """
    Represents a candidate solution (ranking) in the Firework Algorithm
    A firework is essentially a ranking of documents for a query
    """
    
    def __init__(self, ranking, fitness=0.0):
        self.ranking = ranking  # List of (doc_idx, score) tuples
        self.fitness = fitness  # Quality of this ranking
        self.explosion_amplitude = EXPLOSION_AMPLITUDE
    
    def explode(self, num_sparks):
        """
        Generate sparks (variations) of this firework
        Sparks represent local search around current solution
        """
        sparks = []
        
        for _ in range(num_sparks):
            # Create a spark by perturbing the ranking
            spark_ranking = self.ranking.copy()
            
            # Randomly swap positions (local perturbation)
            num_swaps = random.randint(1, min(3, len(spark_ranking)))
            
            for _ in range(num_swaps):
                if len(spark_ranking) < 2:
                    break
                idx1 = random.randint(0, len(spark_ranking) - 1)
                idx2 = random.randint(0, len(spark_ranking) - 1)
                
                # Swap scores to change ranking
                doc1, score1 = spark_ranking[idx1]
                doc2, score2 = spark_ranking[idx2]
                spark_ranking[idx1] = (doc1, score2)
                spark_ranking[idx2] = (doc2, score1)
            
            # Re-sort by modified scores
            spark_ranking.sort(key=lambda x: x[1], reverse=True)
            
            spark = Firework(spark_ranking, fitness=0.0)
            sparks.append(spark)
        
        return sparks
    
    def gaussian_mutation(self):
        """
        Apply Gaussian mutation to scores
        This adds random noise to explore solution space
        """
        mutated_ranking = []
        
        for doc_idx, score in self.ranking:
            # Add Gaussian noise
            noise = np.random.normal(0, 0.05)  # Small perturbation
            new_score = max(0.0, min(1.0, score + noise))  # Keep in [0,1]
            mutated_ranking.append((doc_idx, new_score))
        
        # Re-sort
        mutated_ranking.sort(key=lambda x: x[1], reverse=True)
        
        return Firework(mutated_ranking, fitness=0.0)

def evaluate_ranking_fitness(ranking, query_ontology, doc_ontologies, 
                             simrank_scores, horns_scores, ontology_similarities):
    """
    Evaluate fitness of a ranking
    
    Fitness considers:
    1. Relevance (high-scored docs should be relevant)
    2. Diversity (avoid too similar docs)
    3. Ontology match (semantic coherence)
    4. Position importance (top positions more critical)
    """
    
    if not ranking:
        return 0.0
    
    fitness = 0.0
    
    # 1. Relevance Score (weighted by position)
    position_weights = [1.0 / (i + 1) for i in range(len(ranking))]  # DCG-like weights
    
    for idx, (doc_idx, score) in enumerate(ranking):
        # Combine SimRank, Horn's, and Ontology
        relevance = (0.4 * simrank_scores.get(doc_idx, 0.0) + 
                    0.3 * horns_scores.get(doc_idx, 0.0) +
                    0.3 * ontology_similarities.get(doc_idx, 0.0))
        
        fitness += position_weights[idx] * relevance
    
    # 2. Diversity Penalty (penalize too similar consecutive documents)
    diversity_score = 0.0
    for i in range(len(ranking) - 1):
        doc1_idx = ranking[i][0]
        doc2_idx = ranking[i + 1][0]
        
        # Lower similarity between consecutive docs is better
        similarity = simrank_scores.get((doc1_idx, doc2_idx), 0.0)
        diversity_score += (1.0 - similarity)
    
    if len(ranking) > 1:
        diversity_score /= (len(ranking) - 1)
    
    # 3. Ontology Coherence (top results should match query ontology)
    ontology_score = 0.0
    top_k = min(5, len(ranking))
    for i in range(top_k):
        doc_idx = ranking[i][0]
        ontology_score += ontology_similarities.get(doc_idx, 0.0)
    
    if top_k > 0:
        ontology_score /= top_k
    
    # Combine components
    fitness = 0.5 * fitness + 0.25 * diversity_score + 0.25 * ontology_score
    
    return fitness

def firework_algorithm(query_idx, simrank_matrix, horns_index, doc_nodes, node_to_idx,
                       query_ontology, doc_ontologies, top_k=20):
    """
    Firework Algorithm for optimizing document ranking
    
    Algorithm:
    1. Initialize population of fireworks (initial rankings)
    2. For each iteration:
       a. Evaluate fitness of each firework
       b. Generate sparks (variations) around best fireworks
       c. Apply Gaussian mutation
       d. Select best fireworks for next generation
    3. Return best ranking found
    """
    
    print(f"\n  Optimizing ranking with Firework Algorithm...")
    print(f"    Population: {NUM_FIREWORKS}, Iterations: {MAX_ITERATIONS}")
    
    # Precompute scores
    simrank_scores = {}
    horns_scores = {}
    ontology_similarities = {}
    
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    for idx in range(len(doc_nodes)):
        if idx == query_idx:
            continue
        
        doc_node = doc_nodes[idx]
        
        # SimRank score
        simrank_scores[idx] = simrank_matrix[query_idx, idx]
        
        # Horn's Index
        horns_scores[idx] = horns_index.get(doc_node, 0.0)
        
        # Ontology similarity
        doc_ont = doc_ontologies[idx]
        ont_sim = compute_ontology_similarity(query_ontology, doc_ont)
        ontology_similarities[idx] = ont_sim
    
    # Initialize population of fireworks
    population = []
    
    for _ in range(NUM_FIREWORKS):
        # Create initial ranking by combining scores with random weights
        alpha = random.uniform(0.3, 0.5)
        beta = random.uniform(0.2, 0.4)
        gamma = 1.0 - alpha - beta
        
        initial_ranking = []
        for idx in range(len(doc_nodes)):
            if idx == query_idx:
                continue
            
            score = (alpha * simrank_scores.get(idx, 0.0) +
                    beta * horns_scores.get(idx, 0.0) +
                    gamma * ontology_similarities.get(idx, 0.0))
            
            initial_ranking.append((idx, score))
        
        # Sort and take top-k
        initial_ranking.sort(key=lambda x: x[1], reverse=True)
        initial_ranking = initial_ranking[:top_k]
        
        firework = Firework(initial_ranking)
        population.append(firework)
    
    # Optimization loop
    best_fitness_history = []
    
    for iteration in range(MAX_ITERATIONS):
        # Evaluate fitness for all fireworks
        for firework in population:
            firework.fitness = evaluate_ranking_fitness(
                firework.ranking, query_ontology, doc_ontologies,
                simrank_scores, horns_scores, ontology_similarities
            )
        
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        best_fitness = population[0].fitness
        best_fitness_history.append(best_fitness)
        
        if (iteration + 1) % 5 == 0:
            print(f"    Iteration {iteration + 1}/{MAX_ITERATIONS}: Best fitness = {best_fitness:.4f}")
        
        # Generate new population
        new_population = []
        
        # Keep best firework (elitism)
        new_population.append(population[0])
        
        # Generate sparks from top fireworks
        num_elite = min(3, len(population))
        for i in range(num_elite):
            sparks = population[i].explode(NUM_SPARKS)
            
            # Evaluate sparks
            for spark in sparks:
                spark.fitness = evaluate_ranking_fitness(
                    spark.ranking, query_ontology, doc_ontologies,
                    simrank_scores, horns_scores, ontology_similarities
                )
            
            # Add best sparks
            sparks.sort(key=lambda x: x.fitness, reverse=True)
            new_population.extend(sparks[:2])
        
        # Apply Gaussian mutation to some fireworks
        for i in range(min(3, len(population))):
            if random.random() < MUTATION_RATE:
                mutated = population[i].gaussian_mutation()
                mutated.fitness = evaluate_ranking_fitness(
                    mutated.ranking, query_ontology, doc_ontologies,
                    simrank_scores, horns_scores, ontology_similarities
                )
                new_population.append(mutated)
        
        # Trim population to size
        new_population.sort(key=lambda x: x.fitness, reverse=True)
        population = new_population[:NUM_FIREWORKS]
    
    # Return best solution
    best_firework = population[0]
    print(f"    ✓ Optimization complete! Best fitness: {best_firework.fitness:.4f}")
    
    return best_firework.ranking, best_fitness_history

# ========== BATCH OPTIMIZATION ==========

def optimize_all_queries(simrank_matrix, horns_index, doc_nodes, node_to_idx, 
                        doc_ontologies, num_test_queries=20):
    """Optimize rankings for multiple test queries"""
    
    print("\n[Phase 1] Optimizing rankings with Firework Algorithm...")
    
    # Sample test queries
    test_query_indices = random.sample(range(len(doc_nodes)), num_test_queries)
    
    optimized_rankings = {}
    
    for i, query_idx in enumerate(test_query_indices):
        query_node = doc_nodes[query_idx]
        print(f"\n[{i+1}/{num_test_queries}] Query: '{query_node}'")
        
        # Create a simple query ontology (in real system, this comes from user query)
        # For testing, we use the document's own ontology as query
        query_ontology = doc_ontologies[query_idx]
        
        # Run Firework Algorithm
        optimized_ranking, fitness_history = firework_algorithm(
            query_idx, simrank_matrix, horns_index, doc_nodes, node_to_idx,
            query_ontology, doc_ontologies, top_k=10
        )
        
        optimized_rankings[query_idx] = {
            'query_node': query_node,
            'ranking': [(doc_nodes[doc_idx], float(score)) for doc_idx, score in optimized_ranking],
            'fitness_history': [float(f) for f in fitness_history]
        }
    
    return optimized_rankings

# ========== MAIN EXECUTION ==========

def main():
    print("="*70)
    print("FIREWORK ALGORITHM OPTIMIZATION")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load SimRank matrix
    print("\n[Loading] SimRank data...")
    simrank_matrix = np.load(SIMRANK_FILE)
    
    with open(SIMRANK_MAPPING_FILE, 'r', encoding='utf-8') as f:
        simrank_data = json.load(f)
    
    doc_nodes = simrank_data['doc_nodes']
    node_to_idx = {node: idx for idx, node in enumerate(doc_nodes)}
    
    print(f"✓ Loaded SimRank matrix: {simrank_matrix.shape}")
    
    # Load Horn's Index
    with open(RANKING_STATS_FILE, 'r', encoding='utf-8') as f:
        ranking_stats = json.load(f)
    
    # Reconstruct Horn's Index from document info
    horns_index = {}
    for doc_info in simrank_data['document_info']:
        horns_index[doc_info['node_id']] = doc_info['horns_index']
    
    print(f"✓ Loaded Horn's Index for {len(horns_index)} documents")
    
    # Load ontologies
    print("\n[Loading] Ontologies...")
    with open(ONTOLOGY_FILE, 'r', encoding='utf-8') as f:
        ontology_data = json.load(f)
    
    doc_ontologies = ontology_data['document_ontologies']
    print(f"✓ Loaded {len(doc_ontologies)} document ontologies")
    
    # Run optimization
    optimized_rankings = optimize_all_queries(
        simrank_matrix, horns_index, doc_nodes, node_to_idx,
        doc_ontologies, num_test_queries=10  # Test on 10 queries
    )
    
    # Save results
    print("\n[Phase 2] Saving optimized rankings...")
    os.makedirs(FIREWORK_DIR, exist_ok=True)
    
    results = {
        'optimized_rankings': optimized_rankings,
        'algorithm_config': {
            'num_fireworks': NUM_FIREWORKS,
            'max_iterations': MAX_ITERATIONS,
            'explosion_amplitude': EXPLOSION_AMPLITUDE,
            'num_sparks': NUM_SPARKS,
            'mutation_rate': MUTATION_RATE
        },
        'statistics': {
            'num_queries_optimized': len(optimized_rankings),
            'avg_final_fitness': np.mean([r['fitness_history'][-1] for r in optimized_rankings.values()])
        },
        'optimization_date': datetime.now().isoformat()
    }
    
    with open(OPTIMIZED_RANKINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved optimized rankings to: {OPTIMIZED_RANKINGS_FILE}")
    
    # Summary
    print("\n" + "="*70)
    print("FIREWORK OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"✅ Optimized rankings for {len(optimized_rankings)} queries")
    print(f"📊 Average final fitness: {results['statistics']['avg_final_fitness']:.4f}")
    print(f"💾 Saved to: {OPTIMIZED_RANKINGS_FILE}")
    print("="*70)

if __name__ == "__main__":
    main()