import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import torch

# Directories
META_DIR = "data/metadata"
EMBEDDINGS_DIR = "data/embeddings"
CLEAN_DIR = "data/cleaned data"

# Files
ENRICHED_META_FILE = os.path.join(META_DIR, "enriched_metadata.json")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "document_embeddings.npy")
MAPPING_FILE = os.path.join(EMBEDDINGS_DIR, "embedding_mapping.json")

# Model configuration
MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast, good quality, 384 dimensions
# Alternative: 'all-mpnet-base-v2' (768 dimensions, slower but better quality)

def load_documents():
    """Load enriched metadata and document texts"""
    print("\n[Phase 1] Loading documents...")
    
    if not os.path.exists(ENRICHED_META_FILE):
        print(f"✗ Error: {ENRICHED_META_FILE} not found!")
        print("Run 2_extract_metadata.py first.")
        return None, None
    
    with open(ENRICHED_META_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded metadata for {len(metadata)} documents")
    
    # Load document texts
    documents = []
    valid_metadata = []
    
    for idx, meta in enumerate(metadata, 1):
        cleaned_path = meta.get('cleaned_path', '')
        
        if os.path.exists(cleaned_path):
            try:
                with open(cleaned_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                    # Create rich text representation for embedding
                    # Combine title, keywords, and content for better embeddings
                    title = meta.get('title', '')
                    keywords = ' '.join(meta.get('keywords_tfidf', [])[:10])
                    
                    # Take first 2000 chars of content (for speed and quality)
                    content_snippet = text[:2000]
                    
                    combined_text = f"{title}. {keywords}. {content_snippet}"
                    
                    documents.append(combined_text)
                    valid_metadata.append(meta)
                    
            except Exception as e:
                print(f"  ⚠ Could not read {cleaned_path}: {e}")
        else:
            print(f"  ⚠ File not found: {cleaned_path}")
    
    print(f"✓ Loaded {len(documents)} document texts")
    
    return valid_metadata, documents

def generate_embeddings(documents, metadata):
    """Generate embeddings using Sentence Transformers"""
    print("\n[Phase 2] Initializing embedding model...")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device.upper()}")
    
    # Load model
    print(f"  Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"  ✓ Model loaded (embedding dimension: {model.get_sentence_embedding_dimension()})")
    
    # Generate embeddings
    print(f"\n[Phase 3] Generating embeddings for {len(documents)} documents...")
    print("  This may take 5-10 minutes depending on your hardware...\n")
    
    # Process in batches for efficiency
    batch_size = 32
    embeddings = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_embeddings = model.encode(
            batch,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        embeddings.extend(batch_embeddings)
        
        # Progress update
        processed = min(i + batch_size, len(documents))
        print(f"  Processed: {processed}/{len(documents)} documents")
    
    embeddings = np.array(embeddings)
    
    print(f"\n✓ Generated embeddings with shape: {embeddings.shape}")
    print(f"  (Documents: {embeddings.shape[0]}, Dimensions: {embeddings.shape[1]})")
    
    return embeddings, model.get_sentence_embedding_dimension()

def save_embeddings(embeddings, metadata, embedding_dim):
    """Save embeddings and mapping"""
    print("\n[Phase 4] Saving embeddings...")
    
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Save embeddings as numpy array
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"  ✓ Saved embeddings to: {EMBEDDINGS_FILE}")
    
    # Create mapping file
    mapping = {
        'embedding_dimension': embedding_dim,
        'model_name': MODEL_NAME,
        'total_documents': len(metadata),
        'created_at': datetime.now().isoformat(),
        'documents': []
    }
    
    for idx, meta in enumerate(metadata):
        doc_info = {
            'index': idx,
            'title': meta.get('title', ''),
            'file_name': meta.get('file_name', ''),
            'heritage_types': meta.get('classifications', {}).get('heritage_types', []),
            'domains': meta.get('classifications', {}).get('domains', []),
            'time_period': meta.get('classifications', {}).get('time_period', ''),
            'region': meta.get('classifications', {}).get('region', ''),
            'source': meta.get('source', '')
        }
        mapping['documents'].append(doc_info)
    
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved mapping to: {MAPPING_FILE}")

def compute_statistics(embeddings):
    """Compute embedding statistics"""
    print("\n[Phase 5] Computing statistics...")
    
    # Basic statistics
    mean_norm = np.linalg.norm(embeddings, axis=1).mean()
    std_norm = np.linalg.norm(embeddings, axis=1).std()
    
    print(f"  Mean embedding norm: {mean_norm:.4f}")
    print(f"  Std embedding norm: {std_norm:.4f}")
    
    # Compute pairwise similarity for sample
    print("\n  Computing similarity matrix for first 10 documents...")
    sample_embeddings = embeddings[:10]
    similarity_matrix = np.dot(sample_embeddings, sample_embeddings.T)
    
    print(f"  Average pairwise similarity: {similarity_matrix.mean():.4f}")
    print(f"  Min similarity: {similarity_matrix.min():.4f}")
    print(f"  Max similarity: {similarity_matrix.max():.4f}")
    
    return similarity_matrix

def test_embeddings(embeddings, metadata):
    """Test embeddings with a similarity search"""
    print("\n[Phase 6] Testing embeddings...")
    
    # Pick a random document
    test_idx = 0  # First document
    test_doc = metadata[test_idx]
    test_embedding = embeddings[test_idx]
    
    print(f"\n  Test document: '{test_doc['title']}'")
    print(f"  Heritage Type: {test_doc.get('classifications', {}).get('heritage_types', [])}")
    
    # Find most similar documents
    similarities = np.dot(embeddings, test_embedding)
    top_k = 6  # Top 5 + the document itself
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\n  Top {top_k-1} most similar documents:")
    for rank, idx in enumerate(top_indices[1:], 1):  # Skip the first (itself)
        similar_doc = metadata[idx]
        sim_score = similarities[idx]
        print(f"    {rank}. {similar_doc['title'][:60]}")
        print(f"       Similarity: {sim_score:.4f}")
        print(f"       Type: {similar_doc.get('classifications', {}).get('heritage_types', [])}")

def main():
    print("="*70)
    print("DOCUMENT EMBEDDING GENERATION")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load documents
    metadata, documents = load_documents()
    
    if metadata is None or documents is None:
        return
    
    # Generate embeddings
    embeddings, embedding_dim = generate_embeddings(documents, metadata)
    
    # Save embeddings
    save_embeddings(embeddings, metadata, embedding_dim)
    
    # Compute statistics
    compute_statistics(embeddings)
    
    # Test embeddings
    test_embeddings(embeddings, metadata)
    
    # Summary
    print("\n" + "="*70)
    print("EMBEDDING GENERATION COMPLETE")
    print("="*70)
    print(f"✅ Generated embeddings for {len(metadata)} documents")
    print(f"📊 Embedding dimension: {embedding_dim}")
    print(f"💾 Files created:")
    print(f"   - {EMBEDDINGS_FILE}")
    print(f"   - {MAPPING_FILE}")
    print(f"\n📈 Storage info:")
    print(f"   Embeddings size: {embeddings.nbytes / (1024*1024):.2f} MB")
    print("="*70)

if __name__ == "__main__":
    main()