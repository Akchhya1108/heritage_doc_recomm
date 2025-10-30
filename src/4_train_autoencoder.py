import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Directories
EMBEDDINGS_DIR = "data/embeddings"
CLASSIFIED_DIR = "data/classified"
MODELS_DIR = "models/autoencoder"
META_DIR = "data/metadata"

# Files
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "document_embeddings.npy")
MAPPING_FILE = os.path.join(EMBEDDINGS_DIR, "embedding_mapping.json")
ENRICHED_META_FILE = os.path.join(META_DIR, "enriched_metadata.json")
CLASSIFIED_FILE = os.path.join(CLASSIFIED_DIR, "classified_documents.json")
MODEL_FILE = os.path.join(MODELS_DIR, "autoencoder_model.pth")

# Hyperparameters
INPUT_DIM = 384  # From sentence transformer
ENCODING_DIM = 64  # Compressed representation
HIDDEN_DIM = 128  # Hidden layer
NUM_CLUSTERS = 12  # Number of heritage clusters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ========== AUTOENCODER ARCHITECTURE ==========

class HeritageAutoencoder(nn.Module):
    """Autoencoder for heritage document embeddings"""
    
    def __init__(self, input_dim, hidden_dim, encoding_dim):
        super(HeritageAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Get encoded representation"""
        return self.encoder(x)

# ========== TRAINING FUNCTIONS ==========

def load_data():
    """Load embeddings and metadata"""
    print("\n[Phase 1] Loading data...")
    
    # Load embeddings
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"✗ Error: {EMBEDDINGS_FILE} not found!")
        print("Run 3_generate_embeddings.py first.")
        return None, None
    
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"✓ Loaded embeddings: {embeddings.shape}")
    
    # Load metadata
    with open(ENRICHED_META_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded metadata for {len(metadata)} documents")
    
    return embeddings, metadata

def train_autoencoder(embeddings):
    """Train the autoencoder"""
    print("\n[Phase 2] Training autoencoder...")
    
    # Prepare data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(embeddings).to(device)
    
    # Create DataLoader
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = HeritageAutoencoder(INPUT_DIM, HIDDEN_DIM, ENCODING_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n  Model architecture:")
    print(f"    Input: {INPUT_DIM} → Hidden: {HIDDEN_DIM} → Encoding: {ENCODING_DIM}")
    print(f"    Training for {EPOCHS} epochs...\n")
    
    # Training loop
    losses = []
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        
        for batch_idx, (batch_X,) in enumerate(dataloader):
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_X)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")
    
    print(f"\n  ✓ Training complete!")
    print(f"    Final loss: {losses[-1]:.6f}")
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': INPUT_DIM,
        'hidden_dim': HIDDEN_DIM,
        'encoding_dim': ENCODING_DIM,
        'final_loss': losses[-1]
    }, MODEL_FILE)
    
    print(f"  ✓ Model saved to: {MODEL_FILE}")
    
    return model, losses

def encode_documents(model, embeddings):
    """Encode all documents using trained autoencoder"""
    print("\n[Phase 3] Encoding documents...")
    
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        X = torch.FloatTensor(embeddings).to(device)
        encoded = model.encode(X).cpu().numpy()
    
    print(f"  ✓ Encoded to shape: {encoded.shape}")
    
    return encoded

def cluster_documents(encoded_embeddings, metadata):
    """Cluster documents using K-Means"""
    print(f"\n[Phase 4] Clustering documents into {NUM_CLUSTERS} groups...")
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(encoded_embeddings)
    
    print(f"  ✓ Clustering complete")
    
    # Analyze clusters
    print(f"\n  Cluster distribution:")
    cluster_counts = np.bincount(cluster_labels)
    for cluster_id, count in enumerate(cluster_counts):
        print(f"    Cluster {cluster_id}: {count} documents")
    
    # Assign semantic labels to clusters based on dominant heritage types
    cluster_info = assign_cluster_labels(cluster_labels, metadata)
    
    return cluster_labels, cluster_info

def assign_cluster_labels(cluster_labels, metadata):
    """Assign semantic labels to clusters"""
    print(f"\n  Analyzing cluster characteristics...")
    
    cluster_info = {}
    
    for cluster_id in range(NUM_CLUSTERS):
        # Get documents in this cluster
        cluster_docs = [metadata[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        
        # Count heritage types
        heritage_types = []
        domains = []
        time_periods = []
        regions = []
        
        for doc in cluster_docs:
            classifications = doc.get('classifications', {})
            heritage_types.extend(classifications.get('heritage_types', []))
            domains.extend(classifications.get('domains', []))
            time_periods.append(classifications.get('time_period', 'unknown'))
            regions.append(classifications.get('region', 'unknown'))
        
        # Find most common characteristics
        from collections import Counter
        
        top_heritage = Counter(heritage_types).most_common(2)
        top_domain = Counter(domains).most_common(1)
        top_period = Counter(time_periods).most_common(1)
        top_region = Counter(regions).most_common(1)
        
        # Create cluster label
        heritage_label = ', '.join([h[0] for h in top_heritage]) if top_heritage else 'mixed'
        domain_label = top_domain[0][0] if top_domain else 'general'
        period_label = top_period[0][0] if top_period else 'various'
        region_label = top_region[0][0] if top_region else 'various'
        
        cluster_info[cluster_id] = {
            'label': f"{domain_label.title()} {heritage_label.title()}",
            'heritage_types': [h[0] for h in top_heritage[:3]],
            'dominant_domain': domain_label,
            'dominant_period': period_label,
            'dominant_region': region_label,
            'size': len(cluster_docs),
            'sample_titles': [doc['title'] for doc in cluster_docs[:3]]
        }
        
        print(f"\n    Cluster {cluster_id}: {cluster_info[cluster_id]['label']}")
        print(f"      Size: {cluster_info[cluster_id]['size']}")
        print(f"      Period: {period_label}, Region: {region_label}")
        print(f"      Samples: {', '.join(cluster_info[cluster_id]['sample_titles'][:2])}")
    
    return cluster_info

def visualize_clusters(encoded_embeddings, cluster_labels, cluster_info):
    """Visualize clusters using t-SNE"""
    print(f"\n[Phase 5] Generating visualization...")
    
    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(encoded_embeddings)
    
    # Create plot
    plt.figure(figsize=(15, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_CLUSTERS))
    
    for cluster_id in range(NUM_CLUSTERS):
        mask = cluster_labels == cluster_id
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[cluster_id]],
            label=f"C{cluster_id}: {cluster_info[cluster_id]['label']}",
            alpha=0.6,
            s=50
        )
    
    plt.title('Heritage Document Clusters (t-SNE Visualization)', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save visualization
    viz_file = os.path.join(CLASSIFIED_DIR, 'cluster_visualization.png')
    os.makedirs(CLASSIFIED_DIR, exist_ok=True)
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Visualization saved to: {viz_file}")
    
    plt.close()

def save_classifications(metadata, cluster_labels, encoded_embeddings, cluster_info):
    """Save classified documents"""
    print(f"\n[Phase 6] Saving classifications...")
    
    os.makedirs(CLASSIFIED_DIR, exist_ok=True)
    
    classified_docs = []
    
    for idx, (meta, cluster_id, encoding) in enumerate(zip(metadata, cluster_labels, encoded_embeddings)):
        classified = {
            **meta,  # Keep all original metadata
            'cluster_id': int(cluster_id),
            'cluster_label': cluster_info[cluster_id]['label'],
            'cluster_domain': cluster_info[cluster_id]['dominant_domain'],
            'encoded_representation': encoding.tolist(),
            'classification_date': datetime.now().isoformat()
        }
        classified_docs.append(classified)
    
    # Save to JSON
    with open(CLASSIFIED_FILE, 'w', encoding='utf-8') as f:
        json.dump(classified_docs, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved {len(classified_docs)} classified documents")
    print(f"  ✓ Output: {CLASSIFIED_FILE}")
    
    # Save cluster info
    cluster_info_file = os.path.join(CLASSIFIED_DIR, 'cluster_info.json')
    with open(cluster_info_file, 'w', encoding='utf-8') as f:
        json.dump(cluster_info, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Cluster info: {cluster_info_file}")

def main():
    print("="*70)
    print("AUTOENCODER TRAINING & DOCUMENT CLASSIFICATION")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Input dim: {INPUT_DIM}")
    print(f"  Encoding dim: {ENCODING_DIM}")
    print(f"  Clusters: {NUM_CLUSTERS}")
    print(f"  Epochs: {EPOCHS}")
    
    # Load data
    embeddings, metadata = load_data()
    
    if embeddings is None:
        return
    
    # Train autoencoder
    model, losses = train_autoencoder(embeddings)
    
    # Encode documents
    encoded_embeddings = encode_documents(model, embeddings)
    
    # Cluster documents
    cluster_labels, cluster_info = cluster_documents(encoded_embeddings, metadata)
    
    # Visualize clusters
    visualize_clusters(encoded_embeddings, cluster_labels, cluster_info)
    
    # Save classifications
    save_classifications(metadata, cluster_labels, encoded_embeddings, cluster_info)
    
    # Summary
    print("\n" + "="*70)
    print("CLASSIFICATION COMPLETE")
    print("="*70)
    print(f"✅ Trained autoencoder: {INPUT_DIM}D → {ENCODING_DIM}D")
    print(f"✅ Classified {len(metadata)} documents into {NUM_CLUSTERS} clusters")
    print(f"\n📊 Files created:")
    print(f"   - {CLASSIFIED_FILE}")
    print(f"   - {MODEL_FILE}")
    print(f"   - cluster_visualization.png")
    print("="*70)

if __name__ == "__main__":
    main()