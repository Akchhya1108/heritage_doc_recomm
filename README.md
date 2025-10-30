# Heritage Document Recommendation System

A multi-source heritage document recommendation system using Knowledge Graphs, autoencoders, and graph-based ranking algorithms.

## 🎯 Project Overview

This system collects heritage documents from multiple authoritative sources, constructs a rich Knowledge Graph, and provides intelligent recommendations based on semantic similarity and graph topology.

### Key Features
- **Multi-source data collection**: Wikipedia, UNESCO, Indian Heritage, Archive.org
- **Automated metadata extraction**: Using NLP and entity recognition
- **Deep learning classification**: Autoencoder-based document clustering
- **Rich Knowledge Graph**: Lesk similarity + external KG integration
- **Advanced ranking**: SimRank + Horn's Index + Firework Algorithm

## 📊 Dataset Statistics

- **Total Documents**: 150-200 heritage documents
- **Sources**: 4 diverse heritage databases
- **Coverage**: Global + India-specific monuments
- **Metadata**: Categories, entities, temporal data, geographic info

## 🏗️ Architecture
```
Data Collection → Metadata Extraction → Classification (Autoencoder) 
→ Knowledge Graph Construction → Ranking (SimRank + Horn's Index) 
→ Query Processing → Recommendations
```

## 📁 Project Structure
```
heritage-doc-recommender/
├── data/                      # Data storage
│   ├── raw/                   # Raw collected documents
│   ├── cleaned/               # Preprocessed documents
│   └── metadata/              # Extracted metadata
├── src/                       # Source code
│   ├── 1a_collect_wikipedia.py
│   ├── 1b_collect_unesco.py
│   ├── 1c_collect_indian_heritage.py
│   ├── 1d_collect_archives.py
│   ├── 1_collect_all_sources.py
│   ├── 2_data_cleaning.py
│   ├── 3_metadata_extraction.py
│   ├── 4_autoencoder_classification.py
│   ├── 5_kg_construction.py
│   ├── 6_kg_integration.py
│   ├── 7_ranking_simrank.py
│   ├── 8_firework_algorithm.py
│   └── 9_query_processing.py
├── models/                    # Trained models
├── knowledge_graph/           # KG data and visualizations
├── utils/                     # Utility functions
├── requirements.txt
└── README.md
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/heritage-doc-recommender.git
cd heritage-doc-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"
```

## 📖 Usage

### 1. Data Collection
Collect heritage documents from multiple sources:
```bash
python src/1_collect_all_sources.py
```

This will scrape:
- Wikipedia heritage articles
- UNESCO World Heritage Sites
- Indian monuments and archaeological sites
- Archive.org historical documents

### 2. Data Preprocessing
Clean and normalize collected documents:
```bash
python src/2_data_cleaning.py
```

### 3. Metadata Extraction
Extract categories, entities, and keywords:
```bash
python src/3_metadata_extraction.py
```

### 4. Document Classification
Train autoencoder and classify documents:
```bash
python src/4_autoencoder_classification.py
```

### 5. Knowledge Graph Construction
Build and integrate Knowledge Graph:
```bash
python src/5_kg_construction.py
python src/6_kg_integration.py
```

### 6. Query and Recommend
Process queries and get recommendations:
```bash
python src/9_query_processing.py
```

## 🔬 Methodology

### Data Collection
- **Automated scraping** from 4 diverse sources
- **Validation pipeline** to ensure content quality
- **Deduplication** across sources

### Knowledge Graph
- **Entities**: Documents, monuments, locations, time periods, cultural themes
- **Relationships**: part_of, similar_to, located_in, belongs_to_period
- **Enrichment**: Integration with Google Knowledge Graph API

### Ranking Algorithm
1. **SimRank**: Structural similarity in KG
2. **Horn's Index**: Entity importance weighting  
3. **Firework Algorithm**: Metaheuristic optimization

## 📊 Evaluation Metrics

- Precision@K
- Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- Mean Average Precision (MAP)
- Knowledge Graph density and connectivity

## 🎓 Academic Context

This project was developed as a final year project for [Your University Name] and is intended for submission to [Conference Name].

### Citation
If you use this work, please cite:
```
[Your Name]. (2025). Heritage Document Recommendation System using 
Knowledge Graphs and Deep Learning. [Conference/Journal Name].
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 👤 Author

**[Your Name]**
- GitHub: [@your_username](https://github.com/your_username)
- Email: your.email@example.com

## 🙏 Acknowledgments

- UNESCO World Heritage Centre
- Wikipedia contributors
- Archaeological Survey of India
- Internet Archive

## 📚 References

1. [Key papers on Knowledge Graphs]
2. [SimRank algorithm paper]
3. [Autoencoder architectures]
4. [Heritage information systems]

---

**Status**: 🚧 In Development | **Version**: 1.0.0 | **Last Updated**: October 2025