## Code Overview

### Main Components:

- **Embedding Model**: Uses [wespeaker model](https://github.com/wenet-e2e/wespeaker) for voice embeddings.
- **Clustering**: Embedding vectors are clustered using HDBSCAN based on cosine similarity.
