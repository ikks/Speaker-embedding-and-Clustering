## Code Overview

### Main Components:

- **Diarization Pipeline**: Uses the pre-trained `pyannote/speaker-diarization-3.1` pipeline for diarization.
- **Embedding Model**: Uses the pre-trained `pyannote/embedding` model to extract speaker embeddings.
- **Clustering**: Embedding vectors are clustered using DBSCAN based on cosine similarity.
- **Database Interaction**: The script interacts with a database to retrieve and update audio file metadata using SQLAlchemy.
