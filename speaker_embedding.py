import sqlite3
import sqlite_vec
from typing import List
import struct
import concurrent.futures
from pathlib import Path
import os
import torch
import numpy as np
import wespeaker
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances
import warnings
from contextlib import contextmanager

warnings.filterwarnings("ignore")

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

try:
    print("Initializing embedding model...")
    embedding_model = model = wespeaker.load_model("english")
    print("Embedding model initialized")
except Exception as e:
    print(f"error initializing embedding model: {e}")

# Create a folder for saving embeddings if it doesn't exist
EMBEDDING_DIR = "embeddings"
if not os.path.exists(EMBEDDING_DIR):
    os.makedirs(EMBEDDING_DIR)
    print(f"Created directory for embeddings: {EMBEDDING_DIR}")


@contextmanager
def no_grad_context():
    with torch.no_grad():
        yield


def extract_speaker_embeddings(audio_files, save_path):
    all_embeddings = []
    speaker_info = []

    for audio_file in audio_files:
        print(f"\nProcessing file: {audio_file.name}")

        # Extract embedding
        embedding = embedding_model.extract_embedding(audio_file)
        if not isinstance(embedding, np.ndarray):
            embedding = embedding.numpy()

        embedding = embedding / np.linalg.norm(embedding)
        all_embeddings.append(embedding)
        speaker_info.append(
            {
                "audio_file": audio_file.name,
                "local_speaker_label": audio_file.name,
                "embedding": embedding,
            }
        )
        try:
            np.savez_compressed(
                save_path, embeddings=all_embeddings, speaker_info=speaker_info
            )
            print(f"Saved embeddings and speaker info to {save_path}")
        except Exception as e:
            print(f"Error embedding { e }")

        # Free GPU memory after each embedding extraction
        torch.cuda.empty_cache()

    return all_embeddings, speaker_info


def load_embeddings(save_path):
    data = np.load(save_path, allow_pickle=True)
    all_embeddings = data["embeddings"]
    speaker_info = data["speaker_info"]

    return all_embeddings, speaker_info


def compute_similarity_matrix(embeddings, batch_size=1000):
    try:
        """Compute cosine similarity in batches to avoid memory overload."""
        num_embeddings = len(embeddings)
        similarity_matrix = []

        for start_idx in range(0, num_embeddings, batch_size):
            end_idx = min(start_idx + batch_size, num_embeddings)
            batch = embeddings[start_idx:end_idx]

            # Calculate similarities for the batch
            batch_similarities = []
            for embedding in embeddings:
                similarities = 1 - cosine_distances(batch, [embedding])
                batch_similarities.append(similarities.flatten())

            similarity_matrix.extend(batch_similarities)

        # Convert to NumPy array
        similarity_matrix = np.array(similarity_matrix, dtype=np.float32)
        print("Computed cosine similarity matrix in batches.")
        return similarity_matrix
    except Exception as e:
        print(f"Error in compute similarity matrix: {e}")


def cluster_speaker(embeddings):
    embeddings_array = np.vstack(embeddings)
    print(f"Embeddings array shape: {embeddings_array.shape}")

    cosine_sim_matrix = 1 - cosine_distances(embeddings_array)
    print("Computed cosine similarity matrix.")

    distance_matrix = 1 - cosine_sim_matrix
    clustering = HDBSCAN(min_cluster_size=2)
    clustering.fit(distance_matrix)
    labels = clustering.labels_

    print(
        f"Clustering completed. Number of clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}"
    )
    return labels


def assign_global_speaker_ids(labels, speaker_info):
    speaker_mapping = {}
    for idx, info in enumerate(speaker_info):
        key = (info["audio_file"], info["local_speaker_label"])
        cluster_label = labels[idx]
        if cluster_label == -1:
            continue
        speaker_mapping[key] = cluster_label
    return speaker_mapping


def generate_embedding(audio_file):
    save_path = os.path.join(EMBEDDING_DIR, f"embedding_{audio_file.name}.npz")
    return extract_speaker_embeddings([audio_file], save_path=save_path)


def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)


def process_speaker_id(db, path_dir):
    audios = list(Path(path_dir).glob("*.wav"))

    if not audios:
        print("No valid audio files found for processing")
        return

    all_embeddings = []
    all_speaker_info = []
    new_embeddings = []

    # Review if the embeddings have already been calculated
    print("Loading previously calculated embeddings...")

    for audio_file in audios:
        # Save embeddings in the 'embeddings' folder with the audio file name
        save_path = os.path.join(EMBEDDING_DIR, f"embedding_{audio_file.name}.npz")
        # If the embedding is already calculated, load it.
        rows = db.execute(
            """
            SELECT e.embedding FROM files f, embeddings e WHERE f.filename = ?
            """,
            [audio_file.name],
        ).fetchall()
        if os.path.exists(save_path):
            embeddings, speaker_info = load_embeddings(save_path)
            if len(rows) > 0:
                embeddings2 = np.array(
                    [struct.unpack("256f", rows[0][0])], dtype=np.float32
                )
                for i in range(len(embeddings[0])):
                    assert embeddings[0][i] == embeddings2[0][i]
                embeddings = embeddings2
            all_embeddings.extend(embeddings)
            all_speaker_info.extend(speaker_info)
        else:
            new_embeddings.append(audio_file)

    print("Embeddings to be calculated: {}".format(len(new_embeddings)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_embedding = {
            executor.submit(generate_embedding, audio_file): audio_file
            for audio_file in new_embeddings
        }
        for future in concurrent.futures.as_completed(future_embedding):
            # Save embeddings in the 'embeddings' folder with the audio file name
            audio_file = future_embedding[future]
            try:
                embeddings, speaker_info = future.result()
                db.execute(
                    """INSERT INTO files(filename, filesize) VALUES(?, ?)""",
                    [audio_file.name, audio_file.stat().st_size],
                )
                db.execute(
                    """INSERT INTO embeddings(rowid, embedding) 
                       SELECT rowid,? FROM files WHERE filename = ?
                    """,
                    [serialize_f32(embeddings[0]), audio_file.name],
                )
                db.commit()
            except Exception as e:
                print(f"Error in generating embeddings for {audio_file}: {e}")
                raise e
            else:
                all_embeddings.extend(embeddings)
                all_speaker_info.extend(speaker_info)

        # Free GPU memory after processing each file

        torch.cuda.empty_cache()

    print("Calculating cluster")

    # now that we have all embeddings, perform clustering and assign speaker IDs
    if all_embeddings:
        # cosine_sim_matrix = compute_similarity_matrix(all_embeddings)
        labels = cluster_speaker(all_embeddings)
        speaker_mapping = assign_global_speaker_ids(labels, all_speaker_info)

        file_speakers = {}
        update_db_cluster_info(db, speaker_mapping)
        for info in all_speaker_info:
            # Check if 'info' is structured correctly as a dictionary
            if "audio_file" in info and "local_speaker_label" in info:
                audio_file = info["audio_file"]
                key = (audio_file, info["local_speaker_label"])
                global_speaker_id = speaker_mapping.get(key, -1)
                if global_speaker_id == -1:
                    continue
                if audio_file not in file_speakers:
                    file_speakers[audio_file] = set()
                file_speakers[audio_file].add(global_speaker_id)
    # print(file_speakers)

    print("Processing completed")


def update_db_cluster_info(db, speaker_mapping):
    for key, val in speaker_mapping.items():
        res = db.execute(
            """
          UPDATE files SET label = ?
          WHERE filename = ?  
        """,
            [str(val), key[0]],
        )
        if res.rowcount > 0:
            db.commit()


def prepare_db(db_file, clean=False):
    db = sqlite3.connect(db_file)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    if clean:
        db.execute("DELETE FROM embeddings")
        db.execute("DELETE FROM files")
    db.execute("""
        CREATE TABLE IF NOT EXISTS files(
            filename VARCHAR(2048),
            filesize INT,
            label TEXT NULL,
            userinfo TEXT NULL
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS files_name_idx ON files(filename)")
    db.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(embedding float[256])"
    )
    return db


if __name__ == "__main__":
    db_file = "db.sqlite"
    path_wavs = "/tmp/wavs/"
    db = prepare_db(db_file)
    process_speaker_id(db, path_wavs)
    print("done")
