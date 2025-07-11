from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
import sqlite3
import sqlite_vec
from typing import List
import struct
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import wespeaker
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances
import warnings
from contextlib import contextmanager


@contextmanager
def no_grad_context():
    with torch.no_grad():
        yield


def extract_speaker_embedding(embedding_model, audio_file):
    # Extract embedding
    embedding = embedding_model.extract_embedding(audio_file)
    if not isinstance(embedding, np.ndarray):
        embedding = embedding.numpy()

    embedding = embedding / np.linalg.norm(embedding)
    speaker_info = {
        "audio_file": audio_file.name,
        "embedding": embedding,
    }

    # Free GPU memory after each embedding extraction
    torch.cuda.empty_cache()

    return embedding, speaker_info


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
        f"Clustering completed. Number of speakers found: {len(set(labels)) - (1 if -1 in labels else 0)}"
    )
    return labels


def assign_global_speaker_ids(labels, speaker_info):
    speaker_mapping = {}
    for idx, info in enumerate(speaker_info):
        cluster_label = labels[idx]
        if cluster_label == -1:
            continue
        speaker_mapping[info["audio_file"]] = cluster_label
    return speaker_mapping


def generate_embedding(embedding_model, audio_file):
    return extract_speaker_embedding(embedding_model, audio_file)


def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)


def process_speaker_id(embedding_model, df, db, path_dir):
    audios = list(Path(path_dir).glob("*.wav"))

    if not audios:
        print("No valid audio files found for processing")
        return

    n_audios = len(audios)
    all_embeddings = []
    all_speaker_info = []
    new_embeddings = []

    processed = 0
    print(f"Reviewing {n_audios} files")
    # Review if the embeddings have already been calculated
    print("Loading previously calculated embeddings...")

    for audio_file in audios:
        rows = db.execute(
            """
            SELECT e.embedding FROM files f, embeddings e
            WHERE e.rowid = f.rowid AND f.filename = ?
            """,
            [audio_file.name],
        ).fetchall()
        # If the embedding is already calculated, load it.
        if len(rows) > 0:
            embeddings = np.array([struct.unpack("256f", rows[0][0])], dtype=np.float32)
            speaker_info = {
                "audio_file": audio_file.name,
                "embedding": embeddings,
            }
            all_embeddings.append(embeddings[0])
            all_speaker_info.append(speaker_info)
            processed += 1
        else:
            new_embeddings.append(audio_file)

    print("Embeddings to be calculated: {}".format(len(new_embeddings)))
    if is_notebook():
        progress_bar = tqdm_nb
    else:
        progress_bar = tqdm
    for audio_file in progress_bar(new_embeddings):
        # Save embeddings in the 'embeddings' folder with the audio file name
        try:
            embedding, speaker_info = generate_embedding(embedding_model, audio_file)

            userinfo = f"""{{"transcription": "{df[df["fn"] == audio_file.name].iloc[0]["transcription"]}"}}"""
            db.execute(
                """INSERT INTO files(filename, filesize, userinfo) VALUES(?, ?, ?)""",
                [
                    audio_file.name,
                    audio_file.stat().st_size,
                    userinfo,
                ],
            )
            db.execute(
                """INSERT INTO embeddings(rowid, embedding)
                   SELECT rowid,? FROM files WHERE filename = ?
                """,
                [serialize_f32(embedding), audio_file.name],
            )
            db.commit()
        except Exception as e:
            print(f"Error in generating embeddings for {audio_file}: {e}")
            raise e
        else:
            all_embeddings.append(embedding)
            all_speaker_info.append(speaker_info)

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
            if "audio_file" in info:
                audio_file = info["audio_file"]
                global_speaker_id = speaker_mapping.get(audio_file, -1)
                if global_speaker_id == -1:
                    continue
                if audio_file not in file_speakers:
                    file_speakers[audio_file] = set()
                file_speakers[audio_file].add(global_speaker_id)


def update_db_cluster_info(db, speaker_mapping):
    for key, val in speaker_mapping.items():
        res = db.execute(
            """
          UPDATE files SET label = ?
          WHERE filename = ?  
        """,
            [str(val), key],
        )
        if res.rowcount > 0:
            db.commit()


def prepare_db(db_file, clean=False):
    db = sqlite3.connect(db_file)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    if clean:
        db.execute("DROP TABLE embeddings")
        db.execute("DROP TABLE files")
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


def load_metadata(csv_filename, delimiter="|"):
    # Has two columns delimited by |
    # filename|transcription
    df = pd.read_csv(
        csv_filename, header=None, delimiter=delimiter, names=["fn", "transcription"]
    )
    return df


def load_sp_metadata(csv_filename, delimiter=","):
    # Has three columns
    # filename with prefix audio/,filesize,transcription
    df = pd.read_csv(
        csv_filename,
        header=1,
        delimiter=delimiter,
        usecols=[0, 2],
        names=["fn", "transcription"],
    )
    df["fn"] = df["fn"].str.replace("audios/", "")
    return df


def nb_cluster_task(db_file, csv_file, path_wavs, delimiter=","):
    df = load_sp_metadata(csv_file, delimiter)
    cluster_task(db_file, df, path_wavs)


def local_cluster_task(db_file, csv_file, path_wavs, delimiter=","):
    df = load_metadata(csv_file, delimiter)
    cluster_task(db_file, df, path_wavs)


def cluster_task(db_file, df, path_wavs):
    db = prepare_db(db_file)
    warnings.filterwarnings("ignore")

    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    try:
        print("Initializing embedding model...")
        embedding_model = wespeaker.load_model("english")
        print("Embedding model initialized")
    except Exception as e:
        print(f"error initializing embedding model: {e}")

    process_speaker_id(embedding_model, df, db, path_wavs)
    print("done")


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def main():
    db_file = "db.sqlite"
    csv_file = "/tmp/metadata.csv"
    path_wavs = "/tmp/wavs/"
    local_cluster_task(db_file, csv_file, path_wavs, "|")


if __name__ == "__main__":
    main()
