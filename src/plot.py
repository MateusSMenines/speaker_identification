import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import glob
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_distances


def initialize_recognizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recognizer = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmp_model",
        run_opts={"device": device},
    )
    return recognizer, device


def process_audio_file(file_path, recognizer, device):
    with torch.no_grad():
        signal, fs = torchaudio.load(file_path)
        signal = signal / torch.abs(signal).max()
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)
        embedding = recognizer.encode_batch(signal.to(device))
        embedding = embedding.squeeze(0).cpu().numpy()
        return embedding


def collect_embeddings(file_list, recognizer, device):
    embeddings = {}
    for class_path in glob.glob(file_list + "/*"):
        class_id = int(class_path[-4:])
        for img_path in glob.glob(class_path + "/*"):
            embedding_np = process_audio_file(img_path, recognizer, device)
            if class_id not in embeddings:
                embeddings[class_id] = []
            embeddings[class_id].append(embedding_np[0])
        print(f"Processado lasse: {class_id}")
    return embeddings


def plot_tsne(label_embedding_dict):
    embeddings = []
    labels = []
    for label, embedding_list in label_embedding_dict.items():
        embeddings.extend(embedding_list)
        labels.extend([label] * len(embedding_list))
    n_samples = len(embeddings)
    perplexity = min(30, n_samples - 1)
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    data_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        plt.scatter(
            data_2d[idx, 0],
            data_2d[idx, 1],
            label=f"Label {label}",
            s=50,
        )
    plt.title("Visualização de Clusters usando t-SNE")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend()
    plt.show()


def main():
    recognizer, device = initialize_recognizer()
    file_list = "teste/"
    embeddings = collect_embeddings(file_list, recognizer, device)
    plot_tsne(embeddings)


if __name__ == "__main__":
    main()
