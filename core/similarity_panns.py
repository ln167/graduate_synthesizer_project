import torch
import numpy as np
from panns_inference import AudioTagging

try:
    from core.profiler import get_profiler
except ImportError:
    from profiler import get_profiler


class PANNsSimilarityMetric:

    def __init__(self, device='cuda', sample_rate=44100):
        self.device = device
        self.sample_rate = sample_rate
        self.profiler = get_profiler()

        print(f"[PANNs] Initializing Cnn14 model on {device}...")

        # Load pre-trained model (downloads checkpoint on first use)
        self.model = AudioTagging(checkpoint_path=None, device=device)
        self.model.model.eval()  # Set to evaluation mode

        # Cache for target embedding (computed once)
        self.target_embedding = None

        print("[PANNs] Model loaded successfully")

    def _get_embedding(self, audio):
        # Convert to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Ensure mono and add batch dimension
        if audio.ndim == 1:
            audio = audio[None, :]  # (1, n_samples)

        # Extract embedding
        with torch.no_grad():
            _, embedding = self.model.inference(audio)

        # Convert to numpy if needed (model.inference returns numpy already)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()

        return embedding.squeeze()  # (2048,)

    def set_target(self, target_audio):
        self.profiler.start("panns_set_target")
        self.target_embedding = self._get_embedding(target_audio)
        self.profiler.end("panns_set_target")
        print(f"[PANNs] Target embedding cached (shape: {self.target_embedding.shape})")

    def compute_distance(self, audio1, audio2):
        self.profiler.start("panns_compute_distance")

        # Extract embeddings
        emb1 = self._get_embedding(audio1)
        emb2 = self._get_embedding(audio2)

        # Compute cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        cosine_similarity = dot_product / (norm_product + 1e-8)

        # Convert to distance (lower = more similar)
        distance = 1.0 - cosine_similarity

        self.profiler.end("panns_compute_distance")
        return float(distance)

    def compute_distance_batch(self, audio_batch, target_audio):
        self.profiler.start("panns_compute_distance_batch")

        # Pre-compute target embedding if not already cached
        if self.target_embedding is None:
            self.set_target(target_audio)

        target_emb = self.target_embedding
        target_norm = np.linalg.norm(target_emb)

        # Convert batch to list if it's a tensor
        if isinstance(audio_batch, torch.Tensor):
            audio_list = [audio_batch[i].cpu().numpy() for i in range(audio_batch.shape[0])]
        else:
            audio_list = audio_batch

        distances = []

        # Process each sample
        for audio in audio_list:
            # Extract embedding
            emb = self._get_embedding(audio)

            # Compute cosine similarity with cached target
            dot_product = np.dot(emb, target_emb)
            emb_norm = np.linalg.norm(emb)
            cosine_similarity = dot_product / (emb_norm * target_norm + 1e-8)

            # Convert to distance
            distance = 1.0 - cosine_similarity
            distances.append(float(distance))

        self.profiler.end("panns_compute_distance_batch")
        return distances

    def compute_similarity(self, audio1, audio2):
        distance = self.compute_distance(audio1, audio2)
        return 1.0 - distance  # Convert distance back to similarity
