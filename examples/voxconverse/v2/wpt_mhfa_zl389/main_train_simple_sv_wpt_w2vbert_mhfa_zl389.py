"""
Simple SV with WPT (Wavelet Prompt Tuning) + W2V-BERT-2.0 + Improved MHFA Head
(zl389 SV encoder checkpoint variant)
================================================================================

Vendored under examples/voxconverse/v2/wpt_mhfa_zl389/ for WeSpeaker diarization
(`run_w2vbert_wpt_mhfa_zl389.sh`). Training imports `dataset_asv` only inside
`train()` so inference can import this module with `losses.py` alone.
Upstream: Encode-explore/USM_FTcode (keep in sync if architecture changes).
================================================================================

Same as main_train_simple_sv_wpt_w2vbert_mhfa.py, but can load zl389 `model_base_*.pth`
(and optionally `model_lmft_*.pth`) into the frozen HF Wav2Vec2BertModel encoder.
HF `facebook/w2v-bert-2.0` (Hub or local) still supplies config + feature extractor.

This implementation combines:
1. WPT (Wavelet Prompt Tuning) with W2V-BERT-2.0 backbone (frozen)
2. IMPROVED MHFA (Multi-Head Factorized Attention) head with Adapter + Deep MLP
   - Aggregates features from ALL transformer layers (not just last layer)
   - Uses factorized Key/Value streams for efficient attention pooling
   - Adds Adapter module (256→128→256) with residual connection
   - Adds Deep Embedding MLP (256→256→256) for richer representations

Architecture:
    W2V-BERT-2.0 (frozen) + WPT (Wavelet Prompt Tuning)
        ↓ Extract features from ALL layers
    All L layers: X ∈ ℝ^(L×T×D)
        ↓ MHFA Head (layer-wise weighted aggregation)
    Key stream: K_feat = Σ softmax(w^k_l) · Z_l
    Value stream: V_feat = Σ softmax(w^v_l) · Z_l
        ↓ Dimension compression + Attention pooling
    Embedding projection (1024→256)
        ↓ Adapter (256→128→256) with residual
        ↓ Deep MLP (256→256→256)
    Final Embedding (256D) → AAM-Softmax Classifier

Improvements over original MHFA:
    - Adapter module: Provides residual learning and better optimization
    - Deep embedding MLP: Richer representations for better discrimination
    - Expected improvement: 3.0% → 2.5-2.8% EER

Reference:
    Paper: https://wildspoof.github.io/pdfs/technical_report/SpoofCeleb_BUT_Speech.pdf
    MHFA: Multi-Head Factorized Attention (Peng et al., SLT 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
from typing import Optional
import numpy as np
import math
from transformers import Wav2Vec2BertModel, AutoFeatureExtractor, AutoConfig
from tqdm import tqdm
from losses import ArcFaceLoss

torch.set_default_dtype(torch.float32)


def load_zl389_sv_encoder_weights(model: Wav2Vec2BertModel, pth_path: str) -> None:
    """
    Load zl389 SpeechBrain-style checkpoint (`model_base_*.pth` or `model_lmft_*.pth`) into HF Wav2Vec2BertModel.
    Expects top-level dict with modules['spk_model'] and keys prefixed `front.encoder.`.
    """
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "modules" not in ckpt:
        raise ValueError(f"Expected dict with 'modules' in {pth_path}")
    mods = ckpt["modules"]
    if "spk_model" not in mods:
        raise ValueError(f"Expected modules['spk_model'] in {pth_path}, got {list(mods.keys())}")
    raw = mods["spk_model"]
    mapped = {}
    for k, v in raw.items():
        if k.startswith("front.encoder."):
            mapped[k[len("front.encoder.") :]] = v
    if not mapped:
        raise ValueError(f"No keys starting with 'front.encoder.' in spk_model ({pth_path})")
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(f"  Loaded SV encoder weights from {pth_path}")
    print(f"    load_state_dict: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    if missing[:5]:
        print(f"    missing (sample): {missing[:5]}")
    if unexpected[:5]:
        print(f"    unexpected (sample): {unexpected[:5]}")


def compute_eer(target_scores, nontarget_scores):
    """Compute Equal Error Rate"""
    all_scores = np.concatenate([target_scores, nontarget_scores])
    labels = np.concatenate([np.ones(len(target_scores)), 
                             np.zeros(len(nontarget_scores))])
    
    thresholds = np.sort(np.unique(all_scores))
    far = np.zeros(len(thresholds))
    frr = np.zeros(len(thresholds))
    
    for i, threshold in enumerate(thresholds):
        predictions = (all_scores >= threshold).astype(int)
        far[i] = np.sum((predictions == 1) & (labels == 0)) / np.sum(labels == 0)
        frr[i] = np.sum((predictions == 0) & (labels == 1)) / np.sum(labels == 1)
    
    abs_diff = np.abs(far - frr)
    min_index = np.argmin(abs_diff)
    eer = (far[min_index] + frr[min_index]) / 2
    threshold = thresholds[min_index]
    
    return eer * 100, threshold


class WaveletBlock(nn.Module):
    """Wavelet transformation block for prompt tokens"""
    
    def __init__(self, wave='haar', J=1, input_dim=1024, output_dim=1024):
        super(WaveletBlock, self).__init__()
        from pytorch_wavelets import DWTForward
        self.dwt = DWTForward(J=J, wave=wave)  
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        B, T, D = x.shape
        assert D == self.input_dim, f"Input dimension (dim={D}) must match WaveletBlock's input_dim ({self.input_dim})"

        x = x.unsqueeze(dim=1)
        LL, band = self.dwt(x)
        bands = band[0]
        LL = LL.unsqueeze(dim=2)
        features = torch.cat((LL, bands), dim=2).view(B, -1, D)
        return features


class WPTW2VBERTMultiLayer(nn.Module):
    """
    W2V-BERT-2.0 with WPT (Wavelet Prompt Tuning) - extracts features from ALL layers
    
    This extracts features from all L transformer layers for MHFA aggregation.
    The backbone is frozen, only prompt tokens are trainable.
    """
    
    def __init__(
        self,
        model_dir,
        num_prompt_tokens=6,
        num_wavelet_tokens=4,
        prompt_dim=1024,
        dropout=0.1,
        encoder_ckpt_path: Optional[str] = None,
        encoder_ckpt_lmft_path: Optional[str] = None,
        gpu_mel_frontend: bool = False,
    ):
        super().__init__()
        
        self.num_prompt_tokens = num_prompt_tokens
        self.num_wavelet_tokens = num_wavelet_tokens
        self.prompt_dim = prompt_dim
        self.gpu_mel = None
        
        # Load W2V-BERT-2.0 (frozen backbone)
        self.config = AutoConfig.from_pretrained(model_dir)
        self.processor = AutoFeatureExtractor.from_pretrained(model_dir)
        self.model = Wav2Vec2BertModel.from_pretrained(model_dir)

        if gpu_mel_frontend:
            try:
                from wespeaker.diar.seamless_m4t_fbank_gpu import SeamlessM4TLogMelGpu
            except ImportError as e:
                raise ImportError(
                    "gpu_mel_frontend=True requires the WeSpeaker package on PYTHONPATH "
                    "(wespeaker.diar.seamless_m4t_fbank_gpu)."
                ) from e
            self.gpu_mel = SeamlessM4TLogMelGpu.from_hf_processor(self.processor)
            print("  Using GPU/torch SeamlessM4T log-mel frontend (no HF CPU round-trip).")

        if encoder_ckpt_path:
            load_zl389_sv_encoder_weights(self.model, encoder_ckpt_path)
        if encoder_ckpt_lmft_path:
            load_zl389_sv_encoder_weights(self.model, encoder_ckpt_lmft_path)
        
        # Freeze backbone
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        num_layers = self.config.num_hidden_layers  # 24 for W2V-BERT-2.0
        
        # Regular prompt embeddings (per layer)
        self.prompt_embeddings = nn.Parameter(
            torch.zeros(num_layers, num_prompt_tokens, prompt_dim)
        )
        
        # Wavelet prompt embeddings (per layer)
        self.wavelet_prompt_embeddings = nn.Parameter(
            torch.zeros(num_layers, num_wavelet_tokens, prompt_dim)
        )
        
        # Wavelet transformation block
        self.wavelet_block = WaveletBlock(wave='haar', J=1, input_dim=prompt_dim, output_dim=prompt_dim)
        
        # DWT with haar can change the token count for odd inputs (ceil(T/2)
        # produces +1 token). Compute the actual output size so we strip the
        # correct number of tokens in the forward pass.
        with torch.no_grad():
            _dummy = torch.zeros(1, num_wavelet_tokens, prompt_dim)
            _dummy_out = self.wavelet_block(_dummy)
            self.actual_wavelet_tokens = _dummy_out.shape[1]
        
        # Xavier initialization
        val = math.sqrt(6. / float(2 * prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        nn.init.uniform_(self.wavelet_prompt_embeddings.data, -val, val)
        
        # Dropout for prompts
        self.prompt_dropout = nn.Dropout(p=dropout)
        
        print(f"  WPT + W2V-BERT-2.0 initialized:")
        print(f"    Regular prompt tokens per layer: {num_prompt_tokens}")
        print(f"    Wavelet prompt tokens per layer: {num_wavelet_tokens} (DWT output: {self.actual_wavelet_tokens})")
        print(f"    Total layers: {num_layers}")
        print(f"    Feature dimension: {prompt_dim}")
    
    def forward(self, audio_data):
        """
        Forward pass with WPT - extracts features from ALL layers
        
        Returns:
            layer_features: List of (B, T, D) tensors, one per layer
        """
        # Store original device
        original_device = audio_data.device

        if self.gpu_mel is not None:
            if audio_data.dim() == 1:
                audio_data = audio_data.unsqueeze(0)
            feat = self.gpu_mel(audio_data.to(dtype=torch.float32))
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)
        else:
            # Process audio - AutoFeatureExtractor expects CPU tensors or numpy arrays
            if isinstance(audio_data, torch.Tensor) and audio_data.is_cuda:
                audio_data_cpu = audio_data.cpu()
                if isinstance(audio_data_cpu, torch.Tensor):
                    audio_data_cpu = audio_data_cpu.numpy()
            else:
                audio_data_cpu = audio_data
                if isinstance(audio_data_cpu, torch.Tensor):
                    audio_data_cpu = audio_data_cpu.numpy()

            processed = self.processor(
                audio_data_cpu,
                sampling_rate=16000,
                return_tensors="pt",
            )

            if "input_features" in processed:
                feat = processed["input_features"].to(original_device)
            elif hasattr(processed, "input_features"):
                feat = processed.input_features.to(original_device)
            elif "input_values" in processed:
                feat = processed["input_values"].to(original_device)
            else:
                feat = list(processed.values())[0].to(original_device)

            if feat.dim() == 2:
                feat = feat.unsqueeze(0)

        feat = feat.to(device=original_device, dtype=torch.float32)
        
        batch_size = feat.size(0)
        
        # Get initial features (frozen)
        with torch.no_grad():
            hidden_state, extract_features = self.model.feature_projection(feat)
            hidden_state = self.model.encoder.dropout(hidden_state)
        
        total_prompt_tokens = self.actual_wavelet_tokens + self.num_prompt_tokens
        
        # Store outputs from ALL layers
        layer_features = []
        
        # Pass through transformer layers WITH prompt + wavelet tuning
        for layer_idx in range(self.config.num_hidden_layers):
            # Get regular prompts for this layer
            prompt = self.prompt_embeddings[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)
            prompt = self.prompt_dropout(prompt)
            
            # Get wavelet prompts for this layer and transform them
            wavelet_prompt = self.wavelet_prompt_embeddings[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)
            wavelet_prompt = self.wavelet_block(wavelet_prompt)
            wavelet_prompt = self.prompt_dropout(wavelet_prompt)
            
            if layer_idx == 0:
                hidden_state = torch.cat([wavelet_prompt, prompt, hidden_state], dim=1)
            else:
                audio_features = hidden_state[:, total_prompt_tokens:, :]
                hidden_state = torch.cat([wavelet_prompt, prompt, audio_features], dim=1)
            
            # Pass through transformer layer
            hidden_state = self.model.encoder.layers[layer_idx](hidden_state)[0]
            
            # Store audio features from this layer (remove prompts)
            audio_only = hidden_state[:, total_prompt_tokens:, :].clone()
            layer_features.append(audio_only)
        
        # Safety: truncate all layers to same time dimension (handles any
        # residual off-by-one from conv padding across transformer versions)
        min_t = min(f.size(1) for f in layer_features)
        if any(f.size(1) != min_t for f in layer_features):
            layer_features = [f[:, :min_t, :] for f in layer_features]
        
        return layer_features


class MHFAHeadImproved(nn.Module):
    """
    Improved Multi-Head Factorized Attention (MHFA) Head
    Based on BUT paper implementation + Adapter + Deep MLP
    
    Architecture:
    1. Layer-wise weighted aggregation (Key and Value streams)
    2. Dimension compression
    3. Multi-head attention pooling
    4. Embedding projection (1024→256)
    5. Adapter module (256→128→256) with residual connection
    6. Deep embedding MLP (256→256→256)
    
    Improvements:
    - Adapter: Provides residual learning for easier optimization
    - Deep MLP: Richer representations for better speaker discrimination
    """
    
    def __init__(self, feature_dim=1024, num_layers=24, num_heads=8, 
                 compression_dim=128, embedding_dim=256, adapter_bottleneck=128,
                 dropout=0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.compression_dim = compression_dim
        self.embedding_dim = embedding_dim
        self.adapter_bottleneck = adapter_bottleneck
        
        # Layer-wise attention weights for Key and Value streams (factorized)
        # w^k ∈ ℝ^L and w^v ∈ ℝ^L
        self.layer_weights_key = nn.Parameter(torch.zeros(num_layers))
        self.layer_weights_value = nn.Parameter(torch.zeros(num_layers))
        
        # Initialize weights uniformly
        nn.init.uniform_(self.layer_weights_key.data, -0.1, 0.1)
        nn.init.uniform_(self.layer_weights_value.data, -0.1, 0.1)
        
        # Dimension compression: feature_dim → compression_dim
        self.key_projection = nn.Linear(feature_dim, compression_dim)
        self.value_projection = nn.Linear(feature_dim, compression_dim)
        
        # Attention projection for multi-head attention
        self.attention_projection = nn.Linear(compression_dim, num_heads)
        
        # Embedding projection: (num_heads * compression_dim) → embedding_dim
        self.embedding_projection = nn.Sequential(
            nn.Linear(num_heads * compression_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )
        
        # Adapter module: embedding_dim → adapter_bottleneck → embedding_dim
        # With residual connection for easier optimization
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, adapter_bottleneck),
            nn.BatchNorm1d(adapter_bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_bottleneck, embedding_dim),
        )
        self.adapter_norm = nn.LayerNorm(embedding_dim)
        
        # Deep embedding MLP: embedding_dim → embedding_dim → embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        print(f"  Improved MHFA Head initialized:")
        print(f"    Feature dimension: {feature_dim}")
        print(f"    Number of layers: {num_layers}")
        print(f"    Number of heads: {num_heads}")
        print(f"    Compression dimension: {compression_dim}")
        print(f"    Embedding dimension: {embedding_dim}")
        print(f"    Adapter bottleneck: {adapter_bottleneck}")
        print(f"    Improvements: Adapter + Deep MLP")
    
    def _init_weights(self):
        """Initialize weights for adapter and embedding MLP"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, layer_features):
        """
        Forward pass through Improved MHFA head
        
        Args:
            layer_features: List of (B, T, D) tensors, one per layer
                           or Tensor of shape (L, B, T, D)
        
        Returns:
            embeddings: (B, embedding_dim) - utterance-level embeddings
        """
        # Convert list to tensor if needed: (L, B, T, D)
        if isinstance(layer_features, list):
            # Stack: (L, B, T, D)
            layer_features = torch.stack(layer_features, dim=0)
        
        L, B, T, D = layer_features.shape
        assert L == self.num_layers, f"Expected {self.num_layers} layers, got {L}"
        assert D == self.feature_dim, f"Expected feature_dim={self.feature_dim}, got {D}"
        
        # Step 1: Layer-wise weighted aggregation
        # Normalize layer weights with softmax
        w_k = F.softmax(self.layer_weights_key, dim=0)  # (L,)
        w_v = F.softmax(self.layer_weights_value, dim=0)  # (L,)
        
        # Weighted sum: K_feat = Σ softmax(w^k_l) · Z_l
        # layer_features: (L, B, T, D)
        # w_k: (L,) -> (L, 1, 1, 1) for broadcasting
        w_k = w_k.view(L, 1, 1, 1)
        w_v = w_v.view(L, 1, 1, 1)
        
        K_feat = (layer_features * w_k).sum(dim=0)  # (B, T, D)
        V_feat = (layer_features * w_v).sum(dim=0)  # (B, T, D)
        
        # Step 2: Dimension compression
        K = self.key_projection(K_feat)  # (B, T, compression_dim)
        V = self.value_projection(V_feat)  # (B, T, compression_dim)
        
        K = self.dropout(K)
        V = self.dropout(V)
        
        # Step 3: Multi-head attention pooling
        # Compute attention weights from Key stream
        A = self.attention_projection(K)  # (B, T, num_heads)
        A = F.softmax(A, dim=1)  # (B, T, num_heads) - normalize over time
        
        # Apply attention to Value stream
        # A: (B, T, num_heads) -> (B, num_heads, T, 1)
        # V: (B, T, compression_dim) -> (B, 1, T, compression_dim)
        A = A.transpose(1, 2).unsqueeze(-1)  # (B, num_heads, T, 1)
        V = V.unsqueeze(1)  # (B, 1, T, compression_dim)
        
        # Weighted pooling: (B, num_heads, compression_dim)
        pooled = (A * V).sum(dim=2)  # (B, num_heads, compression_dim)
        
        # Flatten heads: (B, num_heads * compression_dim)
        pooled = pooled.view(B, self.num_heads * self.compression_dim)
        
        # Step 4: Embedding projection
        embeddings = self.embedding_projection(pooled)  # (B, embedding_dim)
        
        # Step 5: Adapter module with residual connection
        adapted = self.adapter(embeddings)  # (B, embedding_dim)
        adapted = self.adapter_norm(adapted + embeddings)  # Residual connection
        
        # Step 6: Deep embedding MLP
        embeddings = self.embedding_layer(adapted)  # (B, embedding_dim)
        
        return embeddings


class SimpleSVModelWPTW2VBERTMHFA(nn.Module):
    """
    Simple Speaker Verification with WPT + W2V-BERT-2.0 + MHFA Head
    
    Combines:
    - WPT (Wavelet Prompt Tuning) for efficient fine-tuning
    - W2V-BERT-2.0 as frozen backbone
    - MHFA head for layer aggregation and pooling
    """
    
    def __init__(
        self,
        model_dir,
        num_speakers,
        embedding_dim=256,
        num_prompt_tokens=6,
        num_wavelet_tokens=4,
        prompt_dropout=0.1,
        num_heads=8,
        compression_dim=128,
        adapter_bottleneck=128,
        head_dropout=0.1,
        use_arcface=True,
        arcface_margin=0.3,
        arcface_scale=30.0,
        w2vbert_encoder_ckpt: Optional[str] = None,
        w2vbert_encoder_ckpt_lmft: Optional[str] = None,
        gpu_mel_frontend: bool = False,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.use_arcface = use_arcface
        
        # Load W2V-BERT-2.0 with WPT
        print(f"\nLoading W2V-BERT-2.0 with WPT from {model_dir}...")
        self.wpt_w2vbert = WPTW2VBERTMultiLayer(
            model_dir=model_dir,
            num_prompt_tokens=num_prompt_tokens,
            num_wavelet_tokens=num_wavelet_tokens,
            prompt_dim=1024,
            dropout=prompt_dropout,
            encoder_ckpt_path=w2vbert_encoder_ckpt,
            encoder_ckpt_lmft_path=w2vbert_encoder_ckpt_lmft,
            gpu_mel_frontend=gpu_mel_frontend,
        )
        
        # Get number of layers from config
        num_layers = self.wpt_w2vbert.config.num_hidden_layers
        
        # Improved MHFA Head
        print("\nInitializing Improved MHFA Head (with Adapter + Deep MLP)...")
        self.mhfa_head = MHFAHeadImproved(
            feature_dim=1024,
            num_layers=num_layers,
            num_heads=num_heads,
            compression_dim=compression_dim,
            embedding_dim=embedding_dim,
            adapter_bottleneck=adapter_bottleneck,
            dropout=head_dropout
        )
        
        # Classifier
        if use_arcface:
            print(f"\nInitializing ArcFace loss...")
            self.arcface_loss = ArcFaceLoss(
                in_features=embedding_dim,
                out_features=num_speakers,
                scale=arcface_scale,
                margin=arcface_margin,
                easy_margin=False
            )
            self.classifier = None  # ArcFace has its own weight matrix
            print(f"  Using ArcFace: margin={arcface_margin}, scale={arcface_scale}")
        else:
            self.classifier = nn.Linear(embedding_dim, num_speakers)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
            self.arcface_loss = None
            print(f"  Using Linear classifier")
        
        print(f"\n  Model: WPT + W2V-BERT-2.0 + MHFA Head")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Num speakers: {num_speakers}")
        print(f"  Use ArcFace: {use_arcface}")
    
    def extract_embedding(self, audio_data, normalize=True):
        """Extract speaker embedding"""
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)
        
        # Get features from all layers
        layer_features = self.wpt_w2vbert(audio_data)
        
        # Pass through MHFA head
        embeddings = self.mhfa_head(layer_features)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(self, audio_data, labels=None):
        """
        Forward pass - returns embeddings and logits
        
        Args:
            audio_data: Input audio waveforms
            labels: Optional labels for ArcFace loss computation
        
        Returns:
            If labels provided and using ArcFace:
                (embeddings_norm, embeddings_unnorm, logits, loss)
            Otherwise:
                (embeddings_norm, embeddings_unnorm, logits)
        """
        # Get embeddings (NOT normalized for classification)
        embeddings_unnorm = self.extract_embedding(audio_data, normalize=False)
        
        # Also return normalized embeddings for verification
        embeddings_norm = F.normalize(embeddings_unnorm, p=2, dim=1)
        
        # Classify
        if self.use_arcface and self.arcface_loss is not None:
            # ArcFace: compute logits using weight matrix
            weight = F.normalize(self.arcface_loss.weight, p=2, dim=1)
            cosine = F.linear(embeddings_norm, weight)
            
            # Apply ArcFace margin if labels provided
            if labels is not None:
                # Compute ArcFace loss and adjusted logits
                loss, logits = self.arcface_loss(embeddings_norm, labels)
                return embeddings_norm, embeddings_unnorm, logits, loss
            else:
                # For inference, just scale cosine similarity
                logits = cosine * self.arcface_loss.scale
                return embeddings_norm, embeddings_unnorm, logits
        else:
            # Linear classifier
            logits = self.classifier(embeddings_unnorm)
            return embeddings_norm, embeddings_unnorm, logits


def train(args):
    """Training function"""
    from dataset_asv import SpoofCelebASV, SpeakerFolderASV, WavBatchSampler
    from torch.utils.data import DataLoader

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine variable-length mode
    use_variable_length = args.dur_range is not None
    if use_variable_length:
        dur_range = tuple(args.dur_range)
        print(f"\n*** Variable-length training enabled ***")
        print(f"  dur_range: {dur_range[0]}-{dur_range[1]}s (random crop per batch)")
        print(f"  max_eval_dur: {args.max_eval_dur}s")
    else:
        dur_range = None
        print(f"\n  Fixed-length mode: {args.audio_len} samples ({args.audio_len/16000:.3f}s)")

    # Load dataset
    print("\nLoading dataset...")
    if args.train_label:
        # SpoofCeleb-style: CSV protocol file
        train_dataset = SpoofCelebASV(
            path_to_features=args.train_audio,
            path_to_protocol=args.train_label,
            audio_length=args.audio_len,
            bonafide_only=True,
            rawboost=args.rawboost,
            musanrir=args.musanrir,
            reverb_lmdb_path=args.reverb_lmdb_path,
            noise_lmdb_path=args.noise_lmdb_path,
            aug_prob=args.aug_prob,
            variable_length=use_variable_length,
            max_eval_dur=args.max_eval_dur,
        )
    else:
        # Folder-per-speaker structure (e.g. VoxCeleb)
        train_dataset = SpeakerFolderASV(
            path_to_features=args.train_audio,
            audio_length=args.audio_len,
            rawboost=args.rawboost,
            musanrir=args.musanrir,
            reverb_lmdb_path=args.reverb_lmdb_path,
            noise_lmdb_path=args.noise_lmdb_path,
            aug_prob=args.aug_prob,
            variable_length=use_variable_length,
            max_eval_dur=args.max_eval_dur,
        )

    if args.eval_label:
        val_dataset = SpoofCelebASV(
            path_to_features=args.eval_audio,
            path_to_protocol=args.eval_label,
            audio_length=args.audio_len,
            bonafide_only=True,
            rawboost=False,
            musanrir=False,
            variable_length=use_variable_length,
            max_eval_dur=args.max_eval_dur,
        )
    else:
        val_dataset = SpeakerFolderASV(
            path_to_features=args.eval_audio,
            audio_length=args.audio_len,
            rawboost=False,
            musanrir=False,
            variable_length=use_variable_length,
            max_eval_dur=args.max_eval_dur,
        )
    
    if use_variable_length:
        # Variable-length training: use WavBatchSampler
        train_batch_sampler = WavBatchSampler(
            train_dataset,
            dur_range=dur_range,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        # Variable-length eval: each sample keeps its full length, collate with padding
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # batch_size=1 for variable-length eval (no padding needed)
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        # Legacy fixed-length mode
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Get number of speakers
    num_speakers = train_dataset.num_speakers
    print(f"Number of speakers: {num_speakers}")
    
    # Initialize model
    print("\nInitializing model...")
    model = SimpleSVModelWPTW2VBERTMHFA(
        model_dir=args.xlsr,
        num_speakers=num_speakers,
        embedding_dim=args.embedding_dim,
        num_prompt_tokens=args.num_prompt_tokens,
        num_wavelet_tokens=args.num_wavelet_tokens,
        prompt_dropout=args.prompt_dropout,
        num_heads=args.num_heads,
        compression_dim=args.compression_dim,
        adapter_bottleneck=args.adapter_bottleneck,
        head_dropout=args.head_dropout,
        use_arcface=args.use_arcface,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        w2vbert_encoder_ckpt=args.w2vbert_encoder_ckpt,
        w2vbert_encoder_ckpt_lmft=args.w2vbert_encoder_ckpt_lmft,
    ).to(device)
    
    # Optimizer - only train prompt tokens and MHFA head
    trainable_params = []
    trainable_params += [model.wpt_w2vbert.prompt_embeddings]
    trainable_params += [model.wpt_w2vbert.wavelet_prompt_embeddings]
    trainable_params += list(model.mhfa_head.parameters())
    if model.use_arcface and model.arcface_loss is not None:
        trainable_params += list(model.arcface_loss.parameters())
    else:
        trainable_params += list(model.classifier.parameters())
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01
    )
    
    # Training loop
    best_eer = 100.0
    os.makedirs(args.out_fold, exist_ok=True)
    
    print("\nStarting training...")
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        # Keep backbone frozen
        model.wpt_w2vbert.model.eval()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for batch_idx, (waveform, filename, speaker_labels) in enumerate(pbar):
            waveform = waveform.to(device)
            speaker_labels = speaker_labels.to(device)
            
            optimizer.zero_grad()
            
            if args.use_arcface:
                embeddings_norm, embeddings_unnorm, logits, loss = model(waveform, labels=speaker_labels)
            else:
                embeddings_norm, embeddings_unnorm, logits = model(waveform, labels=speaker_labels)
                loss = F.cross_entropy(logits, speaker_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            # ArcFace builds CE logits by shrinking the *target* logit (angular margin).
            # Argmax on those logits is not "predicted speaker" — another class can score
            # higher than the true class even when training is healthy. Use scaled cosine
            # (same as inference) for train accuracy.
            if args.use_arcface and model.arcface_loss is not None:
                w = F.normalize(model.arcface_loss.weight, p=2, dim=1, eps=1e-8)
                cosine_logits = F.linear(embeddings_norm, w) * model.arcface_loss.scale
                preds = cosine_logits.argmax(dim=1)
            else:
                preds = logits.argmax(dim=1)
            train_correct += (preds == speaker_labels).sum().item()
            train_total += speaker_labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        scheduler.step()
        
        # Validation
        if (epoch + 1) % args.interval == 0:
            model.eval()
            val_embeddings = []
            val_filenames = []
            val_labels = []
            
            with torch.no_grad():
                for waveform, filename, speaker_labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                    waveform = waveform.to(device)
                    embeddings = model.extract_embedding(waveform, normalize=True)
                    val_embeddings.append(embeddings.cpu())
                    val_filenames.extend(filename)
                    val_labels.append(speaker_labels)
            
            val_embeddings = torch.cat(val_embeddings, dim=0)
            val_labels = torch.cat(val_labels, dim=0)
            
            # Efficient EER computation: use trial file if provided, otherwise sample pairs
            if args.trial_file and os.path.exists(args.trial_file):
                # Use trial file approach (same as working script)
                filename_to_emb = {}
                for fn, emb in zip(val_filenames, val_embeddings):
                    # Store with full relative path (without extension) as key to avoid
                    # basename collisions (e.g. many VoxCeleb files share names like 00001.wav)
                    key = os.path.splitext(fn)[0]
                    filename_to_emb[key] = emb.numpy()

                target_scores = []
                nontarget_scores = []

                with open(args.trial_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 3:
                            continue

                        label = int(parts[0])
                        enroll_file = parts[1]
                        test_file = parts[2]

                        enroll_basename = os.path.splitext(enroll_file)[0]
                        test_basename = os.path.splitext(test_file)[0]
                        
                        if enroll_basename not in filename_to_emb or test_basename not in filename_to_emb:
                            continue
                        
                        score = np.dot(filename_to_emb[enroll_basename], filename_to_emb[test_basename])
                        
                        if label == 1:
                            target_scores.append(score)
                        else:
                            nontarget_scores.append(score)
                
                if len(target_scores) == 0 or len(nontarget_scores) == 0:
                    print("  Warning: No valid trials found in trial file")
                    eer, threshold = 100.0, 0.0
                else:
                    target_scores = np.array(target_scores)
                    nontarget_scores = np.array(nontarget_scores)
                    eer, threshold = compute_eer(target_scores, nontarget_scores)
            else:
                # Fallback: sample pairs to limit computation (max 100K pairs)
                emb_np = val_embeddings.cpu().numpy()
                labels_np = val_labels.cpu().numpy()
                num_samples = emb_np.shape[0]
                
                # Limit to max 100K pairs for efficiency
                max_pairs = 100000
                total_pairs = num_samples * (num_samples - 1) // 2
                
                if total_pairs > max_pairs:
                    # Sample pairs randomly (vectorized for efficiency)
                    np.random.seed(42)  # For reproducibility
                    idx_i, idx_j = np.triu_indices(num_samples, k=1)
                    sample_indices = np.random.choice(len(idx_i), max_pairs, replace=False)
                    idx_i_sampled = idx_i[sample_indices]
                    idx_j_sampled = idx_j[sample_indices]
                    
                    # Compute scores for sampled pairs (vectorized)
                    emb_i = emb_np[idx_i_sampled]  # (max_pairs, emb_dim)
                    emb_j = emb_np[idx_j_sampled]  # (max_pairs, emb_dim)
                    pair_scores = np.sum(emb_i * emb_j, axis=1)  # (max_pairs,)
                    same_speaker = labels_np[idx_i_sampled] == labels_np[idx_j_sampled]
                else:
                    # Compute all pairs (vectorized)
                    sim_matrix = np.matmul(emb_np, emb_np.T)
                    idx_i, idx_j = np.triu_indices(num_samples, k=1)
                    pair_scores = sim_matrix[idx_i, idx_j]
                    same_speaker = labels_np[idx_i] == labels_np[idx_j]
                
                target_scores = pair_scores[same_speaker]
                nontarget_scores = pair_scores[~same_speaker]
                
                eer, threshold = compute_eer(target_scores, nontarget_scores)
            
            print(f"\nEpoch {epoch+1} Validation:")
            print(f"  EER: {eer:.4f}%")
            print(f"  Threshold: {threshold:.4f}")
            
            # Save best model
            if eer < best_eer:
                best_eer = eer
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'eer': eer,
                }, os.path.join(args.out_fold, 'best_model.pt'))
                print(f"  ✓ Saved best model (EER: {eer:.4f}%)")
            
            # Log results
            with open(os.path.join(args.out_fold, 'val_eer.log'), 'a') as f:
                f.write(f"{epoch+1} {eer:.4f}\n")
    
    print(f"\nTraining complete! Best EER: {best_eer:.4f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='WPT + W2V-BERT-2.0 + MHFA for Simple SV (zl389 encoder .pth optional)'
    )
    
    # Data paths
    parser.add_argument('--train_audio', type=str,
                       default='/home/aref.farhadipour/DATASETS/vox2_dev/dev/aac')
    parser.add_argument('--train_label', type=str, default=None,
                       help='Path to training labels CSV (SpoofCeleb). If not provided, uses folder-per-speaker structure.')
    parser.add_argument('--eval_audio', type=str,
                       default='/home/aref.farhadipour/DATASETS/vox1_test_wav/wav')
    parser.add_argument('--eval_label', type=str, default=None,
                       help='Path to evaluation labels CSV (SpoofCeleb). If not provided, uses folder-per-speaker structure.')
    parser.add_argument('--trial_file', type=str, default=None,
                       help='Path to trial file for efficient evaluation (optional). If not provided, will sample pairs.')
    
    # Audio parameters
    parser.add_argument('--audio_len', type=int, default=64600,
                       help='Audio length in samples (default: 64600 = 4.0375s @ 16kHz). '
                            'Only used when --dur_range is NOT set (legacy fixed-length mode).')
    parser.add_argument('--dur_range', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                       help='Variable-length training: min and max duration in seconds, e.g. --dur_range 2 6. '
                            'Each batch gets a random duration in [MIN, MAX]. '
                            'When set, --audio_len is ignored for training.')
    parser.add_argument('--max_eval_dur', type=float, default=60.0,
                       help='Maximum audio duration in seconds for evaluation (default: 60). '
                            'Only used when --dur_range is set.')
    
    # Augmentation parameters
    parser.add_argument('--rawboost', action='store_true',
                       help='Enable RawBoost augmentation')
    parser.add_argument('--musanrir', action='store_true',
                       help='Enable MUSAN+RIR augmentation')
    parser.add_argument('--reverb_lmdb_path', type=str, default=None,
                       help='Path to RIRS LMDB database for room reverberation augmentation')
    parser.add_argument('--noise_lmdb_path', type=str, default=None,
                       help='Path to MUSAN LMDB database for noise augmentation')
    parser.add_argument('--aug_prob', type=float, default=0.6,
                       help='Probability of applying MUSAN+RIR augmentation (default: 0.6)')
    
    # Model paths
    parser.add_argument('--xlsr', type=str, default='facebook/w2v-bert-2.0',
                       help='HF id or local folder for W2V-BERT-2.0 (config + feature extractor)')
    parser.add_argument(
        '--w2vbert_encoder_ckpt',
        type=str,
        default=None,
        help='zl389 model_base_*.pth: load modules[spk_model] encoder into Wav2Vec2BertModel',
    )
    parser.add_argument(
        '--w2vbert_encoder_ckpt_lmft',
        type=str,
        default=None,
        help='Optional zl389 model_lmft_*.pth: loaded after base (same key mapping)',
    )
    
    # Training hyperparameters
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--interval', type=int, default=1,
                       help='Validation interval (epochs)')
    
    # WPT parameters
    parser.add_argument('--num_prompt_tokens', type=int, default=6)
    parser.add_argument('--num_wavelet_tokens', type=int, default=4)
    parser.add_argument('--prompt_dropout', type=float, default=0.1)
    
    # MHFA head parameters
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads in MHFA')
    parser.add_argument('--compression_dim', type=int, default=128,
                       help='Compression dimension in MHFA')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Final embedding dimension')
    parser.add_argument('--adapter_bottleneck', type=int, default=128,
                       help='Adapter bottleneck dimension (embedding_dim → adapter_bottleneck → embedding_dim)')
    parser.add_argument('--head_dropout', type=float, default=0.1)
    
    # Loss parameters
    parser.add_argument('--use_arcface', action='store_true',
                       help='Use ArcFace loss instead of standard cross-entropy')
    parser.add_argument('--arcface_margin', type=float, default=0.3)
    parser.add_argument('--arcface_scale', type=float, default=30.0)
    
    # Output
    parser.add_argument('--out_fold', type=str, required=True,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Save arguments
    os.makedirs(args.out_fold, exist_ok=True)
    with open(os.path.join(args.out_fold, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    train(args)

