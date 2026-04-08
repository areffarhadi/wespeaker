"""
Custom loss functions for SASV training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss for speaker recognition
    
    Reference: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    https://arxiv.org/abs/1801.07698
    
    This loss learns more discriminative speaker embeddings by adding an angular margin
    in the angular space, making intra-class samples more compact and inter-class samples
    more separable.
    """
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False):
        """
        Args:
            in_features: Size of input features (embedding dimension)
            out_features: Number of classes (number of speakers)
            scale: Feature scale (s in the paper), controls the radius of decision boundary
            margin: Angular margin (m in the paper), penalty to make classes more separable
            easy_margin: Use easy margin or standard margin
        """
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Weight matrix for classification
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute cos(margin) and sin(margin) for efficiency
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # Threshold
        self.mm = math.sin(math.pi - margin) * margin
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Input embeddings (batch_size, in_features)
            labels: Ground truth labels (batch_size,)
        
        Returns:
            loss: ArcFace loss value
            logits: Classification logits (for compatibility)
        """
        # NUMERICAL STABILITY FIX: Check for NaN/Inf in inputs
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            print(f"WARNING: NaN/Inf detected in embeddings input to ArcFace!")
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize embeddings and weights with epsilon for stability
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
        weight = F.normalize(self.weight, p=2, dim=1, eps=1e-8)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)  # (batch_size, out_features)
        
        # Clip cosine to avoid numerical errors (strict bounds)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Compute sine from cosine: sin(theta) = sqrt(1 - cos^2(theta))
        # Add small epsilon to prevent sqrt of negative numbers
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=0.0, max=1.0))
        
        # Compute cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Clip phi to valid range
        phi = torch.clamp(phi, -1.0, 1.0)
        
        if self.easy_margin:
            # Easy margin: if cos(theta) > 0, use phi, else use cosine
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Standard margin: if cos(theta) > cos(pi - m), use phi, else use cosine - m*sin(pi-m)
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot encoding of labels
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale the output
        output *= self.scale
        
        # NUMERICAL STABILITY CHECK: Verify output before loss computation
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"WARNING: NaN/Inf detected in ArcFace output! Using fallback.")
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(output, labels)
        
        # Final NaN check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf loss detected! Returning zero loss.")
            loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return loss, output


class SASVLoss(nn.Module):
    """
    Combined loss for SASV: Weighted Cross-Entropy (spoof) + ArcFace (speaker)
    """
    def __init__(self, feat_dim, num_speakers, spoof_weight=1.0, speaker_weight=1.0,
                 spoof_class_weights=[10, 1], arcface_scale=30.0, arcface_margin=0.50):
        """
        Args:
            feat_dim: Feature embedding dimension
            num_speakers: Number of speaker classes
            spoof_weight: Weight for spoof detection loss
            speaker_weight: Weight for speaker recognition loss
            spoof_class_weights: Class weights for spoof detection [bonafide_weight, spoof_weight]
            arcface_scale: Scale parameter for ArcFace
            arcface_margin: Angular margin for ArcFace
        """
        super(SASVLoss, self).__init__()
        
        self.spoof_weight = spoof_weight
        self.speaker_weight = speaker_weight
        
        # Spoof detection loss: Weighted Cross-Entropy
        spoof_weights = torch.FloatTensor(spoof_class_weights)
        self.spoof_criterion = nn.CrossEntropyLoss(weight=spoof_weights)
        
        # Speaker recognition loss: ArcFace
        self.arcface = ArcFaceLoss(
            in_features=feat_dim,
            out_features=num_speakers,
            scale=arcface_scale,
            margin=arcface_margin
        )
        
    def forward(self, embeddings, spoof_logits, speaker_embeddings, spoof_labels, speaker_labels):
        """
        Args:
            embeddings: Feature embeddings (not used currently, kept for compatibility)
            spoof_logits: Spoof detection logits (batch_size, 2)
            speaker_embeddings: Speaker embeddings for ArcFace (batch_size, feat_dim)
            spoof_labels: Spoof labels (batch_size,) - 0=bonafide, 1=spoof
            speaker_labels: Speaker labels (batch_size,) - speaker_attack IDs
        
        Returns:
            total_loss: Combined loss
            spoof_loss: Spoof detection loss
            speaker_loss: Speaker recognition loss
            speaker_logits: Speaker classification logits (for monitoring)
        """
        # Spoof detection loss (weighted cross-entropy)
        spoof_loss = self.spoof_criterion(spoof_logits, spoof_labels)
        
        # Speaker recognition loss (ArcFace)
        speaker_loss, speaker_logits = self.arcface(speaker_embeddings, speaker_labels)
        
        # Combined loss
        total_loss = self.spoof_weight * spoof_loss + self.speaker_weight * speaker_loss
        
        return total_loss, spoof_loss, speaker_loss, speaker_logits
    
    def to(self, device):
        """Move all components to device"""
        super().to(device)
        self.spoof_criterion.weight = self.spoof_criterion.weight.to(device)
        return self


def get_sasv_loss(feat_dim, num_speakers, spoof_weight=1.0, speaker_weight=1.0,
                  spoof_class_weights=[10, 1], arcface_scale=30.0, arcface_margin=0.50):
    """
    Factory function to create SASV loss
    
    Args:
        feat_dim: Feature embedding dimension
        num_speakers: Number of speaker classes
        spoof_weight: Weight for spoof detection loss
        speaker_weight: Weight for speaker recognition loss
        spoof_class_weights: Class weights for spoof detection [bonafide_weight, spoof_weight]
        arcface_scale: Scale parameter for ArcFace (default: 30.0)
        arcface_margin: Angular margin for ArcFace (default: 0.50)
    
    Returns:
        SASVLoss instance
    """
    return SASVLoss(
        feat_dim=feat_dim,
        num_speakers=num_speakers,
        spoof_weight=spoof_weight,
        speaker_weight=speaker_weight,
        spoof_class_weights=spoof_class_weights,
        arcface_scale=arcface_scale,
        arcface_margin=arcface_margin
    )

