import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config

class Wav2Vec2SpeechClassifier(nn.Module):
    """
    Wav2Vec2 Deep Speech Affect Model Architecture:
    Raw waveform -> Wav2Vec2 Encoder -> Mean Pooling -> Dropout -> Linear Classifier Head

    Supports configurable freezing of the Wav2Vec2 encoder backbone.
    """
    def __init__(
        self,
        model_name_or_path: str = "facebook/wav2vec2-base-960h",
        num_classes: int = 6,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        freeze_encoder: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_encoder_flag = freeze_encoder

        # Load pretrained Wav2Vec2 backbone model
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name_or_path)
        except Exception:
            # Fallback to local default initialization if offline
            cfg = Wav2Vec2Config.from_pretrained(model_name_or_path) if hasattr(Wav2Vec2Config, 'from_pretrained') else Wav2Vec2Config()
            self.wav2vec2 = Wav2Vec2Model(cfg)

        # Set encoder freeze status
        self.set_freeze_encoder(freeze_encoder)

        encoder_hidden_size = self.wav2vec2.config.hidden_size  # e.g., 768

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(encoder_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def set_freeze_encoder(self, freeze: bool):
        """Enable or disable gradient calculation for the Wav2Vec2 encoder."""
        self.freeze_encoder_flag = freeze
        for param in self.wav2vec2.parameters():
            param.requires_grad = not freeze

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass:
        input_values: [batch_size, sequence_length] raw 16kHz audio waveform
        Returns logits: [batch_size, num_classes]
        """
        if self.freeze_encoder_flag:
            with torch.no_grad():
                outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        else:
            outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)

        # Extract sequence representation: hidden_states [batch_size, time_steps, hidden_size]
        hidden_states = outputs.last_hidden_state

        # Perform Mean Pooling over time dimension
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = torch.mean(hidden_states, dim=1)

        # Classification Logits
        logits = self.classifier(pooled)
        return logits
