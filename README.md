# Non-contact-Cross-modal
Non-contact Cross-modal Emotion Recognition via Multiple Instance Learning and Hierarchical Attention Fusion from Single Facial Video

Visual Backbone & Multi-Instance Learning (AMIL):

Utilizes a pre-trained ResNet-50 to extract frame-level spatial features from facial video sequences (handling 60 input frames per sample).

Implements an Attention-based Multiple Instance Learning (AMIL with Top-K) mechanism to automatically select the 16 most emotionally informative frames, filtering out irrelevant or noisy visual contexts.

Physiological Signal Extraction (HRV & IPPG):

Processes HRV (Heart Rate Variability) features through normalization, non-linear projection, and dropout layers.

Employs a dedicated 1D Convolutional Neural Network (IPPG1DCNN) to extract dynamic physiological representations from non-contact IPPG (Imaging Photoplethysmography) waveforms.

Hierarchical Cross-Modal Fusion:

Fuses the extracted HRV and IPPG features, and subsequently interacts them with the selected visual frame embeddings via an advanced Cross-Modal Attention Mechanism (allowing deep mutual guidance between visual expressions and physiological responses).

Combines a learnable [CLS] token with the fused features and feeds them into a multi-layer Transformer Encoder to capture long-range global contextual dependencies.
