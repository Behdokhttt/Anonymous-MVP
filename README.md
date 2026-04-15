# MVP: Multimodal Analysis

## Overview

MAINS (Multimodal Affective INteraction State Modeling) is a multimodal framework for modeling social interaction in natural conversations. It integrates affective state classification from audio, visual, and textual signals with network-based analysis to capture both the co-occurrence structure and temporal dynamics of affective and interactional states. Built on turn-level conversational data, the framework combines multimodal predictions with interactional discourse codes to provide interpretable representations of how social interaction is organized and unfolds over time.
<div align="center">
  <img src="model.png" alt="Flowchart of Data Processing" width="80%">
  <p><strong>Figure 1:</strong> ESP Model Framework: The architecture of the proposed Multimodal Emotional State Classifier consists of four key components: Audio Extraction, Pose Estimation, Text Features integrated with a Knowledge Graph, and a Fusion Network with a classification head. This framework assesses whether participants exhibit positive emotions—supportive and uplifting behaviors—during video segments.</p>
</div>

## Process

1. **Knowledge Graph Representation**: 
   - The text data is converted into a graph-like structure for knowledge graph representation. The text is sent to the ConceptNet API, which extracts information about related words and their relationships. 
   - This process utilizes the Knowledge Graph code, and the text data is also used to fine-tune the RoBERTa model for improved understanding.

2. **Data Extraction**: 
   - Audio data and 3D skeleton data are extracted and integrated with the text-derived features.

3. **Model Training**: 
   - All three modalities—text, audio, and 3D human pose data—are combined to train the model for emotional prediction.
