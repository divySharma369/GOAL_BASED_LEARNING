# Goal-Based Learning: A Proof of Concept in Dynamic Predictive AI

## Project Overview
This repository serves as a technical proof of concept for Goal-Based Learning (GBL), an experimental approach in artificial intelligence that focuses on dynamic, context-aware objective activation. Moving beyond traditional monolithic predictive models, this system demonstrates how an AI can selectively engage specific sub-networks or "goals" based on the latent characteristics of the input data.

The prototype is currently implemented within the context of educational analytics, using student feature sets to drive a multi-task inference engine.

## Core Philosophy
The central hypothesis of this research is that complex predictive tasks are better addressed by modular systems that can prioritize specific outcomes depending on the situational context. In this model, "Goal-Based" refers to the system's ability to:
1. Extract high-level abstract representations from raw input.
2. Evaluate the contextual relevance of various predictive objectives.
3. Dynamically activate the appropriate specialized modules to fulfill those objectives.

## Technical Architecture
The system is built on a modular neural network architecture using the PyTorch framework.

### Shared Latent Encoder
At the foundation of the model is a shared encoder designed to map input features into a lower-dimensional latent space. This representation captures the fundamental underlying patterns of the data, which are then used for subsequent decision-making.

### Modular Task Heads
The model employs a Multi-Task Learning (MTL) structure where separate neural "heads" are trained to address specific goals, such as:
*   Pass/Fail classification.
*   Quantitative score prediction across multiple domains (Mathematics, Reading, Writing).

### Contextual Gating Mechanism
The defining experimental feature of this repository is the Gating Mechanism. This component analyzes the output of the shared encoder and determines which task-specific heads are relevant for a given input. This selective activation allows the system to focus its computational resources and predictive logic on contextually significant goals, effectively simulating a form of targeted attention.

## Implementation Details
The project is divided into several key components:
*   **inference.py**: The core logic containing the neural network definitions, the gating function, and the inference pipeline.
*   **app.py**: A Streamlit-based interface providing a functional demonstration of the GBL engine.
*   **model_weights/**: Serialized PyTorch state dictionaries for the encoder and multi-task heads.

## Deployment
A live demonstration of this proof of concept is available via Streamlit:
[Goal-Driven Student AI Prototype](https://goalbasedlearning-369.streamlit.app/)

## Research Status
This repository is a discovery-phase prototype. The primary focus is on the validation of the dynamic gating architecture and the assessment of shared latent representations in multi-goal environments. Future iterations will explore more complex gating logic and the expansion of the goal-selection criteria.
