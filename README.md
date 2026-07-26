# 🧠 visual-cortex-fmri-decoder

This repository contains an advanced, cloud-native fMRI decoding pipeline designed to analyze and classify human visual cortex activation patterns using Deep Learning.

The project processes brain imaging data to predict which of the 8 distinct visual categories a subject is looking at, utilizing neural networks trained on fMRI BOLD (Blood-Oxygen-Level-Dependent) signals.

## 🚀 Key Features & Engineering

* **Deep Learning Architecture:** Custom PyTorch neural network replacing baseline SVMs, optimized for complex biological data.
* **Cloud-Native & GPU Accelerated:** Fully compatible with Google Colab T4 GPU (CUDA) environments for rapid matrix computations and model training.
* **Neuro-Optimization:** 
  * Implemented a 2 TR (Time Repetition) temporal shift to account for hemodynamic response lag.
  * Dynamically filtered out 'rest' states to focus strictly on active visual stimuli.
* **Containerized BIDS Pipeline:** Structured to support Dockerized environments for standardized neuroimaging data processing.

## 📊 Results & Performance

The deep learning model was trained and evaluated on 8 distinct visual categories. 

* **Task:** Multi-class classification of visual stimuli.
* **Chance Level:** 12.5% (1/8 categories)
* **Model Accuracy (True Test Set):** **~66.47%**

The model significantly outperforms the random chance level and traditional machine learning baselines, demonstrating a strong capability to decode complex fMRI spatial patterns.

## 🛠️ Tech Stack

* **Machine Learning / AI:** `PyTorch`, `Scikit-learn`, `SciPy`
* **Neuroimaging:** `Nilearn`
* **Data Processing:** `Pandas`, `NumPy`
* **Infrastructure:** `Docker`, `CUDA` (Google Colab)

## 💻 Quick Start

To clone and run this repository locally or in a cloud environment:

```bash
git clone [https://github.com/senaayy/visual-cortex-fmri-decoder.git](https://github.com/senaayy/visual-cortex-fmri-decoder.git)
cd visual-cortex-fmri-decoder
Note: Ensure you have a CUDA-enabled environment (like Google Colab) for optimal training performance.

📈 Future Roadmap
Integration of cross-subject generalization (transfer learning) to evaluate universal neural patterns across different brain anatomies.

Application of Brain Mapping (Weight Visualization) to map the neural network's activation back to 3D NIfTI brain space.

Developed for research and exploration in Brain-Computer Interfaces (BCI) and Neurotechnology.
