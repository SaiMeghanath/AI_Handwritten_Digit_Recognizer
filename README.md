# AI Handwritten Digit Recognizer

CNN trained on MNIST for real-time handwritten digit classification, deployed as an interactive web app via Gradio. A full-loop deep learning project architecture: training → evaluation → deployment.

---

## Overview

This project implements a Convolutional Neural Network (CNN) for handwritten digit recognition on the MNIST benchmark, then packages the trained model into a real-time interactive web application using Gradio. The goal is not just classification accuracy, but the full pipeline: designing an architecture, training and evaluating it rigorously, and making the result accessible to anyone without a coding environment.

---

## Model Architecture

```
Input: 28×28 grayscale image
        ↓
  Conv2D (32 filters, 3×3, ReLU)
        ↓
  MaxPooling2D (2×2)
        ↓
  Conv2D (64 filters, 3×3, ReLU)
        ↓
  MaxPooling2D (2×2)
        ↓
    Flatten
        ↓
  Dense (128, ReLU)
        ↓
  Dropout (0.5)
        ↓
  Dense (10, Softmax)
        ↓
Output: Class probabilities for digits 0-9
```

### Training Details:
- **Dataset:** MNIST (60,000 train, 10,000 test)
- **Optimizer:** Adam
- **Loss:** Categorical cross-entropy
- **Epochs:** 10
- **Test Accuracy:** 99%

---

## Why CNN for MNIST?

Standard fully-connected networks treat each pixel independently, ignoring spatial relationships. CNNs use convolutional filters that slide across the image, learning local patterns—edges, curves, strokes—that compose into digit-level features. This spatial inductive bias is what makes CNNs the right tool for image classification, and MNIST is the cleanest possible dataset to demonstrate this.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8 |
| Deep Learning | TensorFlow 2.x, Keras |
| Dataset | MNIST via tensorflow.keras.datasets |
| Visualization | Matplotlib |
| Deployment | Gradio |
| Environment | Google Colab, Jupyter Notebook |

---

## Project Structure

```
AI_Handwritten_Digit_Recognizer/
├── AI_Handwritten_Digit_Recognizer.ipynb  # Full pipeline notebook
├── requirements.txt                        # Dependencies
└── README.md
```

---

## Setup & Usage

### Option A: Google Colab (Recommended)

Open `AI_Handwritten_Digit_Recognizer.ipynb` in Google Colab and run all cells. Gradio will generate a public share link automatically.

### Option B: Local

```bash
git clone https://github.com/SaiMeghanath/AI_Handwritten_Digit_Recognizer.git
cd AI_Handwritten_Digit_Recognizer
pip install -r requirements.txt
jupyter notebook AI_Handwritten_Digit_Recognizer.ipynb
```

### Running the Gradio App

```python
import gradio as gr
# interface defined in notebook
interface.launch(share=True)  # generates public URL
```

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 99% |
| Test Loss | 0.03 |
| Parameters | 93,000 |
| Training Time (Colab GPU) | ~2 min |

The model generalizes well to real hand-drawn digits through the Gradio interface, even when writing style differs from the MNIST training distribution.

---

## Future Directions

- Extend to full handwritten text → word-level OCR (a natural step toward document AI)
- Experiment with data augmentation (rotation, shear) to improve robustness to real-world writing variation
- Deploy as persistent HuggingFace Space
- Explore few-shot generalization to non-Latin scripts (Devanagari digits, Telugu numerals)

---

## Author

**Aladurthi Sai Meghanath**  
MCA AI Specialization | Amrita Vishwa Vidyapeetham  
📧 saimeghanath052@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/saimeghanath) | [GitHub](https://github.com/SaiMeghanath)
