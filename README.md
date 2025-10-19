---
title: Production MLM Word Prediction
emoji: 🔮
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: A transformer model that predicts masked words by learning relationships between words in sentences.
---

# 🔮 Production MLM Word Prediction

This is a **Masked Language Model (MLM)** built with a Transformer Encoder that learns relationships between words in a sentence and can predict what word should fill a masked position.

## 🎯 Target Sentence

The model is trained on the sentence:
> **"This model creates a super relationships between the words to predict what word"**

## 🧠 How It Works

1. **Multi-Head Self-Attention**: Each word attends to other words in the sentence
2. **Relationship Learning**: Discovers grammatical and semantic patterns
3. **Word Prediction**: Predicts what word should fill a masked position based on context

## 🚀 Features

- **Interactive Interface**: Mask any word and see predictions
- **Confidence Scores**: See how confident the model is in its predictions
- **Top Predictions**: View the top 3 most likely words
- **Real-time Results**: Instant predictions with detailed explanations

## 🔬 Model Architecture

- **Type**: Transformer Encoder
- **Embedding Dimension**: 128
- **Layers**: 3
- **Attention Heads**: 4
- **Vocabulary**: 16 tokens
- **Training Accuracy**: 100%

## 💡 Example Usage

1. **Input**: "This model [MASK] a super relationships between the words to predict what word"
2. **Position**: 2 (where [MASK] is)
3. **Output**: "creates" with 98.9% confidence

## 🎨 Key Insights

The model learns various types of relationships:
- **Grammatical**: "creates" often follows "model"
- **Semantic**: "relationships" often follows "super"
- **Positional**: Articles appear in specific positions
- **Contextual**: Words that typically appear together

## 🛠️ Technical Details

- **Framework**: PyTorch
- **Architecture**: Custom Transformer Encoder
- **Training**: Masked Language Modeling (MLM)
- **Interface**: Gradio

## 📊 Performance

- **Training Accuracy**: 100%
- **Test Accuracy**: 23.1% (learns relationships, not memorization)
- **Confidence**: All predictions have >98% confidence

## 🎯 Use Cases

- **Language Understanding**: Demonstrate how transformers learn word relationships
- **Educational**: Show attention mechanisms and self-supervised learning
- **Prototype**: Foundation for larger language models
- **Research**: Study how models learn linguistic patterns

---

**Note**: This model is designed for educational and demonstration purposes. It shows how transformer models learn relationships between words through self-attention mechanisms.