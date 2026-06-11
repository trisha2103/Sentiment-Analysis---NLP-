# 🌟 Sentiment Analysis with NLP & BERT

> Analyze customer reviews from the web in real-time using a pre-trained multilingual BERT model — and get star ratings (1–5) for any piece of text.

---

## 📌 Overview

This project performs **sentiment analysis** on web-scraped customer reviews using **BERT** (Bidirectional Encoder Representations from Transformers). It scrapes reviews from review platforms, processes them through a multilingual NLP model, and returns a sentiment score from **1 (most negative) to 5 (most positive)**.

---

## 🚀 Features

- ✅ Real-time **web scraping** of customer reviews using `BeautifulSoup`
- ✅ Sentiment scoring powered by **`nlptown/bert-base-multilingual-uncased-sentiment`**
- ✅ Supports **multilingual** reviews (6 languages: EN, FR, DE, NL, IT, ES)
- ✅ Outputs structured results in a **Pandas DataFrame**
- ✅ Works on **GPU** (CUDA 11.8) for fast inference

---

## 🧠 Model

| Property | Details |
|---|---|
| **Model** | `nlptown/bert-base-multilingual-uncased-sentiment` |
| **Task** | Sequence Classification (Sentiment) |
| **Output** | Star rating: 1 ⭐ to 5 ⭐⭐⭐⭐⭐ |
| **Languages** | English, French, German, Dutch, Italian, Spanish |
| **Source** | 🤗 Hugging Face Transformers |

---

## 🗂️ Project Structure

```
Sentiment-Analysis---NLP-/
│
├── Sentiment Analysis-YELP.ipynb   # Main Jupyter notebook
└── README.md                        # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/trisha2103/Sentiment-Analysis---NLP-.git
cd Sentiment-Analysis---NLP-
```

### 2. Install dependencies

```bash
# Install PyTorch with CUDA 11.8 support (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install NLP and scraping libraries
pip install transformers requests beautifulsoup4 pandas numpy
```

> 💡 If you don't have a GPU, PyTorch will fall back to CPU automatically.

---

## 📖 How It Works

### Step 1 — Load the BERT Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
```

### Step 2 — Scrape Reviews from the Web

```python
import requests
from bs4 import BeautifulSoup
import re

r = requests.get('https://www.trustpilot.com/review/hotchocolatedesign.com')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*typography_body.-*')
results = soup.find_all('p', {'class': regex})
reviews = [result.text for result in results]
```

### Step 3 — Score Each Review

```python
import torch

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1
```

### Step 4 — Build a DataFrame with Scores

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(np.array(reviews), columns=['reviews'])
df['sentiment'] = df['reviews'].apply(sentiment_score)
```

---

## 📊 Sample Output

| Review | Sentiment Score |
|---|---|
| "Absolutely loved this product!" | ⭐⭐⭐⭐⭐ (5) |
| "It was okay, nothing special." | ⭐⭐⭐ (3) |
| "I hated this, absolutely the worst." | ⭐ (1) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| 🤗 `transformers` | BERT model & tokenizer |
| 🔥 `torch` | Deep learning inference |
| 🌐 `requests` + `BeautifulSoup` | Web scraping |
| 🐼 `pandas` + `numpy` | Data handling |

---

## 📋 Requirements

- Python 3.8+
- CUDA 11.8 (optional, for GPU support)
- Jupyter Notebook or Google Colab

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👩‍💻 Author

**Trisha** — [@trisha2103](https://github.com/trisha2103)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ and 🤗 Transformers</p>
