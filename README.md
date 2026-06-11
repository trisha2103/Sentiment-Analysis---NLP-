# 📊 Transformer-Based Web Scraping & Sentiment Analysis Pipeline

An end-to-end NLP intelligence tool designed to scrape live, real-world consumer reviews from the web and automatically classify customer satisfaction using a state-of-the-art Deep Learning **Transformer (BERT) model**.

This project bypasses traditional static datasets, utilizing dynamic data engineering and deep learning inference to turn messy, unstructured HTML text into structured, actionable business intelligence.

---

## 🚀 Key Highlights & Architecture

* **Live Web Data Scraping:** Dynamically establishes network connections to pull customer feedback directly from the web using `BeautifulSoup`.
* **State-of-the-Art NLP:** Leverages a pre-trained, multilingual **BERT-base** architecture fine-tuned specifically for localized review sentiment.
* **Granular Analytics:** Replaces simple binary (positive/negative) classifications with a highly specific **1-to-5 Star** satisfaction grading scale.
* **Production-Ready Workflow:** Built with a clean, modular pipeline format emphasizing time-to-value and rapid prototype iteration.

---

## 🛠️ Tech Stack & Dependencies

| Layer | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Deep Learning Engine** | `PyTorch` (torch) | Tensor computation & GPU acceleration |
| **Model Hub** | `Hugging Face Transformers` | Pre-trained BERT Tokenizer & Sequence Classifier |
| **Data Scraping** | `BeautifulSoup4` & `requests` | HTML parsing, DOM traversal, and live web requests |
| **Data Engineering** | `pandas` & `numpy` | DataFrame manipulation and structural formatting |
| **Text Processing** | `re` (Regular Expressions) | Advanced text pattern matching and cleaning |

---

## 📦 Deep Learning Model Details

The pipeline initializes the **`nlptown/bert-base-multilingual-uncased-sentiment`** model. This multi-layered neural network reads contextual sub-word tokens and maps them directly onto a discrete scale of customer sentiment:

* **1 Star** ➡️ Strongly Dissatisfied (Extreme Negative)
* **2 Stars** ➡️ Dissatisfied (Negative)
* **3 Stars** ➡️ Neutral
* **4 Stars** ➡️ Satisfied (Positive)
* **5 Stars** ➡️ Strongly Satisfied (Extreme Positive)

---

## 💻 Code Architecture & Execution

<details>
<summary>📂 <b>Step 1: Environment Installation</b> (Click to expand)</summary>

Ensure your local system has the required deep learning dependencies and web scrapers ready. Run the following commands in your environment:

```bash
# Install PyTorch (CUDA 11.8 Accelerated for faster inference)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install Hugging Face Transformers and Web Parsing Libraries
pip install transformers requests beautifulsoup4 pandas numpy

## 📈 Before vs. After Insights

* **Before (Unstructured Text Chaos):** Messy, unformatted HTML code blocks buried deep within web page DOM elements, completely inaccessible for analytics.
* **After (Structured Analytical Dashboard):** A clean, highly actionable data view pairing explicit user reviews with concrete, numeric star-satisfaction indicators (1-5 stars). This system enables product, support, and engineering teams to filter feedback and triage critical user friction points instantly.

### The Data Transformation:
```text
[Raw HTML DOM] ──> [BeautifulSoup] ──> [BERT Transformer] ──> [Structured Output]
                                                              ┌──────────────────────┬───────────┐
                                                              │ Review Text          │ Sentiment │
                                                              ├──────────────────────┼───────────┤
                                                              │ "I hated this, bad!" │  1 Star   │
                                                              └──────────────────────┴───────────┘
