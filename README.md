# ⌨️ Smart Keyboard V1

A next-word prediction keyboard built with statistical N-gram language models, featuring Add-k smoothing and back-off strategies, wrapped in a modern Streamlit web interface.
## live demo https://smart-keyboard-v1-majrisalma.streamlit.app/
## 👤 Author
* **Salma Majri** 
* **Program:** Master SDIA-2 

## 📸 Preview

![Smart Keyboard UI](1.png)

---

## 🚀 Features

- **Statistical N-gram Models:** Supports Unigram, Bigram, and Trigram language modeling.
- **Add-k Smoothing:** Handles unseen words and prevents zero-probability issues.
- **Smart Back-off Strategy:** Automatically falls back from Trigram to Bigram and Unigram contexts.
- **Interactive Web UI:** Modern, dark-themed Streamlit interface with glassmorphism styling and live clickable word suggestions.
- **Evaluation Pipeline:** Computes dataset perplexity and Top-$k$ accuracy efficiently.

---

## 📂 Project Structure

```text
02_smart_keyboard_v1/
│
├── data/
│   ├── prepare_data.py   # Dataset preparation script (Wikitext-2)
│   └── corpus.csv        # Processed corpus dataset
│
├── src/
│   ├── pipeline.py       # Core N-gram classes, smoothing, and back-off logic
│   └── evaluate.py       # Evaluation metrics script (Perplexity & Accuracy)
│
├── .gitignore
├── app.py                # Streamlit modern web application
└── README.md
