# Decoding Parameter and Sampling Analysis

## Project Overview
This project analyzes how Temperature, Top-K, and Top-P sampling parameters affect the outputs of Large Language Models (LLMs), including determinism, hallucination, and output structure.

## Models Used
- Llama 3
- Qwen 2.5:3b

## Tools and Libraries
- Python
- Ollama
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly
- Requests

## Project Structure
- `app.py` — Streamlit application for running the experiments.
- `Sampling_Analysis_Report.pdf` — Project report.
- `results/` — Raw experimental data and summary CSV files.

## How to Run
1. Install Python and Ollama.
2. Download the required models using Ollama:
   ```bash
   ollama pull llama3
   ollama pull qwen2.5:3b
   ```
3. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the application:
   ```bash
   streamlit run app.py
   ```

## Experimental Parameters
- Temperature: 0.2, 0.7, 1.2
- Top-K: 1, 20, 40
- Top-P: 0.5, 0.9, 1.0

The project report describes the experimental methodology, results, analysis, and limitations.