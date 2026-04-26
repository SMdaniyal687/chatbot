# Mammogram Chatbot with RAG

A Retrieval-Augmented Generation (RAG) chatbot designed to process mammogram reports (PDFs and Images) and answer questions using a local LLM.

## Features
- **Local AI**: Uses Mistral-7B-Instruct-v0.2 (GGUF) for 100% private processing.
- **Multimodal**: Supports both PDF reports and image scans (OCR).
- **GPU Accelerated**: Optimized for NVIDIA GPUs (RTX 3050+) using CUDA.
- **Interactive UI**: Built with Gradio for easy file uploads and chatting.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install Poppler (required for PDF processing):
   - Windows: `winget install -e --id oschwartz10612.Poppler`
3. Download the model:
   - Download `mistral-7b-instruct-v0.2.Q5_K_M.gguf` and place it in the root directory.

## Usage
Run the application:
```bash
python chatbot_app.py
```
