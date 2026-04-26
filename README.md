# AI Health Report Chatbot with RAG

A Retrieval-Augmented Generation (RAG) chatbot designed to process any medical or health reports (PDFs and Images) and answer questions using a local LLM.

## Features
- **General Health Assistant**: Works with any medical report, blood test, or health document.
- **Local AI**: Uses Mistral-7B-Instruct-v0.2 (GGUF) for 100% private processing.
- **Multimodal**: Supports both PDF reports and image scans/photos (OCR).
- **GPU Accelerated**: Optimized for NVIDIA GPUs (RTX 3050+) using CUDA.
- **Interactive UI**: Built with Gradio for easy file uploads and chatting.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
