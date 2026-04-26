import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import ctransformers
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import easyocr
import torch
import os
import numpy as np
from PIL import Image
import shutil

# Configuration - Use relative paths for portability
DATA_PATH = './data/'
TEMP_FOLDER = os.path.join(DATA_PATH, "TEMP")
Img_folder = os.path.join(TEMP_FOLDER, "images")

VECTORSTORE_DIR = './vectorstore'
PDF_DB_PATH = os.path.join(VECTORSTORE_DIR, 'db_pdf')
IMAGE_DB_PATH = os.path.join(VECTORSTORE_DIR, 'db_image')
MERGED_DB_PATH = os.path.join(VECTORSTORE_DIR, 'merged_db')

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(Img_folder, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Device setup for LLM and Embeddings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# EasyOCR initialization
reader = easyocr.Reader(['en'])

# Custom prompt template for chatbot
custom_prompt_template = """You are a sophisticated chatbot designed to provide detailed and accurate information based on the medical or health report provided.
Your goal is to assist users by answering their questions, offering insights, and providing recommendations based on the merged content of these sources.
If you don't know the answer, just say that you don't know, and don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# Load LLM once at startup
def load_llm():
    # Update this path to where your model is actually located
    model_path = "./mistral-7b-instruct-v0.2.Q5_K_M.gguf"
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}. Please update the path.")
    
    llm = ctransformers.CTransformers(
        model=model_path,
        model_type="mistral",
        max_new_tokens=512,
        temperature=0.5,
        config={'gpu_layers': 50}  # Moves 50 layers to GPU (RTX 3050)
    )
    return llm

print("Loading LLM...")
global_llm = None
try:
    global_llm = load_llm()
except Exception as e:
    print(f"Error loading LLM: {e}")

embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2', 
    model_kwargs={'device': device}
)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", clean_up_tokenization_spaces=True)

def create_vector_db(pdf_path):
    try:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_folder = os.path.join(TEMP_FOLDER, pdf_name)
        os.makedirs(output_folder, exist_ok=True)

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        texts = text_splitter.split_documents(documents)

        db = FAISS.from_documents(texts, embeddings)
        db.save_local(PDF_DB_PATH)

        # Convert PDF pages to images for gallery preview
        images = convert_from_path(pdf_path)
        image_paths = []
        for i, img in enumerate(images):
            img_path = os.path.join(output_folder, f'page_{i + 1}.png')
            img.save(img_path, 'PNG')
            image_paths.append(img_path)
            
        merge_message = merge_vector_stores()
        return f"Successfully vectorized {len(images)} pages. {merge_message}", image_paths
    except Exception as e:
        return f"Error processing PDF: {e}", []

def extract_text_from_image(image):
    image_array = np.array(image)
    result = reader.readtext(image_array)
    return " ".join([text for _, text, _ in result])

def vectorize_image_text(image_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20) 
    text_chunks = splitter.split_text(image_text) 
    db = FAISS.from_texts(text_chunks, embeddings)
    db.save_local(IMAGE_DB_PATH)
    return merge_vector_stores()

def merge_vector_stores():
    try:
        pdf_exists = os.path.exists(PDF_DB_PATH)
        image_exists = os.path.exists(IMAGE_DB_PATH)

        if not pdf_exists and not image_exists:
            return "No vector store exists to merge."

        db_pdf = FAISS.load_local(PDF_DB_PATH, embeddings, allow_dangerous_deserialization=True) if pdf_exists else None
        db_image = FAISS.load_local(IMAGE_DB_PATH, embeddings, allow_dangerous_deserialization=True) if image_exists else None

        if db_pdf and db_image:
            db_pdf.merge_from(db_image)
            db_pdf.save_local(MERGED_DB_PATH)
            return f"Merged vector stores successfully."
        elif db_pdf:
            db_pdf.save_local(MERGED_DB_PATH)
            return f"Saved PDF store to merged database."
        elif db_image:
            db_image.save_local(MERGED_DB_PATH)
            return f"Saved Image store to merged database."
        return "No data to merge."
    except Exception as e:
        return f"Error merging vector stores: {e}"

def chunk_text(text, max_tokens=512):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks

def final_result(query):
    if global_llm is None:
        return "Error: LLM not loaded. Please check your model path and restart."
    try:
        if not os.path.exists(MERGED_DB_PATH):
            return "Please upload a document or image first to initialize the vector store."
            
        db = FAISS.load_local(MERGED_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        chunks = chunk_text(query)
        results = []
        
        qa = RetrievalQA.from_chain_type(
            llm=global_llm,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs={'k': 1}),
            return_source_documents=True
        )
        
        for chunk in chunks:
            response = qa.invoke({'query': chunk})
            results.append(response['result'])

        return " ".join(results)  
    except Exception as e:
        return f"Error with LLM or vector store: {e}"

def clear_vectorstores():
    try:
        for path in [PDF_DB_PATH, IMAGE_DB_PATH, MERGED_DB_PATH, TEMP_FOLDER]:
            if os.path.exists(path):
                shutil.rmtree(path)
        
        os.makedirs(TEMP_FOLDER, exist_ok=True)
        os.makedirs(Img_folder, exist_ok=True)
        return "Cleared successfully."
    except Exception as e:
        return f"Error clearing: {e}"

def move_image_to_data_folder(image_path, from_explorer=False):
    if from_explorer:
        return "Selected from data folder."
    try:
        target_path = os.path.join(Img_folder, os.path.basename(image_path))
        shutil.copy(image_path, target_path)
        return f"Copied to {target_path}."
    except Exception as e:
        return f"Error copying image: {e}"

def handle_file_upload(file_paths, from_explorer=False):
    status_messages = []
    all_images = []

    if not file_paths:
        return None, "No files selected.", [], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    for file_path in file_paths:
        if file_path.lower().endswith('.pdf'):
            status, images = create_vector_db(file_path)
            status_messages.append(status)
            all_images.extend(images)
        else:
            try:
                image = Image.open(file_path)
                all_images.append(image)
                extracted_text = extract_text_from_image(image)
                merge_message = vectorize_image_text(extracted_text)
                status_messages.append(f"Image processed: {merge_message}")
                move_image_to_data_folder(file_path, from_explorer)
            except Exception as e:
                status_messages.append(f"Error processing {file_path}: {e}")

    merge_vector_stores()
    return None, "\n".join(status_messages), all_images, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

def setup_interface():
    with gr.Blocks() as app:
        chat_history = gr.State([])
        
        with gr.Tab("Upload PDFs or Images"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_explorer_input = gr.FileExplorer(
                        label="Select from Data Folder", 
                        root_dir=DATA_PATH,
                        file_count="multiple", 
                        min_width=200
                    )
                    file_input = gr.File(
                        label="Or Upload from Anywhere", 
                        file_count="multiple", 
                        min_width=200
                    )
                    clear_button = gr.Button("Clear Data & Reset", variant="danger")

                with gr.Column(scale=1):
                    image_preview = gr.Image(label="Preview", visible=True)
                    gallery_output = gr.Gallery(label="Extracted Pages", visible=False)
                    vectorization_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=3):
                    chatbox_output = gr.Chatbot(label="ChatBot", height=400)
                    question_input = gr.Textbox(label="Ask a Question", placeholder="Type here...", visible=False)
                    submit_button = gr.Button("Submit", visible=False, variant="primary")

            def handle_query(query, history):
                answer = final_result(query)
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": answer})
                return history, history, ""

            file_explorer_input.change(
                lambda paths: handle_file_upload(paths, from_explorer=True),
                inputs=file_explorer_input,
                outputs=[image_preview, vectorization_status, gallery_output, gallery_output, question_input, submit_button]
            )

            file_input.change(
                lambda files: handle_file_upload([f.name for f in files] if files else []),
                inputs=file_input,
                outputs=[image_preview, vectorization_status, gallery_output, gallery_output, question_input, submit_button]
            )

            submit_button.click(
                handle_query, 
                inputs=[question_input, chat_history], 
                outputs=[chatbox_output, chat_history, question_input]
            )
            
            question_input.submit(
                handle_query, 
                inputs=[question_input, chat_history], 
                outputs=[chatbox_output, chat_history, question_input]
            )

            clear_button.click(
                lambda: ([], [], "", None, gr.update(visible=False), gr.update(visible=False), clear_vectorstores()),
                outputs=[chatbox_output, chat_history, question_input, image_preview, gallery_output, submit_button, vectorization_status]
            )

        with gr.Tab("FAQs"):
            gr.Markdown("""# FAQs
            **1. How to upload?** Use the file input or explorer.
            **2. What happens on clear?** All processed data and vector stores are deleted.
            **3. How to ask questions?** Upload a document first, then type in the box.
            """)
    
    app.launch()

if __name__ == "__main__":
    setup_interface()
