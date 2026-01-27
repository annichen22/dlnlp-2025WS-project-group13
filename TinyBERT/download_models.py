"""
Script to download Hugging Face models for TinyBERT task distillation.
Downloads:
- Student model: TinyBERT General 4L-312D
- Teacher model: BERT-base fine-tuned on MRPC
- BERT-base model: Base BERT model (uncased)
"""
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, BertConfig, BertForMaskedLM
import torch
import os

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model paths
STUDENT_MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
TEACHER_MODEL_NAME = "Intel/bert-base-uncased-mrpc"
BERT_BASE_MODEL_NAME = "google-bert/bert-base-uncased"

STUDENT_DIR = os.path.join(MODELS_DIR, "student_tinybert")
TEACHER_DIR = os.path.join(MODELS_DIR, "teacher_bert_mrpc")
BERT_BASE_DIR = os.path.join(MODELS_DIR, "bert_base_uncased")

def download_models():
    """Download and save both teacher and student models"""
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Downloading Student Model: TinyBERT General 4L-312D")
    print("=" * 60)
    
    # Download student model (General TinyBERT)
    print(f"Loading {STUDENT_MODEL_NAME}...")
    student_model = BertModel.from_pretrained(STUDENT_MODEL_NAME)
    student_tokenizer = BertTokenizer.from_pretrained(STUDENT_MODEL_NAME)
    student_config = BertConfig.from_pretrained(STUDENT_MODEL_NAME)
    
    # Save student model
    os.makedirs(STUDENT_DIR, exist_ok=True)
    print(f"Saving to {STUDENT_DIR}...")
    
    # Save config and tokenizer
    student_config.save_pretrained(STUDENT_DIR)
    student_tokenizer.save_pretrained(STUDENT_DIR)
    
    # Save model weights in PyTorch format (for TinyBERT compatibility)
    torch.save(student_model.state_dict(), os.path.join(STUDENT_DIR, "pytorch_model.bin"))
    print("✓ Student model saved successfully\n")
    
    print("=" * 60)
    print("Downloading Teacher Model: BERT-base fine-tuned on MRPC")
    print("=" * 60)
    
    # Download teacher model (Fine-tuned BERT-base on MRPC)
    print(f"Loading {TEACHER_MODEL_NAME}...")
    teacher_model = BertForSequenceClassification.from_pretrained(TEACHER_MODEL_NAME)
    teacher_tokenizer = BertTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    teacher_config = BertConfig.from_pretrained(TEACHER_MODEL_NAME)
    
    # Save teacher model
    os.makedirs(TEACHER_DIR, exist_ok=True)
    print(f"Saving to {TEACHER_DIR}...")
    
    # Save config and tokenizer
    teacher_config.save_pretrained(TEACHER_DIR)
    teacher_tokenizer.save_pretrained(TEACHER_DIR)
    
    # Save model weights in PyTorch format (for TinyBERT compatibility)
    torch.save(teacher_model.state_dict(), os.path.join(TEACHER_DIR, "pytorch_model.bin"))
    print("✓ Teacher model saved successfully\n")
    
    print("=" * 60)
    print("Downloading BERT-base Model (uncased)")
    print("=" * 60)
    
    # Download BERT-base model
    print(f"Loading {BERT_BASE_MODEL_NAME}...")
    bert_base_model = BertModel.from_pretrained(BERT_BASE_MODEL_NAME)
    bert_base_tokenizer = BertTokenizer.from_pretrained(BERT_BASE_MODEL_NAME)
    bert_base_config = BertConfig.from_pretrained(BERT_BASE_MODEL_NAME)
    
    # Save BERT-base model
    os.makedirs(BERT_BASE_DIR, exist_ok=True)
    print(f"Saving to {BERT_BASE_DIR}...")
    
    # Save config and tokenizer
    bert_base_config.save_pretrained(BERT_BASE_DIR)
    bert_base_tokenizer.save_pretrained(BERT_BASE_DIR)
    
    # Save model weights in PyTorch format (for compatibility with TinyBERT's old transformer code)
    # Load as BertForMaskedLM for data augmentation compatibility
    bert_base_mlm = BertForMaskedLM.from_pretrained(BERT_BASE_MODEL_NAME)
    torch.save(bert_base_mlm.state_dict(), os.path.join(BERT_BASE_DIR, "pytorch_model.bin"))
    print("✓ BERT-base model saved successfully\n")
    
    return STUDENT_DIR, TEACHER_DIR, BERT_BASE_DIR

def print_summary(student_dir, teacher_dir, bert_base_dir):
    print("=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print("\nModel Directories:")
    print(f"  Student Model: {student_dir}")
    print(f"  Teacher Model: {teacher_dir}")
    print(f"  BERT-base Model: {bert_base_dir}")

if __name__ == "__main__":
    student_dir, teacher_dir, bert_base_dir = download_models()
    print_summary(student_dir, teacher_dir, bert_base_dir)
