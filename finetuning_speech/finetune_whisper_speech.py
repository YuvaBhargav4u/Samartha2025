import torch
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import librosa
import os

# --- Configuration & Global Objects ---
BASE_MODEL = "openai/whisper-base"
DATASET_PATH = "training_data.jsonl"
OUTPUT_DIR = "../backend/whisper_finetuned_speech"

# MOVED: Processor is now defined in the global scope so all functions can access it.
processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="english", task="transcribe")

# --- Function and Class Definitions ---
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def prepare_sample(batch):
    """This function now correctly sees the global 'processor' object."""
    audio_path = batch["audio_path"]
    waveform, sample_rate = librosa.load(audio_path, sr=16000)

    batch["input_features"] = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_features[0]
    batch["labels"] = processor.tokenizer(batch["description"], return_tensors="pt").input_ids[0]
    return batch

# --- Main Execution Block ---
if __name__ == '__main__':
    # 1. Load Model
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, load_in_8bit=True)

    # 2. Configure LoRA Adapters
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 3. Load and Prepare the Dataset
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    train_dataset = dataset.map(prepare_sample, num_proc=1)

    # 4. Set up the Trainer
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=50,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 5. Start Training
    print("Starting LoRA fine-tuning for speech-specific details...")
    trainer.train()

    # 6. Save the final adapter model
    print("Training complete. Saving LoRA adapter model.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")