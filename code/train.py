import argparse
import os
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig
# Import GradientAscentTrainer
from trainer.gradient_ascent import GradientAscentTrainer


# Set up argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model using LoRA fine-tuning.")

    # Add arguments
    parser.add_argument('--epochs', type=int, default=1,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Per-device batch size during training.")
    parser.add_argument('--learning_rate', type=float,
                        default=2e-4, help="Learning rate for training.")
    parser.add_argument('--output_dir', type=str, default="./models/",
                        help="Directory for saving the model.")
    parser.add_argument('--save_steps', type=int, default=5000,
                        help="Save the model every N steps.")
    parser.add_argument('--wandb_project', type=str,
                        default="news", help="Weights & Biases project name.")
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Meta-Llama-3.1-8B", help="Model name to fine-tune.")
    parser.add_argument('--dataset_name', type=str,
                        default="TaiMingLu/news-truthful", help="Dataset name for loading.")
    parser.add_argument('--lora_r', type=int, default=2048, help="LoRA rank.")
    parser.add_argument('--lora_alpha', type=int,
                        default=64, help="LoRA alpha.")
    parser.add_argument('--lora_dropout', type=float,
                        default=0.1, help="LoRA dropout.")
    parser.add_argument('--unlearn', action='store_true',
                        help="Use Gradient Ascent instead of Gradient Descent.")

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Set environment variables
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # Load dataset
    print("Loading Dataset")
    dataset = load_dataset(args.dataset_name, split="train")
    print("Loaded Dataset")

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "final_model"),
        num_train_epochs=args.epochs,
        logging_dir="./logs",
        logging_steps=5,
        save_steps=args.save_steps,
        report_to="wandb",
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    # Configure LoRA parameters
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Choose the trainer based on the 'unlearn' argument
    TrainerClass = GradientAscentTrainer if args.unlearn else SFTTrainer

    # Initialize the trainer
    trainer = TrainerClass(
        model_name=args.model_name,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=1024,
        packing=True
    )

    # Start training
    print("Start Training")
    trainer.train()

    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
