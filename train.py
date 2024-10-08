# Copyright (c) 2024 Dr. Tristan Behrens
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import datetime
from itertools import chain
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import time

import torch
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (DataCollatorForLanguageModeling,
                          PreTrainedTokenizerFast, get_scheduler)

try:
    import xlstm
    from model import xLSTMConfig, xLSTMForCausalLM
except ImportError as e:
    print(e)
    print("xLSTM package not installed or not available.")
    print("Please visit https://github.com/NX-AI/xlstm")
    exit()

from source.utilities import (display_logo, human_readable_number,
                              validate_config)

# Import the LinearWarmupCosineAnnealing scheduler from the experiments module.
# Source: https://github.com/NX-AI/xlstm/tree/main
if not os.path.exists("experiments/lr_scheduler.py"):
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/NX-AI/xlstm/main/experiments/lr_scheduler.py"
    )
    os.makedirs("experiments", exist_ok=True)
    urllib.request.urlretrieve(url, "experiments/lr_scheduler.py")

from experiments.lr_scheduler import LinearWarmupCosineAnnealing

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def run_training(config_path: str):
    """
    Run the training process based on the provided configuration file.
    Args:
        config_path (str): The path to the configuration file.
    Raises:
        FileNotFoundError: If the configuration file is not found.
    Returns:
        None
    """

    # Load the configuration.
    config = load_config(config_path)

    # Specify the output_dir.
    run_dir = "run_" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
    output_dir = os.path.join(config.training.output_dir, run_dir)

    # Initialize the loggers.
    loggers = []
    if (
        "wandb_project" in config.training
        and config.training.wandb_project is not None
        and config.training.wandb_project != ""
    ):
        loggers.append("wandb")

    # Get gradient accumulation steps.
    gradient_accumulation_steps = config.training.get("gradient_accumulation_steps", 1)
    # config.training.batch_size = config.training.batch_size * gradient_accumulation_steps

    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        log_with=loggers,
        project_dir=output_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )

    # Display the logo.
    if accelerator.is_local_main_process:
        display_logo()

    # Create the output directory.
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.print(f"Output directory: {output_dir}")

    # Set log every step to save every step.
    if "log_every_step" not in config.training:
        config.training.log_every_step = 1
    if "save_every_step" not in config.training:
        config.training.save_every_step = -1

    # Preprocess the dataset and tokenizer.
    tokenized_datasets, tokenizer = preprocess(config, accelerator)

    if config.model.context_length != tokenizer.model_max_length:
        tokenizer.model_max_length = config.model.context_length

    # Get the vocabulary size.
    vocab_size = tokenizer.vocab_size
    config.model.vocab_size = vocab_size

    # Create the data collator.
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Create the model.
    accelerator.print("Creating model...")
    hf_model_config = xLSTMConfig(vocab_size=vocab_size, config=config.model)
    model = xLSTMForCausalLM(hf_model_config)
    model.reset_parameters()

    # Apply precision.
    training_dtype = get_torch_dtype(config.training.weight_precision)
    model = model.to(dtype=training_dtype)
    accelerator.print(f"Training dtype: {training_dtype}")

    # Print the model.
    accelerator.print(model)
    num_params = sum(p.numel() for p in model.parameters())
    num_params_human = human_readable_number(num_params)
    accelerator.print(f"Number of parameters: {num_params:_} ({num_params_human})")

    # Prepare the DataLoader from the tokenized dataset.
    accelerator.print("Preparing DataLoader...")
    train_dataloader = DataLoader(
        tokenized_datasets,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    # Estimate the number of steps.
    num_steps = (
        config.training.num_epochs
        * len(tokenized_datasets)
        // config.training.batch_size
    )
    num_steps = num_steps // accelerator.num_processes
    accelerator.print(f"Estimated number of steps: {num_steps:_}")


    # Prepare the optimizer and learning rate scheduler.
    try:
        optimizer_groups = model._create_weight_decay_optim_groups()
        optimizer = torch.optim.AdamW(
            (
                {
                    "weight_decay": config.training.weight_decay,
                    "params": optimizer_groups[0],
                },
                {"weight_decay": 0.0, "params": optimizer_groups[1]},
            ),
            lr=config.training.lr,
        )
    except:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)

    
    if config.training.get("lr_scheduler", "cosine_with_warmup") == "cosine_with_warmup":
        lr_scheduler = LinearWarmupCosineAnnealing(
            optimizer=optimizer,
            warmup_steps=config.training.lr_warmup_steps * accelerator.num_processes,
            decay_until_step=config.training.lr_decay_until_steps if config.training.lr_decay_until_steps != "auto"
            else num_steps * accelerator.num_processes,
            max_lr=config.training.lr,
            min_lr=config.training.lr_decay_factor * config.training.lr,
        )
    else:
        lr_scheduler = get_scheduler(
            config.training.lr_scheduler,
            optimizer,
            num_warmup_steps=config.training.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=num_steps * accelerator.num_processes,
        )


    # Prepare model, optimizer, and dataloader for accelerator.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Get some parameters.
    save_every_step = config.training.save_every_step
    log_every_step = config.training.log_every_step
    generate_every_step = config.training.generate_every_step
    num_epochs = config.training.num_epochs
    enable_mixed_precision = config.training.enable_mixed_precision
    max_grad_norm = config.training.max_grad_norm
    wandb_project = config.training.get("wandb_project", None)
    wandb_enabled = wandb_project is not None and wandb_project != ""

    # Get a subset of the config that includes only the model.
    model_config = OmegaConf.select(config, "model")

    # Create the readme.
    create_readme(output_dir, config)

    # Save the config as yaml and delete it.
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config, f)
    del config

    # Save the tokenizer.
    tokenizer.save_pretrained(output_dir)

    # Enable trackers.
    if wandb_enabled:
        accelerator.print(f"Enabling wandb logging for project: {wandb_project}")
        config_dict = OmegaConf.to_container(model_config)
        # Add num_params to the config.
        config_dict["num_params"] = num_params
        config_dict["num_params_human"] = num_params_human
        accelerator.init_trackers(
            project_name=wandb_project,
            config=config_dict,
            init_kwargs={"wandb": {"name": run_dir}},
        )

    # Training loop.
    step = 0
    running_loss = []
    history = {
        "loss": [],
        "lr": [],
        "step": [],
    }

    # Add a green progress bar.
    progress_bar = tqdm(total=num_steps, desc="Training", unit="step", colour="YELLOW")

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):  
                # Assuming batch only contains 'input_ids'
                inputs = batch["input_ids"]

                # Get the labels by shifting the inputs. Remove the first token. Fill the last token with 0.
                labels = torch.roll(inputs, -1, dims=1)
                labels[:, -1] = 0

                outputs = model(inputs, return_dict=True)
                logits = outputs.logits
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, vocab_size),
                    labels.view(-1),
                    ignore_index=-1,
                )
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.zero_grad()
                running_loss.append(loss.item())

                # Next step.
                step += 1

                # Save every step.
                if step % save_every_step == 0 and step > 0 and save_every_step > 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
                    accelerator.wait_for_everyone()
                    save_model(
                        accelerator.unwrap_model(model),
                        hf_model_config,
                        tokenizer,
                        checkpoint_dir,
                    )

                # Log every step.
                if step % log_every_step == 0 and step > 0 and log_every_step > 0:
                    # Update the log.
                    average_loss = sum(running_loss) / len(running_loss)
                    last_lr = lr_scheduler.get_last_lr()[0]
                    history["loss"].append(average_loss)
                    history["lr"].append(last_lr)
                    history["step"].append(step)
                    running_loss = []

                    # Log to wandb.
                    if wandb_enabled:
                        accelerator.log({"loss": average_loss, "lr": last_lr}, step=step)

                    # Update the progressbar. Use the step as the total. Also display the loss and lr.
                    progress_bar.set_postfix({"loss": average_loss, "lr": last_lr})
                    progress_bar.update(log_every_step)

                if step % generate_every_step == 0 and step > 0 and generate_every_step > 0:
                    accelerator.unwrap_model(model).eval()
                    prompt = "The "
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                        accelerator.device
                    )
                    output = accelerator.unwrap_model(model).generate(
                        input_ids, max_length=100, temperature=0.7, do_sample=True
                    )
                    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                    # Print with green highlight.
                    accelerator.print(f"\033[92mGenerated text:\033[0m {generated_text}")
                    accelerator.unwrap_model(model).train()

    # End training.
    progress_bar.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()

    # Print some information.
    accelerator.print(f"Training completed. Epochs: {epoch}, Steps: {step}")

    # Save the last model.
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}-last")
    accelerator.wait_for_everyone()

    save_model(
        accelerator.unwrap_model(model), hf_model_config, tokenizer, checkpoint_dir
    )

    # Save the history as JSON.
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)


def load_config(config_path: str) -> OmegaConf:
    """
    Load the configuration from the specified path.
    Args:
        config_path (str): The path to the configuration file.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    Returns:
        OmegaConf: The configuration object.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config_yaml = f.read()
    config = OmegaConf.create(config_yaml)
    OmegaConf.resolve(config)
    validate_config(config)
    return config


def train_whitespace_tokenizer(raw_datasets):
    """
    Trains a whitespace tokenizer using the provided raw datasets.
    Args:
        raw_datasets (dict): A dictionary containing the raw datasets.
    Returns:
        PreTrainedTokenizerFast: The trained whitespace tokenizer.
    """

    # Initialize the tokenizer.
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    # Train the tokenizer.
    def get_training_corpus():
        dataset = raw_datasets["train"]
        for start_idx in range(0, len(dataset), 1000):
            samples = dataset[start_idx : start_idx + 1000]
            yield samples["text"]

    training_corpus = get_training_corpus()
    tokenizer.train_from_iterator(training_corpus, trainer=trainer)

    # Convert the tokenizer to a fast tokenizer.
    with tempfile.TemporaryDirectory() as tempdir:
        tokenizer_path = os.path.join(tempdir, "tokenizer.json")
        tokenizer.save(tokenizer_path)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Return the tokenizer.
    return tokenizer


def get_torch_dtype(dtype: str) -> torch.dtype:
    """
    Returns the corresponding torch.dtype for the given dtype string.

    Args:
        dtype (str): The dtype string.

    Returns:
        torch.dtype: The corresponding torch.dtype.

    Raises:
        ValueError: If the dtype is unknown.
    """

    if dtype == "float32":
        return torch.float32
    elif dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float16":
        return torch.float16
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def save_model(model, model_config, tokenizer, output_dir):
    """
    Save the model and its configuration to the specified output directory.

    Args:
        model (torch.nn.Module): The model to be saved.
        model_config (OmegaConf.DictConfig): The configuration of the model.
        output_dir (str): The directory where the model and configuration will be saved.

    Returns:
        None
    """
    model_config.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def create_readme(output_dir, config):
    """
    Create a README file based on a template and provided configuration.
    Args:
        output_dir (str): The directory where the README file will be saved.
        config (dict): The configuration dictionary containing the necessary information.
    Raises:
        FileNotFoundError: If the template or banner file is not found.
    Returns:
        None
    """

    # Load the template.
    template_path = os.path.join("assets", "readmetemplate.md")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    # Load the template.
    with open(template_path, "r") as f:
        readme_text = f.read()

    # Project name.
    model_name = config.training.model_name

    # Configuration convert the configuration to a yaml string.
    configuration = OmegaConf.to_yaml(config)

    # Base model.
    base_model = "None"
    if "base_model" in config.model:
        base_model = config.model.base_model

    # Tags.
    tags = ["NLP"]
    if "tags" in config.model:
        tags = config.model.tags.split(",")
    tags = "\n".join([f"  - {tag}" for tag in tags])

    # Languages.
    languages = ["en"]
    if "languages" in config.model:
        languages = config.model.languages.split(",")
    languages = "\n".join([f"  - {language}" for language in languages])

    # Datasets.
    datasets = [config.dataset.hugging_face_id]
    datasets = "\n".join([f"  - {dataset}" for dataset in datasets])

    # License.
    license = "mit"

    # Format the template.
    readme_text = readme_text.format(
        model_name=model_name,
        configuration=configuration,
        base_model=base_model,
        tags=tags,
        languages=languages,
        datasets=datasets,
        license=license,
    )

    # Save the readme.
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_text)

    # Copy the banner.
    banner_path = os.path.join("assets", "trainedwithhelibrunna.jpg")
    if not os.path.exists(banner_path):
        raise FileNotFoundError(f"Banner not found: {banner_path}")
    banner_target_path = os.path.join(output_dir, "banner.jpg")
    shutil.copy(banner_path, banner_target_path)


def preprocess_only(config_path):
    # Load the configuration.
    config = load_config(config_path)

    # Initialize the accelerator.
    accelerator = Accelerator()

    _ = preprocess(config, accelerator, ask_for_overwrite=True)


def preprocess(config, accelerator=None, ask_for_overwrite=False):
    """
    Preprocess the dataset and tokenizer. Only the main process should perform this task.

    Args:
        config (OmegaConf): The configuration object.
        accelerator (Accelerator): The Accelerator instance.

    Returns:
        datasets.DatasetDict: The tokenized datasets.
        PreTrainedTokenizerFast: The tokenizer.
    """

    # Load the dataset.
    hugging_face_id = config.dataset.hugging_face_id

    if isinstance(hugging_face_id, str):
        hugging_face_id = (hugging_face_id,)

    model_name = config.training.model_name
    output_path = config.dataset.output_path

    data_path = os.path.join(output_path, f"preprocessed/{model_name}/data")
    tokenizer_path = os.path.join(output_path, f"preprocessed/{model_name}/tokenizer")
    tokenized_data_path = os.path.join(output_path, f"preprocessed/{model_name}/tokenized_datasets")

    # If tokenizer and tokenized datasets exist, and ask_for_overwrite is True, ask for overwrite.
    if (
        os.path.exists(tokenizer_path)
        and os.path.exists(tokenized_data_path)
        and ask_for_overwrite
    ):
        overwrite = input("Preprocessed data already exists. Overwrite? [y/n]: ")
        if overwrite.lower() == "y":
            accelerator.print("Deleting existing preprocessed data...")
            shutil.rmtree(data_path)
            shutil.rmtree(tokenizer_path)
            shutil.rmtree(tokenized_data_path)

    # If tokenizer and tokenized datasets exist, load them.
    if os.path.exists(tokenizer_path) and os.path.exists(tokenized_data_path):
        accelerator.print("Loading preprocessed data...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        tokenized_datasets = load_from_disk(tokenized_data_path)
        return tokenized_datasets, tokenizer

    # Download the dataset.
    if accelerator.is_local_main_process:
        accelerator.print(f"Loading dataset: {hugging_face_id}")
        raw_datasets = load_dataset(*hugging_face_id, split=config.dataset.split)

        # Save the dataset to disk to be reused by other processes.
        raw_datasets.save_to_disk(data_path)
        accelerator.print("Dataset downloaded and saved.")
    else:
        # Other processes wait for the dataset to be downloaded and saved.
        while not os.path.exists(data_path):
            time.sleep(1)
        raw_datasets = load_dataset(data_path)

    accelerator.wait_for_everyone()

    # Tokenizer creation.
    if config.tokenizer.type == "whitespace":
        if accelerator.is_local_main_process:
            accelerator.print("Training whitespace tokenizer...")
            tokenizer = train_whitespace_tokenizer(raw_datasets)
            tokenizer.save_pretrained(tokenizer_path)
            vocab_size = tokenizer.vocab_size
        else:
            while not os.path.exists(f"{tokenizer_path}/tokenizer.json"):
                time.sleep(1)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            vocab_size = tokenizer.vocab_size
    elif config.tokenizer.type == "pretrained":
        from transformers import AutoTokenizer

        if accelerator.is_local_main_process:
            tokenizer_id = config.tokenizer.pretrained_id
            accelerator.print(f"Loading pre-trained tokenizer: {tokenizer_id}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            if (
                tokenizer.pad_token is None
            ):  # and "GPT2TokenizerFast" in str(type(tokenizer)):
                assert (
                    tokenizer.eos_token is not None
                ), "Tokenizer does not have a eos token."
                tokenizer.pad_token = tokenizer.eos_token
            # else:
            #     #tokenizer.add_tokens("[PAD]")
            #     #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            #     assert False, f"Tokenizer type not supported: {type(tokenizer)}"
            tokenizer.save_pretrained(tokenizer_path)
        else:
            while not os.path.exists(f"{tokenizer_path}/tokenizer_config.json"):
                time.sleep(1)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {config.tokenizer.type}")

    accelerator.wait_for_everyone()

    # Assign the vocabulary size to the model configuration.
    # assert vocab_size > 0
    # config.model.vocab_size = vocab_size

    if config.dataset.shuffle:
        if accelerator.is_local_main_process:
            accelerator.print("Shuffling dataset...")
            raw_datasets = raw_datasets.shuffle(seed=config.dataset.seed)

    # Tokenize the datasets.
    def tokenize_function(example):
        tokenized_example = tokenizer(
            example["text"],
        )
        return tokenized_example

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.

        block_size = config.model.context_length
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    if accelerator.is_local_main_process:
        accelerator.print("Tokenizing datasets...")
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_datasets.column_names,
            num_proc=multiprocessing.cpu_count(),
        )
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=multiprocessing.cpu_count(),
            desc=f"Grouping texts in chunks of {config.model.context_length}",
        ).remove_columns("attention_mask")

        tokenized_datasets.save_to_disk(tokenized_data_path)
    else:
        while not os.path.exists(tokenized_data_path):
            time.sleep(1)
        tokenized_datasets = load_from_disk(tokenized_data_path)

    accelerator.wait_for_everyone()

    # Check a sample.
    if accelerator.is_local_main_process:
        accelerator.print("Sample tokenized text:")
        sample = raw_datasets[0]
        tokenized = tokenized_datasets[0]
        assert list(tokenized.keys()) == ["input_ids"], list(tokenized.keys())
        accelerator.print(f"Original text: {sample}")
        accelerator.print(f"Tokenized text: {tokenized}")

        print("Data saved to: ", data_path)

    return tokenized_datasets, tokenizer


# Run the training.
if __name__ == "__main__":
    if sys.argv[1] == "preprocess":
        config_path = sys.argv[2]
        preprocess_only(config_path)
    else:
        config_path = sys.argv[1]
        run_training(config_path)
