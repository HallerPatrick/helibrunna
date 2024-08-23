# Helibrunna - A HuggingFace compatible xLSTM trainer.
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

import json
import fire
import os
import glob
from huggingface_hub import create_repo
from huggingface_hub import HfApi


def update_readme_md(readme_path: str, model_name: str = "xLSTM", orga: str = "PatrickHaller"):
    with open(readme_path, "r") as f:
        content = f.read()

    content = content.replace("<PLACEHOLDER_MODEL_NAME>", model_name)
    content = content.replace("<PLACEHOLDER_ORGA>", orga)

    with open(readme_path, "w") as f:
        f.write(content)


def update_config_json(config_path: str):
    with open(config_path, "r") as f:
        content = json.load(f)
        if "auto_map" not in content:
            content["auto_map"] = {
                "AutoConfig": "configuration_xlstm.xLSTMConfig",
                "AutoModelForCausalLM": "modeling_xlstm.xLSTMForCausalLM",
                "AutoModel": "modeling_xlstm.xLSTMModel",
            }

    with open(config_path, "w") as f:
        json.dump(content, f, indent=2)

def push_to_huggingface(
        model_path: str,
        username_or_orga: str,
        repo_name: str,
        private: bool = False,
):
    """
    Pushes a model to Hugging Face Model Hub repository.
    Args:
        model_path (str): The path to the model directory.
        username_or_orga (str): The username or organization name on Hugging Face Model Hub.
        repo_name (str): The name of the repository on Hugging Face Model Hub.
        private (bool, optional): Whether the repository should be private. Defaults to False.
    Raises:
        ValueError: If the model path does not exist or if no checkpoints are found.
    """

    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist.")
    
    # Add the files that are required.
    files = [
        # os.path.join(model_path, "tokenizer.json"),
        # os.path.join(model_path, "special_tokens_map.json"),
        # os.path.join(model_path, "tokenizer_config.json"),
        # os.path.join(model_path, "README.md"),
        # os.path.join(model_path, "banner.jpg"),

        os.path.join("model", "README.md"),
        os.path.join("model", "configuration_xlstm.py"),
        os.path.join("model", "modeling_xlstm.py"),
        os.path.join(model_path, "tokenizer_config.json"),
        os.path.join(model_path, "tokenizer.json"),
        os.path.join(model_path, "generation_config.json"),
        os.path.join(model_path, "model.safetensors"),
        os.path.join(model_path, "special_tokens_map.json"),
        os.path.join(model_path, "config.json"),
    ]
    
    update_readme_md(files[0], model_name=repo_name, orga=username_or_orga)
    # Update config.json so we include the auto mapping from the hub repo
    update_config_json(files[-1])


    # Find all the checkpoints. Make sure these are folders.
    checkpoints = glob.glob(os.path.join(model_path, "checkpoint-*"))
    checkpoints = [checkpoint for checkpoint in checkpoints if os.path.isdir(checkpoint)]
    if len(checkpoints) > 0:

        # Find the last checkpoint. It begins with "checkpoint-" and ends with a "-last".
        last_checkpoint = None
        for checkpoint in checkpoints:
            if checkpoint.endswith("-last"):
                last_checkpoint = checkpoint
                break
        if last_checkpoint is None:
            raise ValueError("No last checkpoint found.")
        files += [os.path.join(last_checkpoint, "model.safetensors")]
        files += [os.path.join(last_checkpoint, "config.yaml")]
    else:
        print("No checkpoints found.")

    # Check if all files exist.
    for file in files:
        if not os.path.exists(file):
            raise ValueError(f"File {file} does not exist.")
    print(f"Files to be uploaded: {len(files)}")
    for file in files:
        print(f"  - {file}")
    
    # Create the repository.
    print(f"Creating repository {username_or_orga}/{repo_name}...")
    create_repo(
        f"{username_or_orga}/{repo_name}",
        private=private,
    )

    # Push the files.
    api = HfApi()
    print(f"Pushing files to repository {username_or_orga}/{repo_name}...")
    for file in files:
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=os.path.basename(file),
            repo_id=f"{username_or_orga}/{repo_name}",
            repo_type="model",
        )

    # Done.
    print(f"Pushed {len(files)} files to {username_or_orga}/{repo_name}.")


# Entry point.
if __name__ == "__main__":
    fire.Fire(push_to_huggingface)
