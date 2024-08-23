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

import fire
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from source.utilities import display_logo

# Register the xLSTM model and configuration.
import model


def generate(
        model_path_or_repo: str,
        prompt: str = "The ",
        temperature: float = 1.0,
        max_length: int = 100,
) -> None:
    """
    Generates text continuation based on a given prompt using a pre-trained language model.
    Args:
        model_path_or_repo (str): The path to the model or the Hugging Face repository ID.
        prompt (str): The prompt text to generate continuation from.
        temperature (float, optional): The temperature value for sampling from the distribution. Defaults to 0.5.
        max_length (int, optional): The maximum length of the generated text. Defaults to 100.
    Raises:
        ValueError: If the model weights, tokenizer, or config are not found at the specified paths.
    Returns:
        None
    """

    # Display the logo.
    display_logo()

    # Set the device.
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Create the model from the config.
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path_or_repo).to(device)

    # Load the tokenizer.
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_repo)

    # Tokenize the prompt.
    print("Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    assert inputs.shape[0] == 1

    # Generate the continuation.
    start_time = time.time()
    output = model.generate(
        inputs,
        do_sample=True,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
    )

    tokens_count = inputs.shape[1]

    # Print the elapsed time and tokens per second.
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f}s")
    tokens_per_second = tokens_count / elapsed_time
    print(f"Tokens per second: {tokens_per_second:.2f}")

    # Decode the output.
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\033[92mGenerated text:\033[0m")
    print(output)


# Entry point.
if __name__ == "__main__":
    fire.Fire(generate)
