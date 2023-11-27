# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import random
import sys

def split(input_file, training_output_file, validation_output_file, test_output_file, train_proportion, validation_proportion, seed): 
    if seed is not None:
        random.seed(seed)
        
    # Read the JSONL file and shuffle the JSON objects
    with open(input_file, "r") as f:
        lines = f.readlines()
        random.shuffle(lines)

    # Calculate split indices
    total_lines = len(lines)
    train_index = int(total_lines * train_proportion)
    val_index = int(total_lines * validation_proportion)

    # Distribute JSON objects into training and validation sets
    train_data = lines[:train_index]
    validation_data = lines[train_index:train_index+val_index]
    test_data = lines[train_index+val_index:]

    # Write JSON objects to training file
    with open(training_output_file, "w") as f:
        for line in train_data:
            f.write(line.strip() + "\n")

    # Write JSON objects to validation file
    with open(validation_output_file, "w") as f:
        for line in validation_data:
            f.write(line.strip() + "\n")

    # Write JSON objects to training file
    with open(test_output_file, "w") as f:
        for line in test_data:
            f.write(line.strip() + "\n")


def main(arguments):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--train', type=str, required=True, help='Path to save the training set JSONL file')
    parser.add_argument('--valid', type=str, required=True, help='Path to save the validation set JSONL file')
    parser.add_argument('--test',  type=str, required=True, help='Path to save the test set JSONL file')
    parser.add_argument('--seed',  type=int, required=False, help='Split ratio for validate (default: 0.15)')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Split ratio for training (default: 0.8)')
    parser.add_argument('--valid-ratio', type=float, default=0.15, help='Split ratio for validate (default: 0.15)')

    args = parser.parse_args(arguments)
    split(args.input, args.train, args.valid, args.test, args.train_ratio, args.valid_ratio, args.seed)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
