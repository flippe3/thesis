#!/bin/python3
import matplotlib.pyplot as plt
import torch
import json
import sys

colors = ['#000000','#1E93FF','#F93C31','#4FCC30','#FFDC00', '#999999','#E53AA3','#FF851B','#87D8F1','#921231','#555555']
colormap = plt.matplotlib.colors.ListedColormap(colors)

base_path = "data/"

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

training_challenges = load_json(base_path + 'arc-agi_training_challenges.json')
evaluation_challenges = load_json(base_path + 'arc-agi_evaluation_challenges.json')
test_challenges = load_json(base_path + 'arc-agi_test_challenges.json')
sample_submission = load_json(base_path + 'sample_submission.json')


def plot_task(task, task_name):
    training_pairs = task.get('train', [])
    evaluating_pairs = task.get('test', [])
    num_train_pairs = len(training_pairs)
    num_eval_pairs = len(evaluating_pairs)

    fig, axs = plt.subplots(num_train_pairs + num_eval_pairs, 2, figsize=(10, (num_train_pairs + num_eval_pairs) * 4))

    for i, pair in enumerate(training_pairs):
        input_data = pair['input']
        output_data = pair['output']

        axs[i, 0].imshow(input_data, cmap='viridis', interpolation='nearest')
        axs[i, 0].set_title(f'Training Input - Pair {i+1}')

        axs[i, 1].imshow(output_data, cmap='viridis', interpolation='nearest')
        axs[i, 1].set_title(f'Training Output - Pair {i+1}')

    for i, pair in enumerate(evaluating_pairs):
        input_data = pair['input']

        axs[i + num_train_pairs, 0].imshow(input_data, cmap='viridis', interpolation='nearest')
        axs[i + num_train_pairs, 0].set_title(f'Evaluating Input - Pair {i+1}')
        axs[i + num_train_pairs, 1].axis('off')

    plt.tight_layout()
    plt.savefig(str(task_name))

if len(sys.argv) != 2:
    print("Please enter a task number.")
    print(f"Examples:{training_challenges.keys()}")
else:
    t = sys.argv[1]
    task = training_challenges[t]
    print(f'Task: {t}')
    plot_task(task, t)

