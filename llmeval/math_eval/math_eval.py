import argparse
import json
import os

from llmeval.math_eval.math_score import compute_scores


def get_after_think(text):
    parts = text.split('\n</think>\n\n', 1)
    if len(parts) > 1:
        return parts[1]
    else:
        return text


def process_item(item, task_name):
    item['task'] = task_name
    return item


def main():
    parser = argparse.ArgumentParser(description='Evaluate model outputs')
    parser.add_argument('--input_path',
                        type=str,
                        required=True,
                        help='Path to input jsonl file')
    parser.add_argument('--cache_path',
                        type=str,
                        required=True,
                        help='Path to save cache results')
    parser.add_argument(
        '--task_name',
        type=str,
        required=True,
        help=
        "Task should be in ['math_opensource/aime24', 'math_opensource/aime25' ,'livecodebench', 'ifeval']"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)

    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    data = [process_item(item.copy(), args.task_name) for item in data]

    if 'math_opensource' in args.task_name:
        acc = compute_scores(data, args.cache_path)
        print(f'Task: {args.task_name}, Accuracy: {acc}')
    else:
        print(f'No evaluation function found for task name: {args.task_name}')

    print('Evaluation complete!')


if __name__ == '__main__':
    main()
