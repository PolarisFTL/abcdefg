import os
import math
import numpy as np
import functools
from collections import Counter
import matplotlib.pyplot as plt


def parse_detection_file(file_path, num_classes):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    list_label2indices = [[] for _ in range(num_classes)]
    image_paths = []
    labels_per_image = []

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        image_path = parts[0]
        bboxes = parts[1:]

        image_paths.append(line)
        class_ids = set()

        for box in bboxes:
            x1, y1, x2, y2, cls = box.split(',')
            class_ids.add(int(cls))

        for cls_id in class_ids:
            list_label2indices[cls_id].append(idx)

    return list_label2indices, image_paths


def partition_balance(idxs, num_split: int):
    num_per_part, r = len(idxs) // num_split, len(idxs) % num_split
    parts = []
    i, r_used = 0, 0
    while i < len(idxs):
        if r_used < r:
            parts.append(idxs[i:(i + num_per_part + 1)])
            i += num_per_part + 1
            r_used += 1
        else:
            parts.append(idxs[i:(i + num_per_part)])
            i += num_per_part
    return parts


def build_non_iid_by_dirichlet(seed, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers):
    random_state = np.random.RandomState(seed)
    random_state.shuffle(indices2targets)

    idx_batch = [[] for _ in range(n_workers)]
    targets = np.array(indices2targets)

    for cls in range(num_classes):
        idx_cls = np.where(targets[:, 1] == cls)[0]
        idx_cls = targets[idx_cls, 0]
        if len(idx_cls) == 0:
            continue

        proportions = random_state.dirichlet(np.repeat(non_iid_alpha, n_workers))
        proportions = (np.cumsum(proportions) * len(idx_cls)).astype(int)[:-1]
        split = np.split(idx_cls, proportions)
        for i, part in enumerate(split):
            idx_batch[i].extend(part.tolist())

    return idx_batch


def clients_indices(list_label2indices, num_classes, num_clients, non_iid_alpha, seed=None):
    indices2targets = []
    for label, indices in enumerate(list_label2indices):
        for idx in indices:
            indices2targets.append((idx, label))

    batch_indices = build_non_iid_by_dirichlet(
        seed=seed,
        indices2targets=indices2targets,
        non_iid_alpha=non_iid_alpha,
        num_classes=num_classes,
        num_indices=len(indices2targets),
        n_workers=num_clients
    )
    list_client2indices = partition_balance(
        functools.reduce(lambda x, y: x + y, batch_indices),
        num_clients
    )
    return list_client2indices


def save_split_files(client_indices, image_paths, output_dir="aaaaa"):
    os.makedirs(output_dir, exist_ok=True)
    for i, indices in enumerate(client_indices):
        with open(os.path.join(output_dir, f"client{i+1}.txt"), 'w') as f:
            for idx in indices:
                f.write(image_paths[idx] + "\n")


def parse_detection_line(line):
    parts = line.strip().split()
    bboxes = parts[1:]
    return [int(b.split(',')[-1]) for b in bboxes]


def compute_class_distribution(file_paths, num_classes):
    distributions = []
    for file_path in file_paths:
        counter = Counter()
        with open(file_path, 'r') as f:
            for line in f:
                counter.update(parse_detection_line(line))
        distributions.append([counter[i] for i in range(num_classes)])
    return np.array(distributions)


def plot_class_distribution(distributions, num_classes, output_path="class_distribution.png", non_iid_alpha=0.5):
    num_clients = distributions.shape[0]
    x = np.arange(num_classes)
    width = 0.28  

    plt.figure(figsize=(8, 5))
    for i in range(num_clients):
        plt.bar(x + i * width, distributions[i], width, label=f'Client {i+1}')
    
    plt.xlabel('Class ID', fontsize=14)
    plt.ylabel('Sample Count', fontsize=14)
    plt.title(f'Client-wise Class Distribution (α={non_iid_alpha})', fontsize=20)
    plt.xticks(x + width * (num_clients - 1) / 2, [str(i) for i in range(num_classes)], fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(pad=1.0) 
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    input_txt = "datasets_split/train.txt"
    num_classes = 5
    num_clients = 3
    non_iid_alpha = 100
    seed = 42

    list_label2indices, image_paths = parse_detection_file(input_txt, num_classes)
    client_indices = clients_indices(list_label2indices, num_classes, num_clients, non_iid_alpha, seed)
    save_split_files(client_indices, image_paths, output_dir=f"demo/split_clients_{non_iid_alpha}")
    
    client_files = [f"demo/split_clients_{non_iid_alpha}/client{i+1}.txt" for i in range(num_clients)]
    distributions = compute_class_distribution(client_files, num_classes)
    for i, dist in enumerate(distributions):
        print(f"Client {i+1}: {dist}")

    plot_class_distribution(distributions, num_classes, output_path=f"demo/split_clients_{non_iid_alpha}/class_distribution.png", non_iid_alpha=non_iid_alpha)

if __name__ == "__main__":
    main()
