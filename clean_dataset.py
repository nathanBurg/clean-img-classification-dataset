import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import random
import numpy as np
import umap
from datetime import datetime
from typing import Tuple, List


NPZ_PATH = "path/to/your/npz/file.npz"
EXCLUDED_PATHS = []

def load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data = np.load(npz_path)
    return data["embeddings"], data["labels"], data["image_paths"]

def reduce_embeddings(embeddings: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(n_components=3)
    return reducer.fit_transform(embeddings)

def save_to_txt(paths: List[str]) -> None:
    unique_paths = list(set(paths))
    now = datetime.now()
    file_name = f"images_to_exclude_from_training_{now.strftime('%m_%d_%Y')}.txt"
    with open(file_name, "w") as file:
        file.writelines(f"{path}\n" for path in unique_paths)

def on_pick(event, labels: np.ndarray, paths: List[str]) -> None:
    ind = event.ind[0]
    image_path = paths[ind]
    image = Image.open(image_path)
    label = labels[ind]
    
    fig, ax = plt.subplots()
    plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.axis('off')

    def exclude(event) -> None:
        EXCLUDED_PATHS.append(image_path)
        save_to_txt(EXCLUDED_PATHS)
        plt.close(fig)
        print(f"Excluded: {image_path}")

    ax_button = plt.axes([0.8, 0.01, 0.1, 0.075])
    btn = Button(ax_button, 'Exclude')
    btn.on_clicked(exclude)

    plt.show()
    print(f"Label: {label}, Image Path: {image_path}")

def graph_embeddings(reduced_embeddings: np.ndarray, labels: np.ndarray, paths: List[str]) -> None:
    def random_color() -> Tuple[float, float, float]:
        return random.random(), random.random(), random.random()

    unique_labels = np.unique(labels)
    label_to_color = {label: random_color() for label in unique_labels}
    point_colors = [label_to_color[label] for label in labels]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], 
                         c=point_colors, s=5, picker=True)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_to_color[label], markersize=10) for label in unique_labels]
    ax.legend(handles, unique_labels, loc='best', title='Classes', fontsize='small', markerscale=2, frameon=True)

    ax.set_title('3D Cluster Map of Mammal Embeddings')
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_zlabel('UMAP-3')

    fig.canvas.mpl_connect('pick_event', lambda event: on_pick(event, labels, paths))

    plt.show()

if __name__ == "__main__":
    embeddings, labels, paths = load_npz(NPZ_PATH)
    reduced_embeddings = reduce_embeddings(embeddings)
    graph_embeddings(reduced_embeddings, labels, paths)