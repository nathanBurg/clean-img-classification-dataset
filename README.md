# clean-img-classification-dataset
    - [Medium Article](https://medium.com/@nathan_49206/finding-needles-in-the-haystack-2fbbe45da9dd)

## Setting up the project environment
### Run in root of repository
1. Install virtualenv if you haven't already:
    - **Note:** Depending on your system and how Python is installed, you may need to use `pip` instead of `pip3` to ensure you are using Python 3.

   ```
   pip3 install virtualenv
   ```
2. Create a virtual environment in the project directory:
   ```
   virtualenv clean-img-classification-dataset-env
   ```
3. Activate the virtual environment:
   - On Windows:
     ```
     .\clean-img-classification-dataset-env\Scripts\activate
     ```
   - On Unix or MacOS:
     ```
     source clean-img-classification-dataset-env/bin/activate
     ```
4. Install the required dependencies:
   ```
   pip3 install -r requirements.txt
   ```


## Run the script
1.  First ensure you have saved your embeddings, labels, and paths to a .npz file. You can insert the following code into your script where the embeddings, labels, and paths are generated.
    ```
    import numpy as np

    embeddings = [] # list of embeddings
    labels = [] # list of labels
    image_paths = [] # list of img paths

    np.savez_compressed(
        "image_embeddings_labels_paths.npz",
        embeddings=np.vstack(embeddings),  # Stack embeddings into a 2D array
        labels=np.array(labels),  # Convert labels to a 1D numpy array
        image_paths=np.array(image_paths)  # Convert image paths to a 1D numpy array
    )
    ```
2. Update the `NPZ_PATH` variable in `clean_dataset.py` to the path of your .npz file.
3. Run the script:
    ```
    python3 clean_dataset.py
    ```
