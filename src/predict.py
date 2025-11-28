import numpy as np
from PIL import Image


def preprocess_image(img: Image.Image, target_size=(32, 32)) -> np.ndarray:
    """
    Preprocess a PIL image for CIFAR-10 models:
    - resize to target_size
    - convert to float32
    - normalize to [0, 1]
    - add batch dimension
    """
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0

    # If grayscale, convert to 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    # Ensure shape (32, 32, 3)
    if arr.shape[-1] == 4:  # RGBA → RGB
        arr = arr[..., :3]

    arr = np.expand_dims(arr, axis=0)  # (1, 32, 32, 3)
    return arr


def ensemble_predict(models, img: Image.Image):
    """
    Run an unweighted soft-voting ensemble over a list of Keras models.

    Args:
        models: list of loaded Keras models
        img: PIL Image object

    Returns:
        pred_idx: predicted class index (int)
        confidence: probability of the predicted class (float)
        probs: full probability vector (np.ndarray of shape (num_classes,))
    """
    x = preprocess_image(img)
    # Collect probability predictions from each model
    prob_list = [m.predict(x, verbose=0)[0] for m in models]

    # Simple average across models (soft voting)
    probs = np.mean(prob_list, axis=0)

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    return pred_idx, confidence, probs
