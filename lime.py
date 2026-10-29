import os
import json
import urllib.request
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from sklearn.linear_model import Ridge

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
LABELS_PATH = "imagenet_labels.json"

def download_and_load_imagenet_labels(labels_path: str = LABELS_PATH, labels_url: str = LABELS_URL) -> List[str]:
  """ Downloads and caches ImageNet class labels if not already present. """
  if not os.path.exists(labels_path):
    urllib.request.urlretrieve(labels_url, labels_path)
  with open(labels_path, "r", encoding="utf-8") as f:
    return json.load(f)

class ModelWrapper:
  def __init__(
    self,
    model: torch.nn.Module,
    device: torch.device
  ):
    self.device = device
    self.model = model.eval().to(self.device)
    self.class_names = download_and_load_imagenet_labels()

    self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
    self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

  def predict_proba(self, images: Union[np.ndarray, List[np.ndarray]], batch_size: int = 64) -> np.ndarray:
    """
    Computes the matrix of probabilities over all classes for the passed list of images (H, W, 3) in [0, 1].
    Added support for single images as well, so this can be used for computing the predicted class through argmax
    Returns:
      (matrix_of_probabilities) 
    """
    if isinstance(images, list):
      images = np.stack(images, axis=0)

    num_images = images.shape[0]
    all_probs = []
    with torch.inference_mode():
      for i in range(0, num_images, batch_size):
        batch_np = images[i : i + batch_size]
        batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(self.device)
        batch_tensor = (batch_tensor - self.mean) / self.std

        logits = self.model(batch_tensor)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)

  def predict_class(self, image: np.ndarray) -> Tuple[int, str]:
    """
    Runs inference on a single image (H, W, 3) in [0, 1].
    Returns:
      (predicted_class_id, predicted_label_name)
    """
    input_batch = np.expand_dims(image, axis=0)
    probs = self.predict_proba(input_batch, batch_size=1)
    class_id = int(np.argmax(probs[0]))
    label_name = self.class_names[class_id]
    return class_id, label_name

class ImageExplanation:
  def __init__(
    self,
    image: np.ndarray,
    segments: np.ndarray,
    target_class: int,
    target_label: str,
    coefficients: np.ndarray,
    intercept: float,
    score: float,
    predicted_proba: float,
    explainer_model: Ridge,
    masks: np.ndarray,
    predicted_probabilities: np.ndarray
  ):
    self.image = image
    self.segments = segments
    self.target_class = target_class
    self.target_label = target_label
    self.coefficients = coefficients
    self.intercept = intercept
    self.score = score
    self.predicted_proba = predicted_proba
    self.explainer_model = explainer_model
    self.masks = masks
    self.predicted_probabilities = predicted_probabilities

  def get_top_features(self, num_features: int = 5) -> List[Tuple[int, float]]:
    """ Returns the IDs of the top-k most important regions, alongisde their model coefficients. """
    sorted_indices = np.argsort(np.abs(self.coefficients))[::-1]
    top_indices = sorted_indices[:num_features]
    return [(int(idx), float(self.coefficients[idx])) for idx in top_indices]

  def get_image_and_mask(
    self,
    num_features: int = 5,
    mask_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
  ) -> Tuple[np.ndarray, np.ndarray]:
    """ Generates an image showing only the top regions. """
    top_features = self.get_top_features(num_features=num_features)
    top_segment_ids = [feat[0] for feat in top_features]

    mask = np.isin(self.segments, top_segment_ids)
    result_img = np.array(mask_color, dtype=np.float32) * np.ones_like(self.image)
    result_img[mask] = self.image[mask]

    return result_img, mask

  def visualize(
    self,
    num_features: int = 5,
    figsize: Tuple[int, int] = (20, 5)
  ):
    """ Plots the explanation figure. """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # - Original image -
    axes[0].set_title(f"Original image\nLabel: {self.target_label} (Class {self.target_class}: {self.predicted_proba:.2%})")
    axes[0].imshow(self.image)

    # - Regions -
    axes[1].set_title(f"Regions")
    partitioned_view = mark_boundaries(self.image, self.segments, color=(1, 1, 0), mode="thick")
    axes[1].imshow(partitioned_view)

    # - Heatmap of weights -
    axes[2].set_title(f"Importance of each region\nModel Fidelity R2 = {self.score:.2f}")
    heat_map = np.zeros(self.segments.shape, dtype=np.float32)
    for seg_id, weight in enumerate(self.coefficients):
      heat_map[self.segments == seg_id] = weight
    
    max_bound = max(float(np.max(np.abs(heat_map))), 1e-6)
    im = axes[2].imshow(heat_map, cmap="RdBu_r", vmin=-max_bound, vmax=max_bound)
    boundaries = find_boundaries(self.segments, mode="thick")
    boundary_overlay = np.zeros((*boundaries.shape, 4))
    boundary_overlay[boundaries] = (0, 0, 0, 0.6)
    axes[2].imshow(boundary_overlay)

    # - Most important regions -
    axes[3].set_title(f"Top {num_features} most important regions")
    masked_img, _ = self.get_image_and_mask(num_features=num_features)
    axes[3].imshow(masked_img)

    for ax in axes:
      ax.axis("off")

    plt.tight_layout()
    plt.show()

class LimeImageExplainer:
  def __init__(
    self,
    model: torch.nn.Module,
    device: torch.device,
    kernel_width: float = 0.25
  ):
    self.model_wrapper = ModelWrapper(model, device)
    self.kernel_width = kernel_width

  @staticmethod
  def load_and_preprocess_image(
    image_input: str,
    target_size: Tuple[int, int] = (224, 224)
  ) -> np.ndarray:
    """ Loads and resizes an image (exclusively from its path), returning a float32 array in [0, 1] of shape (H, W, 3). """
    raw_image = Image.open(image_input).convert("RGB")
    resized_image = raw_image.resize(target_size, Image.Resampling.BILINEAR)
    return np.array(resized_image, dtype=np.float32) / 255.0

  def segment_image(
    self,
    image: np.ndarray,
    n_segments: int = 45,
    compactness: float = 10.0,
    sigma: float = 1.0
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes regions using SLIC algorithm (?).
    Returns:
      segments: (H, W) array with contiguous 0-indexed segment labels
      unique_segments: (K,) array of unique segment labels
    """
    segments = slic(image, compactness=compactness, n_segments=n_segments, sigma=sigma, start_label=0)
    unique_segments = np.unique(segments)
    return segments, unique_segments

  def explain_instance(
    self,
    image_input: str,
    num_samples: int = 800,
    batch_size: int = 64,
    mask_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    n_segments: int = 45,
    compactness: float = 10.0,
    sigma: float = 1.0,
    kernel_width: Optional[float] = None,
    random_state: Optional[int] = None
  ) -> ImageExplanation:
    """
    Explains the prediction of `self.model_wrapper` on image_input for the top class.

    Parameters:
      image_input: file path
      num_samples: number of perturbed neighborhood samples to generate
      batch_size: batch size for batched classifier inference
      mask_color: RGB fill color for masked regions
      n_segments, compactness, sigma: segmentation parameters
      kernel_width: kernel parameter
    """

    # Added support for different random seeds
    if random_state is not None:
      np.random.seed(random_state)

    kw = kernel_width or self.kernel_width
    image = self.load_and_preprocess_image(image_input)

    # Segment the image into regions
    segments, unique_segments = self.segment_image(
      image,
      n_segments=n_segments,
      compactness=compactness,
      sigma=sigma
    )
    n_features = len(unique_segments)

    # Compute the initial probability for the predicted class
    initial_probs = self.model_wrapper.predict_proba(np.expand_dims(image, axis=0))[0]
    target_class = int(np.argmax(initial_probs))
    initial_target_proba = float(initial_probs[target_class])

    wrapper_class_names = self.model_wrapper.class_names
    target_label = wrapper_class_names[target_class]

    # Compute binary perturbation masks
    perturbation_masks = np.random.binomial(1, 0.5, size=(num_samples, n_features)).astype(np.float32)
    perturbation_masks[0, :] = 1.0  # original image instance

    # Generate the perturbed images and predict target class probabilities
    target_probabilities = np.zeros(num_samples, dtype=np.float32)
    mask_color_arr = np.array(mask_color, dtype=np.float32).reshape(1, 1, 1, 3)

    for i in range(0, num_samples, batch_size):
      batch_masks = perturbation_masks[i : i + batch_size]
      batch_pixel_masks = batch_masks[:, segments, None]
      batch_perturbed_imgs = batch_pixel_masks * image + (1.0 - batch_pixel_masks) * mask_color_arr

      batch_probs = self.model_wrapper.predict_proba(batch_perturbed_imgs)
      target_probabilities[i : i + batch_size] = batch_probs[:, target_class]

    # Compute the distances + weights
    original_vector = np.ones(n_features, dtype=np.float32)
    distances = np.linalg.norm(perturbation_masks - original_vector, axis=1) / np.sqrt(n_features)
    weights = np.exp(-(distances ** 2) / (kw ** 2))

    # Fit the linear model
    surrogate_model = Ridge(alpha=1.0, fit_intercept=True)
    surrogate_model.fit(perturbation_masks, target_probabilities, sample_weight=weights)

    coefficients = surrogate_model.coef_
    intercept = float(surrogate_model.intercept_)
    score = float(surrogate_model.score(perturbation_masks, target_probabilities, sample_weight=weights))

    return ImageExplanation(
      image=image,
      segments=segments,
      target_class=target_class,
      target_label=target_label,
      coefficients=coefficients,
      intercept=intercept,
      score=score,
      predicted_proba=initial_target_proba,
      explainer_model=surrogate_model,
      masks=perturbation_masks,
      predicted_probabilities=target_probabilities
    )

  def explain(
    self,
    image_input: str,
    **kwargs
  ) -> ImageExplanation:
    """ Wrapper over explain_instance for ease of use. """
    return self.explain_instance(image_input=image_input, **kwargs)
