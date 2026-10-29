import numpy as np
import torch 
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from skimage.segmentation import slic
from PIL import Image

class ModelWrapper():
  def __init__(self, model, device):
    self.device = device
    self.model = model.eval().to(self.device)
    self.image_tf = T.Compose([
      T.ToImage(),
      T.ToDtype(torch.float32, scale=True),
      T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
      )
    ])

  def predict_class(self, image):
    """
    image: np.array of shape (H, W, 3)
    """

    image_tensor = self.image_tf(image).unsqueeze(0).to(self.device)
    with torch.no_grad():
      logits = self.model(image_tensor)
      predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

  def predict_target_class_proba(self, X, target_class, batch_size=32):
    """
    runs inference in batches
    returns the probability assigned by the model to the target_class
    over all examples

    batch_size   : size of a single batch (default 32)
    X            : np.array of shape (N, H, W, 3)
    target_class : the class for which we extract probabilities

    note: N is length of dataset
    """
    target_probs = []
    with torch.no_grad():
      for i in range(0, len(X), batch_size):
        X_batch = X[i : i + batch_size]
        X_batch = torch.stack([self.image_tf(img) for img in X_batch])
        X_batch = X_batch.to(self.device)

        batch_logits = self.model(X_batch)
        batch_probs = F.softmax(batch_logits, dim=1)
        target_probs.extend(batch_probs[:, target_class].cpu().numpy())
    return target_probs

class LimeImageExplainer():
  def __init__(self, model, device):
    self.model_wrapper = ModelWrapper(model, device)

  def load_and_preprocess_image(self, image_path):
    """
    image_path: path to the image we want to load
    """
    raw_image = Image.open(image_path).convert("RGB")
    resized_image = raw_image.resize((224, 224), Image.Resampling.BILINEAR)
    np_image = np.array(resized_image).astype(np.float32) / 255.0
    return np_image

  def segment_image(self, image):
    """
    computes the "superpixels" of an image

    image: np.array of shape (H, W, 3)
           with float values in [0, 1]
    """
    segments = slic(image, compactness=100, n_segments=400)
    unique_segments = np.unique(segments)
    n_features = len(unique_segments)
    return (n_features, segments, unique_segments)

  def generate_neighborhood(self, image):
    """
    generates binary perturbations

    image: np.array of shape (H, W, 3)
           with float values in [0, 1]
    """
    n_features, segments, unique_segments = self.segment_image(image)

    n_samples = 1000 # how many perturbations to generate

    # generate n_samples masks for the segments of the image
    # for a mask, we only keep a segment if it's one
    z = np.random.binomial(1, 0.5, size=(n_samples, n_features))
    z[0, :] = 1.0 # make sure we also have the original image

    mask_color = np.array([0.5, 0.5, 0.5])
    X_perturbed = []
    for mask in z:
      perturbed_img = image.copy()
      masked_features = unique_segments[mask == 0]
      pixels_mask = np.isin(segments, masked_features)
      perturbed_img[pixels_mask] = mask_color
      X_perturbed.append(perturbed_img)
    return X_perturbed

  def train_explainer_model(self):
    pass

  def explain(self, image_path):
    """
    image_path: local path to the image we want to explain
    """
    image = self.load_and_preprocess_image(image_path)

    X_perturbed = self.generate_neighborhood(image)
    predicted_class = self.model_wrapper.predict_class(image)
    y_pred_perturbed = self.model_wrapper.predict_target_class_proba(
      X_perturbed, predicted_class)

    return y_pred_perturbed
