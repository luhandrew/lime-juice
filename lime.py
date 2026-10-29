import torch 
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from skimage.segmentation import slic

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

        batch_logits = self.model(X_batch)
        batch_probs = F.softmax(batch_logits, dim=1)
        target_probs.extend(batch_probs[:, target_class].cpu().numpy())
    return target_probs

class LimeImageExplainer():
  def __init__(self, model, device):
    self.model_wrapper = ModelWrapper(model, device)

  def generate_neighborhood(self):
    pass 

  def train_explainer_model(self):
    pass

  def explain(self, image):
    """
    image: np.array of shape (H, W, 3)
           with float values in [0, 1]
    """

    segments = slic(image, compactness=100, n_segments=400)

    # we want to explain the top class predicted for this image
    predicted_class = self.model_wrapper.predict_class(image)