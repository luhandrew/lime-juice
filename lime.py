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

  def batched_predict(self, X, target_class, batch_size=32):
    """
    runs inference in batches

    batch_size   : size of a single batch (default 32)
    X            : np.array of shape [N, H, W, 3]
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

  def explain(self, img):
    """
    img_np: np.array of shape (H, W, 3)
            with float values in [0, 1]
    """

    segments = slic(img, compactness=100, n_segments=400)
    