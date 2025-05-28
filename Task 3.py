import torch
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load image
def load_image(image_path, max_size=400):
    image = Image.open(image_path)
    size = max(image.size)
    if size > max_size:
        ratio = max_size / float(size)
        new_size = tuple([int(x * ratio) for x in image.size])
        image = image.resize(new_size, Image.ANTIALIAS)
    return transforms.ToTensor()(image).unsqueeze(0).cuda()

# Load images (content and style)
content_img = load_image("content_image.jpg")
style_img = load_image("style_image.jpg", max_size=500)

# Define normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Load VGG19 model (pre-trained)
vgg = models.vgg19(pretrained=True).features.cuda().eval()

# Extract features from layers of the VGG model
def get_features(image, model, layers=None):
    if layers is None:
        layers = [0, 5, 10, 19, 28]  # Layers for feature extraction
    features = []
    x = image
    for i, layer in enumerate(model):
        x = layer(x)
        if i in layers:
            features.append(x)
    return features

# Compute content and style features
content_features = get_features(content_img, vgg)
style_features = get_features(style_img, vgg)

# Compute Gram Matrix for Style Transfer
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (d * h * w)

# Compute style loss and content loss
def compute_loss(content, style, target, content_weight=1e5, style_weight=1e10):
    content_loss = torch.mean((target - content)**2)
    style_loss = 0
    for target_feature, style_feature in zip(target, style):
        target_gram = gram_matrix(target_feature)
        style_gram = gram_matrix(style_feature)
        style_loss += torch.mean((target_gram - style_gram)**2)
    return content_weight * content_loss + style_weight * style_loss

# Initialize target image (copy of content image)
target = content_img.clone().requires_grad_(True).cuda()

# Optimizer
optimizer = optim.LBFG
