import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision.transforms as transforms 
import torchvision.models as models 
from PIL import Image 
from torchvision.utils import save_image

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]
    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
imsize = 356

loader = transforms.Compose([ 
    transforms.Resize((imsize, imsize)), 
    transforms.ToTensor()
])

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

orig_img = load_image('orig.png')
style_img = load_image('style.jpg')
generated = orig_img.clone().requires_grad_(True)

model = VGG().to(device).eval()
total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated], lr=learning_rate)

for step in range(total_steps):
    orig_features = model(orig_img)
    style_features = model(style_img)
    generated_features = model(generated)

    style_loss = orig_loss = 0

    for orig_feature, style_feature, generated_feature in zip(orig_features, style_features, generated_features):
        orig_loss += torch.mean((orig_feature-generated_feature)**2)
        batch_size, channel, height, width = orig_feature.shape 
        # Gram Matrix
        G = generated_feature.view(channel, height*width).mm(generated_feature.view(channel, height*width).t())
        A = style_feature.view(channel, height*width).mm(style_feature.view(channel, height*width).t())
        style_loss += torch.mean((G-A)**2)
    total_loss = alpha*orig_loss + beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step%200 == 0:
        print(total_loss)
        save_image(generated, 'c.png')