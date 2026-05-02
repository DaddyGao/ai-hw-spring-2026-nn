import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# ==========================================
# 1. DATA PREPARATION & AUGMENTATION
# ==========================================
# Standard transform (No augmentation)
standard_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std for MNIST
])

# Augmented transform (Improves robustness)
augmented_transform = transforms.Compose([
    transforms.RandomRotation(10), # Rotate by up to 10 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Shift up to 10%
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_dataloaders(use_augmentation=False, batch_size=64):
    train_transform = augmented_transform if use_augmentation else standard_transform
    
    # Train on the train part
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Test on the test part (NEVER augment the test set)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=standard_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader

# ==========================================
# 2. MODELS
# ==========================================

# A. Shallow Multi-Layer Perceptron (MLP)
class ShallowMLP(nn.Module):
    def __init__(self):
        super(ShallowMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# B. Convolutional Neural Network (CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

# C. Vision Transformer (Encoder)
# Note: ViTs usually need large datasets. This is a miniaturized version for 28x28 images.
class MiniViT(nn.Module):
    def __init__(self):
        super(MiniViT, self).__init__()
        self.patch_size = 7 # 28x28 image -> 16 patches of 7x7
        self.embed_dim = 64
        self.num_patches = (28 // self.patch_size) ** 2
        
        # Linear projection of flattened patches
        self.patch_embed = nn.Linear(self.patch_size * self.patch_size, self.embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(self.embed_dim, 10)

    def forward(self, x):
        B = x.size(0)
        # Extract patches: [B, 1, 28, 28] -> [B, 16, 49]
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, self.num_patches, -1)
        
        # Embed patches and add class token + positional embeddings
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Pass through Transformer
        x = self.transformer(x)
        
        # Use the class token for classification
        cls_out = x[:, 0]
        return self.fc(cls_out)

# ==========================================
# 3. TRAIN & TEST FUNCTIONS
# ==========================================
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# ==========================================
# 4. EXECUTION SCRIPT
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    epochs = 5
    models = {
        "MLP": ShallowMLP().to(device),
        "CNN": SimpleCNN().to(device),
        "MiniViT": MiniViT().to(device)
    }
    
    criterion = nn.CrossEntropyLoss()

    for aug_state, aug_name in [(False, "Standard Data"), (True, "Augmented Data")]:
        print(f"\n{'='*40}\nRunning with: {aug_name}\n{'='*40}")
        train_loader, test_loader = get_dataloaders(use_augmentation=aug_state)
        
        for name, model in models.items():
            # Reset model weights for a fair comparison between data states
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                    
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            print(f"\nTraining {name}...")
            
            start_time = time.time()
            for epoch in range(1, epochs + 1):
                train(model, device, train_loader, optimizer, criterion, epoch)
            
            test_loss, accuracy = test(model, device, test_loader, criterion)
            elapsed_time = time.time() - start_time
            
            print(f"{name} Results -> Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}% (Time: {elapsed_time:.1f}s)")