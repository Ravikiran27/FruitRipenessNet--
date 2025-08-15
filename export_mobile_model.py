import torch
import torch.nn as nn

# Define the model class (copy from notebook)
class FruitRipenessCNN(nn.Module):
	def __init__(self, num_classes):
		super(FruitRipenessCNN, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 32, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(64, 128, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(128 * 16 * 16, 256),
			nn.ReLU(),
			nn.Linear(256, num_classes)
		)
	def forward(self, x):
		x = self.features(x)
		x = self.classifier(x)
		return x

# Set number of classes manually (update if needed)
num_classes = 3  # Overripe, Ripe, Unripe

model = FruitRipenessCNN(num_classes)
model.load_state_dict(torch.load('best_fruit_ripeness_cnn.pth', map_location='cpu'))
model.eval()

example_input = torch.rand(1, 3, 128, 128)
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save('fruit_ripeness_mobile.pt')
print('TorchScript model saved as fruit_ripeness_mobile.pt')
