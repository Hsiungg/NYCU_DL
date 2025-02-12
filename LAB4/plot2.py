from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
# Example of OneCycleLR scheduler

model = nn.Linear(10, 1)  # Example model
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = OneCycleLR(optimizer, total_steps=80, max_lr=0.0003, three_phase=True )# Example scheduler
criterion = nn.MSELoss()

# Dummy data
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# Store learning rates
lrs = []

# Training loop
for epoch in range(80):  # Example number of epochs
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # Capture learning rate
    current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)

# Plot learning rates
plt.plot(lrs)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Over Epochs')
plt.savefig("plot_lr.png")