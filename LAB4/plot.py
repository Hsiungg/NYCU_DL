import matplotlib.pyplot as plt

# Example data collection
epochs = []
teacher_forcing_rates = []
tf = 1.
rate = 0.05
# Simulate data collection during training
num_epochs = 80
for epoch in range(num_epochs):
    # Simulated teacher forcing rate, e.g., decreasing over epochs
    
    if epoch > 15:
        tf = tf - rate

    teacher_forcing_rate = max(0, tf)
    
    # Store epoch and teacher forcing rate
    epochs.append(epoch)
    teacher_forcing_rates.append(teacher_forcing_rate)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, teacher_forcing_rates, marker='o', linestyle='-', color='b', label='Teacher Forcing Rate')

# Adding titles and labels
plt.title('Teacher Forcing Rate')
plt.xlabel('Epoch')
plt.ylabel('Teacher Forcing Rate')
plt.legend()
plt.grid(True)
plt.savefig("plot.png")
plt.show()