import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Input: 3 channels, Output: 6 channels, Kernel: 5x5
        self.pool = nn.MaxPool2d(2, 2)  # Kernel: 2x2, Stride: 2
        self.conv2 = nn.Conv2d(6, 16, 5)  # Input: 6 channels, Output: 16 channels, Kernel: 5x5
        self.fc1 = nn.Linear(16 * 317 * 253, 120)  # Adjusted input size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply conv1 -> ReLU -> pool
        x = self.pool(F.relu(self.conv2(x)))  # Apply conv2 -> ReLU -> pool
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = F.relu(self.fc2(x))  # Fully connected layer 2
        x = self.fc3(x)  # Fully connected layer 3 (output layer)
        return x

if __name__ == "__main__":
    # Set device type
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the path to your custom dataset
    data_dir = r"C:\Users\ekros\OneDrive\Documents\Textbooks\Spring 2025\CCBER Machine Learning\synthetic_images - Copy"

    # Define transformations for the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the dataset using ImageFolder
    dataset = ImageFolder(root=data_dir, transform=transform)

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create DataLoaders for training and testing
    batch_size = 4
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Get the class names from the dataset
    classes = dataset.classes
    print("Classes:", classes)

    # Initialize the network, loss function, and optimizer
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Define the path where the model will be saved
    PATH = r"C:\Users\ekros\OneDrive\Documents\Textbooks\Spring 2025\CCBER Machine Learning\mlm_test1.pth"

    # Save the model's state dictionary
    torch.save(net.state_dict(), PATH)

    # Load the saved model for evaluation
    net.load_state_dict(torch.load(PATH))
    net.eval()  # Set the model to evaluation mode

    # Evaluate the model on the test dataset
    correct = 0
    total = 0
    # Since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # Calculate outputs by running images through the network
            outputs = net(images)

            # The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

    # Prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # Again, no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # Collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # Print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


