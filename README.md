# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load and Preprocess Data

### STEP 2: 

Get the shape of the first image in the training dataset

### STEP 3: 

Get the shape of the first image in the test dataset

### STEP 4:

Train the Model

### STEP 5: 

Test the Model

### STEP 6: 

Predict on a Single Image

### STEP 7: 

Display the image



## PROGRAM

### Name:R.SUBHASHRI

### Register Number:212223230219

```python


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Initialize model, loss function, and optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    for images,labels in train_loader:
      optimizer.zero_grad()
      outputs=model(images)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss+=loss.item()
    print('Name:R.SUBHASHRI')
    print('Register Number:   212223230219    ')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


```

### OUTPUT

## Training Loss per Epoch


<img width="304" height="172" alt="image" src="https://github.com/user-attachments/assets/70e7b58e-239a-4a1e-8108-ef74bc6dd91b" />


## Confusion Matrix

<img width="869" height="699" alt="image" src="https://github.com/user-attachments/assets/4616f148-e840-4c26-887d-5380fbffe987" />

## Classification Report


<img width="476" height="342" alt="image" src="https://github.com/user-attachments/assets/34710362-0ba9-4c8d-9ec2-c090b41cbb9b" />


### New Sample Data Prediction

<img width="684" height="520" alt="image" src="https://github.com/user-attachments/assets/9da0d9ef-6c1c-40e3-b45a-224a6bd70119" />


## RESULT


Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
