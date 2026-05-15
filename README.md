# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Developing a Convolutional Deep Neural Network (CNN) for Image Classification

Image classification is a fundamental problem in computer vision where the goal is to assign a label to an image based on its visual content. Traditional machine learning methods struggle with high-dimensional image data and fail to capture spatial features effectively.

The objective of this project is to develop a Convolutional Neural Network (CNN) that can automatically extract features from images and accurately classify them into predefined categories.

## Neural Network Model
A Neural Network Model is a computational model inspired by the human brain, used in machine learning and deep learning to recognize patterns, learn from data, and make predictions.
<img width="998" height="698" alt="image" src="https://github.com/user-attachments/assets/a757df16-cd3e-4a0a-99c3-8ac7e249f2af" />

## DESIGN STEPS 
1. Load and Preprocess Data
2. Get the shape of the first image in the training dataset
3. Get the shape of the first image in the test dataset
4. Train the Model
5. Test the Model
6. Predict on a Single Image
7. Display the image


## PROGRAM

### Name: NITHEESH YEGAVINTI 

### Register Number: 212224040370

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name:Y.NITHEESH ')
        print('Register Number: 212224040370')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

### OUTPUT

## Training Loss per Epoch

<img width="467" height="282" alt="image" src="https://github.com/user-attachments/assets/8ee04db3-fba3-4d8e-9595-7e09e47f7ced" />

## Confusion Matrix

<img width="802" height="732" alt="image" src="https://github.com/user-attachments/assets/7a6a329a-5803-46bc-a75b-c8cfd4922a9c" />


## Classification Report

<img width="570" height="340" alt="image" src="https://github.com/user-attachments/assets/f0f74cbb-d54f-476d-a585-1cbc142149bc" />


### New Sample Data Prediction

<img width="521" height="527" alt="image" src="https://github.com/user-attachments/assets/6907b95e-743f-46af-9c1b-6dfc117e4a81" />


## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
