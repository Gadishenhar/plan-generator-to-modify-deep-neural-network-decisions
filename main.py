import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

#Hyper parameters
BATCH_SIZE = 500
LEARNING_RATE = 0.0001
EPOCH_COUNT = 30
DROPOUT_RATE = 0.1

class Dataset(torch.utils.data.Dataset):
    def __init__(self,csv_file):

        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature = self.df.iloc[idx,:-1]
        label = self.df.iloc[idx,-1]
        sample = {'feature': torch.tensor(feature), 'label': torch.tensor(label)}

        return sample

# Define train , validation and test sets
df = pd.read_csv('Dataset/prep.txt')
DF_LEN = len(df)
TRAIN_LEN = round(DF_LEN * 0.6)
VAL_LEN = round(DF_LEN * 0.2)
TEST_LEN = DF_LEN - TRAIN_LEN - VAL_LEN

train = df.iloc[1:TRAIN_LEN, :]
train.to_csv('Dataset/train.txt', index=False)
val = df.iloc[(1+TRAIN_LEN):(TRAIN_LEN+VAL_LEN), :]
val.to_csv('Dataset/val.txt', index=False)
test = df.iloc[-(TEST_LEN-1):, :]
test.to_csv('Dataset/test.txt', index=False)

train_dataset = Dataset('Dataset/train.txt')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)






class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.INPUT_NEURONS = 21
        self.NEURONS_PER_LAYER = 200
        self.input_layer = nn.Linear(self.INPUT_NEURONS, self.NEURONS_PER_LAYER)
        self.hidden_layer = nn.Linear(self.NEURONS_PER_LAYER, self.NEURONS_PER_LAYER)
        self.output_layer = nn.Linear(self.NEURONS_PER_LAYER, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.input_layer(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.hidden_layer(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.hidden_layer(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.hidden_layer(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.output_layer(x)
        x = self.softmax(x)

        return x

net = Net()

# Define loss
critertion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Train the network
for epoch in range(EPOCH_COUNT):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs = data['feature']
        labels = data['label']
        optimizer.zero_grad()
        outputs = net(inputs)
        labels = labels.view(500, 1)
        loss = critertion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 19:  # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

# Test the net
correct = 0
total = 0
test_dataset = Dataset('Dataset/test.txt')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
with torch.no_grad():
    for data in test_dataloader:
        features, labels = data
        outputs = net(features)
        _, predicted = outputs.data
        predicted[predicted >= 0.5] = 1
        predicted[predicted < 0.5] = 0
        correct += (predicted == labels).sum().item()


print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

