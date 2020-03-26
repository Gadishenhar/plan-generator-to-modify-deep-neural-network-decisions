import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# TODO Add column names to data frame
# Load data set
df = pd.read_table('./tmp_dataset.txt', '|')
# Clean up NaNs and nulls
#df.dropna(inplace=True)
# Convert labels Y->1, N->0
df = (df.iloc[:, 24]).replace('Y', 1)
df = (df.iloc[:, 24]).replace('N', 0)

# Define train , validation and test sets
DF_LEN = len(df)
TRAIN_LEN = round(DF_LEN * 0.6)
VAL_LEN = round(DF_LEN * 0.2)
TEST_LEN = DF_LEN - TRAIN_LEN - VAL_LEN
x_train = df.iloc[:TRAIN_LEN, :23]
y_train = df.iloc[:TRAIN_LEN:, 24]
x_val = df.iloc[TRAIN_LEN:(TRAIN_LEN+VAL_LEN), :23]
y_val = df.iloc[TRAIN_LEN:(TRAIN_LEN+VAL_LEN), 24]
x_test = df.iloc[-TEST_LEN:, :23]
y_test = df.iloc[-TEST_LEN:, 24]

#Hyper parameters
BATCH_SIZE = 500
LEARNING_RATE = 0.0001
EPOCH_COUNT = 30
DROPOUT_RATE = 0.1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.INPUT_NEURONS = 24
        self.NEURONS_PER_LAYER = 200
        self.input_layer = nn.Linear(self.INPUT_NEURONS, self.NEURONS_PER_LAYER)
        self.hidden_layer = nn.Linear(self.NEURONS_PER_LAYER, self.NEURONS_PER_LAYER)
        self.output_layer = nn.Linear(self.NEURONS_PER_LAYER, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.softmax = nn.Softmax()

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
critertion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Train the network
# TODo fIX THIS
train_loader = torch.utils.data.DataLoader(df, batch_size=BATCH_SIZE, shuffle=True)
for epoch in range(EPOCH_COUNT):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = critertion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 19:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

# Test the net
correct = 0
total = 0
# TODO Fix this
test_loader = torch.utils.data.DataLoader(df, batch_size=BATCH_SIZE, shuffle=True)
with torch.no_grad()
    for data in test_loader:
        features, labels = data
        outputs = net(features)
        _, predicted = outputs.data
        predicted[predicted >= 0.5] = 1
        predicted[predicted < 0.5] = 0
        correct += (predicted == labels).sum().item()


print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

