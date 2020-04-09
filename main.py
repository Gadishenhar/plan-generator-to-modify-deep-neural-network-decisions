import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class Dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        feature = self.df.iloc[idx,:-1]
        label = self.df.iloc[idx,-1]
        sample = {'features': torch.tensor(feature), 'label': torch.tensor(label)}
        return sample


class Net(nn.Module):

    def __init__(self, DROPOUT_RATE, DEBUG=False):

        super(Net, self).__init__()
        self.DEBUG = DEBUG

        self.INPUT_NEURONS = 21
        self.NEURONS_PER_LAYER = 200

        self.input_layer = nn.Linear(self.INPUT_NEURONS, self.NEURONS_PER_LAYER)
        self.hidden_layer = nn.Linear(self.NEURONS_PER_LAYER, self.NEURONS_PER_LAYER)
        self.output_layer = nn.Linear(self.NEURONS_PER_LAYER, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        if self.DEBUG:
            print('Input features:')
            print(x)

        x = self.input_layer(x)
        x = self.dropout(x)
        x = self.relu(x)

        if self.DEBUG:
            print('After input layer:')
            print(x)

        x = self.hidden_layer(x)
        x = self.dropout(x)
        x = self.relu(x)

        if self.DEBUG:
            print('After second layer:')
            print(x)

        x = self.hidden_layer(x)
        x = self.dropout(x)
        x = self.relu(x)

        if self.DEBUG:
            print('After third layer:')
            print(x)

        x = self.hidden_layer(x)
        x = self.dropout(x)
        x = self.relu(x)

        if self.DEBUG:
            print('After dourth layer:')
            print(x)

        x = self.output_layer(x)

        if self.DEBUG:
            print('After output layer:')
            print(x)

        x = self.sigmoid(x)

        if self.DEBUG:
            print('After sigmoid layer:')
            print(x)

        return x


def main(PREP_PATH):

    # Hyper parameters
    BATCH_SIZE = 500
    LEARNING_RATE = 0.0001
    EPOCH_COUNT = 30
    DROPOUT_RATE = 0.1

    # Initialize train dataset and loader
    train_dataset = Dataset(PREP_PATH + 'train.txt')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

    # Instantiate the net
    net = Net(DROPOUT_RATE, DEBUG=False)

    # Define loss
    critertion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Train the network
    for epoch in range(EPOCH_COUNT):
        running_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):

            features = data['features']
            labels = data['label']

            optimizer.zero_grad()

            outputs = net(features)

            labels = labels.view(BATCH_SIZE, 1)
            loss = critertion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:  # print every 20 mini-batches
                #print('[%d, %5d] loss: %.3f' %
                #      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    # Test the network
    test_dataset = Dataset(PREP_PATH + 'test.txt')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    correct, total = 0, 0
    with torch.no_grad():
        for data in test_dataloader:
            features = data['features']
            labels = data['label']
            outputs = net(features)
            _, predicted = outputs.data
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            correct += (predicted == labels).sum().item()

    print('Test accuracy: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    main('Dataset/prep/')
