import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class Dataset(torch.utils.data.Dataset): #Defining the dataset

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file) #Load the data table

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() #We prefer working with lists rather than tensors at this stage, so if we find a tensor - turn it into a list
        feature = self.df.iloc[idx,:-1] #Our features are all the columns except the last one: "default" - whether the client payed all their payements on time
        label = self.df.iloc[idx,-1] #Our label is the last column: "default" - whether the client payed all their payements on time. 0 is yes, 1 is no.
        sample = {'features': torch.tensor(feature), 'label': torch.tensor(label)} #We represent our data as 'samples' which are two seperate tensors - one for features and one for the labels
        return sample


class Net(nn.Module):

    def __init__(self, DROPOUT_RATE, DEBUG=False): #Definitions for our fully-connected network

        super(Net, self).__init__()
        self.DEBUG = DEBUG

        self.INPUT_NEURONS = 21 #The input neurons are in the amount of our features
        self.NEURONS_PER_LAYER = 200 #We decided to use 200 output neurons as they did in Polaris

        self.input_layer = nn.Linear(self.INPUT_NEURONS, self.NEURONS_PER_LAYER) #the first layer is linear
        self.hidden_layer = nn.Linear(self.NEURONS_PER_LAYER, self.NEURONS_PER_LAYER) #the middle layers are linear
        self.output_layer = nn.Linear(self.NEURONS_PER_LAYER, 1) #the last layer in linear

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): #Defining our forward path

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
    LEARNING_RATE = 0.001 #The optimal learning rate for the Adam optimizer
    EPOCH_COUNT = 30
    DROPOUT_RATE = 0.1

    # Initialize train dataset and loader
    train_dataset = Dataset(PREP_PATH + 'train.txt') #Sending to the class "dataset" only the samples that we chose to be the training samples (60% of the data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE) #Calling the dataloader which will iterate over the training data and arrange it in batches

    # Instantiate the net
    net = Net(DROPOUT_RATE, DEBUG=False)

    # Define loss
    critertion = nn.BCELoss() #We shall use Binary Cross-Entropy Loss
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE) #We shall use the optimizer Adam

    # Train the network
    for epoch in range(EPOCH_COUNT):
        running_loss = 0.0 #Defining a running loss that is initialized to 0 but will contain at the end of the batch the total loss of all the batch's samples

        for i, data in enumerate(train_dataloader, 0):

            features = data['features']
            labels = data['label']

            optimizer.zero_grad() #Initializing the gradients to zero

            outputs = net(features) #Fetching the predictions from the FC network

            labels = labels.view(BATCH_SIZE, 1) #Fetching the labels
            loss = critertion(outputs, labels) #Using the Binary Cross-Entropy Loss

            loss.backward() #Starting the backwards path
            optimizer.step()  #Calling the optimizer

            running_loss += loss.item() #Summing the loss of each of this batch's samples into the total running loss
            if i % 20 == 19:  # print every 20 mini-batches
                #print('[%d, %5d] loss: %.3f' %
                #      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    # Test the network
    test_dataset = Dataset(PREP_PATH + 'test.txt')  #Sending to the class "dataset" only the samples that we chose to be the test samples (20% of the data)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE) #Calling the dataloader which will iterate over the test data and arrange it in batches

    correct, total = 0, 0
    with torch.no_grad(): #Initializing the gradients to zero
        for data in test_dataloader: #Iterate over the test samples
            features = data['features']
            labels = data['label']
            outputs = net(features) #Fetching the network's predictions
            _, predicted = outputs.data
            threshold = 0.5 #Defining a threshold so results above it will be considered as 1 and below it - 0
            predicted[predicted >= threshold] = 1
            predicted[predicted < threshold] = 0
            correct += (predicted == labels).sum().item() #Testing how much of our predictions were same as the label

    print('Test accuracy: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    main('Dataset/prep/')
