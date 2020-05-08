import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

class Dataset(torch.utils.data.Dataset): #Defining the dataset

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file) #Load the data table

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # We prefer working with lists rather than tensors at this stage, so if we find a tensor - turn it into a list
        feature = self.df.iloc[idx,:-1]  # Our features are all the columns except the last one: "default" - whether the client payed all their payements on time
        label = self.df.iloc[idx, -1]  # Our label is the last column: "default" - whether the client payed all their payements on time. 0 is yes, 1 is no.
        sample = {'features': torch.tensor(feature), 'label': torch.tensor(label)}  # We represent our data as 'samples' which are two seperate tensors - one for features and one for the labels
        return sample


class Net(nn.Module):

    def __init__(self, DROPOUT_RATE, DEBUG=False):  # Definitions for our fully-connected network

        super(Net, self).__init__()
        self.DEBUG = DEBUG

        self.INPUT_NEURONS = 21  # The input neurons are in the amount of our features
        self.NEURONS_PER_LAYER = 200  # We decided to use 200 output neurons as they did in Polaris

        self.input_layer = nn.Linear(self.INPUT_NEURONS, self.NEURONS_PER_LAYER)  # the first layer is linear
        self.hidden_layer = nn.Linear(self.NEURONS_PER_LAYER, self.NEURONS_PER_LAYER)  # the middle layers are linear
        self.output_layer = nn.Linear(self.NEURONS_PER_LAYER, 1)  # the last layer in linear

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # Defining our forward path

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


def compute_loss(net, dataloader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = 0
    count = 0
    count0 = 0
    count1 = 0
    sens = 0
    spec = 0
    critertion = nn.BCELoss()
    with torch.no_grad():  # Initializing the gradients to zero
        for data in dataloader:  # Iterate over the test samples

            features = data['features'].to(device)
            labels = data['label'].view(-1, 1).to(device)

            outputs = net(features)  # Fetching the network's predictions
            loss += critertion(outputs, labels)
            labeled0 = labels==0
            labeled1 = labels==1
            outputs0 = outputs==0
            outputs1 = outputs==1
            count += len(labels)
<<<<<<< HEAD
            count0 += sum(labeled0)
            count1 += sum(labeled1)
            tmp1 = outputs1*labeled1
            tmp0 = labeled0*outputs0
            sens += sum(tmp1)
            spec += sum(tmp0)

    print('There were', count0, 'zeros and', count1, 'ones')
    print('Network guessed', num_of_zeros, 'zeros and', num_of_ones, 'ones')

    a = (loss/count)
    b = (100*float(sens)/count1)
    c = (100*float(spec)/count0)

    return a, b, c
=======
            count0 += len(labeled0)
            count1 += len(labeled1)
            sens += sum(outputs1==labeled1)
            spec += sum(labeled0==outputs0)

    return (loss/count) , (sens/count1) , (spec/count0)
>>>>>>> parent of 281f0e3... Fixed spec sens calculation bug


def main(PREP_PATH):

    # Hyper parameters
    BATCH_SIZE = 500
    LEARNING_RATE = 0.001  # The optimal learning rate for the Adam optimizer
<<<<<<< HEAD
    EPOCH_COUNT = 5
=======
    EPOCH_COUNT = 1
>>>>>>> parent of 281f0e3... Fixed spec sens calculation bug
    DROPOUT_RATE = 0.1

    # Initialize train and validation datasets and loaders
    train_dataset = Dataset(PREP_PATH + 'train.txt')  # Sending to the class "dataset" only the samples that we chose to be the training samples (60% of the data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)  # Calling the dataloader which will iterate over the training data and arrange it in batches

    # GADI Change here the path if you want to use the full original validation set
    val_dataset = Dataset(PREP_PATH + 'val.txt')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Instantiate the net
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net(DROPOUT_RATE, DEBUG=False).to(device)

    # Define loss
    critertion = nn.BCELoss()  # We shall use Binary Cross-Entropy Loss
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)  # We shall use the optimizer Adam

    # Train the network
    train_loss = []
    val_loss = []
    for epoch in range(EPOCH_COUNT):
        running_loss = 0.0  # Defining a running loss that is initialized to 0 but will contain at the end of the batch the total loss of all the batch's samples

        for i, data in enumerate(train_dataloader, 0):

            features = data['features'].to(device)
            labels = data['label'].to(device)

            optimizer.zero_grad()  # Initializing the gradients to zero

            outputs = net(features)  # Fetching the predictions from the FC network

            labels = labels.view(-1, 1)  # Fetching the labels
            loss = critertion(outputs, labels)  # Using the Binary Cross-Entropy Loss

            loss.backward()  # Starting the backwards path
            optimizer.step()  # Calling the optimizer

            running_loss += loss.item()  # Summing the loss of each of this batch's samples into the total running loss

            PRINT_FREQ = 25
            if i % PRINT_FREQ == (PRINT_FREQ - 1):  # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / PRINT_FREQ))
                running_loss = 0.0

        # At the end of each epoch, remember the current train and validation loss
<<<<<<< HEAD
        print('Computing total training loss...')
        new_loss, _, _ = compute_loss(net, train_dataloader)
        train_loss.append(new_loss)
        print('Current training losses are', train_loss)
        print('Computing total validation loss, sensitivity and specificity...')
        new_loss, new_sens, new_spec = compute_loss(net, val_dataloader)
        val_loss.append(new_loss)
        val_sens.append(new_sens)
        val_spec.append(new_spec)
        print('Current validation losses are', val_loss)
        print('Current validation sensitivities are', val_sens)
        print('Current validation specificities are', val_spec)
=======
        print('Computing total training loss, sensitivity and specificity...')
        train_loss , train_sens , train_spec = compute_loss(net, train_dataloader)
        #train_loss.append(train_loss)
        print('Computing total validation loss, sensitivity and specificity...')
        val_loss , val_sens , val_spec = compute_loss(net, val_dataloader)
        #val_loss.append(val_sens)
>>>>>>> parent of 281f0e3... Fixed spec sens calculation bug

        # GADI Please note this is the output model name and it is saved after every epoch. Be careful not to override files :)
        torch.save(net.state_dict(), 'models/split_33_66_batchsize_' + str(BATCH_SIZE) + '_lr_' + str(LEARNING_RATE) + '_dropout_'+ str(DROPOUT_RATE) + '_epoch_' + str(epoch) +'.pkl')

    print('Final training losses are', train_loss)
    print('Final validation losses are', val_loss)
<<<<<<< HEAD
    print('Final validation sensitivities are', val_sens)
    print('Final validation specificities are', val_spec)
=======
    print('Final validation sesitivity is', val_sens)
    print('Final validation specificity is', val_spec)
>>>>>>> parent of 281f0e3... Fixed spec sens calculation bug

    # GADI Uncomment this if you want to test the test set too
    """
    # Test the network
    # GADI Change here the path if you want to use the full original test set
    test_dataset = Dataset(PREP_PATH + 'test.txt')  # Sending to the class "dataset" only the samples that we chose to be the test samples (20% of the data)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)  # Calling the dataloader which will iterate over the test data and arrange it in batches

    labels_zeros = 0
    labels_ones = 0
    outputs_zeros = 0
    outputs_ones = 0

    correct, total = 0, 0
    with torch.no_grad():  # Initializing the gradients to zero
        for data in test_dataloader:  # Iterate over the test samples

            features = data['features'].to(device)
            labels = data['label'].view(-1, 1).to(device)

            labels_zeros += int(sum(labels == 0))
            labels_ones += int(sum(labels == 1))

            outputs = net(features)  # Fetching the network's predictions
            predicted = outputs

            threshold = 0.5  # Defining a threshold so results above it will be considered as 1 and below it - 0
            predicted[predicted >= threshold] = 1
            predicted[predicted < threshold] = 0

            outputs_zeros += int((predicted == 0).sum())
            outputs_ones += int((predicted == 1).sum())

            x = (predicted == labels)
            correct += int(sum(predicted == labels))  # Testing how much of our predictions were same as the label
            total += int(len(predicted))

    print('Test accuracy:', (100 * correct / total), '%')
    print('There were', labels_zeros, 'zero labels and ', labels_ones, 'one labels')
    print('The network predicted', outputs_zeros, 'zeros and', outputs_ones, 'ones')"""


if __name__ == '__main__':
    main('Dataset/prep_biased_33_66/')


