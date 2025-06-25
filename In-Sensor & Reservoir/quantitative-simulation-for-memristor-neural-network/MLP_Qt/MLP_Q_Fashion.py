import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

BATCH_SIZE = 64
isQuant = True
Data = np.loadtxt('ExpData/DeviceData_31.txt') 
# MAKE DATASET
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5),
                                transforms.Resize([14, 14])])

data_train = datasets.FashionMNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = True)

data_test = datasets.FashionMNIST(root="./data/",
                           transform = transform,
                           train = False,
                           download = True)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = BATCH_SIZE,
                                                shuffle = True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size = BATCH_SIZE,
                                               shuffle = True)

# NETWORK MODEL
class Model(torch.nn.Module):
    
    def __init__(self, in_features, hidden_features, out_features):
        super(Model, self).__init__()
        self.in_features = in_features
        self.Input2Hiden = nn.Linear(in_features, hidden_features)
        self.Hiden2Output = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.Input2Hiden(x)
        x = nn.ReLU()(x)
        x = self.Hiden2Output(x)
        return x

def QuantW(W, D):
    W = W.detach().numpy()
    upper = np.max(W)
    lower = np.min(W)
    D = D/(np.max(D)-np.min(D))*(upper-lower)
    W_q = W+0
    for i in range(len(W[:, 0])):
        for j in range(len(W[0, :])):
            ind = np.argmin(np.abs(W[i, j]-D))
            W_q[i, j] = D[ind]
    W_q = torch.from_numpy(W_q).float()
    return W_q

def QuantB(B, D):
    B = B.detach().numpy()
    upper = np.max(B)
    lower = np.min(B)
    D = D/(np.max(D)-np.min(D))*(upper-lower)
    B_q = B+0
    for i in range(len(B)):
        ind = np.argmin(np.abs(B[i]-D))
        B_q[i] = D[ind]
    B_q = torch.from_numpy(B_q).float()
    return B_q

# # TRAINING PROCESURE
model = Model(14*14, 200, 10)
print(model)

cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 50
ACC = []
for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-"*10)
    for data in data_loader_train:
        X_train, y_train = data
        outputs = model(X_train)
        _,pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    if isQuant:
        W1 = model.Input2Hiden.weight.data
        model.Input2Hiden.weight.data = QuantW(W1, Data)
        B1 = model.Input2Hiden.bias.data
        model.Input2Hiden.bias.data = QuantB(B1, Data)
        W2 = model.Hiden2Output.weight.data
        model.Hiden2Output.weight.data = QuantW(W2, Data)
        B2 = model.Hiden2Output.bias.data
        model.Hiden2Output.bias.data = QuantB(B2, Data)
        
    for data in data_loader_test:
        X_test, y_test = data
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss/len(data_train),
                                                                                      100*running_correct/len(data_train),
                                                                                      100*testing_correct/len(data_test)))
    ACC.append(testing_correct.cpu().data.numpy()/len(data_test))
torch.save(model.state_dict(), "./save/MLP_m_model_parameter.pkl")
plt.figure()
plt.plot(ACC)
plt.show()
acc = np.array(ACC)
# CONFUSION MATRIX
model.load_state_dict(torch.load('./save/MLP_m_model_parameter.pkl'))
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                          batch_size = 5000,
                                          shuffle = True)
X_test, y_test = next(iter(data_loader_test))
inputs = X_test
pred = model(inputs)
_,pred = torch.max(pred, 1)

final_pred = pred.cpu().data.numpy()
final_targ = y_test.data.numpy()

confusion_matrix = metrics.confusion_matrix(final_pred, final_targ)

LABELS = [
    '‌T-shirt',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix/(np.sum(confusion_matrix, axis=1, keepdims=1)), xticklabels=LABELS, yticklabels=LABELS, annot=True)
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
