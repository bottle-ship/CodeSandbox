import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

dtype = torch.float
device = torch.device("cuda")


class BostonHousingDataset(Dataset):

    def __init__(self, n_noise_feature: int = 13):
        self.n_noise_feature = n_noise_feature

        df = pd.read_csv("../housing.csv", sep="\\s+", header=None)
        df.columns = [
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
        ]
        df = df.astype(float)

        x_scaler = StandardScaler()
        x = df.drop(columns=["MEDV"]).values
        x = x_scaler.fit_transform(x)
        self.n_original_featrues = x.shape[1]
        x_noise = np.random.randn(len(x), n_noise_feature)
        self.x = np.concatenate([x, x_noise], axis=-1)

        self.y = df.reindex(columns=["MEDV"]).values

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index, ...], self.y[index, ...]


class Model(nn.Module):

    def __init__(self, n_inputs: int, n_outputs: int, n_hidden_nodes: int = 10):
        super(Model, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_nodes = n_hidden_nodes

        self.model = nn.Sequential(
            nn.Linear(self.n_inputs, self.n_hidden_nodes, bias=True),
            nn.Tanh(),
            nn.Linear(self.n_hidden_nodes, self.n_hidden_nodes, bias=True),
            nn.Tanh(),
            nn.Linear(self.n_hidden_nodes, self.n_hidden_nodes, bias=True),
            nn.Tanh(),
            nn.Linear(self.n_hidden_nodes, self.n_outputs, bias=True)
        )

    def forward(self, x):
        return self.model(x)

    def reset(self):
        for i in range(0, len(self.model)):
            if hasattr(self.model[i], "reset_parameters"):
                self.model[i].reset_parameters()

    def get_weights(self, gamma: float = 2):
        return torch.norm(self.model[0].weight, dim=0).pow(gamma).data

    def proximal(self, lam: float, eta: float):
        alpha = torch.clamp(torch.norm(self.model[0].weight, dim=0) - lam * eta, min=0)
        v = torch.nn.functional.normalize(self.model[0].weight, dim=0) * alpha
        self.model[0].weight.data = v


def main():
    N, D_in, H, D_out = 506, 26, 10, 1
    lam_factors = [0.01, 0.05, 0.1, 0.5, 1, 2]
    xi_factors = [0.01, 0.05, 0.1, 0.5, 1, 2]

    epoch = 1000
    eta = 1e-2

    tr_errors = np.zeros(len(lam_factors))
    tst_errors = np.zeros(len(lam_factors))
    res = np.zeros(4)

    dataset = BostonHousingDataset(n_noise_feature=13)

    x = torch.from_numpy(dataset.x).to(dtype=dtype, device=device)
    y = torch.from_numpy(dataset.y).to(dtype=dtype, device=device)

    model = Model(n_inputs=D_in, n_outputs=D_out, n_hidden_nodes=H)
    model.to(dtype=dtype, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    criterion = nn.MSELoss()

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.25)

    for r in np.arange(0, len(lam_factors)):
        model.reset()
        model.train()

        for t in range(epoch):
            y_train_pred = model(x_train)
            loss = criterion(y_train_pred, y_train)

            # if t % 1000 == 0:
            #     print(t, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.proximal(lam_factors[r], eta)

        model.eval()
        with torch.no_grad():
            y_test_pred = model(x_test)
        tst_errors[r] += (y_test_pred - y_test).pow(2).mean()

    lam_final = lam_factors[np.argmin(tst_errors)]
    print(tst_errors, "(loss)")
    print("Optimization now.")

    model.reset()
    model.train()

    for t in range(epoch):

        y_pred = model(x)
        loss = criterion(y_pred, y)

        if t % 1000 == 0:
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.proximal(lam_final, eta)

    # Adaptive
    weights = model.get_weights(gamma=2)
    norm = 1 / weights
    I = np.arange(0, D_in)
    a = torch.norm(model.model[0].weight, dim=0)

    I = np.arange(0, 13)
    J = np.arange(13, 26)
    a = torch.norm(model.model[0].weight, dim=0)

    data1 = a.detach().cpu().numpy()

    res[0] = sum(a[I] > 0)
    res[1] = sum(a[J] > 0)

    temp1 = torch.norm(model.model[0].weight, dim=0)
    print("Sparsity original", temp1[0:13])
    print("Sparsity random Gaussian", temp1[13:26])

    tr_errors = np.zeros(len(xi_factors))
    tst_errors = np.zeros(len(xi_factors))

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.25)

    for r in np.arange(0, len(xi_factors)):
        model.reset()
        model.train()

        for t in range(epoch):
            y_train_pred = model.forward(x_train)
            loss = criterion(y_train_pred, y_train)

            # if t % 1000 == 0:
            #     print(t, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.proximal(xi_factors[r] * norm, eta)

        model.eval()
        with torch.no_grad():
            y_test_pred = model.forward(x_test)
        tst_errors[r] += (y_test_pred - y_test).pow(2).mean()

    print(tst_errors[r], "(Test loss)")

    xi_final = lam_factors[np.argmin(tst_errors)]
    print(tst_errors, "(loss)")
    print("Optimization now.")

    model.reset()
    model.train()

    for t in range(epoch):

        y_pred = model.forward(x)
        loss = criterion(y_pred, y)

        if t % 1000 == 0:
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.proximal(xi_final * norm, eta)

    I = np.arange(0, 13)
    J = np.arange(13, 26)
    a = torch.norm(model.model[0].weight, dim=0)

    res[2] = sum(a[I] > 0)
    res[3] = sum(a[J] > 0)

    data2 = a.detach().cpu().numpy()

    temp2 = torch.norm(model.model[0].weight, dim=0)
    print("Sparsity original", temp2[0:13])
    print("Sparsity random Gaussian", temp2[13:26])

    plt.figure()
    plt.bar(np.arange(0, 26) - 0.1, temp1.detach().cpu().numpy())
    plt.bar(np.arange(0, 26) + 0.1, temp2.detach().cpu().numpy())
    plt.show()

    # np.savetxt('housing.csv', res, delimiter=",")
    # np.savetxt('housing-data1.csv', data1, delimiter=",")
    # np.savetxt('housing-data2.csv', data2, delimiter=",")


if __name__ == "__main__":
    main()
