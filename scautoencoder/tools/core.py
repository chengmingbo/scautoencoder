import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler



class Autoencoder(nn.Module):
    """Makes the main denoising auto

    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(in_shape, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, enc_shape),
        )

        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, in_shape)

        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


def autoencoder(adata, epochs=500, learning_rate=1e-3,  is_copy=False):
    device ='cuda' if torch.cuda.is_available() else 'cpu'

    adata = adata.copy() if is_copy else adata


    ## normalize data
    X = MinMaxScaler().fit_transform(adata.X.toarray())
    X = torch.from_numpy(X).to(device)


    ## Module creation
    encoder = Autoencoder(in_shape=adata.n_vars, enc_shape=32).float().to(device)
    error = nn.L1Loss()
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    ## Training
    encoder.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        output = encoder(X)
        loss = error(output, X)
        loss.backward()
        optimizer.step()

        if epoch % int(0.1*epochs) == 0:
            print(f'epoch {epoch} \t Loss: {loss.item():.4g}')



    with torch.inference_mode():
        encoded = encoder.encode(X)
        decoded = encoder.decode(encoded)
        mse = error(decoded, X).item()
        enc = encoded.cpu().detach().numpy()
        dec = decoded.cpu().detach().numpy()


    print(enc)
    print(type(enc))
    adata.obsm['X_encoded'] = np.array(enc)

    return adata if is_copy else None


