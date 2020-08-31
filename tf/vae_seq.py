'''
code taken from:
http://pyro.ai/examples/vae.html
'''
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import vae

ALPH_LENGTH = 5
EMBEDDING_DIM = 3
BATH_SIZE = 10

embedding_layer = nn.Embedding(ALPH_LENGTH, EMBEDDING_DIM)


class Encoder(nn.Module):
    def __init__(self, in_dim, z_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.in_dim = in_dim

        self.embedding_layer = embedding_layer
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, self.in_dim)

        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


class VAE(vae.VAE):
    def __init__(self, img_size=784, z_dim=50, hidden_dim=400, *args):
        vae.VAE.__init__(self, img_size=img_size, z_dim=z_dim,
                         hidden_dim=hidden_dim, *args)
        self.encoder = Encoder(img_size, z_dim, hidden_dim)


def get_sequences(filename="test_sequence_Coloeus_monedula.txt"):
    alph = ['A', 'C', 'G', 'T', 'N', 'R', 'Y']

    def tokenizer(word):
        # 0 reserved for empty:
        return(alph.index(word)+1)

    with open(filename) as f:
        data = []
        length = 0
        count = 0
        for line in f:
            entry = list(map(tokenizer, line[:-1]))
            entry_length = len(entry)
            
            if entry_length > length:
                length = entry_length
                # for padding:
                count += 1

            data.append(entry)

    if count > 1:
        # padding:
        for entry in data:
            entry_length = len(entry)
            if entry_length < length:
                entry.extend([0 for i in range(length-entry_length)])
    # batching:
    data_bathed = [data[bath_idx*BATH_SIZE: (bath_idx+1)*BATH_SIZE]
                   for bath_idx in range(len(data)//BATH_SIZE)]
    if len(data_bathed[-1]) < BATH_SIZE:
        data_bathed = data_bathed[:-1]

    return(torch.Tensor(data_bathed))


def test():
    print("ALPH_LENGTH: ", ALPH_LENGTH)
    print("EMBEDDING_DIM: ", EMBEDDING_DIM)
    print("BATCH_SIZE: ", BATH_SIZE)

    data = get_sequences()
    x = data[0]
    img_size = data.shape[-1]

    # x is batch
    print("x:")
    print(x)
    encoder = Encoder(x.shape[-1], 3, 10)
    # encoder = Encoder(data.shape[-1], 3, 10)
    z_loc, z_scale = encoder.forward(x)
    print("z_loc, z_scale:")
    print(z_loc)
    print(z_scale)

    decoder = vae.Decoder(x.shape[-1], 3, 10)
    # decoder = vae.Decoder(data.shape[-1], 3, 10)
    z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
    loc_img = decoder.forward(z)
    print("loc_img:")
    print(loc_img)
    
    obs = pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1),
                      obs=x.reshape(-1, img_size))
    print("obs (same as x because of sampling restriction):")
    print(obs)

    ovae = VAE(img_size=img_size, z_dim=3, hidden_dim=10)
    print("vae.model(x):")
    # x is batch
    ovae.model(x)
    print("done")
    print("vae.guide(x)")
    ovae.guide(x)
    print("done")
