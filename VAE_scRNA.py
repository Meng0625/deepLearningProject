import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.io.mmio import mmread as read
from sklearn import preprocessing
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#pre-operation
#torch.cuda.empty_cache()
#torch.set_default_tensor_type('torch.cuda.LongTensor') #Error: Long Tensor can't be default
matrix=read("./matrix.mtx").toarray()   # """Row: Cell , Col: Gene"""
genes, cells=matrix.shape

#preprocessing
# gene_flitered=[]
# threadhold=matrix.mean()*cells
# for i in range(genes):
#     if matrix[i].sum()>=threadhold:
#         #gene_flitered.append(preprocessing.scale(matrix[i]))
#         gene_flitered.append(matrix[i])


#take PCA instead
decomposer = PCA(n_components=0.99999)
decomposer.fit(matrix)
gene_flitered=decomposer.components_.transpose()

cells, fliteredGenes =gene_flitered.shape
data=Variable(torch.tensor(gene_flitered,dtype=torch.float).transpose(0,1).reshape(cells,1,1,fliteredGenes))



codeSize=100
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        #Encoder
        self.fc1 = nn.Linear(fliteredGenes, 1000)
        self.fc2 = nn.Linear(1000, 400)
        self.fc21 = nn.Linear(400, codeSize)
        self.fc22 = nn.Linear(400, codeSize)
        #Decoder
        self.fc3 = nn.Linear(codeSize, 500)
        self.fc4 = nn.Linear(500, fliteredGenes)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.relu(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        code = self.reparametrize(mu, logvar)
        return code, self.decode(code), mu, logvar


model = VAE()


if torch.cuda.is_available():
    model.cuda()

reconstruction_function = nn.MSELoss(size_average=True)


def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD



optimizer = optim.Adam(model.parameters(), lr=1e-4)

if torch.cuda.is_available():
    data = data.cuda()

num_epochs = 500 #Total rounds for train

for epoch in range(num_epochs):
    #Use the whole dataset for each train here due to the small size of the dataset
    model.train()
    data = Variable(data)
    train_loss = 0
    optimizer.zero_grad()
    code, recon_batch, mu, logvar = model(data)
    #print(code)
    loss = loss_function(recon_batch, data, mu, logvar)
    loss.backward()
    train_loss += loss.data.item()
    optimizer.step()
    if epoch%100==0:
        print("The epoch is {} and the loss is {}".format(epoch,loss))


tsneCode = TSNE(n_components=2,learning_rate=100).fit_transform(code.cpu().detach().numpy().reshape(cells,codeSize))
estimator=cluster.KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=5, algorithm='auto')
estimator.fit(tsneCode)
label_pred = estimator.labels_
centroids = estimator.cluster_centers_
inertia = estimator.inertia_
plt.figure()
plt.scatter(tsneCode[:,0],tsneCode[:,1],c=label_pred, s=3)
plt.show()
