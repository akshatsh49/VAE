from global_variables import *

class encoder(nn.Module):
  def __init__(self):
    super(encoder,self).__init__()
    self.fc1=nn.Linear(784,400)
    self.fc21=nn.Linear(400,20)
    self.fc22=nn.Linear(400,20)
  
  def forward(self,x):
    x=nn.ReLU()(self.fc1(x.view(batch_size,784)))
    mu=self.fc21(x)
    logvar=self.fc22(x)
    return mu,logvar
  
  def reparametrize(self,mu,logvar):
    sigma=torch.exp(0.5*logvar)
    eps=torch.randn_like(mu)
    return mu+eps*sigma

class decoder(nn.Module):
  def __init__(self):
    super(decoder,self).__init__()
    self.fc3=nn.Linear(20,400)
    self.fc4=nn.Linear(400,784)

  def forward(self,z):
    x=nn.ReLU()(self.fc3(z))
    x=nn.Sigmoid()(self.fc4(x))
    return x

class model(nn.Module):
  def __init__(self):
    super(model,self).__init__()
    self.encoder=encoder()
    self.decoder=decoder()

  def forward(self,x):
    mu,logvar=self.encoder(x)
    z=self.encoder.reparametrize(mu,logvar)
    reconst_x=self.decoder(z)
    return reconst_x,mu,logvar
  