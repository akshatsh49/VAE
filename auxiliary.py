from model import *

def loss_function(reconst_x,x,mu,logvar):
  loss1=nn.BCELoss(reduction='sum')(reconst_x.view(batch_size,784),x.view(batch_size,784))
  loss2=-0.5*torch.sum(1+ logvar -mu.pow(2)-torch.exp(logvar))
  return loss1+loss2

def validation_loss(test_loader,m):
  m.eval()
  loss=0
  with torch.no_grad():
    for x,y in test_loader:
      x=x.to(device)
      reconst_x,mu,logvar=m.forward(x)
      loss+=loss_function(reconst_x,x,mu,logvar).item()
  
  m.train()
  return loss/len(test_loader)

def show(tensor,num_images=batch_size,title='',path=sample_folder):
  img=torchvision.utils.make_grid(tensor.view(-1,1,28,28),nrow=8)
  npimg=img.cpu().detach().numpy()
  plt.imshow(np.transpose(npimg,(1,2,0)))
  plt.title('Epoch_{}'.format(title))
  plt.axis('off')
  plt.savefig(path+'/{}.png'.format(title))

def samples(m,epoch,path=sample_folder):
  m.eval()
  z=torch.randn(64,20,device=device)
  show(m.decoder(z),title=epoch,path=path)
  m.train()

def reconst(m,batch):
  m.eval()
  with torch.no_grad():
    recon,_,_=m(batch)
  m.train()
  return batch,recon

def latent_space_interpolation(m_,title='',path=space_interpolations):
    z1=torch.randn(1,20).to(device)
    z2=torch.randn(1,20).to(device)
    steps=10
    l=[]
    for i in range(steps+1):
        l.append(z1+i/steps*(z2-z1))

    tensor=torch.stack(l)
    tensor=tensor.to(device)
    show(m_.decoder(tensor),title=title,num_images=steps+1,path=path)

def show_losses(training_loss,validation_loss):
    plt.close()
    plt.plot( range(1,1+len(training_loss)) , training_loss , label='Training_loss')
    plt.title('Training_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(loss + '/Training_loss.png')

    plt.close()
    plt.plot( range(1,1+len(validation_loss)) , validation_loss , label='Validation_loss')
    plt.title('Validation_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(loss + '/Validation_loss.png')
