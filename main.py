from auxiliary import *

m=model()
m=m.to(device)

training_loss_list=[]
validation_loss_list=[]

max_training_epochs=200
op=optim.Adam(m.parameters(),lr=1e-3)

for epoch in range(1,2+max_training_epochs):
    m.train()
    train_loss=0
    start=time.time()
    for i ,(x,y) in enumerate(train_loader):
        x=x.to(device)
        reconst_x,mu,logvar=m(x)
        op.zero_grad()
        loss=loss_function(reconst_x,x,mu,logvar)
        train_loss+=loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_value_(m.parameters(), 10)
        op.step()

    training_loss_list.append(train_loss)
    validation_loss_list.append(validation_loss(test_loader,m))

    if(epoch%10==1):
      path=saved_models + '/model_{}.pt'.format(epoch)
      torch.save({'epoch': epoch, 'model_state_dict':m.state_dict(), 'optim_state_dict':op.state_dict() },path)

    if(epoch%1==0):
        samples(m,epoch)
        show_losses(training_loss_list,validation_loss_list)
        print('Done epoch {} in {:0.2f} sec'.format(epoch,time.time()-start))
