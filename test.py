from auxiliary import *

batch,_=next(iter(test_loader))
batch=batch.to(device)

m_=model()
m_=m_.to(device)

#samples
for file in sorted(os.listdir(saved_models)):
  if(os.path.splitext(file)[-1]=='.pt'):
    checkpoint=torch.load(saved_models + '/' + file)
    m_.load_state_dict(checkpoint['model_state_dict'])
    samples(m_,checkpoint['epoch'],path=test_folder)

#reconstructions
show(batch,title='Data',path=reconst_folder)
for file in sorted(os.listdir(saved_models)):
  if(os.path.splitext(file)[-1]=='.pt'):
    checkpoint=torch.load(saved_models + '/' + file)
    m_.load_state_dict(checkpoint['model_state_dict'])
    _,recon=reconst(m_,batch)
    show(recon,title='Reconstructions_{}'.format(checkpoint['epoch']),path=reconst_folder)

#space_interpolations
file=os.listdir(saved_models)[-1]
checkpoint=torch.load(saved_models + '/' + file)
m_.load_state_dict(checkpoint['model_state_dict'])
latent_space_interpolation(m_,title=checkpoint['epoch'],path=space_interpolations)