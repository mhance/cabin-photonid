import torch

from cabin.LossFunctions import lossvars,loss_fn,effic_loss_fn,smooth_loss_fn


class photonlossvars(lossvars):
    def __init__(self,other=None):
        self.ptloss = 0
        self.muloss = 0
        if other is not None:
            self.copy(other)
         
    def copy(self,other):
        super().copy(other)
        
        if hasattr(other, 'ptloss'):
            self.ptloss = other.ptloss

        if hasattr(other, 'muloss'):
            self.muloss = other.muloss
        
    def totalloss(self):
        return super().totalloss()+self.ptloss+self.muloss

    def scale(self,scale):
        super().scale(scale)
        self.ptloss = scale * self.ptloss
        self.muloss = scale * self.muloss

    def __add__(self,other):
        third = photonlossvars(super().__add__(other))
        
        if hasattr(other, 'ptloss'):
            third.ptloss = self.ptloss + other.ptloss
        else:
            third.ptloss = self.ptloss

        if hasattr(other, 'muloss'):
            third.muloss = self.muloss + other.muloss
        else:
            third.muloss = self.muloss
            
        return third
        

    

def full_loss_fn(y_pred, y_true, features, net,
                 alpha=1., beta=1., gamma=0.001, delta=0., 
                 eps_ef=0.001, eps_pt=0.001, eps_mu=0.001,
                 debug=False):

    loss=None    
    for i in range(len(net.pt)):
        for j in range(len(net.mu)):
            efl=effic_loss_fn(y_pred[i][j], 
                              y_true[i][j], 
                              features, 
                              net.nets[i][j],
                              alpha, beta, gamma, delta, eps_ef, debug)
            if loss==None:
                loss = photonlossvars(efl)
            else:
                loss = loss + efl

    # something like this would make sense.  but it really screws things up.
    # loss.scale(1./(len(net.pt)*len(net.mu)))
    
    # now smooth over pT, for each efficiency/mu bin:
    if len(net.pt)>=3:
        for k in range(len(net.effics)):
            for j in range(len(net.mu)):
                cuts=[net.nets[i][j][k].get_cuts() for i in range(len(net.pt))]
                l=smooth_loss_fn([net.pt[z][0] for z in range(len(net.pt))],cuts)
                if loss.ptloss == 0:
                    loss.ptloss = l
                else:
                    loss.ptloss = loss.ptloss + l
        loss.ptloss = loss.ptloss*eps_pt/(len(net.effics)*len(net.mu))

    # now smooth over mu, for each efficiency/pt bin:
    if len(net.mu)>=3:
        for k in range(len(net.effics)):
            for i in range(len(net.pt)):
                cuts=[net.nets[i][j][k].get_cuts() for j in range(len(net.mu))]
                l=smooth_loss_fn([net.mu[z][0] for z in range(len(net.mu))],cuts)
                if loss.muloss == 0:
                    loss.muloss = l
                else:
                    loss.muloss = loss.muloss + l
        loss.muloss = loss.muloss*eps_mu/(len(net.pt)*len(net.effics))

        
    return loss