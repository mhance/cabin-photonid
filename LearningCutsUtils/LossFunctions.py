import torch

from cabin.LossFunctions import lossvars,loss_fn,effic_loss_fn

class photonlossvars(lossvars):
    def __init__(self,other=None):
        self.ptloss = 0
        self.muloss = 0
        #print("hi")
        if other is not None:
            #print("hi2")
            self.__copy__(other)
         
    def __copy__(self,other):
        self.efficloss = other.efficloss
        self.backgloss = other.backgloss
        self.cutszloss = other.cutszloss
        self.monotloss = other.monotloss
        self.BCEloss   = other.BCEloss
        self.signaleffic = other.signaleffic
        self.backgreffic = other.backgreffic
        if hasattr(other, 'ptloss'):
            self.ptloss = other.ptloss

        if hasattr(other, 'muloss'):
            self.muloss = other.muloss
        #print("hi3")
        
    def totalloss(self):
        return super().totalloss()+self.ptloss+self.muloss

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
        

# This should really go into CABIN at some point.
def smooth_loss_fn(xaxis,cuts):
    loss = None

    if len(xaxis)<3: 
        return loss

    for i in range(1,len(xaxis)-1):
        # y is the cut, x is the dependent variable (e.g. efficiency, or pT, or mu)
        y_i   = cuts [i  ]
        y_im1 = cuts [i-1]
        y_ip1 = cuts [i+1]
        x_i   = xaxis[i  ]
        x_im1 = xaxis[i-1]
        x_ip1 = xaxis[i+1]

        fl = None

        # ------------------------------------------------------------------
        # This method just forces cut i to be in between cut i+1 and cut i-1. 
        #
        # add some small term so that when cutrange=0 the loss doesn't become undefined
        xrange               = (x_ip1 - x_im1)
        yrange               = (y_ip1 - y_im1)

        slope                = (yrange / xrange)
        interp               = (slope*(x_i-x_im1) + y_im1)
        distance_from_interp = (y_i   - interp)
        
        # add some offset to denominator to avoid case where cutrange=0.
        # playing with the exponent doesn't change behavior much.
        # it's important that this term not become too large, otherwise
        # the training won't converge.  just a modest penalty for moving
        # away from the linear interpolation should do the trick.
        exponent=2.  # if this changes, e.g. to 4, then epsilon will also need to increase
        fl=(distance_from_interp**exponent)/((yrange**exponent)+0.1)
        # ------------------------------------------------------------------
        
        # ------------------------------------------------------------------
        ## can also do it this way, which just forces all sequential cuts to be similar.
        #fl = torch.pow(cuts_i-cuts_im1,2) + torch.pow(cuts_i-cuts_ip1,2) + torch.pow(cuts_im1-cuts_ip1,2)
        # ------------------------------------------------------------------
        if loss == None:
            loss = fl
        else:
            loss = loss + fl

    # sum over all cuts, and normalize to the number of xaxis points
    return torch.sum(loss)/len(xaxis)

    
# this also needs to go into CABIN
def effic_loss_fn_updated(
    y_pred,
    y_true,
    features,
    net,
    alpha=1.0,
    beta=1.0,
    gamma=0.001,
    delta=0.0,
    epsilon=0.001,
    debug=False,
):

    # probably a better way to do this, but works for now
    sumefficlosses = None
    for i in range(len(net.effics)):
        effic = net.effics[i]
        efficnet = net.nets[i]
        loss_i = loss_fn(
            y_pred[i],
            y_true,
            features,
            efficnet,
            effic,
            alpha,
            beta,
            gamma,
            delta,
            debug,
        )
        if sumefficlosses is None:
            sumefficlosses = loss_i
        else:
            sumefficlosses = sumefficlosses + loss_i

    loss = sumefficlosses

    if len(net.effics) >= 3:
        cuts=[net.nets[k].get_cuts() for k in range(len(net.effics))]
        l=smooth_loss_fn(net.effics,cuts)
        if loss.monotloss == 0:
            loss.monotloss = l
        else:
            loss.monotloss = loss.monotloss + l

        loss.monotloss = epsilon * loss.monotloss

    return loss
    

def full_loss_fn(y_pred, y_true, features, net,
                 alpha=1., beta=1., gamma=0.001, delta=0., 
                 eps_ef=0.001, eps_pt=0.001, eps_mu=0.001,
                 debug=False):

    loss=None    
    for i in range(len(net.pt)):
        for j in range(len(net.mu)):
            l=effic_loss_fn_updated(y_pred[i][j], 
                                    y_true[i][j], 
                                    features, 
                                    net.nets[i][j],
                                    alpha, beta, gamma, delta, eps_ef, debug)
            if loss==None:
                loss = photonlossvars(l)
            else:
                loss = loss + l

    # should scale down loss.monotloss in the same way as ptloss and muloss below?
    
    
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