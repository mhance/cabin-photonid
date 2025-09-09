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
        

def smooth_loss_fn(xaxis,cuts):
    featureloss = None

    if len(xaxis)<3: 
        return featureloss

    for i in range(1,len(xaxis)-1):
        # y is the cut, x is the dependent variable (e.g. efficiency, or pT, or mu)
        y_i   = cuts [i  ]
        y_im1 = cuts [i-1]
        y_ip1 = cuts [i+1]
        x_i   = xaxis[i  ]
        x_im1 = xaxis[i-1]
        x_ip1 = xaxis[i+1]

        # calculate distance between cuts.  
        # would be better to implement this as some kind of distance away from the region 
        # between the two other cuts.
        #
        # maybe some kind of dot product?  think about Ising model.
        #
        # maybe we just do this for the full set of biases, to see how many transitions there are?  no need for a loop?
        #
        # otherwise just implement as a switch that calculates a distance if outside of the range of the two cuts, zero otherwise
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
        # away from the mean should do the trick.
        exponent=2.  # if this changes, e.g. to 4, then epsilon will also need to increase
        fl=(distance_from_interp**exponent)/((yrange**exponent)+0.1)
        # ------------------------------------------------------------------
        
        # ------------------------------------------------------------------
        ## can also do it this way, which just forces all sequential cuts to be similar.
        #fl = torch.pow(cuts_i-cuts_im1,2) + torch.pow(cuts_i-cuts_ip1,2) + torch.pow(cuts_im1-cuts_ip1,2)
        # ------------------------------------------------------------------
        if featureloss == None:
            featureloss = fl
        else:
            featureloss = featureloss + fl

    # sum over all cuts, and normalize to the number of xaxis points
    return torch.sum(featureloss)/len(xaxis)
    

def full_loss_fn(y_pred, y_true, features, net,
                 alpha=1., beta=1., gamma=0.001, delta=0., 
                 eps_ef=0.001, eps_pt=0.001, eps_mu=0.001,
                 debug=False):
    loss=None    
    for i in range(len(net.pt)):
        for j in range(len(net.mu)):
            for k in range(len(net.effics)):
                pt     = net.pt[i][0]
                mu     = net.mu[j][0]
                effic  = net.effics[k]
                subnet = net.nets[i][j][k]
                l=loss_fn(y_pred[i][j][k], 
                          y_true[i][j], 
                          features, 
                          subnet, effic,
                          alpha, beta, gamma, delta, debug)
                if loss==None:
                    loss = photonlossvars(l)
                    #print(type(loss))
                else:
                    loss = loss + l
                    #print(type(loss))


    # this broke everything!  don't do this scaling.  not sure why it doesn't work
    #totalpoints=(len(net.pt)*len(net.mu)*len(net.effics))
    #loss.efficloss /= totalpoints
    #loss.backgloss /= totalpoints
    #loss.cutszloss /= totalpoints

    #print(type(loss))

    # now smooth over efficiency, for each pT/mu bin:
    if len(net.effics)>=3:
        for i in range(len(net.pt)):
            for j in range(len(net.mu)):
                cuts=[net.nets[i][j][k].get_cuts() for k in range(len(net.effics))]
                l=smooth_loss_fn(net.effics,cuts)
                if loss.monotloss == 0:
                    loss.monotloss = l
                else:
                    loss.monotloss = loss.monotloss + l
        loss.monotloss = loss.monotloss*eps_ef/(len(net.pt)*len(net.mu))

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
    


# scott's version, does this simultaneously in all dimensions. not sure I understand how this works.
def scott_full_loss_fn(y_pred, y_true, features, net,
                  alpha=1., beta=1., gamma=0.001, delta=0., epsilon=0.001,
                  debug=False):
    sumptlosses=None    
    for i in range(len(net.pt)):
        for j in range(len(net.mu)):
            for k in range(len(net.effics)):
                pt=net.pt[i][0]
                mu=net.mu[j][0]
                effic = net.effics[k]
                subnet = net.nets[i][j][k]
                l=loss_fn(y_pred[i][j][k], y_true[i][j], features, 
                          subnet, effic,
                          alpha, beta, gamma, delta, debug)
                if sumptlosses==None:
                    sumptlosses=l
                else:
                    sumptlosses = sumptlosses + l

    loss=sumptlosses

    

    if len(net.pt)>=3 and len(net.mu)>=3:
        featurelosspt = None
        featurelossmu = None
        featurelosseffic = None
        for i in range(1,len(net.pt)-1):
            for j in range(1,len(net.mu)-1):
                for k in range(1,len(net.effics)-1):
                    cuts_ijk   = net.nets[i  ][j  ][k  ].get_cuts()
                    cuts_im1jk = net.nets[i-1][j  ][k  ].get_cuts()
                    cuts_ip1jk = net.nets[i+1][j  ][k  ].get_cuts()
                    cuts_ijm1k = net.nets[i  ][j-1][k  ].get_cuts()
                    cuts_ijp1k = net.nets[i  ][j+1][k  ].get_cuts()
                    cuts_ijkm1 = net.nets[i  ][j  ][k-1].get_cuts()
                    cuts_ijkp1 = net.nets[i  ][j  ][k+1].get_cuts()
                    flpt = None
                    flmu = None
                    fleffic = None
        
                    cutrange_pt           =  cuts_ip1jk-cuts_im1jk
                    mean_pt               = (cuts_ip1jk+cuts_im1jk)/2.
                    distance_from_mean_pt = (cuts_ijk  -mean_pt)
                    
                    cutrange_mu           =  cuts_ijp1k-cuts_ijm1k
                    mean_mu               = (cuts_ijp1k+cuts_ijm1k)/2.
                    distance_from_mean_mu = (cuts_ijk  -mean_mu)
                    
                    cutrange_effic           =  cuts_ijkp1-cuts_ijkm1
                    mean_effic               = (cuts_ijkp1+cuts_ijkm1)/2.
                    distance_from_mean_effic = (cuts_ijk  -mean_effic)                
                    exponent=2.  
                    
                    flpt=(distance_from_mean_pt**exponent)/((cutrange_pt**exponent)+0.1) 
                    flmu=(distance_from_mean_mu**exponent)/((cutrange_mu**exponent)+0.1) 
                    fleffic = (distance_from_mean_effic**exponent)/((cutrange_effic**exponent)+0.1)
                    # -----------------------------------------------------
                  
                    if featurelosspt == None:
                        featurelosspt = flpt
                    else:
                        featurelosspt = featurelosspt + flpt
                        
                    if featurelossmu == None:
                        featurelossmu = flmu
                    else:
                        featurelossmu = featurelossmu + flmu
                    
                    if featurelosseffic == None:
                        featurelosseffic = fleffic
                    else:
                        featurelosseffic = featurelosseffic + fleffic
                    
        sumptlosses = torch.sum(featurelosspt)/features
        summulosses = torch.sum(featurelossmu)/features
        summonotlosses = torch.sum(featurelosseffic)/features #/(len(net.pt)-2)
        loss.ptloss = epsilon*sumptlosses
        loss.muloss = epsilon*summulosses
        loss.monotloss = epsilon*summonotlosses

    ### For pt>=3 and mu<3
    if len(net.pt)>=3 and len(net.mu)<3:
        featurelosspt = None
        featurelosseffic = None
        for i in range(1,len(net.pt)-1):
            for j in range(len(net.mu)):
                for k in range(1,len(net.effics)-1):
                    cuts_ijk   = net.nets[i  ][j  ][k  ].get_cuts()
                    cuts_im1jk = net.nets[i-1][j  ][k  ].get_cuts()
                    cuts_ip1jk = net.nets[i+1][j  ][k  ].get_cuts()
                    cuts_ijkm1 = net.nets[i  ][j  ][k-1].get_cuts()
                    cuts_ijkp1 = net.nets[i  ][j  ][k+1].get_cuts()
                    flpt = None
                    flmu = None
                    fleffic = None
        
                    cutrange_pt           =  cuts_ip1jk-cuts_im1jk
                    mean_pt               = (cuts_ip1jk+cuts_im1jk)/2.
                    distance_from_mean_pt = (cuts_ijk  -mean_pt)
                    
                    cutrange_effic           =  cuts_ijkp1-cuts_ijkm1
                    mean_effic               = (cuts_ijkp1+cuts_ijkm1)/2.
                    distance_from_mean_effic = (cuts_ijk  -mean_effic)                
                    exponent=2.  
                    
                    flpt=(distance_from_mean_pt**exponent)/((cutrange_pt**exponent)+0.1)
                    fleffic = (distance_from_mean_effic**exponent)/((cutrange_effic**exponent)+0.1)
                    # -----------------------------------------------------
                  
                    if featurelosspt == None:
                        featurelosspt = flpt
                    else:
                        featurelosspt = featurelosspt + flpt
                    
                    if featurelosseffic == None:
                        featurelosseffic = fleffic
                    else:
                        featurelosseffic = featurelosseffic + fleffic
                    
        sumptlosses = torch.sum(featurelosspt)/features
        summonotlosses = torch.sum(featurelosseffic)/features #/(len(net.pt)-2)
        loss.ptloss = epsilon*sumptlosses
        loss.monotloss = epsilon*summonotlosses

    return loss