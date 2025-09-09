import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import torch
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd

# Utility function for seeing what happens when taking gradient of loss function.
def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])
                

def ListToGraph(l,bins,color,low=0,high=1,weight=0,label=""):
    counts,bin_edges = np.histogram(l,bins,range=(low,high))
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    err = np.sqrt(counts)
    if weight==0:
        weight=1./(sum(counts)*(bin_edges[1]-bin_edges[0]))
    counts = weight*counts
    err = weight*err
    return plt.errorbar(bin_centres, counts, yerr=err, fmt='o',color=color,label=label)




def make_ROC_curve(y_test, y_pred_test):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test.numpy())
    roc_auc = roc_auc_score(y_test, y_pred_test.numpy())
    
    plt.figure(1)
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", label="AUC = {:.3f}".format(roc_auc))
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    #plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    

def plot_classifier_output(y_train, y_pred_train,y_test, y_pred_test, nbins=20, range=(0,1)):
    signal_train=[]
    signal_test =[]
    backgr_train=[]
    backgr_test =[]
    for y,y_p in zip(y_train,y_pred_train):
        if y==1: signal_train.append(float(y_p))
        else:    backgr_train.append(float(y_p))
    for y,y_p in zip(y_test,y_pred_test):
        if y==1: signal_test.append(float(y_p))
        else:    backgr_test.append(float(y_p))
    
    signal_train_hist=plt.hist(signal_train,nbins,density=True,range=range,histtype='stepfilled',alpha=0.5,color='red',label="Train: signal")
    backgr_train_hist=plt.hist(backgr_train,nbins,density=True,range=range,histtype='stepfilled',alpha=0.5,color='blue', label="Train: background")
    signal_test=ListToGraph(signal_test,nbins,"red",low=range[0], high=range[1], label="Test: signal")
    backgr_test=ListToGraph(backgr_test,nbins,"blue", low=range[0], high=range[1], label="Test: background")
    plt.yscale("log")
    plt.xlabel('Output Score')
    plt.ylabel('Arbitrary Units')
    plt.legend(loc="upper center")
    #plt.show()


def plotlosses(losses, test_losses):
    plt.plot([l.totalloss().detach().numpy() for l in losses], '.', label="Train")
    plt.plot([l.totalloss().detach().numpy() for l in test_losses], '.', label="Test")
    plt.plot([l.efficloss.detach().numpy() for l in losses], '.', label="Train: effic")
    plt.plot([l.backgloss.detach().numpy() for l in losses], '.', label="Train: backg")
    plt.plot([l.cutszloss.detach().numpy() for l in losses], '.', label="Train: cutsz")
    # plt.plot([l.BCEloss.detach().numpy() for l in losses], '.', label="Train: BCE")
    if type(losses[0].ptloss) is not int:
        # this particular term can get very small, just cut it off for super small values
        plt.plot([max(l.ptloss.detach().numpy(),1e-12) for l in losses], '.', label="Train: pt smooth")
    if type(losses[0].efficloss) is not int:
        # this particular term can get very small, just cut it off for super small values
        plt.plot([max(l.efficfeatloss.detach().numpy(),1e-12) for l in losses], '.', label="Train: effic smooth")
    plt.legend()
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.yscale('log');
    


def ploteffics(losses, targeteffics):
    if type(losses[0].signaleffic) is list:
        for e in range(len(losses[0].signaleffic)):
            p = plt.plot([l.signaleffic[e].detach().numpy() for l in losses], '.', label=f"Effic: {100.*targeteffics[e]:3.1f}%")
            p = plt.plot([l.backgreffic[e].detach().numpy() for l in losses], '-', color=plt.gca().lines[-1].get_color(), label="BG effic")
    else:
        plt.plot([l.signaleffic.detach().numpy() for l in losses], '.', label=f"Effic: {100.*targeteffics[0]:3.1f}%")
        plt.plot([l.backgreffic.detach().numpy() for l in losses], '.', color=plt.gca().lines[-1].get_color(), label="BG effic")
    plt.legend()
    plt.xlabel('Training Epoch')
    plt.ylabel('Signal Efficiency')
    #plt.yscale('log')


def check_effic(x_test_tensor, y_test, net, printout=True):
    num_pass_test=0.
    num_bg_pass_test=0.
    test_outputs = net.apply_cuts(x_test_tensor).detach().cpu()
    m=test_outputs.shape[1]
    trues=torch.tensor(m*[True])
    for i in range(len(test_outputs)):
    
        tt=torch.zeros(m)
        t=torch.gt(test_outputs[i],tt)
    
        if torch.equal(t,trues) and y_test[i]==1.:
            num_pass_test+=1
        elif torch.equal(t,trues) and y_test[i]!=1.:
            num_bg_pass_test+=1.
        
    
    effic_test    = num_pass_test    / np.sum(y_test)
    bg_effic_test = num_bg_pass_test / np.sum(1.-y_test)

    if printout:
        print(f"Signal     efficiency with net outputs: {100*effic_test:4.1f}%")
        print(f"Background efficiency with net outputs: {100*bg_effic_test:8.5f}%")

    return effic_test,bg_effic_test


def plotgenericcuts(net):
    fig = plt.figure(figsize=(20,5))
    fig.tight_layout()
    targeteffics=net.effics
    m=net.features
    
    scaled_cuts=[len(targeteffics)*[0] for i in range(m)]
    for n in range(len(targeteffics)):
        cuts=net.nets[n].get_cuts().detach().numpy()
        for f in range(m):
            cutval=cuts[f]
            scaled_cuts[f][n]=cutval
    for b in range(m):
        ax=fig.add_subplot(2,5,1+b)
        plt.subplots_adjust(hspace=0.6,wspace=0.5)
        ax.plot(targeteffics,scaled_cuts[b])
        ax.set_xlabel(f"Target Efficiency")
        ax.set_ylabel("Cut value")
        ax.set_title(f"Feature {b}")
        ax.set_ylim([-10,10])


def plotfeatures(net,x_signal,x_backgr,sc):
    # Distributions after scaling
    targeteffics=net.effics
    m=net.features
    for n in range(len(targeteffics)):
        print(f"Target efficiency: {targeteffics[n]*100}%")
        fig = plt.figure(figsize=(20,5))
        fig.tight_layout()
        nbins=50
        
        weights=net.nets[n].weight.detach().numpy()
        scaled_cuts=net.nets[n].get_cuts().detach().numpy()
        
        x_signal_scaled=sc.transform(x_signal)
        x_backgr_scaled=sc.transform(x_backgr)
        
        for b in range(m):
            ax=fig.add_subplot(2,5,1+b)
            plt.subplots_adjust(hspace=0.5,wspace=0.5)
            plt.yscale('log')
            ax.hist(x_signal_scaled[:,b],nbins,density=True,histtype='stepfilled',alpha=0.5,color='red')
            ax.hist(x_backgr_scaled[:,b],nbins,density=True,histtype='stepfilled',alpha=0.5,color='blue')
            ax.set_xlabel(f"Feature {b}")
            ax.set_ylabel("Events/Bin")
            if weights[b] < 0:
                ax.axvline(x = scaled_cuts[b], color='g') # cut is "less than"
            else:
                ax.axvline(x = scaled_cuts[b], color='y') # cut is "greater than"
        plt.show()

def noisify(x_tensor, y_tensor, weights, noiserate=0.10):

    signal_noise = noiserate*torch.rand(x_tensor.shape[0],x_tensor.shape[1])*y_tensor.unsqueeze(1)
    backgr_noise = noiserate*torch.rand(x_tensor.shape[0],x_tensor.shape[1])*(1-y_tensor).unsqueeze(1)
    weightsigns=torch.sign(weights)
    for i in range(len(weightsigns)):
        if weightsigns[i]>0:
            signal_noise[:,i] = -signal_noise[:,i]
        else:
            backgr_noise[:,i] = -backgr_noise[:,i]
    
    return x_tensor + signal_noise + backgr_noise
    

def getcuteffic(y_test_pred, y_test_true, thresh, signal=True):
    y_test = y_test_true if signal else (1-y_test_true)
    passes = (y_test_pred*y_test > thresh).sum().item()
    total  = (y_test > 0.5).sum().item()
    return passes/total

def getefficcut(y_test_pred,y_test_true,targeteffic):
    fpr, tpr, thresholds = roc_curve(y_test_true, y_test_pred.numpy())
    thresh=0
    #print(thresholds)
    #print(tpr)
    #print(fpr)
    for i in range(len(tpr)):
        if tpr[i]<=targeteffic and tpr[i+1]>targeteffic:
            return thresholds[i],tpr[i]
        
def plotcuts(net):
    latex_string = ['$R_{{had}}$','$R_{{\\eta}}$','$R_{{\\phi}}$','$w_{{\\eta^{{2}}}}$','$E_{{ratio}}$','$\\Delta E [MeV]$','$w^{{tot}}_{{\\eta 1}}$','$F_{{side}}$','$w_{{\\eta^{{3}}_{{\\eta 1}}}}$']
    fig = plt.figure(figsize=(12,8))
    fig.tight_layout()
    targeteffics=net.effics
    m=net.features
    
    scaled_cuts=[len(targeteffics)*[0] for i in range(m)]
    for n in range(len(targeteffics)):
        cuts=net.nets[n].get_cuts().detach().numpy()
        for f in range(m):
            cutval=cuts[f]
            scaled_cuts[f][n]=cutval
    for b in range(m):
        ax=fig.add_subplot(3,3,1+b)
        plt.subplots_adjust(hspace=0.6,wspace=0.5)
        ax.plot(targeteffics,scaled_cuts[b])
        ax.set_xlabel(f"Target Efficiency")
        ax.set_ylabel("Cut value")
        ax.set_title(f"{latex_string[b]}")
        ax.set_ylim([-3,3])


def data_concat(fp,fn,tn,feats):
    full_path = [f"{fp}{file}" for file in fn]

    data=[]
    for file_path in full_path:
        with uproot.open(file_path) as events:
            data.append(events[tn].arrays(feats))
    
    return ak.concatenate(data)

def data_mask(d,e,eta,mu):
    # print('.')
    # takes awk array and converts to numpy and sorts features
    datalist=['ph.pt', 'ph.eta', 'ph.rhad1', 'ph.reta', 'ph.rphi', 'ph.weta2', 'ph.eratio', 'ph.deltae', 'ph.wstot', 'ph.fside', 'ph.w1', 'ph.truth_pdgId', 'ph.truth_type', 'ph.convFlag', 'EventInfo.actualIntPerXing']
    # for i in datalist:
    #     print(f'{i:7f},{len(data[i])},{type(data[i])}')
    mask = (d['ph.pt'] >= e[0]) & (d['ph.pt'] < e[1]) & (abs(d['ph.eta']) < eta[1]) & (abs(d['ph.eta']) >= eta[0]) & (d['ph.wstot'] >= 0) & (d['ph.w1'] >= 0) & (d['ph.convFlag'] == 0)
    # do unconv for now ph.convFlag == 0
    # print('.')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,3))
    
    myrange_pt = (0,100)
    nbins_pt=20
    binwidth_pt=(myrange_pt[1]-myrange_pt[0])/nbins_pt
    ax1.hist(((d['ph.pt'])/1000)[mask],density=True, bins=nbins_pt, range=myrange_pt,color = 'grey')
    ax1.set_xlabel('Transverse Momentum $p_{\\mathrm{T}}$ [GeV]',loc = 'right')
    ax1.set_ylabel(f"1/N dN/d({'$p_{t}$'})",loc = 'top')
    plt.tight_layout()
    
    nbins_eta=20
    myrange_eta=(-4,4)
    binwidth_eta=(myrange_eta[1]-myrange_eta[0])/nbins_eta
    ax2.hist(d["ph.eta"][mask],density=True, bins=nbins_eta, range=myrange_eta, color = 'grey')
    ax2.set_xlabel('Psuedorapidity $\\eta$',loc = 'right')
    ax2.set_ylabel("1/N dN/d($\\eta$)",loc = 'top')
    plt.tight_layout()

    nbins_mu=20
    myrange_mu=mu
    binwidth_mu=(myrange_mu[1]-myrange_mu[0])/nbins_mu
    ax3.hist(d['EventInfo.actualIntPerXing'][mask],density=True, bins=nbins_mu, range=myrange_mu, color = 'grey')
    ax3.set_xlabel('Pileup $\\mu$',loc = 'right')
    ax3.set_ylabel("1/N dN/d($\\mu$)",loc = 'top')
    plt.tight_layout()
    
    numpy_data = {}
    
    for i in datalist:
        numpy_array = ak.to_numpy(d[i][mask])
        numpy_data[i] = numpy_array
        # print(i,len(numpy_array),type(numpy_array))
    print(f'This mask includes {len(numpy_array)} event candidates:')
    return numpy_data
    
def get_labels(d):
    labels = []
    for i in range(len(d['ph.pt'])):
        if d['ph.truth_pdgId'][i] == 22 and (d['ph.truth_type'][i]==15 or d['ph.truth_type'][i]==13 or d['ph.truth_type'][i]==14):
            labels.append(1)
        else:
            labels.append(0)
    return torch.Tensor(labels)
    
def true_sort(d,ls):
    dlist=['ph.pt', 'ph.eta', 'ph.rhad1', 'ph.reta', 'ph.rphi', 'ph.weta2', 'ph.eratio', 'ph.deltae', 'ph.wstot', 'ph.fside', 'ph.w1', 'ph.truth_pdgId', 'ph.truth_type', 'ph.convFlag']
    truedata = {}
    falsedata = {}
    for i in range(0,11):
        truelist = []
        falselist = []
        for j in range(0,len(ls)):
            if ls[j] == 1:
                truelist.append(d[dlist[i]][j])
            else:
                falselist.append(d[dlist[i]][j])
        truedata[i] = truelist
        falsedata[i] = falselist
        
    print('We are working with:')
    print(f'{len(truelist)} Signal Events')
    print(f'{len(falselist)} Backround Events')
    print(f'Signal:Backround = {(len(truelist)/len(falselist)):.3f}')
    return truedata, falsedata

def plot_signal(t,f):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
    ranges = [(-0.05,0.25),(0.2,1.1),(0.2,1.1),(0,0.025),(0,1),(0,8000),(0,15),(0,1.),(0,1.)]
    latex_string = ['$R_{{had}}$','$R_{{\\eta}}$','$R_{{\\phi}}$','$w_{{\\eta^{{2}}}}$','$E_{{ratio}}$','$\\Delta E [MeV]$','$w^{{tot}}_{{\\eta 1}}$','$F_{{side}}$','$w_{{\\eta^{{3}}_{{\\eta 1}}}}$']
    nbins = 20
    for i, ax in enumerate(axes.flatten()):
        ax.hist(f[i+2],density=True, range = ranges[i],bins=nbins,histtype = 'stepfilled',color='grey',alpha = 0.9, edgecolor='black', label = 'Backround')
        ax.hist(t[i+2], density=True, range = ranges[i],bins=nbins,histtype = 'stepfilled',color='white',alpha = 0.75, edgecolor='black', label = 'Signal')
        ax.set_xlabel(latex_string[i],loc = 'right')
        ax.set_ylabel(f"1/N dN/d({latex_string[i]})",loc = 'top')
        if i == 0:
            ax.legend()
            # ax.set_title('**ATLAS**\nPrelimiary\nData',y=0.85)
        ax.set_yscale('log')
    fig.tight_layout()

def plot_feats(d):
    discriminating_vars = ['ph.rhad1','ph.reta','ph.rphi','ph.weta2','ph.eratio','ph.deltae','ph.wstot','ph.fside','ph.w1']
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 8))
    ranges = [(-0.05,0.25),(0.2,1.1),(0.2,1.1),(0,0.025),(0,1),(0,8000),(0,15),(0,1.),(0,1.)]
    latex_string = ['$R_{{had}}$','$R_{{\\eta}}$','$R_{{\\phi}}$','$w_{{\\eta^{{2}}}}$','$E_{{ratio}}$','$\\Delta E [MeV]$','$w^{{tot}}_{{\\eta 1}}$','$F_{{side}}$','$w_{{\\eta^{{3}}_{{\\eta 1}}}}$']
    nbins = 20
    for i, ax in enumerate(axes.flatten()):
        ax.hist(d[discriminating_vars[i]], density=True, range = ranges[i],bins=nbins,histtype = 'stepfilled',color='grey',alpha = 0.75, edgecolor='black')
        ax.set_xlabel(latex_string[i],loc = 'right')
        ax.set_ylabel(f"1/N dN/d({latex_string[i]})",loc = 'top')
        ax.set_yscale('log')
    fig.tight_layout()




def test_train_split(d,pt,mu,ratio):

    

    data_bucket = [[[] for j in range(len(mu))] for i in range(len(pt))]
    label_bucket = [[[] for j in range(len(mu))] for i in range(len(pt))]
    dlist = ['ph.rhad1', 'ph.reta', 'ph.rphi', 'ph.weta2', 'ph.eratio', 'ph.deltae', 'ph.wstot', 'ph.fside', 'ph.w1']
    loosetrues = []
    for i in range(len(d['ph.pt'])):
        if d['ph.truth_pdgId'][i] == 22 and (d['ph.truth_type'][i]==15 or 
                                             d['ph.truth_type'][i]==13 or
                                             d['ph.truth_type'][i]==14):
            true = 1
        else:
            true = 0
        
        for j in range(len(pt)):
            for k in range(len(mu)):
                if (d['ph.pt'][i] >= pt[j][0] and d['ph.pt'][i] < pt[j][1]) \
                & (d['EventInfo.actualIntPerXing'][i] >= mu[k][0] and d['EventInfo.actualIntPerXing'][i] < mu[k][1]):
    
                    label_bucket[j][k].append(true)
                    data_bucket[j][k].append([d[key][i] for key in dlist])
                    break

    loosetrues = torch.Tensor(loosetrues)
    
    x_train_tensor_list = [[[] for j in range(len(mu))] for i in range(len(pt))]
    x_test_tensor_list = [[[] for j in range(len(mu))] for i in range(len(pt))]
    y_train_tensor_list = [[[] for j in range(len(mu))] for i in range(len(pt))]
    y_test_tensor_list = [[[] for j in range(len(mu))] for i in range(len(pt))]
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    sc_list = [[StandardScaler() for j in range(len(mu))] for i in range(len(pt))]
    
    for i in range(len(pt)):
        for j in range(len(mu)):
            # print(len(data_bucket[i][j]))
            X = np.array(data_bucket[i][j], dtype=np.float32)
            Y = np.array(label_bucket[i][j], dtype=np.float32)
            X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=ratio, random_state=42)
            
        
            
            
            X_train = sc_list[i][j].fit_transform(X_train)
            X_test = sc_list[i][j].transform(X_test)
            
            x_train_tensor_list[i][j] = torch.tensor(X_train, dtype=torch.float).detach()
            y_train_tensor_list[i][j] = torch.tensor(y_train, dtype=torch.float).detach()
            
            x_test_tensor_list[i][j] = torch.tensor(X_test, dtype=torch.float).detach()
            y_test_tensor_list[i][j] = torch.tensor(y_test, dtype=torch.float).detach()
        
    return x_train_tensor_list, y_train_tensor_list, x_test_tensor_list, y_test_tensor_list, sc_list

def cutsheatmap(net):
    latex_string = ['$R_{{had}}$','$R_{{\\eta}}$','$R_{{\\phi}}$','$w_{{\\eta^{{2}}}}$','$E_{{ratio}}$','$\\Delta E [MeV]$','$w^{{tot}}_{{\\eta 1}}$','$F_{{side}}$','$w_{{\\eta^{{3}}_{{\\eta 1}}}}$']
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,9))
    targeteffics=net.effics
    pt=net.pt
    m=net.features
    
    scaled_cuts=[[[[0] for _ in range(len(targeteffics))] for _ in range(len(pt))] for i in range(m)]
    for i in range(len(pt)):
        for j in range(len(targeteffics)):
            cuts=net.nets[i][j].get_cuts().detach().numpy()
            for f in range(m):
                cutval=cuts[f]
                scaled_cuts[f][i][j]=cutval
    
    
    for b, ax in enumerate(axes.flatten()):
        im = ax.imshow([[scaled_cuts[b][i][j] for j in range(len(targeteffics))]for i in range(len(pt))])
        cbar = plt.colorbar(im, ax=ax)
        # cbar.set_label("Scaled Cut Val", fontsize=10)
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(range(len(targeteffics)), labels=targeteffics,)
        ax.set_yticks(range(len(pt)), labels=pt)
        
        # # Loop over data dimensions and create text annotations.
        # for i in range(len(pt)):
        #     for j in range(len(targeteffics)):
        #         # text = ax.text(j, i, scaled_cuts[b][i][j],
        #         #                ha="center", va="center", color="w")
        
        ax.set_title(f"Cuts Over Pt and Efficency for {latex_string[b]}")
    fig.tight_layout()

def make_ROC_curves(y_test,x_test, net):
    net.eval() # configure the model for evaluation (testing)
    pt=net.pt
    effics = net.effics
    m=net.features
    y_pred = net(x_test)
    fig, axes = plt.subplots(nrows=len(pt), ncols=len(effics), figsize=(12,12))
    for i in range(len(pt)):
        for j in range(len(effics)):
            ax = axes[i, j]
            y_pred_ij = y_pred[i][j].detach().numpy()
            y_test_i = y_test[i]
            # print(len(y_pred_ij),len(y_test_i))
            fpr, tpr, thresholds = roc_curve(y_test_i, y_pred_ij)
            roc_auc = roc_auc_score(y_test_i, y_pred_ij)
            
            lw = 2
            ax.plot(fpr, tpr, color="darkorange", label="AUC = {:.3f}".format(roc_auc))
            ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            # ax.set_xlabel('False positive rate')
            # ax.set_ylabel('True positive rate')
            ax.set_title(f'pt={pt[i]} \n effic={effics[j]}')
            ax.legend(loc="lower right")
    fig.tight_layout()

def check_noisy_test_inputs(x_test_tensor, y_test_tensor, net, weights, targeteffic=0.95, noiserate=0.10):
    x_test_tensor_noisy = noisify(x_test_tensor, y_test_tensor, weights, noiserate)
    y_pred_test_noisy   = net(x_test_tensor_noisy).detach().cpu()
    y_pred_test_clean   = net(x_test_tensor).detach().cpu()

    thresh,thresheffic=getefficcut(y_pred_test_clean,y_test_tensor,targeteffic)    
    print(f"Output threshold for {targeteffic*100:4.1f}% efficiency: {thresh} gives {thresheffic*100:4.1f}%")
    #print(thresheffic)
    
    print()
    # signal 
    if len(y_pred_test_noisy.shape)>1: 
        print(f"Signal efficiency for nominal, output>{thresh}: {getcuteffic(y_pred_test_clean.squeeze(1), y_test_tensor, thresh, signal=True)}")
        print(f"Signal efficiency for noisy  , output>{thresh}: {getcuteffic(y_pred_test_noisy.squeeze(1), y_test_tensor, thresh, signal=True)}")
    else:
        print(f"Signal efficiency for nominal, output>{thresh}: {getcuteffic(y_pred_test_clean, y_test_tensor, thresh, signal=True)}")
        print(f"Signal efficiency for noisy  , output>{thresh}: {getcuteffic(y_pred_test_noisy, y_test_tensor, thresh, signal=True)}")
    print()
    
    # background efficiency
    if len(y_pred_test_noisy.shape)>1: 
        print(f"Backgr efficiency for nominal, output>{thresh}: {getcuteffic(y_pred_test_clean.squeeze(1), y_test_tensor, thresh, signal=False)}")
        print(f"Backgr efficiency for noisy  , output>{thresh}: {getcuteffic(y_pred_test_noisy.squeeze(1), y_test_tensor, thresh, signal=False)}")
    else:
        print(f"Backgr efficiency for nominal, output>{thresh}: {getcuteffic(y_pred_test_clean, y_test_tensor, thresh, signal=False)}")
        print(f"Backgr efficiency for noisy  , output>{thresh}: {getcuteffic(y_pred_test_noisy, y_test_tensor, thresh, signal=False)}")
    print()
    
    thresh_noisy,thresheffic_noisy=getefficcut(y_pred_test_noisy,y_test_tensor,targeteffic)    
    print(f"Output threshold for {targeteffic*100:4.1f}% efficiency if trained on noisy data: {thresh_noisy} gives {thresheffic_noisy*100:4.1f}%")

def get_scores_by_label(x_tensor, y_tensor, net):
    tp_scores=[]
    tn_scores=[]
    fp_scores=[]
    fn_scores=[]
    outputs = net.apply_cuts(x_tensor).detach().cpu()
    scores  = net(x_tensor).detach().cpu()
    m=outputs.shape[1]
    trues=torch.tensor(m*[True])
    for i in range(len(outputs)):
    
        tt=torch.zeros(m)
        t=torch.gt(outputs[i],tt)
    
        if torch.equal(t,trues) and y_tensor[i]==1.:
            tp_scores.append(scores[i])
        elif torch.equal(t,trues) and y_tensor[i]!=1.:
            fp_scores.append(scores[i])
        elif y_tensor[i]==1.:
            fn_scores.append(scores[i])
        elif y_tensor[i]!=1.:
            tn_scores.append(scores[i])

    return tp_scores, tn_scores, fp_scores, fn_scores

def plot_classifier_output_by_label(y_tp,y_fp,y_tn,y_fn, nbins=10, range=(0,1)):

    tp_hist=plt.hist(y_tp,nbins,density=True,range=range,histtype='stepfilled',alpha=0.5,color='red'   , label="True Signal: Pass")
    fn_hist=plt.hist(y_fn,nbins,density=True,range=range,histtype='stepfilled',alpha=0.5,color='yellow', label="True Signal: Fail")
    tn_hist=plt.hist(y_tn,nbins,density=True,range=range,histtype='stepfilled',alpha=0.5,color='blue'  , label="True Background: Fail")
    fp_hist=plt.hist(y_fp,nbins,density=True,range=range,histtype='stepfilled',alpha=0.5,color='green' , label="True Background: Pass")

    plt.yscale("log")
    plt.xlabel('Output Score')
    plt.ylabel('Arbitrary Units')
    plt.legend(loc="upper center")

def load_random_data():
    # array of means and widths for gaussians,
    # indices are:
    # - number of dimensions
    # - signal (0) vs background (1)
    # - mean (0) vs width (1)

    N=20000 # number of points
    m=10 # dimensions
    
    np.random.seed(123)
    width_scale_factor=2.
    mean_scale_factor=1.
    
    # if we want more control over the function, for example
    means_widths=[[[4,0.4],
                   [-2,.10]],
                  [[3,1.0],
                   [-1,0.5]]]
    
    means_widths=np.random.randn(m,2,2)               
    
    # now construct the training data.  after taking transpose, should be N rows by m columns.  
    x_signal=np.array([mean_scale_factor*means_widths[i][0][0]+width_scale_factor*np.fabs(means_widths[i][0][1])*np.random.randn(N) for i in range(m)]).T
    y_signal=np.ones(N)
    x_backgr=np.array([mean_scale_factor*means_widths[i][1][0]+width_scale_factor*np.fabs(means_widths[i][1][1])*np.random.randn(N) for i in range(m)]).T
    y_backgr=np.zeros(N)

    return x_signal, y_signal, x_backgr, y_backgr, [f"Feature {b}" for b in range(m)]


def load_SUSY_data():
    # Prep Data using Pandas
    x_sig_data=None
    y_sig_data=None
    
    x_bkg_data=None
    y_bkg_data=None
    
    num_sig_events=0
    num_bkg_events=0
    
    # Need to copy the last branch, Rll, so that we have a total of 5 branches
    branches=[
        'lep1MT_Met','lep1_DPhiMet',
        'lep2MT_Met','lep2_DPhiMet',
        'jet0Pt','Rll','met_Et',
    #    ,'lep1Pt','lep2Pt',
    #    ,'lep1Eta','lep1Phi', 
    #    ,'lep2Eta','lep2Phi'
    #    ,'met_Phi'
    #    'nJet30','mt2leplsp_100'
    ]
    
    # open signal
    mass=200
    split=30
    filepath='/data/mhance/SUSY/Compressed/'
    filebase='SusySkimSlep_v0.2_SlepSignals__'
    filename='MGPy8EG_A14N23LO_SlepSlep_dir_2L2MET75_%dp0_%dp0_NoSys' % (mass,mass-split)
    filesuff='.hf5'
    
    # Makes data frame for signal events
    fullname=filepath+filebase+filename+filesuff
    df = pd.read_hdf(fullname, key=filename)
    mask = (df['lep1Pt']>10) & (df['lep2Pt']>10) & (df['nBJet30']==0) & (df['met_Et']>300) & (df['minDPhiAllJetsMet']>0.4)
    df = df[mask]
    df1 = df[branches].to_numpy()
    num_sig_events = len(df1)
    x_signal = df1
    y_signal = np.ones(num_sig_events)
    #print("Extracted %7d signal events" % num_sig_events)

    # Now background
    fullname=filepath+"SusySkimSlep_v0.2_diboson2L__diboson2L_NoSys"+filesuff
    df = pd.read_hdf(fullname, key='diboson2L_NoSys')
    mask = (df['lep1Pt']>10) & (df['lep2Pt']>10) & (df['nBJet30']==0) & (df['met_Et']>300) & (df['minDPhiAllJetsMet']>0.4)
    df = df[mask]
    df2 = df[branches].to_numpy()
    num_bkg_events = len(df2)
    x_backgr = df2
    y_backgr=np.zeros(num_bkg_events)
    #print("Extracted %7d background events" % num_bkg_events)    

    return x_signal, y_signal, x_backgr, y_backgr, branches