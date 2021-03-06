import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings('ignore')

# Shuffle
# func1.1
def get_time_shuffle(Trace):
    l = Trace.shape[0]
    ts_Trace = Trace[np.random.choice(l,l,replace=False),:]
    return ts_Trace
    
# func1.2
def locate_transients(t):
    thr = np.mean(t)+3*np.std(t)
    thr_cross_or_not = np.append(np.where(t>thr)[0], len(t)+1)
    thr_cross_idx = np.where(np.diff(thr_cross_or_not)>1)[0]
    mean_cross_or_not = np.where(t<np.mean(t))[0]

    trsn_s=[]
    trsn_e=[]
    pidx = thr_cross_or_not[thr_cross_idx]

    for j in range(len(thr_cross_idx)):
        s = np.insert(mean_cross_or_not[np.where(mean_cross_or_not<pidx[j])[0]],0,0)[-1]
        e = np.append(mean_cross_or_not[np.where(mean_cross_or_not>pidx[j])[0]], len(t))[0]
        trsn_s.append(s)
        trsn_e.append(e)
    trsn_s = np.unique(np.array(trsn_s))
    trsn_e = np.unique(np.array(trsn_e))
    
    return trsn_s, trsn_e

# func1.3
def shuffle_transients(trsn_s, trsn_e, t):
    s = np.insert(trsn_e, 0, 0) # startpoint of no_transient segments
    e = np.append(trsn_s, len(t)) # endpoint of no_transient segments
    no_trsn = np.concatenate(([np.arange(s[k],e[k]) for k in range(len(s))]))
    swap_point = np.random.choice(len(no_trsn),len(trsn_e),replace=False) # point for transients to randomly insert
    swap_point = np.append(swap_point, len(no_trsn)+1)
    order = np.argsort(swap_point)
    ordered_swap_point = swap_point[order]

    trsn_shf_idx = no_trsn[:swap_point.min()] # transient shuffle index
    for j in range(len(swap_point)-1):
        trsn_shf_idx = np.append(trsn_shf_idx, np.arange(trsn_s[order[j]], trsn_e[order[j]]))
        trsn_shf_idx = np.append(trsn_shf_idx, no_trsn[ordered_swap_point[j]:ordered_swap_point[j+1]])
        
    return trsn_shf_idx

# func1.4
def get_trsn_shuffle(Trace):
    trsn_shf = []
    for i in range(Trace.shape[1]):
        t = Trace[:,i]
        trsn_s, trsn_e = locate_transients(t)
        trsn_shf_idx = shuffle_transients(trsn_s, trsn_e, t)
        trsn_shf.append(t[trsn_shf_idx])
    
    trsn_shf = np.vstack(trsn_shf).T
    return trsn_shf
    
# func1.5
def plot_shuffle(Trace, trsn_shf, i=9):
    plt.figure(figsize=(10,4))
    plt.plot(Trace[:,i], label = 'original trace')
    plt.plot([0,Trace.shape[0]],[np.mean(Trace[:,i])+3*np.std(Trace[:,i]),np.mean(Trace[:,i])+3*np.std(Trace[:,i])])
    plt.plot(trsn_shf[:,i], label = 'shuffled trace')
    plt.legend()
    plt.xlabel('Timebins')
    plt.ylabel(f'Fluorescence trace of cell #{i}')
    plt.show()

# Save Object
# func2.1
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj

# Visualization
# func3.1
def comp_poststates_pos(origin, Trace, Distance, lengths = None, sort = 'angle', normalize = True):
    """Compute decoded states and state place field.
    
       Output: x, plst, occ, posterior_states, pos_COM
       Parameters
       -----------
       sort: 'angle', 'maximum', or 'none'.
       
       Returns
       -----------
       occ: number of occurrences of specific position"""
       
    
    _, posterior_states = origin.score_samples(Trace, lengths=lengths)
    
    d = np.unique(Distance)
    x = np.zeros((origin.n_components,len(d)))
    occ = np.zeros(len(d))
    for i in range(len(d)):
        x[:,i] = np.mean(posterior_states[np.where(Distance==d[i])[0],:],axis=0)
        occ[i] = np.where(Distance==d[i])[0].shape[0]
        
    if normalize:
        for i in range(origin.n_components):
            if np.sum(x[i,:])!=0:
                x[i,:] = x[i,:]/np.sum(x[i,:])
    
    if sort=='maximum':
        pos_COM = d[np.argmax(x,axis=1)]
        plst = np.argsort(np.argmax(x,axis=1))
        x = x[plst,:]
    elif sort=='angle':
        thetas = 2 * np.pi * d / (d[-1]+d[1]) - np.pi
        vecs = x * np.cos(thetas)[np.newaxis,:] + 1j * x * np.sin(thetas)[np.newaxis,:]
        COM = np.angle(np.sum(vecs, axis=1))
        pos_COM = (COM + np.pi)*d[-1]/2/np.pi
        plst = np.argsort(COM)
        x = x[plst,:]
    else:
        plst = np.arange(origin.n_components)
        pos_COM=[]
    
    return x, plst, occ, posterior_states, pos_COM
        
# func3.2
def plot_poststates_pos(x, vmax=.2, f=None, ax=None):
    """Plot latent state place field"""
    if ax==None:
        f, ax = plt.subplots(figsize=(8,4))
    im = ax.imshow(x, vmax=vmax)
    ax.set_xlabel('Position bins')
    ax.set_ylabel('State')
    ax.set_title('Spatial Probability of Hidden States')
    divider = make_axes_locatable(ax)
    f.colorbar(im, cax=divider.append_axes("right", size=0.1,pad=0.05))

# func3.3
def plot_postprob(posterior_states, plst, lap_end, Distance, Decoded_pos, ax=None, t_st=400, t_duration=300):
    """plot posterior probability"""
    
    if ax==None:
        _, ax = plt.subplots(figsize=(15,3))
    ax.matshow(-posterior_states[:,plst].T, cmap = 'gray')
    ax.plot(Distance/Distance.max()*20, label='actual pos')
    ax.plot(Decoded_pos/Distance.max()*20, label='deocded pos')
    
    if lap_end!=[]:
        for i in lap_end:
            ax.plot([i,i],[0,plst.size-1], 'r--')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title('Posterior Probability of States')
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.set_xlim(t_st,t_st+t_duration)
    plt.legend(bbox_to_anchor=(.8, 1))
    
#    ax.matshow(-posterior_states[t_st:t_st+t_duration,plst].T, cmap = 'gray')
#    num = np.where((lap_end>t_st)&(lap_end<t_st+t_duration))[0]
#    lap_end0 = lap_end - t_st
#    ax.plot(Distance[t_st:t_st+t_duration]*10, label='position')
#    for i in num:
#        ax.plot([lap_end0[i],lap_end0[i]],[0,plst.size-1], 'r--')
#    ax.xaxis.set_ticks_position('bottom')
#    ax.set_title('Posterior Probability')
#    ax.set_xlabel('Time')
#    ax.set_ylabel('State')
#    plt.legend(bbox_to_anchor=(.8, 1))
    

# func3.4
def switch_rc(T, i, count):
    """Switch i_th column and row to count_th column and row"""
    t = T[:,i].copy()
    T[:,i]=T[:,count].copy()
    T[:,count] = t.copy()
    t = T[i,:].copy()
    T[i,:]=T[count,:].copy()
    T[count,:] = t.copy()
    return T

# func3.5
def order_transmat(transmat, startprob, order=None):
    """Reorder transmat according to given order or by maximum transiting prob"""
    T = transmat.copy()
    T = np.vstack((T,np.arange(T.shape[0])))
    
    if order.all()==None:
        i = np.argmax(startprob) # the ith neuron to put to count th
        count = 0
        T = switch_rc(T, i, count)
        for count in range(1,T.shape[1]):
            i = np.argmax(T[count-1,count:])+count
            T = switch_rc(T, i, count)
    else:
        for count in range(T.shape[1]):
            i = np.where(T[-1,:]==order[count])[0][0]
            T = switch_rc(T, i, count)
    
    Tr = T[:-1,:]
    order = T[-1,:].astype('int64')
        
    return Tr, order

# func3.6
def plot_transM(T, f=None, ax=None):
    if ax==None:
        f, ax = plt.subplots(figsize=(3,3))
    im = ax.matshow(T)
    ax.set_xlabel('state j')
    ax.set_ylabel('state i')
    ax.set_title('Transition Matrix A')
    ax.xaxis.set_ticks_position('bottom')
    divider = make_axes_locatable(ax)
    f.colorbar(im, cax=divider.append_axes("right", size=0.1,pad=0.05))

# func3.7
def plot_means(T, f=None, ax=None):
    if ax==None:
        f, ax = plt.subplots(figsize=(6,1.5))
    im = ax.matshow(T)
    ax.set_xlabel('Neuron')
    ax.set_ylabel('State')
    ax.set_title(r'Observation Matrix $\mu$')
    ax.xaxis.set_ticks_position('bottom')
    divider = make_axes_locatable(ax)
    f.colorbar(im, cax=divider.append_axes("right", size=0.1,pad=0.05))

# func3.8
def show_all_plots(origin, Trace, Distance, lap_end, lengths=None, sort = 'angle', t_st=400, t_duration=200, vmax=.2):
    x, plst, occ, posterior_states, pos_COM = comp_poststates_pos(origin, Trace, Distance, lengths = lengths, sort = sort)
    T, order = order_transmat(origin.transmat_, origin.startprob_, order=plst)
    parameters = {'axes.labelsize': 12,
              'axes.titlesize': 12,
              'legend.fontsize': 12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12 }
    plt.rcParams.update(parameters)
    fig = plt.figure(figsize=(12,9),constrained_layout=True)
    plt.subplots_adjust(wspace=0.25)
    gs = fig.add_gridspec(4, 14, wspace=2)
    ax1 = fig.add_subplot(gs[:2,:4])
    plot_transM(T, f=fig, ax=ax1)
    ax2 = fig.add_subplot(gs[:2,5:],sharey=ax1)
    plot_poststates_pos(x,vmax=vmax,f=fig,ax=ax2)
    ax3 = fig.add_subplot(gs[2,:])
    plot_means(origin.means_[order, :], f=fig, ax=ax3)
    ax4 = fig.add_subplot(gs[3,:])
    dec_state = np.argmax(posterior_states, axis = 1)
    Decoded_pos = pos_COM[dec_state]
    plot_postprob(posterior_states, plst, lap_end, Distance, Decoded_pos, ax=ax4,t_st=t_st, t_duration=t_duration)

#    plot_poststates_pos(x,vmax=vmax)
#    plot_postprob(posterior_states, plst, lap_end, Distance, t_st=t_st, t_duration=t_duration)
#
#    T, order = order_transmat(origin.transmat_, origin.startprob_, order=plst)
#    plot_transM(T)
#    plot_means(origin.means_[order, :])
    
    return x, plst, occ, T, posterior_states, pos_COM

# Compute accuracy
# func4.1
def comp_decoded_pos_acc(Distance, posterior_states, pos_COM, stateselect=[], mode = 'circle', print_c=False):
    
    if stateselect!=[]:
        posterior_states = posterior_states[:,stateselect]
        pos_COM = pos_COM[stateselect]
    
    dec_state = np.argmax(posterior_states, axis = 1)
    if print_c:
        _, c = np.unique(dec_state, return_counts=True)
        print(f'Decoded state counts: {c}')
    Decoded_pos = pos_COM[dec_state]
    
    # compute accuracy
    d = np.unique(Distance)
    m=d[1]+d[-1]
    
    if mode=='circle':
        theta = 2*np.pi*Distance/m
        theta_dc = 2*np.pi*Decoded_pos/m
        error = np.arccos(np.cos(theta - theta_dc))
        dev_angle = np.sqrt(np.mean(error**2))
        err_rate=dev_angle/(2*np.pi)
        dev=dev_angle*m/(2*np.pi)
        
    elif mode=='linear':
        error = Distance - Decoded_pos
        dev = np.sqrt(np.mean(error**2))
        err_rate = dev/m
        print(m)
        
    return err_rate, dev, Decoded_pos


# For animal on real world linear track
from scipy.signal import savgol_filter

def divide_direct(Distance, direct):
    difdis = np.diff(Distance)
    difd = savgol_filter(difdis,15,2)
    if direct=='fwd':
        flag=1
    elif direct=='bwd':
        flag=-1
    gtz = np.where(difd*flag>0)[0]
    e_fwd = np.where(np.diff(gtz)>1)[0]
    s_fwd = np.insert(e_fwd[:-1]+1, 0, 0)
    i = np.where((Distance[gtz[e_fwd]]-Distance[gtz[s_fwd]])*flag<100)[0]
    s_fwd = np.delete(s_fwd,i)
    e_fwd = np.delete(e_fwd,i)
    idx_fwd = np.concatenate(([np.arange(gtz[s_fwd[i]],gtz[e_fwd[i]]+1) for i in range(len(s_fwd))]))
    return idx_fwd


def comp_dec_pos(Distance, posterior_states, idx_fwd, n_components):
    
    Distance_fwd = Distance[idx_fwd]
    d = np.unique(Distance_fwd)
    posterior_states_fwd = posterior_states[idx_fwd,:]
    dec_state = np.argmax(posterior_states, axis = 1)

    x_fwd = np.zeros((n_components,len(d)))
    occ = np.zeros(len(d))
    for i in range(len(d)):
        x_fwd[:,i] = np.mean(posterior_states_fwd[np.where(Distance_fwd==d[i])[0],:],axis=0)
        occ[i] = np.where(Distance_fwd==d[i])[0].shape[0]

    for i in range(n_components):
        if np.sum(x_fwd[i,:])!=0:
            x_fwd[i,:] = x_fwd[i,:]/np.sum(x_fwd[i,:])

    pos_COM = d[np.argmax(x_fwd,axis=1)]
    plst_fwd = np.argsort(np.argmax(x_fwd,axis=1))
    x_fwd = x_fwd[plst_fwd,:]

    dec_pos_fwd = pos_COM[dec_state[idx_fwd]]
    
    return dec_pos_fwd, x_fwd, plst_fwd

def comp_decoded_dir(origin, Spike, Distance, plot=True):
    '''Returns:
    -----------------
        err_rate, idx_fwd, idx_bwd, Decoded_position, Distance_both, x_bwd, x_fwd'''
    idx_fwd = divide_direct(Distance,direct='fwd')
    idx_bwd = divide_direct(Distance,direct='bwd')
    
    _, posterior_states = origin.score_samples(Spike)
    dec_state = np.argmax(posterior_states, axis=1)
    print(np.unique(dec_state, return_counts=True))
    print(origin.n_components)
    dec_pos_bwd, x_bwd, plst_bwd = comp_dec_pos(Distance, posterior_states, idx_bwd, origin.n_components)
    dec_pos_fwd, x_fwd, plst_fwd = comp_dec_pos(Distance, posterior_states, idx_fwd, origin.n_components)

    idx_both = np.zeros_like(Distance)
    idx_both[idx_bwd] = -1
    idx_both[idx_fwd] = 1
    Distance_both = Distance[np.where(idx_both!=0)[0]]

    Decoded_position = np.zeros_like(Distance)
    Decoded_position[idx_fwd]=dec_pos_fwd
    Decoded_position[idx_bwd]=dec_pos_bwd
    Decoded_position = Decoded_position[np.where(idx_both!=0)[0]]
    
    d = np.unique(Distance_both)
    m=d[1]+d[-1]
    error = Distance_both - Decoded_position
    dev = np.sqrt(np.mean(error**2))
    err_rate = dev/m
    print(f'err_rate:{err_rate}')

    if plot:
        plt.figure(figsize=(12,2))
        plt.plot(Distance)
        plt.plot(idx_fwd,Distance[idx_fwd],'.',label='forward')
        plt.plot(idx_bwd,Distance[idx_bwd],'.',label='backward')
        plt.ylabel('position')
        plt.xlabel('time')
        plt.legend()
        
        plt.figure()
        CalHMM.plot_postprob(posterior_states[idx_fwd,:], plst_fwd, lap_end=[], Distance=Distance[idx_fwd], Decoded_pos=dec_pos_fwd, ax=None, t_st=0, t_duration=len(idx_fwd))
        plt.figure()
        CalHMM.plot_postprob(posterior_states[idx_bwd,:], plst_bwd, lap_end=[], Distance=Distance[idx_bwd], Decoded_pos=dec_pos_bwd, ax=None, t_st=0, t_duration=len(idx_bwd))

    return err_rate, idx_fwd, idx_bwd, Decoded_position, Distance_both, x_bwd, x_fwd
