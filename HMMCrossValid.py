import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
import CalHMM
import warnings
warnings.filterwarnings('ignore')
import hmmlearn.hmm
from scipy.stats import norm, poisson


# auxiliary funtions
class datainfo():
    def __init__(self, folder, timebin, name, n_chunks, k=2, win=1, normalize=False):
        self.iter = 0
        self.name = name
        self.k = k
        self.win = win
        self.folder = folder
        file_main = f'{folder}/timebin_{timebin}.npz'
        data = np.load(file_main)
        self.Distance0 = data['Distance']
        self.lap_end0 = data['lap_end']
        self.idx = data['idx']
        if name=='Spike':
            file_spk = f'{folder}/timebin_{timebin}_spks.npy'
            Trace0 = np.load(file_spk)
            if normalize:
                maxvalue = np.where(Trace0.max(axis=0)==0, 1, Trace0.max(axis=0))
                self.Trace0 = np.round(Trace0/maxvalue[np.newaxis,:]*Trace0.max())
            else:
                self.Trace0 = Trace0
        else:
            Trace0 = data['Trace']
            if normalize:
                maxvalue = np.where(Trace0.max(axis=0)==0, 1, Trace0.max(axis=0))
                self.Trace0 = Trace0/maxvalue[np.newaxis,:]*Trace0.max()
            else:
                self.Trace0 = Trace0
            
        self.n_chunks = n_chunks
        self.n_laps_perchunk = int((self.lap_end0.shape[0]-1)/n_chunks)*win
        lap_range = []
        n_laps_perstep = int((self.lap_end0.shape[0]-1)/n_chunks)
        for i_chunk in range(n_chunks+1-win):
            lap_range.append([n_laps_perstep*i_chunk, n_laps_perstep*(i_chunk+win)])
        self.lap_range = lap_range
    
        n = len(lap_range)
        self.logprob_train = [[] for i in range(n)]
        self.errrate_train = [[] for i in range(n)]
        self.logprob_test = [[[] for i in range(n)] for j in range(n)]
        self.errrate_test = [[[] for i in range(n)] for j in range(n)]

    def separate_chunk_set(self, random = True):
        '''Separate the data into chunks and sets within chunk.'''

        train_lap = []
        train_lap_len = []
        train_idx = []

        for i_chunk in range(self.n_chunks+1-self.win):
            if random:
                r = np.random.choice(self.n_laps_perchunk,size=self.n_laps_perchunk,replace=False)
            else:
                r = np.arange(self.n_laps_perchunk)
                
            train_lap_chunk = []
            train_lap_len_chunk = []
            train_idx_chunk = []
            for i in range(self.k):
                train_lap_chunk.append(np.arange(self.lap_range[i_chunk][0], self.lap_range[i_chunk][1])[r[i:self.n_laps_perchunk:self.k]])
                train_lap_len_chunk.append(np.array([self.lap_end0[lap+1] - self.lap_end0[lap] for lap in train_lap_chunk[i]]))
                train_idx_chunk.append(np.concatenate([np.arange(self.lap_end0[lap],self.lap_end0[lap+1]) for lap in train_lap_chunk[i]]))
            train_lap.append(train_lap_chunk)
            train_lap_len.append(train_lap_len_chunk)
            train_idx.append(train_idx_chunk)

        return train_lap, train_lap_len, train_idx
    
    
        
    def HMMSetCrossValid(self, text = True, plot = False, random = True, shuffle = None):
        '''Hidden Markov Model: Cross validate between chunks of data
        Parameters
        -------------
        folder: directory to time-binned behaviour and spike data.
        n_chunks: number of chunks of trace/spike data.
        n_iter: number of times training HMM.
        k: number of sets within one chunk.
        name: str, 'Spike' or 'Trace'.'''

        train_lap, train_lap_len, train_idx = self.separate_chunk_set(random = random)
        logprob_train = self.logprob_train.copy()
        logprob_test = self.logprob_test.copy()
        errrate_train = self.errrate_train.copy()
        errrate_test = self.errrate_test.copy()

        self.train_lap = train_lap
        self.train_lap_len = train_lap_len
        self.train_idx = train_idx
        
        for i_chunk in range(len(self.lap_range)):
            chunkselect = np.delete(np.arange(len(self.lap_range)), obj=i_chunk).astype('int')
            print(f'Train chunk: {i_chunk}')

            for i_set in range(self.k):

                #-------- Train --------#
                setselect = np.delete(np.arange(self.k), obj=i_set).astype('int')
                set_idx = np.concatenate([train_idx[i_chunk][f] for f in setselect])
                
                self.Trace_train = self.Trace0[set_idx,:]
                self.Distance_train = self.Distance0[set_idx]
                self.len_train = np.concatenate([train_lap_len[i_chunk][f] for f in setselect])

                fire_neuron = np.where(self.Trace_train.sum(axis=0)!=0)[0]
                self.Trace_train = self.Trace_train[:,fire_neuron]
                
                print(f'Laps in train set: {np.concatenate([train_lap[i_chunk][f] for f in setselect])}')
                if text:
                    print(f'Train set: chunk {i_chunk} set {setselect}\n')
                    

                logprob_n = np.nan
                while np.isnan(logprob_n):
                    if self.name=='Spike':
                        origin = hmmlearn.hmm.PoissonHMM(n_components=20, verbose=True, n_iter=20)
                    elif self.name=='Trace':
                        origin = hmmlearn.hmm.GaussianHMM(n_components=20, verbose=True, n_iter=20)

                    if shuffle=='trsn':
                        Trace_train = CalHMM.get_trsn_shuffle(self.Trace_train)
                    elif shuffle=='ts':
                        Trace_train = CalHMM.get_time_shuffle(self.Trace_train)
                    else:
                        Trace_train = self.Trace_train
                    if plot:
                        plt.figure(figsize=(15,3))
                        plt.plot(self.Trace_train[:,10])
                        plt.plot(Trace_train[:,10])
                        plt.show()
                        
                    origin.fit(Trace_train, self.len_train)
                    
                    if plot & self.k==1:
                        CalHMM.plot_transM(origin.transmat_)
                    _, plst_train, _, _, pos_COM_train = CalHMM.comp_poststates_pos(origin, self.Trace_train, self.Distance_train, lengths=self.len_train)
                    logprob, err_rate, _ = test_HMM(origin, self.Trace_train, self.Distance_train, pos_COM_train, self.len_train, plot=plot, plst_train=plst_train)
                    logprob_n = logprob/self.len_train.sum()
                    print(f'logprob_n: {logprob_n}, length: {self.len_train.sum()}')


                nest=logprob_train[i_chunk]
                nest.append(logprob/self.len_train.sum()) # record logprob of train set
                nest=errrate_train[i_chunk]
                nest.append(err_rate)


                #--------- Test ---------#
                i_chunk_test = i_chunk
                nest1 = logprob_test[i_chunk][i_chunk_test]
                nest2 = errrate_test[i_chunk][i_chunk_test]
                
                if text:
                    print(f'Test set: chunk {i_chunk_test} set {i_set}')
                    print(f'Laps in test set: {train_lap[i_chunk_test][i_set]}')
                Trace_test = self.Trace0[train_idx[i_chunk_test][i_set],:]
                logprob, err_rate, _ = test_HMM(origin, Trace_test[:,fire_neuron], self.Distance0[train_idx[i_chunk_test][i_set]], pos_COM_train, train_lap_len[i_chunk_test][i_set], plot=plot, plst_train=plst_train)
                nest1.append(logprob/train_lap_len[i_chunk_test][i_set].sum())
                nest2.append(err_rate)

                for i_chunk_test in chunkselect:
                    nest1 = logprob_test[i_chunk][i_chunk_test]
                    nest2 = errrate_test[i_chunk][i_chunk_test]
                    for i_test in range(self.k):
                        if text:
                            print(f'Test set: chunk {i_chunk_test} set {i_test}')
                            print(f'Laps in test set: {train_lap[i_chunk_test][i_test]}')
                        Trace_test = self.Trace0[train_idx[i_chunk_test][i_test],:]
                        logprob, err_rate, _ = test_HMM(origin, Trace_test[:,fire_neuron], self.Distance0[train_idx[i_chunk_test][i_test]], pos_COM_train, train_lap_len[i_chunk_test][i_test], plot=plot, plst_train=plst_train)
                        nest1.append(logprob/train_lap_len[i_chunk_test][i_test].sum())
                        nest2.append(err_rate)

        self.logprob_train = logprob_train.copy()
        self.logprob_test = logprob_test.copy()
        self.errrate_train = errrate_train.copy()
        self.errrate_test = errrate_test.copy()
        self.iter += 1
        print(f'============= {self.iter} =============')

    
    def estimate_mean(self):
        n_chunks = len(self.lap_range)
        logprob_test_mean = np.zeros((n_chunks, n_chunks))
        errrate_test_mean = np.zeros((n_chunks, n_chunks))
        for i in range(n_chunks):
            for j in range(n_chunks):
                logprob_test_mean[i,j]=mean_exc_outlier(self.logprob_test[i][j])
                # np.array(self.logprob_test[i][j]).mean()
                errrate_test_mean[i,j]=mean_exc_outlier(self.errrate_test[i][j])
                # np.array(self.errrate_test[i][j]).mean()

        print(f'logprob_test_mean: {logprob_test_mean}\nerrrate_test_mean: {errrate_test_mean}')
        self.logprob_test_mean = logprob_test_mean
        self.errrate_test_mean = errrate_test_mean
        
        
        
    def GaussianSetCrossValid(self, text = True, plot = False):
            '''Gaussian model: Cross validate between chunks of data
            Parameters
            -------------
            folder: directory to time-binned behaviour and spike data.
            n_chunks: number of chunks of trace/spike data.
            n_iter: number of times training HMM.
            k: number of sets within one chunk.
            name: str, 'Spike' or 'Trace'. '''

            self.flag = 0
            train_lap, train_lap_len, train_idx = self.separate_chunk_set()
            logprob_train = self.logprob_train.copy()
            logprob_test = self.logprob_test.copy()
            errrate_train = self.errrate_train.copy()
            errrate_test = self.errrate_test.copy()
            
            self.train_lap = train_lap
            self.train_lap_len = train_lap_len
            self.train_idx = train_idx
            
            d = np.unique(self.Distance0)
            dbins = d.shape[0]

            for i_chunk in range(len(self.lap_range)):
                chunkselect = np.delete(np.arange(len(self.lap_range)), obj=i_chunk).astype('int')
                print(f'Train chunk: {i_chunk}')

                for i_set in range(self.k):

                    #-------- Train --------#
                    self.Trace_train = self.Trace0[train_idx[i_chunk][i_set],:]
                    self.Distance_train = self.Distance0[train_idx[i_chunk][i_set]]
                                        
                    fire_neuron = np.where(self.Trace_train.sum(axis=0)!=0)[0]
                    self.Trace_train = self.Trace_train[:,fire_neuron]
                    print(len(fire_neuron))
                    
                    mean_cal, std_cal, not_once, once = EstimateModelParam(self.Trace_train, self.Distance_train, d, dbins)
                    
                    
                    err_rate, logprob, decoded_pos_prob, Decoded_pos = test_Model(self.Trace_train, self.Distance_train, d, dbins, mean_cal, std_cal, not_once, once, self.name, plot)
                    
                    if np.std(Decoded_pos)<0.1:
                        self.flag = 1
                        break
                    
                    if plot:
                        plt.matshow(mean_cal)
                        plt.colorbar()
                        plt.show()
                        
                    nest=logprob_train[i_chunk]
                    nest.append(logprob) # record logprob of train set
                    nest=errrate_train[i_chunk]
                    nest.append(err_rate)


                    #--------- Test ---------#
                    setselect = np.delete(np.arange(self.k), obj=i_set).astype('int')
                    i_chunk_test = i_chunk
                    nest1 = logprob_test[i_chunk][i_chunk_test]
                    nest2 = errrate_test[i_chunk][i_chunk_test]
                    for i_test in setselect:
                        if text:
                            print(f'Test set: chunk {i_chunk_test} set {i_test}')
                        Trace_test = self.Trace0[train_idx[i_chunk_test][i_test],:]
                        err_rate, logprob, decoded_pos_prob, Decoded_pos = test_Model(Trace_test[:, fire_neuron], self.Distance0[train_idx[i_chunk_test][i_test]], d, dbins, mean_cal, std_cal, not_once, once, self.name, plot)
                        nest1.append(logprob)
                        nest2.append(err_rate)

                    for i_chunk_test in chunkselect:
                        nest1 = logprob_test[i_chunk][i_chunk_test]
                        nest2 = errrate_test[i_chunk][i_chunk_test]
                        for i_test in range(self.k):
                            if text:
                                print(f'Test set: chunk {i_chunk_test} set {i_test}')
                            Trace_test = self.Trace0[train_idx[i_chunk_test][i_test],:]
                            err_rate, logprob, decoded_pos_prob, Decoded_pos = test_Model(Trace_test[:, fire_neuron], self.Distance0[train_idx[i_chunk_test][i_test]], d, dbins, mean_cal, std_cal, not_once, once, self.name, plot)
                            nest1.append(logprob)
                            nest2.append(err_rate)
                            
                if self.flag==1:
                    break

            self.logprob_train = logprob_train.copy()
            self.logprob_test = logprob_test.copy()
            self.errrate_train = errrate_train.copy()
            self.errrate_test = errrate_test.copy()
            self.iter += 1
            print(f'============= {self.iter} =============')

        
def mean_exc_outlier(a):
    a = np.array(a)
    Q1,Q3 = np.percentile(a , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    m = np.mean(a[(a>lower_range)&(a<upper_range)])
    return m

def test_HMM(origin, Trace, Distance, pos_COM_train, lengths=None, plot=False, plst_train=[]):
    logprob, posterior_states = origin.score_samples(Trace, lengths)
    err_rate, _, Decoded_pos = CalHMM.comp_decoded_pos_acc(Distance, posterior_states, pos_COM_train)
    if plot:
        CalHMM.plot_postprob(posterior_states, plst_train, np.cumsum(lengths), Distance, ax=None, t_st=0, t_duration=Distance.shape[0])
        plt.show()
    return logprob, err_rate, Decoded_pos

def EstimateModelParam(Trace, Distance, d, dbins):
    mean_cal = np.zeros((dbins,Trace.shape[1])) # #(position bins)*#(neurons)
    std_cal = np.zeros((dbins,Trace.shape[1]))

    not_once = []
    once = []
    for i in range(dbins): # loop for each position bin
        if (np.where(Distance==d[i])[0].shape[0])>0:
            not_once.append(i)
            mean_cal[i,:] = np.mean(Trace[Distance==d[i],:],axis=0)
            std_cal[i,:] = np.std(Trace[Distance==d[i],:],axis=0)
            if (np.where(Distance==d[i])[0].shape[0])==1:
                once.append(i)
        else:
            once.append(i)

    not_once = np.array(not_once)
    once = np.array(once)
    return mean_cal, std_cal, not_once, once
    

def test_Model(Trace, Distance, d, dbins, mean_cal, std_cal, not_once, once, name, plot):
    decoded_pos_prob = np.zeros((dbins, Trace.shape[0]))
    if name=='Trace':
        for i in not_once:
            y = norm.pdf(Trace,loc=mean_cal[i,:],scale=std_cal[i,:])
#            y_min = np.sort(np.unique(y))[1]
            y += 1e-100 # avoid zero pdf value
            decoded_pos_prob[i,:] = np.sum(np.log(y),axis=1) # add up across all neurons
    elif name=='Spike':
        for i in not_once:
            y = poisson.pmf(Trace,mu=mean_cal[i,:])
#            y_min = np.sort(np.unique(y))[1]
            y += 1e-100 #y_min # avoid zero pdf value
            decoded_pos_prob[i,:] = np.sum(np.log(y),axis=1) # add up across all neurons
        
    for j in once:
        decoded_pos_prob[j,:] = np.min(decoded_pos_prob[i,:]) # postion occurs only once so the prob would be 1, too large
        
    err_rate, dev, Decoded_pos=CalHMM.comp_decoded_pos_acc(Distance, decoded_pos_prob.T, d)
    pp = decoded_pos_prob[np.argmax(decoded_pos_prob, axis = 0),np.arange(len(Distance))]
    logprob = np.mean(pp)
    
    if plot:
        plt.figure()
        plt.plot(Distance,label='real')
        plt.plot(Decoded_pos, label='decoded')
        plt.legend()
        plt.show()
    return err_rate, logprob, decoded_pos_prob, Decoded_pos
