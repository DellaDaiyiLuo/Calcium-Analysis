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
    def __init__(self, folder, timebin, name, n_chunks, k=2, win=1):
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
            self.Trace0 = np.load(file_spk)
        else:
            self.Trace0 = data['Trace']
        
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

    def separate_chunk_set(self):
        '''Separate the data into chunks and sets within chunk.'''

        train_lap = []
        train_lap_len = []
        train_idx = []

        for i_chunk in range(self.n_chunks+1-self.win):
            r = np.random.choice(self.n_laps_perchunk,size=self.n_laps_perchunk,replace=False)
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
    
    
        
    def HMMSetCrossValid(self, text = True, plot = False):
        '''Hidden Markov Model: Cross validate between chunks of data
        Parameters
        -------------
        folder: directory to time-binned behaviour and spike data.
        n_chunks: number of chunks of trace/spike data.
        n_iter: number of times training HMM.
        k: number of sets within one chunk.
        name: str, 'Spike' or 'Trace'. '''

        train_lap, train_lap_len, train_idx = self.separate_chunk_set()
        logprob_train = self.logprob_train.copy()
        logprob_test = self.logprob_test.copy()
        errrate_train = self.errrate_train.copy()
        errrate_test = self.errrate_test.copy()

        for i_chunk in range(len(self.lap_range)):
            chunkselect = np.delete(np.arange(len(self.lap_range)), obj=i_chunk).astype('int')
            print(f'Train chunk: {i_chunk}')

            for i_set in range(self.k):

                #-------- Train --------#
                Trace_train = self.Trace0[train_idx[i_chunk][i_set],:]
                Distance_train = self.Distance0[train_idx[i_chunk][i_set]]
                len_train = train_lap_len[i_chunk][i_set]

                print(f'Laps in train set: {train_lap[i_chunk][i_set]}')
                if text:
                    print(f'Train set: chunk {i_chunk} set {i_set}\n')

                if self.name=='Spike':
                    origin = hmmlearn.hmm.PoissonHMM(n_components=20, verbose=True)
                elif self.name=='Trace':
                    origin = hmmlearn.hmm.GaussianHMM(n_components=20)

                origin.fit(Trace_train, len_train)
                if plot & self.k==1:
                    CalHMM.plot_transM(origin.transmat_)
                _, plst_train, _, _, pos_COM_train = CalHMM.comp_poststates_pos(origin, Trace_train, Distance_train, lengths=len_train)
                logprob, err_rate, _ = test_HMM(origin, Trace_train, Distance_train, pos_COM_train, len_train, plot=plot, plst_train=plst_train)

                nest=logprob_train[i_chunk]
                nest.append(logprob/len_train.sum()) # record logprob of train set
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
                    logprob, err_rate, _ = test_HMM(origin, self.Trace0[train_idx[i_chunk_test][i_test],:], self.Distance0[train_idx[i_chunk_test][i_test]], pos_COM_train, train_lap_len[i_chunk_test][i_test], plot=plot, plst_train=plst_train)
                    nest1.append(logprob/train_lap_len[i_chunk_test][i_test].sum())
                    nest2.append(err_rate)

                for i_chunk_test in chunkselect:
                    nest1 = logprob_test[i_chunk][i_chunk_test]
                    nest2 = errrate_test[i_chunk][i_chunk_test]
                    for i_test in range(self.k):
                        if text:
                            print(f'Test set: chunk {i_chunk_test} set {i_test}')
                        logprob, err_rate, _ = test_HMM(origin, self.Trace0[train_idx[i_chunk_test][i_test],:], self.Distance0[train_idx[i_chunk_test][i_test]], pos_COM_train, train_lap_len[i_chunk_test][i_test], plot=plot, plst_train=plst_train)
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
                logprob_test_mean[i,j]=np.array(self.logprob_test[i][j]).mean()
                errrate_test_mean[i,j]=np.array(self.errrate_test[i][j]).mean()

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

            train_lap, train_lap_len, train_idx = self.separate_chunk_set()
            logprob_train = self.logprob_train.copy()
            logprob_test = self.logprob_test.copy()
            errrate_train = self.errrate_train.copy()
            errrate_test = self.errrate_test.copy()
            
            d = np.unique(self.Distance0)
            dbins = d.shape[0]

            for i_chunk in range(len(self.lap_range)):
                chunkselect = np.delete(np.arange(len(self.lap_range)), obj=i_chunk).astype('int')
                print(f'Train chunk: {i_chunk}')

                for i_set in range(self.k):

                    #-------- Train --------#
                    Trace_train = self.Trace0[train_idx[i_chunk][i_set],:]
                    Distance_train = self.Distance0[train_idx[i_chunk][i_set]]
                    mean_cal, std_cal, not_once, once = EstimateModelParam(Trace_train, Distance_train, d, dbins)
                    err_rate, logprob = test_Model(Trace_train, Distance_train, d, dbins, mean_cal, std_cal, not_once, once, self.name, plot)
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
                        err_rate, logprob = test_Model(self.Trace0[train_idx[i_chunk_test][i_test],:], self.Distance0[train_idx[i_chunk_test][i_test]], d, dbins, mean_cal, std_cal, not_once, once, self.name, plot)
                        nest1.append(logprob)
                        nest2.append(err_rate)

                    for i_chunk_test in chunkselect:
                        nest1 = logprob_test[i_chunk][i_chunk_test]
                        nest2 = errrate_test[i_chunk][i_chunk_test]
                        for i_test in range(self.k):
                            if text:
                                print(f'Test set: chunk {i_chunk_test} set {i_test}')
                            err_rate, logprob = test_Model(self.Trace0[train_idx[i_chunk_test][i_test],:], self.Distance0[train_idx[i_chunk_test][i_test]], d, dbins, mean_cal, std_cal, not_once, once, self.name, plot)
                            nest1.append(logprob)
                            nest2.append(err_rate)

            self.logprob_train = logprob_train.copy()
            self.logprob_test = logprob_test.copy()
            self.errrate_train = errrate_train.copy()
            self.errrate_test = errrate_test.copy()
            self.iter += 1
            print(f'============= {self.iter} =============')

        

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
        if (np.where(Distance==d[i])[0].shape[0])>1:
            not_once.append(i)
            mean_cal[i,:] = np.mean(Trace[Distance==d[i],:],axis=0)
            std_cal[i,:] = np.std(Trace[Distance==d[i],:],axis=0)
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
            y_min = np.sort(np.unique(y))[1]
            y += y_min # avoid zero pdf value
            decoded_pos_prob[i,:] = np.sum(np.log(y),axis=1) # add up across all neurons
    elif name=='Spike':
        for i in not_once:
            y = poisson.pmf(Trace,mu=mean_cal[i,:])
            y_min = np.sort(np.unique(y))[1]
            y += y_min # avoid zero pdf value
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
    return err_rate, logprob
