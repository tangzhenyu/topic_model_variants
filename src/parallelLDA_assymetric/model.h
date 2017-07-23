#ifndef	_MODEL_H
#define	_MODEL_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>

#include <atomic> 
#include <future>
#include <mutex>
#include <thread>

#include "../commons/circqueue.h"
#include "../commons/dataset.h"
#include "../commons/my_rand.h"
#include "../commons/spvector.h"
#include "../commons/utils.h"

class model {
    
public:
    /****** constructor/destructor ******/
    model();
    virtual ~model();

    /****** interacting functions ******/
    static model* init(int, char**);// initialize the model randomly
    int train();					// train LDA using prescribed algorithm on training data
    int test();						// test LDA according to specified method

protected:
    /****** Enums for testing type model status  ******/
    enum {							// testing types:
            INVALID,					// model not initialised
            NO_TEST,					// do not report any likelihood
            SEPARATE_TEST					// report likelihood on a held-out testing set
    } testing_type;

    /****** DATA ******/
    dataset *trngdata;				// training dataset
    dataset *testdata;				// test dataset

    std::map<unsigned, std::string> id2word;			// word map [int => string]
    std::map<std::string, unsigned> word2id;			// word map [string => int]

    /****** Model Parameters ******/
    unsigned M;							// Number of documents
    unsigned V; 							// Number of words in dictionary
    unsigned short K; 						// Number of topics

    /****** Model Hyper-Parameters ******/
    double alpha, alphaK;					// per document Topic proportions dirichlet prior
	double *alpha_topic;
	double alpha_topic_sum;
	int ** topicDocCounts;
	int * topicDocCountsNorm;
	int ** topicDocHist;
	int * topicDocNormHist;

    double beta, Vbeta;				// Dirichlet language model
	double *beta_word;
	double beta_word_sum;
	int ** wordTopicCounts;
	int * wordTopicCountsNorm;
	int ** wordTopicHist;
	int * wordTopicNormHist;
	
	int * wordCounts;
	int ndMax;
    /****** Model variables ******/
    unsigned short ** z;						// topic assignment for each word
    unsigned ** n_wk;					// number of times word w assigned to topic k
    //std::atomic<int> ** n_wk;		// if on target system atomics are lock_free()
    std::vector< spvector > n_mks;  //sparse representation of n_mk: number of words assigned to topic k in document m
    //std::vector< std::vector< std::pair<int, int> > > n_mks; //sparse representation of n_mk: number of words assigned to topic k in document m
    unsigned * n_k;						// number of words assigned to topic k = sum_w n_wk = sum_m n_mk
    //std::atomic<int> * n_k;		// if on target system atomics are lock_free()
	//unsigned * signals;
	//unsigned signals_sum;

    /****** Initialisation aux ******/
    int read_data();				// Read training (and testing) data
    int parse_args(std::vector<std::string> arguments);	// parse command line inputs
	void initializeHists_doc(int ndMax);
	void initializeHists_word(int *wordCounts);
	void init_beta_word();
	void init_alpha_topic();
	void optimizeParam_alpha(int numIters);
	void optimizeParam_beta(int numIters);
    /****** Training aux ******/
    unsigned n_iters;	 				// Number of Gibbs sampling iterations
    unsigned n_save;			 			// Number of iters in between saving
    unsigned n_topWords; 				// Number of top words to be printed per topic
    int init_train();					// init for training
    virtual int specific_init() { return 0; }	// if sampling algo need some specific inits
    virtual int sampling(unsigned) { return 0; }		// sampling on machine i outsourced to children

	virtual int updater_alpha_topic(unsigned numItns)
	{
		int * limits = new int[K];
//		memset(limits, -1, sizeof(limits));
		for(unsigned k = 0 ; k < K ; ++k)
		{
			limits[k] = -1;
		}

		for (int j=0; j < K; j++) 
		{
			for(int jj = 0 ; jj < (ndMax + 1) ; ++jj)
	        {   
	            topicDocHist[j][jj]=0;
	        }

			for (int d=0; d<M; d++)//for all docs
			{
		        topicDocHist[j][topicDocCounts[j][d]]++;
			}
			for (int n=0; n < (ndMax + 1); n++)
			{
			    if (topicDocHist[j][n] > 0)
				    limits[j] = n;//last n
			}
		}		
		for (int i=0; i<numItns; i++) 
		{
			double denominator = 0.0;
			double currentDigamma = 0.0;

			for (int n = 1; n < ndMax + 1 ; n++) 
			{
		        currentDigamma += 1.0 / (alpha_topic_sum + n - 1);
			    denominator += topicDocNormHist[n] * currentDigamma;
			}

		    alpha_topic_sum = 0.0;

			for (int j=0; j<K; j++) {
				int limit = limits[j];

      	        double oldParam = alpha_topic[j];
	            alpha_topic[j] = 0.0;
		        currentDigamma = 0;
				
				for (int n=1; n<=limit; n++) {
					currentDigamma += 1.0 / (oldParam + n - 1);
					alpha_topic[j] += topicDocHist[j][n] * currentDigamma;
				}
				alpha_topic[j] *= oldParam / denominator; //update alpha of topic j
				if (alpha_topic[j] == 0.0)
        	        alpha_topic[j] = 1e-10;
				alpha_topic_sum += alpha_topic[j];
			}
		}
	}

	virtual int incrementCounts_alpha_topic(int j, int d) {
	    topicDocCounts[j][d]++;
	    topicDocCountsNorm[d]++;
	}
	virtual int decrementCounts_alpha_topic(int j, int d) {
	    topicDocCounts[j][d]--;
        topicDocCountsNorm[d]--;
	}

	virtual int updater_beta_word(unsigned numItns)
	{
		int njMax = wordTopicCountsNorm[0];
		for(unsigned j = 1 ; j < K ; ++j)
		{
			 if (wordTopicCountsNorm[j] > njMax)
			 {
				njMax = wordTopicCountsNorm[j];
			 }
		}
		wordTopicNormHist = new int[njMax + 1];
		for(unsigned j = 0 ; j < K ; ++j)
		{
			wordTopicNormHist[wordTopicCountsNorm[j]]++;
		}

		for (unsigned i=0; i<numItns; i++) 
		{
			double denominator = 0.0;
            double currentDigamma = 0.0;

			for (unsigned n=1; n < (njMax + 1); n++) {
				currentDigamma += 1.0 / (beta_word_sum + n - 1);
			    denominator += wordTopicNormHist[n] * currentDigamma;
			}
			beta_word_sum = 0.0;
			//exit(-1);
			for (unsigned v=0; v < V; v++) {
				double oldParam = beta_word[v];
				beta_word[v] = 0.0;
		        currentDigamma = 0;	
				
				unsigned wordTopicHist_length=wordCounts[v] + 1;
				std::cout<<v<<"<==>"<<wordCounts[v]<<std::endl;
				for ( int n = 1 ; n < wordTopicHist_length ; n++ ) {
					currentDigamma += 1.0 / (oldParam + n - 1);
					std::cout<<"currentDigamma:"<<currentDigamma<<std::endl;
				    beta_word[v] += wordTopicHist[v][n] * currentDigamma;
					std::cout<<"beta_word:"<<v<<"=="<<beta_word[v]<<std::endl;
				}
				beta_word[v] *= oldParam / denominator;//update beta of word w
				if (beta_word[v] == 0.0)
				{
					beta_word[v] = 1e-10;
				}
				beta_word_sum += beta_word[v];
				//std::cout<<"v:"<<v<<"=="<<v<<std::endl;
			}
			exit(-1);
		}
	}

	virtual int incrementCounts_beta_word(int w, int j, bool updateHist) 
	{
		int oldCount = wordTopicCounts[w][j]++;
	    wordTopicCountsNorm[j]++;

		if (updateHist) {
			wordTopicHist[w][oldCount]--;
			//wordTopicHist[w][oldCount + 1]++;
		}
	}

	virtual int decrementCounts_beta_word(int w, int j, bool updateHist) 
	{
		int oldCount = wordTopicCounts[w][j]--;
	    wordTopicCountsNorm[j]--;

		if (updateHist) {
			wordTopicHist[w][oldCount]--;
            //wordTopicHist[w][oldCount - 1]++;
		}
	}

    virtual int updater(unsigned i)					// updating sufficient statistics, can be outsourced to children
    {
        do
        {
            for (unsigned tn = 0; tn<nst; ++tn)
            {
                if (!(cbuff[i*nst + tn].empty()))
                {
                    delta temp = cbuff[i*nst + tn].front();
                    cbuff[i*nst + tn].pop();
                    --n_wk[temp.word][temp.old_topic];
                    ++n_wk[temp.word][temp.new_topic];
                    --n_k[temp.old_topic];
                    ++n_k[temp.new_topic];
                    //n_wk[temp.word][temp.old_topic].fetch_add(-1);
                    //n_wk[temp.word][temp.new_topic].fetch_add(+1);
                    //n_k[temp.old_topic].fetch_add(-1);
                    //n_k[temp.new_topic].fetch_add(+1);
                }
            }
        } while (!inf_stop); //(!done[i]);

        return 0;
    }			

    /****** Testing aux ******/
    unsigned test_n_iters;
    unsigned test_M;
    unsigned short ** test_z;
    unsigned ** test_n_mk;
    int init_test();				// init for testing
    int async_test();				// test asynchronously to the training procedure

    /****** Concurency parameters ******/
    unsigned nst;						// number of sampling threads
    unsigned ntt;						// number of table updating threads
    virtual unsigned num_table_threads(unsigned nt) { return nt/16; }
    volatile bool inf_stop;					// flag for stopping inference
    unsigned *current_iter;					// current iteration of some thread
    std::mutex t_mtx;				// lock for n_k
    std::mutex* mtx;				// lock for data-structures involving n_wk
    struct delta					// sufficient statistic update message
    {
        unsigned word;
        unsigned short old_topic;
        unsigned short new_topic;

        delta()
        {	}

        delta(unsigned a, unsigned short b, unsigned short c) : word(a), old_topic(b), new_topic(c)
        {	}
    };
    circular_queue<delta> *cbuff;	// buffer for messages NST*NTT

    /****** Performance computations ******/
    std::vector<double> time_ellapsed; // time ellapsed after each iteration
    std::vector<double> likelihood; // likelihood after each iteration
    double newllhw() const;			// per word log-likelihood for new (unseen) data based on the estimated LDA model
    double llhw() const;			// per word log-likelihood for training data based on the estimated LDA model

    /****** File and Folder Paths ******/
    std::string name;				// dataset name
    std::string ddir;				// data directory
    std::string mdir;				// model directory
    std::string dfile;				// train data file    
    std::string tfile;				// test data file
    std::string vfile;				// vocabulary file

    /****** save LDA model to files ******/
    int save_model(unsigned iter) const;						// save model: call each of the following:		
    int save_model_time(std::string filename) const;	// model_name.time: time at which statistics calculated
    int save_model_llh(std::string filename) const;		// model_name.llh: Per word likelihood on held out documents
    int save_model_params(std::string filename) const;	// model_name.params: containing other parameters of the model (alpha, beta, M, V, K)
    int save_model_topWords(std::string filename) const;// model_name.twords: Top words in each top
    int save_model_phi(std::string filename) const;		// model_name.phi: topic-word distributions
	int save_model_tassgin(std::string filename) const;
};

#endif
