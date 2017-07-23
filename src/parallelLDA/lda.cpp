#include "lda.h"
#include <mmintrin.h>

#define MH_STEPS 2

int simpleLDA::specific_init()
{
    return 0;
}

int simpleLDA::sampling(unsigned i)
{
    xorshift128plus rng_;
	
    double * p = new double[K]; // temp variable for sampling
    unsigned *nd_m = new unsigned[K];
    unsigned short *rev_mapper = new unsigned short[K];
    for (unsigned short k = 0; k < K; ++k)
    {
        nd_m[k] = 0;
        rev_mapper[k] = K;
    }
    std::chrono::high_resolution_clock::time_point ts, tn;

    unsigned iter; current_iter = &iter;	
    for (iter = 0; iter < n_iters; ++iter)
    {
        ts = std::chrono::high_resolution_clock::now();
        // for each document of worker i
        for (unsigned m = i; m < M; m+=nst)
        {
            unsigned short kc = 0;
            for (const auto& k : n_mks[m])
            {
                nd_m[k.idx] = k.val;
                rev_mapper[k.idx] = kc++;
            }
            for (unsigned n = 0; n < trngdata->docs[m]->length; ++n)
            {
                unsigned w = trngdata->docs[m]->words[n];

                // remove z_ij from the count variables
                unsigned short topic = z[m][n]; unsigned short old_topic = topic;
                --nd_m[topic];
                n_mks[m].decrement(rev_mapper[topic]);

                // do multinomial sampling via cumulative method
                double temp = 0;
                for (unsigned short k = 0; k < K; k++)
                {
                    temp += (n_wk[w][k] + beta) / (n_k[k] + Vbeta) * (nd_m[k] + alphaK);
                    p[k] = temp;
                }

                // scaled sample because of unnormalized p[]
                double u = rng_.rand_double() * temp;

                // Do a binary search instead!
                topic = std::lower_bound(p, p + K, u) - p;

                // add newly estimated z_i to count variables
                if (topic!=old_topic)
                {
                    if(nd_m[topic] == 0)
                        rev_mapper[topic] = n_mks[m].push_back(topic, 1);
                    else
                        n_mks[m].increment(rev_mapper[topic]);

                    nd_m[topic] += 1;
                    if (nd_m[old_topic] == 0)
                    {
			unsigned short pos = n_mks[m].erase_pos(rev_mapper[old_topic]);
                        rev_mapper[pos] = rev_mapper[old_topic];
                        rev_mapper[old_topic] = K;                        
                    }

                    cbuff[nst*(w%ntt)+i].push(delta(w,old_topic, topic));
                }
                else
                {
                    n_mks[m].increment(rev_mapper[topic]);
                    ++nd_m[topic];
                }
                z[m][n] = topic;
            }
            for (const auto& k : n_mks[m])
            {
                    nd_m[k.idx] = 0;
                    rev_mapper[k.idx] = K;
            }
        }
	tn = std::chrono::high_resolution_clock::now();
	std::cout << "In thread " << i << " at iteration " << iter << " ..." 
                  << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;
    }
	
    delete[] p;
    delete[] nd_m;
    delete[] rev_mapper;

    current_iter = NULL;
    
    return 0;
}

int sparseLDA::specific_init()
{
    smtx = new shared_mutex[V];
    std::cout << "Converting to sparseLDA data structure ..." << std::endl;
    nws.resize(V);
    for (unsigned v = 0; v < V; ++v)
    {
        for (unsigned short k = 0; k < K; ++k)
        {
            if (n_wk[v][k] != 0)
                nws[v].sorted_insert(k, n_wk[v][k]);
        }
        //std::sort(nws[v].begin(), nws[v].end(), std::greater<std::pair<unsigned, unsigned>>());
    }

    return 0;
}

int sparseLDA::sampling(unsigned i)
{
    xorshift128plus rng_;
	
    double * p = new double[K]; // temp variable for sampling
    //unsigned *nd_m = new unsigned[K];
    unsigned short *rev_mapper = new unsigned short[K];
    unsigned short *fwd_mapper = new unsigned short[K];
    for (unsigned short k = 0; k < K; ++k)
    {
        //nd_m[k] = 0;
        rev_mapper[k] = K;
        fwd_mapper[k] = K;
    }

    unsigned *nwsum_local = new unsigned[K];
    std::copy(n_k, n_k+K, nwsum_local);

    double ssum, rsum, qsum;
    double *q1 = new double[K];
    // compute ssum
    ssum = 0;
    for (unsigned short k = 0; k < K; ++k)
    {
            q1[k] = alphaK / (nwsum_local[k] + Vbeta);
            ssum += 1 / (nwsum_local[k] + Vbeta);
    }
    ssum *= alphaK * beta;
    std::chrono::high_resolution_clock::time_point ts, tn;

    unsigned iter; current_iter = &iter;	
    for (iter = 0; iter < n_iters; ++iter)
    {
        ts = std::chrono::high_resolution_clock::now();
        // for each document of worker i
        for (unsigned m = i; m < M; m+=nst)
        {
            //synchronization
            if (m % 10 == i)
            {
                ssum = 0;
                for (unsigned short k = 0; k < K; ++k)
                {
                    nwsum_local[k] = n_k[k];
                    q1[k] = alphaK / (nwsum_local[k] + Vbeta);
                    ssum += 1 / (nwsum_local[k] + Vbeta);
                }
                ssum *= alphaK * beta;
            }

            unsigned short kc = 0;
            rsum = 0;
            for (const auto& k : n_mks[m])
            {
                    //nd_m[k.idx] = k.val;
                    rev_mapper[k.idx] = kc++;
                    double temp = k.val / (nwsum_local[k.idx] + Vbeta);
                    rsum += temp;
                    q1[k.idx] += temp;
            }
            rsum *= beta;

            for (unsigned n = 0; n < trngdata->docs[m]->length; ++n)
            {	
                unsigned w = trngdata->docs[m]->words[n];

                // remove z_ij from the count variables
                unsigned short topic = z[m][n]; unsigned short old_topic = topic;
                nwsum_local[topic] -= 1;
                //nd_m[topic] -= 1;
                

                // update the bucket sums
                unsigned nd_mk = n_mks[m].decrement(rev_mapper[topic]);
                double denom = nwsum_local[topic] + Vbeta;
                ssum -= (alphaK * beta) / (denom + 1);
                ssum += (alphaK * beta) / denom;
                rsum -= (beta + nd_mk * beta) / (denom + 1);
                rsum += (nd_mk * beta) / denom;
                //rsum -= beta / denom;
                q1[topic] = (alphaK + nd_mk) / denom;

                //  Divide the full sampling mass into three buckets. [s | r | q]
                qsum = 0;
                unsigned short ii=0;
                smtx[w].lock_shared();
                for (const auto &k : nws[w])
                {
                    fwd_mapper[ii] = k.idx;
                    p[ii] = k.val*q1[k.idx];
                    qsum += p[ii++];
                }
                smtx[w].unlock_shared();
                double total_mass = qsum + rsum + ssum;
                double u = rng_.rand_double() * total_mass;
		
                if (u < ssum)	//smoothing only bucket
                {
                        // In this case, we can step through each topic, calculating and adding up	for that topic, until we reach a value greater than u
                        u /= alphaK * beta;
                        for (topic = -1; u > 0; u -= 1 / (nwsum_local[++topic] + Vbeta));
                }
                else if (u < ssum + rsum) //document topic bucket
                {
                    // In this case, we need only iterate over the set of topics t such that ntd=0 a number that is usually substantially less than the total number of topics
                    u -= ssum;
                    u /= beta;
                    for (const auto& k : n_mks[m])
                    {
                        u -= k.val / (nwsum_local[k.idx] + Vbeta);
                        if (u <= 0)
                        {
                            topic = k.idx;
                            break;
                        }
                    }
                }
                else //topic word bucket
                {
                        u -= ssum + rsum;
                        for (topic = -1; u > 0; u -= p[++topic] );
                        topic = fwd_mapper[topic];
                }
		

                //update the bucket sums
                nwsum_local[topic] += 1;
                denom = nwsum_local[topic] + Vbeta;
                ssum -= (alphaK * beta) / (denom - 1);
                ssum += (alphaK * beta) / denom;
                //rsum += beta / denom;

                // add newly estimated z_i to count variables
                if (topic!=old_topic)
                {
                    if(rev_mapper[topic] == K)
                    {
                        rev_mapper[topic] = n_mks[m].push_back(topic, 1);
                        rsum += beta / denom;
                        q1[topic] = (alphaK + 1) / denom;
                    }
                    else
                    {
                        nd_mk = n_mks[m].increment(rev_mapper[topic]);
                        rsum -= (nd_mk * beta - beta) / (denom - 1);
                        rsum += (nd_mk * beta) / denom;
                	q1[topic] = (alphaK + nd_mk) / denom;
                    }
                    //nd_m[topic] += 1;
                    if (n_mks[m].val_in(rev_mapper[old_topic]) == 0)
                    {
			unsigned short pos = n_mks[m].erase_pos(rev_mapper[old_topic]);
                        rev_mapper[pos] = rev_mapper[old_topic];
                        rev_mapper[old_topic] = K;                        
                    }

                    cbuff[nst*(w%ntt)+i].push(delta(w,old_topic,topic));
                }
                else
                {
                    nd_mk = n_mks[m].increment(rev_mapper[topic]);
                    rsum -= (nd_mk * beta - beta) / (denom - 1);
                    rsum += (nd_mk * beta) / denom;
                    q1[topic] = (alphaK +  nd_mk) / denom;
                    //nd_m[topic] += 1;
                }

                z[m][n] = topic;
            }
            for (const auto& k : n_mks[m])
            {
                q1[k.idx] -= k.val / (nwsum_local[k.idx] + Vbeta);
                //nd_m[k.idx] = 0;
                rev_mapper[k.idx] = K;
            }
        }
        tn = std::chrono::high_resolution_clock::now();
        //std::cout << "In thread " << i << " at iteration " << iter << " ..." 
        //                  << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;
    }

    delete[] p;
    //delete[] nd_m;
    delete[] rev_mapper;
    delete[] fwd_mapper;

    current_iter = NULL;
    
    return 0;
}

int sparseLDA::updater(unsigned i)
{
    do
    {
        for(unsigned tn = 0; tn<nst; ++tn)
        {
            if(!cbuff[i*nst + tn].empty())
            {
                delta temp = cbuff[i*nst + tn].front();
                cbuff[i*nst + tn].pop();
                n_wk[temp.word][temp.old_topic] -= 1;
                n_wk[temp.word][temp.new_topic] += 1;
                n_k[temp.old_topic] -= 1;
                n_k[temp.new_topic] += 1;
                //n_k[temp.old_topic].fetch_add(-1);
                //n_k[temp.new_topic].fetch_add(+1);

                smtx[temp.word].lock();
                nws[temp.word].sorted_dec(temp.old_topic);
                nws[temp.word].sorted_inc(temp.new_topic);
                smtx[temp.word].unlock();
            }
        }
    } while(!inf_stop);

    return 0;
}

int aliasLDA::specific_init()
{
    std::cout << "Initializing the alias tables ..." << std::endl;
    q.resize(V);
    for (unsigned w = 0; w < V; ++w)
    {
        q[w].init(K);
        generateQtable(w);
    }

    return 0;
}

int aliasLDA::sampling(unsigned i)
{
    xorshift128plus rng_;
    
    double * p = new double[K]; // temp variable for sampling
    unsigned *nd_m = new unsigned[K];
    unsigned short *rev_mapper = new unsigned short[K];
    for (unsigned short k = 0; k < K; ++k)
    {
        nd_m[k] = 0;
        rev_mapper[k] = K;
    }
    std::chrono::high_resolution_clock::time_point ts, tn;

    unsigned iter; current_iter = &iter;	
    for (iter = 0; iter < n_iters; ++iter)
    {
        ts = std::chrono::high_resolution_clock::now();
        // for each document of worker i
        for (unsigned m = i; m < M; m+=nst)
        {
            unsigned short kc = 0;
            for (const auto& k : n_mks[m])
            {
                nd_m[k.idx] = k.val;
                rev_mapper[k.idx] = kc++;
            }
            for (unsigned n = 0; n < trngdata->docs[m]->length; ++n)
            {
                unsigned w = trngdata->docs[m]->words[n];

                // remove z_ij from the count variables
                unsigned short topic = z[m][n]; unsigned short new_topic; unsigned short old_topic = topic;
		nd_m[topic] -= 1;
                n_mks[m].decrement(rev_mapper[topic]);

                //Compute pdw
                double psum = 0;
                unsigned short ii = 0;
                /* Travese all non-zero document-topic distribution */
                for (const auto& k : n_mks[m])
                {
                    psum += k.val * (n_wk[w][k.idx] + beta) / (n_k[k.idx] + Vbeta);
                    p[ii++] = psum;
                }

                double p_tot = (psum + alphaK*q[w].wsum);

                //MHV to draw new topic
                for (unsigned r = 0; r < MH_STEPS; ++r)
                {
		    double u = rng_.rand_double() * p_tot;
                    //1. Flip a coin
		    if ( u < psum )
                    {
                        //double u = rng_.rand_double() * psum;
                        new_topic = std::lower_bound(p,p+ii,u) - p;
                        new_topic = n_mks[m].idx_in(new_topic);
                    }
                    else
                    {
                        q[w].noSamples++;
                        if(q[w].noSamples > K)
                        {	
                            if(mtx[w].try_lock())
                            {
                                generateQtable(w);
                                mtx[w].unlock();
                            }
                        }
                        new_topic = q[w].sample(rng_.rand_k(K), rng_.rand_double());
                    }

                    _mm_prefetch((const char *)&(q[w].w[new_topic]), _MM_HINT_T1);
                    _mm_prefetch((const char *)&(q[w].w[topic]), _MM_HINT_T1);

		    if (topic != new_topic)
                    {
                            //2. Find acceptance probability
                	    double fkw_old = (n_wk[w][topic] + beta) / (n_k[topic] + Vbeta);
                            double fkw_new = (n_wk[w][new_topic] + beta) / (n_k[new_topic] + Vbeta);
                            //double temp_old = (old_topic!=topic) ? (nw[w][topic] + beta) / (nwsum[topic] + Vbeta) : (nw[w][topic] - 1 + beta) / (nwsum[topic] - 1 + Vbeta);
                            //double temp_new = (old_topic!=new_topic) ? (nw[w][new_topic] + beta) / (nwsum[new_topic] + Vbeta) : (nw[w][new_topic] - 1 + beta) / (nwsum[new_topic] - 1 + Vbeta);
                            double acceptance = (nd_m[new_topic] + alphaK) / (nd_m[topic] + alphaK) * fkw_new / fkw_old
                                    *(nd_m[topic] * fkw_old + alphaK*q[w].w[topic])
                                    / (nd_m[new_topic] * fkw_new + alphaK*q[w].w[new_topic]);
                            //3. Compare against uniform[0,1]
                            if (rng_.rand_double() < acceptance)
                                    topic = new_topic;
                    }
                }		

                // add newly estimated z_i to count variables
                if (topic!=old_topic)
                {
                    if(nd_m[topic] == 0)
                    {
                        rev_mapper[topic] = n_mks[m].push_back(topic, 1);
                    }
                    else
                    {
                        n_mks[m].increment(rev_mapper[topic]);
                    }
                    nd_m[topic] += 1;
                    if (nd_m[old_topic] == 0)
                    {
			unsigned short pos = n_mks[m].erase_pos(rev_mapper[old_topic]);
                        rev_mapper[pos] = rev_mapper[old_topic];
                        rev_mapper[old_topic] = K;
                    }

                    cbuff[nst*(w%ntt)+i].push(delta(w,old_topic, topic));
                }
                else
                {
                    n_mks[m].increment(rev_mapper[topic]);
                    nd_m[topic] += 1;
                }
                z[m][n] = topic;
            }
            for (const auto& k : n_mks[m])
            {
                nd_m[k.idx] = 0;
                rev_mapper[k.idx] = K;
            }
        }
        tn = std::chrono::high_resolution_clock::now();
        //std::cout << "In thread " << i << " at iteration " << iter << " ..." 
        //          << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;
    }
	
    delete[] p;
    delete[] nd_m;
    delete[] rev_mapper;

    current_iter = NULL;

    return 0;
}

void aliasLDA::generateQtable(unsigned w)
{
    double wsum = 0.0;
    for (unsigned short k = 0; k < K; ++k)
    {
        q[w].w[k] = (n_wk[w][k] + beta) / (n_k[k] + Vbeta);
        wsum += q[w].w[k];
    }
    q[w].wsum = wsum;
    q[w].constructTable();
}

int FTreeLDA::specific_init()
{
    // Construct trees here
    std::cout << "Initializing the Fenwich trees ..." << std::endl;
    trees = new fTree[V];
    double *temp = new double[K];
    for (unsigned w = 0; w < V; ++w)
    {
    	for (unsigned short k = 0; k < K; ++k)
        	temp[k] = (n_wk[w][k] + beta) / (n_k[k] + Vbeta);
        trees[w].init(K);
        trees[w].recompute(temp);
    }
    delete[] temp;
    return 0;
}

int FTreeLDA::sampling(unsigned i)
{
    xorshift128plus rng_;
    
    double * p = new double[K]; // temp variable for sampling
    //unsigned *nd_m = new unsigned[K];
    unsigned short *rev_mapper = new unsigned short[K];
    for (unsigned short k = 0; k < K; ++k)
    {
        //nd_m[k] = 0;
        rev_mapper[k] = K;
    }
    std::chrono::high_resolution_clock::time_point ts, tn;

    unsigned iter; current_iter = &iter;	
    for (iter = 0; iter < n_iters; ++iter)
    {
        ts = std::chrono::high_resolution_clock::now();
        // for each document of worker i
        for (unsigned m = i; m < M; m+=nst)
        {
	   //auto n_ms = n_mks[m];
		
            unsigned short kc = 0;
            for (const auto& k : n_mks[m])
            {
                //nd_m[k.idx] = k.val;
                rev_mapper[k.idx] = kc++;
            }
            for (unsigned n = 0; n < trngdata->docs[m]->length; ++n)
            {
                unsigned w = trngdata->docs[m]->words[n];

                // remove z_ij from the count variables
                unsigned short topic = z[m][n]; unsigned short old_topic = topic;
                //nd_m[topic] -= 1;
                n_mks[m].decrement(rev_mapper[topic]);

                // Multi core approximation: do not update fTree[w] apriori
                // trees[w].update(topic, (nw[w][topic] + beta) / (nwsum[topic] + Vbeta));

                //auto mydoc = &n_mks[m];

                //Compute pdw
                double  psum = 0;
                unsigned short ii = 0;
                /* Travese all non-zero document-topic distribution */
                for (const auto& k : n_mks[m])
                {
                    psum += k.val * trees[w].getComponent(k.idx);
                    p[ii++] = psum;
                }
                //if(ii>K)
                //	std::cout<<"I am mad"<<std::endl;

                double u = rng_.rand_double() * (psum + alphaK*trees[w].w[1]);

                if (u < psum)
                {
                    unsigned short temp = std::lower_bound(p,p+ii,u) - p;
                    topic = n_mks[m].idx_in(temp);
                }
                else
                {
                    topic = trees[w].sample(rng_.rand_double());
                }

                // add newly estimated z_i to count variables
                if (topic!=old_topic)
                {
                    if(rev_mapper[topic] == K)
                    {
                        rev_mapper[topic] = n_mks[m].push_back(topic, 1);
                    }
                    else
                    {
                        n_mks[m].increment(rev_mapper[topic]);
                    }
                    //nd_m[topic] += 1;
                    if (n_mks[m].val_in(rev_mapper[old_topic]) == 0)
                    {
                            unsigned short pos = n_mks[m].erase_pos(rev_mapper[old_topic]);
                            rev_mapper[pos] = rev_mapper[old_topic];
                            rev_mapper[old_topic] = K;
                    }

                    cbuff[nst*(w%ntt)+i].push(delta(w,old_topic,topic));
                }
                else
                {
                    n_mks[m].increment(rev_mapper[topic]);
                    //nd_m[topic] += 1;
                }
                z[m][n] = topic;
            }
            for (const auto& k : n_mks[m])
            {
                    //nd_m[k.idx] = 0;
                    rev_mapper[k.idx] = K;
            }
        }
        tn = std::chrono::high_resolution_clock::now();
        //std::cout << "In thread " << i << " at iteration " << iter << " ..." 
        //                  << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;
    }
    delete[] p;
    //delete[] nd_m;
    delete[] rev_mapper;

    current_iter = NULL;
    
    return 0;	
}
int FTreeLDA::updater(unsigned i)
{
	//unsigned aaa = 1;
    do
    {
        for(unsigned tn = 0; tn<nst; ++tn)
        {
            if(!cbuff[i*nst + tn].empty())
            {
                delta temp = cbuff[i*nst + tn].front();
                cbuff[i*nst + tn].pop();
                n_wk[temp.word][temp.old_topic] -= 1;
                n_wk[temp.word][temp.new_topic] += 1;
                n_k[temp.old_topic] -= 1;
                n_k[temp.new_topic] += 1;
                //n_k[temp.old_topic].fetch_add(-1);
                //n_k[temp.new_topic].fetch_add(+1);

                trees[temp.word].update(temp.old_topic, (n_wk[temp.word][temp.old_topic] + beta) / (n_k[temp.old_topic] + Vbeta));
                trees[temp.word].update(temp.new_topic, (n_wk[temp.word][temp.new_topic] + beta) / (n_k[temp.new_topic] + Vbeta));
            }
        }

	//if (aaa++ == 0) std::cout << " I am inside updater loop " << i << std::endl;
    } while(!inf_stop); //(!done[i]);
	//std::cout << " Exiting updater loop " << i << std::endl;
	//if (aaa % 10'000'000 == 0) std::cout << " I am inside updater loop " << i << std::endl;

    return 0;
}

int lightLDA::specific_init()
{
    std::cout << "Initializing the alias tables ..." << std::endl;
    q.resize(V);
    for (unsigned w = 0; w < V; ++w)
    {
        q[w].init(K);
        generateQtable(w);
    }
    return 0;
}

int lightLDA::sampling(unsigned i)
{
    xorshift128plus rng_;
    
    double * p = new double[K]; // temp variable for sampling
    unsigned *nd_m = new unsigned[K];
    unsigned short *rev_mapper = new unsigned short[K];
    for (unsigned short k = 0; k < K; ++k)
    {
        nd_m[k] = 0;
        rev_mapper[k] = K;
    }
    std::chrono::high_resolution_clock::time_point ts, tn;

    unsigned iter; current_iter = &iter;	
    for (iter = 0; iter < n_iters; ++iter)
    {
        ts = std::chrono::high_resolution_clock::now();
        // for each document of worker i
        for (unsigned m = i; m < M; m+=nst)
        {
            unsigned short kc = 0;
            for (const auto& k : n_mks[m])
            {
                nd_m[k.idx] = k.val;
                rev_mapper[k.idx] = kc++;
            }

            double sumPd = trngdata->docs[m]->length + alpha;

            for (unsigned n = 0; n < trngdata->docs[m]->length; ++n)
            {
                unsigned w = trngdata->docs[m]->words[n];

                // remove z_ij from the count variables
                unsigned short topic = z[m][n]; unsigned short new_topic; unsigned short old_topic = topic;
                nd_m[topic] -= 1;
                n_mks[m].decrement(rev_mapper[topic]);

                // MHV to draw new topic
                for (unsigned r = 0; r < MH_STEPS; ++r)
                {
                    {
                        // Draw a topic from doc-proposal
                        double u = rng_.rand_double() * sumPd;
                        if (u < trngdata->docs[m]->length)
                        {
                            // draw from doc-topic distribution skipping n
                            unsigned pos = (unsigned)(u);
                            new_topic = z[m][pos];
                        }
                        else
                        {
                            // draw uniformly
                            u -= trngdata->docs[m]->length;
                            u /= alphaK;
                            new_topic = (unsigned)(u); // pick_a_number(0,ptrndata->docs[m]->length-1); (int)(d_unif01(urng)*ptrndata->docs[m]->length);
                        }

                        if (topic != new_topic)
                        {
                            //2. Find acceptance probability
                            double temp_old = (nd_m[topic] + alphaK) * (n_wk[w][topic] + beta) / (n_k[topic] + Vbeta) ;
                            double temp_new = (nd_m[new_topic] + alphaK) * (n_wk[w][new_topic] + beta) / (n_k[new_topic] + Vbeta);
                            double prop_old = (topic==old_topic) ? (nd_m[topic] + 1 + alphaK) : (nd_m[topic] + alphaK);
                            double prop_new = (new_topic==old_topic) ? (nd_m[new_topic] + 1 + alphaK) : (nd_m[new_topic] + alphaK);
                            double acceptance = (temp_new * prop_old) / (temp_old *prop_new);

                            //3. Compare against uniform[0,1]
                            if (rng_.rand_double() < acceptance)
                            {
                                topic = new_topic;
                            }
                        }
                    }

                    {
                        // Draw a topic from word-proposal
                        q[w].noSamples++;
                        if (q[w].noSamples > K >> 1)
                        {
                            if(mtx[w].try_lock())
                            {
                                generateQtable(w);
                                mtx[w].unlock();
                            }
                        }
                        new_topic = q[w].sample(rng_.rand_k(K),rng_.rand_double() );


                        if (topic != new_topic)
                        {
                            //2. Find acceptance probability
                            double temp_old = (nd_m[topic] + alphaK) * (n_wk[w][topic] + beta) / (n_k[topic] + Vbeta);
                            double temp_new = (nd_m[new_topic] + alphaK) * (n_wk[w][new_topic] + beta) / (n_k[new_topic] + Vbeta);
                            double acceptance =  (temp_new * q[w].w[topic]) / (temp_old * q[w].w[new_topic]);

                            //3. Compare against uniform[0,1]
                            if (rng_.rand_double() < acceptance)
                            {
                                topic = new_topic;
                            }
                        }
                    }
                }

                // add newly estimated z_i to count variables
                if (topic!=old_topic)
                {
                    if(nd_m[topic] == 0)
                    {
                        rev_mapper[topic] = n_mks[m].push_back(topic, 1);
                    }
                    else
                    {
                        n_mks[m].increment(rev_mapper[topic]);
                    }
                    nd_m[topic] += 1;
                    if (nd_m[old_topic] == 0)
                    {
                        unsigned short pos = n_mks[m].erase_pos(rev_mapper[old_topic]);
                        rev_mapper[pos] = rev_mapper[old_topic];
                        rev_mapper[old_topic] = K;
                    }

                    cbuff[nst*(w%ntt)+i].push(delta(w,old_topic,topic));
                }
                else
                {
                    n_mks[m].increment(rev_mapper[topic]);
                    nd_m[topic] += 1;
                }
                z[m][n] = topic;
            }
            for (const auto& k : n_mks[m])
            {
                nd_m[k.idx] = 0;
                rev_mapper[k.idx] = K;
            }
        }
        tn = std::chrono::high_resolution_clock::now();
        //std::cout << "In thread " << i << " at iteration " << iter << " ..." 
        //                  << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;
    }
	
    delete[] p;
    delete[] nd_m;
    delete[] rev_mapper;

    current_iter = NULL;

    return 0;
}

void lightLDA::generateQtable(unsigned w)
{
    float wsum = 0.0;
    for (int k = 0; k < K; ++k)
    {
        q[w].w[k] = (n_wk[w][k] + beta) / (n_k[k] + Vbeta);
        wsum += q[w].w[k];
    }
    q[w].wsum = wsum;
    q[w].constructTable();
}
