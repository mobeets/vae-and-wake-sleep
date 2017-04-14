import glob
import os.path
import argparse
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from train import BLOCK_SIZE, load_data, load_models
import time

bincrossentropy = lambda x, xhat: (x*np.log(np.maximum(1e-15, xhat)) + (1-x)*np.log(np.maximum(1e-15, 1-xhat)))

def sample_z(args, nsamps=1):
    z_mean, z_stddev = args
    # eps = np.random.randn(*((nsamps, z_mean.flatten().shape[0])))
    eps = np.random.randn(nsamps, z_mean.shape[0], z_mean.shape[1])
    return z_mean + z_stddev*eps

def LL_marginal(X, enc_model, dec_model, batch_size, nsamps=10, latent_dim=2):
    """
    estimate the marginal probability of x via importance sampling

    E_q[f(X)] = E_q[p(X | z)] === E_z[p(X | z)] = p(X)

    Given a datapoint x_t, average v_i over i=1,...,nsamps:
        - draw latent sample z_i ~ q(z | x_t)
        - compute p(z_i) (i.e., relative to the N(0,I))
        - compute q(z_i | x_t)
        - compute p(y_t | z_i) (i.e., binary cross-entropy)
        - v_i = p(y_t | Z_i)p(z_i)/q(z_i)

    note: for some reason this is faster if you compute px once per sample
    """
    # sample: z ~ q(z | x)
    Zargs = enc_model.predict(X, batch_size=batch_size)
    z_means = Zargs[:,latent_dim:]
    z_stddev = Zargs[:,:latent_dim]
    z_ts = sample_z((z_means, z_stddev), nsamps)

    # compute importance sampling weights
    # N(z; 0,1)
    pzs = scipy.stats.norm(0,1).pdf(z_ts).sum(axis=-1)
    # N(z; z_mean, exp(z_log_var/2))
    qzs = scipy.stats.norm(z_means, z_stddev).pdf(z_ts).sum(axis=-1)

    # compute p(X | z) in batches
    npts = z_ts.shape[1]
    pxs = []
    for i in xrange(nsamps):
        Xhats = dec_model.predict(z_ts[i], batch_size=batch_size)
        pxs.append(bincrossentropy(X, Xhats).sum(axis=-1))
    px = np.vstack(pxs)

    Vs = np.exp(np.log(pzs/qzs) + px) # avoid overflow
    # avg over samps then pts
    return np.log(Vs.mean(axis=0)).mean()

def get_outfile(args):
    fnm = args.run_name + '_' + args.task + '.png'
    return os.path.join(args.out_dir, fnm)

def get_model_files(args):
    if '*' not in args.model_file:
        return [args.model_file]
    return glob.glob(args.model_file)

def batch_inds(fnms):
    if len(fnms) == 0:
        return [0]
    bs = [int(f.split('-')[-1].split('.')[0])+1 for f in fnms]

def eval_LL(X, model_file, nsamps):
    batch_size = len(X)
    _, enc_model, dec_model = load_models(model_file, batch_size_x=batch_size, batch_size_z=batch_size)
    return LL_marginal(X, enc_model, dec_model, batch_size, nsamps)

def get_outfile(args):
    fnm = args.run_name + '.csv'
    return os.path.join(args.out_dir, fnm)

def write_results(LLs, fnm):
    with open(fnm, 'w') as f:
        f.write('\n'.join(['{},{},{}'.format(x,y,z) for x,y,z in LLs]))

def main(args):
    """
    for each model saved with a given prefix, compute LL_marginal on train and test data
    """
    x_train, _, x_test, _ = load_data()
    model_files = get_model_files(args)
    LLs = []
    for model_file in model_files:
        LLtr = eval_LL(x_train, model_file, args.nsamps)
        LLte = eval_LL(x_test, model_file, args.nsamps)
        LLs.append([model_file, LLtr, LLte])
        print LLs[-1]
    write_results(LLs, get_outfile(args))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
        help='tag for current run')
    parser.add_argument('--model_file', type=str,
        help='model(s) for loading weights (.h5); (can contain wildcards)')
    parser.add_argument('--nsamps', type=int, default=20,
        help='number of samples for computing LL')
    parser.add_argument('--out_dir', type=str,
        default='outputs',
        help='basedir for saving stats')
    args = parser.parse_args()
    main(args)
