import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from train import BLOCK_SIZE, load_data, load_models

def plot_latents(X, y, enc_model, batch_size, latent_dim=2):
    Zargs = enc_model.predict(X, batch_size=batch_size)
    Z = Zargs[:,:latent_dim]
    plt.figure(figsize=(6, 6))
    plt.scatter(Z[:,0], Z[:,1], c=y)
    plt.colorbar()
    # plt.show()

def plot_number_blocks(X, block_size=BLOCK_SIZE):
    nsamps, nfs = X.shape
    nrows = int(np.sqrt(nsamps))
    ncols = nrows
    X = X.reshape([nrows, ncols, nfs])
    S = X.reshape([nrows, ncols, block_size, block_size])
    pix = np.zeros((block_size*nrows, block_size*ncols))
    for i in xrange(nrows):
        for j in xrange(ncols):
            pix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = S[i,j]
    plt.figure(figsize=(10, 10))
    plt.imshow(pix, cmap='Greys_r')
    # plt.show()

def plot_true_examples(X, n=10, block_size=BLOCK_SIZE):
    block_size = np.sqrt(X.shape[1])
    inds = np.random.randint(0, len(X), np.square(n))
    plot_number_blocks(X[inds])

def plot_model_samples(dec_model, batch_size=1, n=10, block_size=BLOCK_SIZE):
    Z = np.random.randn(np.square(n),2)
    X = dec_model.predict(Z, batch_size=batch_size)
    plot_number_blocks(X)

def plot_manifold(dec_model, zmin, zmax, batch_size=1, n=20):
    Zr = np.linspace(zmin, zmax, n)
    Z1, Z2 = np.meshgrid(Zr, Zr)
    Z = np.dstack([Z1, Z2]).reshape([np.square(n), 2])
    X = dec_model.predict(Z, batch_size=batch_size)
    plot_number_blocks(X)

def find_latent_range(X, enc_model, batch_size, latent_dim=2):
    Zargs = enc_model.predict(X, batch_size=batch_size)
    Z = Zargs[:,:latent_dim]
    return Z.min(), Z.max()

def get_outfile(args):
    fnm = args.run_name + '_' + args.task + '.png'
    return os.path.join(args.out_dir, fnm)

def main(args):
    _, _, x_test, y_test = load_data()
    batch_size_x = len(x_test)
    batch_size_z = 100
    _, enc_model, dec_model = load_models(args.model_file, batch_size_x=batch_size_x, batch_size_z=batch_size_z)

    if args.task == 'examples':
        plot_true_examples(x_test)
    elif args.task == 'samples':
        plot_model_samples(dec_model, batch_size_z)
    elif args.task == 'latents':
        plot_latents(x_test, y_test, enc_model, batch_size_x)
    elif args.task == 'manifold':
        zmin, zmax = find_latent_range(x_test, enc_model, batch_size_x)
        plot_manifold(dec_model, zmin, zmax, batch_size_z)

    plt.savefig(get_outfile(args), bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
        help='tag for current run')
    parser.add_argument('-t', '--task', required=True,
        choices=['examples', 'samples', 'latents', 'manifold', 'all'],
        help='tag for current run')
    parser.add_argument('--model_file', type=str,
        help='model for loading weights (.h5)')
    parser.add_argument('--out_dir', type=str,
        default='outputs',
        help='basedir for saving images')
    args = parser.parse_args()
    if args.task == 'all':
        for t in ['examples', 'samples', 'latents', 'manifold']:
            args.task = t
            main(args)
    else:
        main(args)
