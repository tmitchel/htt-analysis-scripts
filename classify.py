from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'

from tqdm import tqdm
from glob import glob
from os import path, mkdir
from keras.models import load_model
import pandas as pd

training_variables = [
    'Q2V1', 'Q2V2', 'Phi', 'Phi1', 'costheta1', 'costheta2',
    'costhetastar', 'mjj', 'higgs_pT', 'm_sv'
]


def main(args):
    # make sure the file doesn't already exist
    if path.exists(f'Output/dataframes/{args.out}'):
        raise Exception('Output directory: {} already exists'.format(args.out))

    mkdir(f'Output/dataframes/{args.out}')

    model = load_model(args.model)
    nominal_data = pd.HDFStore(f'{args.input}/nominal.h5')
    scaler_info = nominal_data.get('standardization')
    nominal_data.close()

    pbar = tqdm(glob(f'{args.input}/*.h5'))
    for ifile in pbar:
        filename = ifile.split('/')[-1].replace('.h5', '')
        pbar.set_description(f'Processing: {filename}')

        dataframes = pd.HDFStore(ifile)
        ppbar = tqdm(dataframes.keys(), leave=False)
        for name in ppbar:
            if name == '/standardization' or name == '/metadata':
                dataframes[name].to_hdf(f'Output/dataframes/{args.out}/{filename}.h5', name, complevel=9, complib='blosc:lz4')
                continue

            ppbar.set_description(f'File: {name}')
            cl_df = dataframes[name][training_variables].copy(deep=True)

            # standardization
            for var in training_variables:
                cl_df[var] -= scaler_info.loc[var, 'mean']
                cl_df[var] /= scaler_info.loc[var, 'scale']

            # classify and save
            new_df = dataframes[name]
            new_df['NN_disc'] = model.predict(cl_df.values, verbose=False)
            new_df.to_hdf(f'Output/dataframes/{args.out}/{filename}.h5', name, complevel=9, complib='blosc:lz4')


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', required=True, help='name of the model to load')
    parser.add_argument('--input', '-i', required=True, help='path to input files')
    parser.add_argument('--out', '-o', required=True, help='path to store output')
    main(parser.parse_args())
