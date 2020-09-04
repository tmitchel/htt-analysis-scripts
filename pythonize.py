import json
import time
import uproot
from os import path, mkdir
from glob import glob
from tqdm import tqdm
from pprint import pprint
import numpy as np
import pandas as pd


bkgs = [
    'data_obs', 'embed', 'W',
    'ZJ', 'ZTT', 'ZL',
    'VVJ', 'VVT', 'VVL',
    'TTJ', 'TTT', 'TTL',
    'STJ', 'STT', 'STL',
]

acs = [
    'ggh125_JHU', 'vbf125_JHU', 'wh125_JHU', 'zh125_JHU',
    'ggh125_madgraph'
]

powhegs = ['ggh125_powheg', 'vbf125_powheg', 'zh125_powheg']
wh_powhegs = ['wplus125_powheg', 'wminus125_powheg']
mg_couplings = ['a1_filtered', 'a3_filtered', 'a3int_filtered']


def build_groupings(idir: str) -> dict:
    """For an input directory, create a dictionary mapping output names to input ROOT files."""
    bkg_group = {key: [ifile for ifile in glob(f'{idir}/*_{key}_*.root')] for key in bkgs}
    pw_group = {key: [ifile for ifile in glob(f'{idir}/{key}*.root')] for key in powhegs}
    wh_pw_group = [ifile for name in wh_powhegs for ifile in glob(f'{idir}/{name}*.root')]
    ungrouped = [ifile for ifile in glob(f'{idir}/*.root') if 'madgraph' in ifile or 'JHU' in ifile]

    group = {}
    for key, files in bkg_group.items():
        if len(files) > 0:
            group[key] = files

    for key, files in pw_group.items():
        if len(files) > 0:
            group[key] = files

    for ifile in ungrouped:
        name = ifile.split('/')[-1].replace('.root', '')
        name = name.split('_SYST')[0].replace('-', '_')
        name = name.replace('_ggH125', '').replace('_VBF125', '').replace('_WH125', '').replace('_ZH125', '')
        group[name] = [ifile]

    if len(wh_pw_group) > 0:
        group['wh125_powheg'] = wh_pw_group

    return group


def build_filelist(input_dir: str, syst: bool = False) -> dict:
    """Build the dictionary of all files to be processed in appropriate groupings."""
    filedict = {
        idir.split('SYST_')[-1].split('/')[0]: {}
        for idir in glob('{}/*'.format(input_dir)) if 'SYST_' in idir
    }

    filedict['nominal'] = build_groupings(f'{input_dir}/NOMINAL')
    if syst:
        for idir in filedict.keys():
            if idir == 'nominal':
                continue
            elif 'Rivet' in idir:
                continue
            filedict[idir] = build_groupings(f'{input_dir}/SYST_{idir}')
    else:
        filedict = {'nominal': filedict['nominal']}

    pprint(filedict, width=150)
    return filedict


def process_group(directory: str, files: dict, channel: str, year: str) -> dict:
    """
    For the given grouping, convert ROOT files into DataFrames merging groups together. 
    Return a dictionary mapping file names to DataFrames.
    """
    if len(files) == 0:
        raise Exception('empty file list for directory {}'.format(directory)) + 1

    dataframes = {}
    for name, ifile in files.items():
        # equivalent of hadding
        update_dfs = uproot.pandas.iterate(ifile, f'{channel}_tree')
        for update_df in update_dfs:
            update_df.fillna(-999, inplace=True)
            dataframes[name] = update_df

    dataframes['metadata'] = pd.DataFrame({'channel': [channel], 'year': [year]})
    return dataframes


def ac_reweighting(dataframes: dict, reweight: bool, config: dict) -> dict:
    """Reweight JHU and Madgraph signals to different coupling scenarios."""
    vbf = pd.concat([df for key, df in dataframes.items() if 'vbf125_JHU' in key])
    wh = pd.concat([df for key, df in dataframes.items() if 'wh125_JHU' in key])
    zh = pd.concat([df for key, df in dataframes.items() if 'zh125_JHU' in key])

    # scale evtwt with appropriate reweighting factor and give a new name
    for weight, name in config['jhu_ac_reweighting_map']['vbf']:
        df = vbf.copy(deep=True)
        df['evtwt'] *= df[weight]
        dataframes[name] = df

    for weight, name in config['jhu_ac_reweighting_map']['wh']:
        df = wh.copy(deep=True)
        df['evtwt'] *= df[weight]
        dataframes[name] = df

    for weight, name in config['jhu_ac_reweighting_map']['zh']:
        df = zh.copy(deep=True)
        df['evtwt'] *= df[weight]
        dataframes[name] = df

    if reweight:
        # add couplings together then apply weights
        ggh = pd.concat([df for key, df in dataframes.items() if 'ggh125_madgraph' in key])
        for weight, name in config['mg_ac_reweighting_map']['ggh']:
            df = ggh.copy(deep=True)
            df['evtwt'] *= df[weight]
        dataframes[name] = df
    else:
        # just add couplings without weighting
        dataframes['reweighted_ggH_htt_0PM125'] = pd.concat([
            df for key, df in dataframes.items() if 'ggh125_madgraph' in key and 'a1_filtered' in key
        ])
        dataframes['reweighted_ggH_htt_0M125'] = pd.concat([
            df for key, df in dataframes.items() if 'ggh125_madgraph' in key and 'a3_filtered' in key
        ])
        dataframes['reweighted_ggH_htt_0Mf05ph0125'] = pd.concat([
            df for key, df in dataframes.items() if 'ggh125_madgraph' in key and 'a3int_filtered' in key
        ])

    return dataframes


def nn_preprocess(dataframes: dict) -> dict:
    """Store information for NN training (signal labels, scaled evtwts, and info for standardization."""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    for name, df in dataframes.items():
        if name == 'metadata':
            continue

        # store signal label for NN training
        if 'ggh125' in name or 'vbf125' in name:
            df['signalLabel'] = np.ones(len(df))
        else:
            df['signalLabel'] = np.zeros(len(df))

        # normalize sample weights
        df['scaled_evtwt'] = MinMaxScaler(feature_range=(1., 2.)).fit_transform(df.evtwt.values.reshape(-1, 1))

    # handle standardization using data distribution
    scaler = StandardScaler()
    scaler.fit(dataframes['data_obs'].values)
    dataframes['standardization'] = pd.DataFrame({
        'mean': scaler.mean_,
        'scale': scaler.scale_,
        'variance': scaler.var_,
        'nsamples': scaler.n_samples_seen_
    }).set_index(dataframes['data_obs'].columns.values)

    return dataframes


def main(args):
    start = time.time()
    output_path = 'Output/dataframes/{}'.format(args.output_name)
    with open('boilerplate.json') as config_file:
        config = json.load(config_file)

    # make sure the file doesn't already exist
    if path.exists(output_path):
        raise Exception('Output directory: {} already exists'.format(output_path))

    mkdir(output_path)

    filedict = build_filelist(args.input, args.syst)
    if len(filedict['nominal']) == 0:
        raise Exception('no nominal files found')

    pbar = tqdm(filedict.items())
    for idir, files in pbar:
        if len(files) == 0:
            continue

        pbar.set_description(f'Processing: {idir}')
        
        # basic processing of a SYST directory
        dataframes = process_group(idir, files, args.channel, args.year)

        # if nominal, do ac reweighting in advance and store info for NN training
        if idir == 'nominal':
            dataframes = ac_reweighting(dataframes, args.reweight, config)
            dataframes = nn_preprocess(dataframes)

        # save dataframes to h5 files
        for name, df in tqdm(dataframes.items(), leave=False):
            df.to_hdf('{}/{}.h5'.format(output_path, idir), name, complevel=9, complib='blosc:lz4')

    print(f'Finished in {time.time() - start} seconds.')


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='path to input files')
    parser.add_argument('--output', '-o', required=True, dest='output_name', help='name of output file')
    parser.add_argument('--channel', required=True, help='which channel? (et, mt)')
    parser.add_argument('--year', required=True, help='which year? (2016, 2017, 2018)')
    parser.add_argument('--reweight', action='store_true', help='apply madgraph reweighting')
    parser.add_argument('--syst', action='store_true', help='process systematics as well')
    main(parser.parse_args())
