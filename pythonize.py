import time
import uproot
from os import path, mkdir
from glob import glob
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

powhegs = [
    'ggh125_powheg', 'vbf125_powheg', 'zh125_powheg'
]


def build_groupings(idir):
    bkg_group = {key: [ifile for ifile in glob(f'{idir}/*_{key}_*.root')] for key in bkgs}
    pw_group = {key: [ifile for ifile in glob(f'{idir}/{key}*.root')] for key in powhegs}
    ac_group = {key: [ifile for ifile in glob(f'{idir}/{key}*.root')] for key in acs} if 'NOMINAL' in idir else {}
    ungrouped = [ifile for ifile in glob(f'{idir}/*.root')
                 if 'madgraph' in ifile or 'JHU' in ifile] if not 'NOMINAL' in idir else []

    group = {}
    for key, files in bkg_group.items():
        if len(files) > 0:
            group[key] = files

    for key, files in pw_group.items():
        if len(files) > 0:
            group[key] = files

    for key, files in ac_group.items():
        if len(files) > 0:
            group[key] = files

    for ifile in ungrouped:
        name = ifile.split('/')[-1].replace('.root', '')
        name = name.split('_SYST')[0].replace('-', '_')
        name = name.replace('_ggH125', '').replace('_VBF125', '').replace('_WH125', '').replace('_ZH125', '')
        group[name] = [ifile]

    return group


def build_filelist(input_dir, syst=False):
    """Build list of files to be processed and included in the DataFrame."""
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

    pprint(filedict, width=150)
    return filedict


def process_group(directory, files, channel):
    if len(files) == 0:
        raise Exception('empty file list for directory {}'.format(directory))

    print(f'Processing: {directory}')
    dataframes = []
    for name, ifile in files.items():
        update_dfs = uproot.pandas.iterate(ifile, f'{channel}_tree')
        for update_df in update_dfs:
            update_df.fillna(-999, inplace=True)
            dataframes.append((name, update_df))

    return dataframes


def main(args):
    start = time.time()
    output_path = 'Output/dataframes/{}'.format(args.output_name)

    # make sure the file doesn't already exist
    if path.exists(output_path):
        raise Exception('Output directory: {} already exists'.format(output_path))

    mkdir(output_path)

    filedict = build_filelist(args.input, args.syst)
    if len(filedict['nominal']) == 0:
        raise Exception('no nominal files found')

    for idir, files in filedict.items():
        if len(files) == 0:
            continue
        dataframes = process_group(idir, files, args.channel)
        for name, df in dataframes:
            df.to_hdf('{}/{}.h5'.format(output_path, idir), name, complevel=9, complib='blosc:lz4')

    print(f'Finished in {time.time() - start} seconds.')


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='path to input files')
    parser.add_argument('--output', '-o', required=True, dest='output_name', help='name of output file')
    parser.add_argument('--channel', required=True, help='which channel? (et, mt)')
    parser.add_argument('--syst', action='store_true', help='process systematics as well')
    main(parser.parse_args())
