import json
import pandas as pd
from glob import glob
from tqdm import tqdm


def ac_reweighting(dataframes: dict, reweight: bool, config: dict) -> dict:
    """Reweight JHU and Madgraph signals to different coupling scenarios."""
    vbf = pd.concat([df for key, df in dataframes.items() if 'vbf125_JHU' in key])
    wh = pd.concat([df for key, df in dataframes.items() if 'wh125_JHU' in key])
    zh = pd.concat([df for key, df in dataframes.items() if 'zh125_JHU' in key])

    weighted_dataframes = {}

    # scale evtwt with appropriate reweighting factor and give a new name
    for weight, name in config['jhu_ac_reweighting_map']['vbf']:
        df = vbf.copy(deep=True)
        df['evtwt'] *= df[weight]
        weighted_dataframes[name] = df

    for weight, name in config['jhu_ac_reweighting_map']['wh']:
        df = wh.copy(deep=True)
        df['evtwt'] *= df[weight]
        weighted_dataframes[name] = df

    for weight, name in config['jhu_ac_reweighting_map']['zh']:
        df = zh.copy(deep=True)
        df['evtwt'] *= df[weight]
        weighted_dataframes[name] = df

    if reweight:
        # add couplings together then apply weights
        ggh = pd.concat([df for key, df in dataframes.items() if 'ggh125_madgraph' in key])
        for weight, name in config['mg_ac_reweighting_map']['ggh']:
            df = ggh.copy(deep=True)
            df['evtwt'] *= df[weight]
        weighted_dataframes[name] = df
    else:
        # just add couplings without weighting
        weighted_dataframes['reweighted_ggH_htt_0PM125'] = pd.concat([
            df for key, df in dataframes.items() if 'ggh125_madgraph' in key and 'a1_filtered' in key
        ])
        weighted_dataframes['reweighted_ggH_htt_0M125'] = pd.concat([
            df for key, df in dataframes.items() if 'ggh125_madgraph' in key and 'a3_filtered' in key
        ])
        weighted_dataframes['reweighted_ggH_htt_0Mf05ph0125'] = pd.concat([
            df for key, df in dataframes.items() if 'ggh125_madgraph' in key and 'a3int_filtered' in key
        ])

    return weighted_dataframes


def main(args):
    files = [ifile for ifile in glob(f'{args.input}/*.h5') if not 'nominal' in ifile]
    with open('boilerplate.json') as config_file:
        config = json.load(config_file)

    for ifile in tqdm(files):
        dataframes = pd.HDFStore(ifile)
        new_dataframes = ac_reweighting(dataframes, args.reweight, config)

        # save dataframes to h5 files
        for name, df in tqdm(new_dataframes.items(), leave=False):
            df.to_hdf(ifile, name, complevel=9, complib='blosc:lz4')


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='path to input files')
    parser.add_argument('--reweight', action='store_true', help='apply madgraph reweighting')
    main(parser.parse_args())
