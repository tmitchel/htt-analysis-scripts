import time
import pandas as pd
import numpy as np
import boost_histogram as bh
from ApplyFF import FFApplicationTool


def build_histogram(name):
    """Build 2d hist to fill with fake fraction."""
    bh.Histogram(
        bh.axis.Variable([0, 50, 80, 100, 110, 120, 130, 150, 170, 200, 250, 1000]),  # vis_mass
        bh.axis.Variable([-0.5, 0.5, 1.5, 15])  # njets
    )


def main(args):
    start = time.time()
    dataframes = pd.HDFStore(args.input)
    channel = dataframes['metadata'].channel
    year = dataframes['metadata'].year
    categories = [channel + pref for pref in ['_inclusive', '_0jet', '_boosted', '_vbf']]

    inputs = {
        'frac_w': ['W', 'ZJ', 'VVJ', 'STJ'],
        'frac_tt': ['TTJ'],
        'frac_data': ['data_obs'],
        'frac_real': ['STL', 'VVL', 'TTL', 'ZL', 'STT', 'VVT', 'TTT', 'embed'],
    }

    fractions = {
        'frac_w': {cat: build_histogram(f'frac_w_{cat}') for cat in categories},
        'frac_tt': {cat: build_histogram(f'frac_tt_{cat}') for cat in categories},
        'frac_qcd': {cat: build_histogram(f'frac_qcd_{cat}') for cat in categories},
        'frac_data': {cat: build_histogram(f'frac_data_{cat}') for cat in categories},
        'frac_real': {cat: build_histogram(f'frac_real_{cat}') for cat in categories},
    }

    variables = set([
        'evtwt', 'vis_mass', 'mjj', 'njets',
        'is_antiTauIso', 'mt', 'higgs_pT', 't1_pt', 'contamination', 't1_genMatch'
    ])

    for frac, samples in inputs.items():
        for sample in samples:
            events = dataframes[sample]
            anti_iso_events = events[
                (events['is_antiTauIso'] > 0) & (events['contamination'] == 0)
            ]

            zero_jet_events = anti_iso_events[anti_iso_events['njets'] == 0]
            boosted_events = anti_iso_events[
                (anti_iso_events['njets'] == 1) |
                ((anti_iso_events['njets'] > 1) & anti_iso_events['mjj'] < 300)
            ]
            vbf_events = anti_iso_events[(anti_iso_events['njets'] > 1) & (anti_iso_events['mjj'] > 300)]

            fractions[frac][f'{channel}_inclusive'].fill(
                anti_iso_events.vis_mass.values, anti_iso_events.njets.values, weight=anti_iso_events.evtwt.values)
            fractions[frac][f'{channel}_0jet'].fill(
                zero_jet_events.vis_mass.values, zero_jet_events.njets.values, weight=zero_jet_events.evtwt.values)
            fractions[frac][f'{channel}_boosted'].fill(
                boosted_events.vis_mass.values, boosted_events.njets.values, weight=boosted_events.evtwt.values)
            fractions[frac][f'{channel}_vbf'].fill(
                vbf_events.vis_mass.values, vbf_events.njets.values, weight=vbf_events.evtwt.values)

    for cat in categories:
        qcd_values = fractions['frac_data'][cat].copy(deep=True).to_numpy()
        qcd_values -= fractions['frac_w'][cat].to_numpy()
        qcd_values -= fractions['frac_tt'][cat].to_numpy()
        qcd_values -= fractions['frac_real'][cat].to_numpy()
        qcd_values = np.clip(qcd_values, 0, None)
        fractions['frac_qcd'][cat].fill(qcd_values)

        denom = qcd_values + fractions['frac_w'][cat].to_numpy() + fractions['frac_tt'][cat].to_numpy()
        fractions['frac_w'][cat].view().values = fractions['frac_w'][cat].to_numpy() / denom
        fractions['frac_tt'][cat].view().values = fractions['frac_tt'][cat].to_numpy() / denom
        fractions['frac_qcd'][cat].view().values = fractions['frac_qcd'][cat].to_numpy() / denom



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='path to input files')
    main(parser.parse_args())
