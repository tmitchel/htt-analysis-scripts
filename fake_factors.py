import time
import pandas as pd
import numpy as np
import boost_histogram as bh
# from ApplyFF import FFApplicationTool
np.seterr(divide='ignore', invalid='ignore')

systs_names = [
    'ff_qcd_0jet_unc1', 'ff_qcd_0jet_unc2', 'ff_qcd_1jet_unc1', 'ff_qcd_1jet_unc2', 'ff_qcd_2jet_unc1', 'ff_qcd_2jet_unc2',
    'ff_w_0jet_unc1', 'ff_w_0jet_unc2', 'ff_w_1jet_unc1', 'ff_w_1jet_unc2', 'ff_w_2jet_unc1', 'ff_w_2jet_unc2',
    'ff_tt_0jet_unc1', 'ff_tt_0jet_unc2',
    'mtclosure_w_unc1', 'mtclosure_w_unc2',
    'lptclosure_xtrg_qcd', 'lptclosure_xtrg_w', 'lptclosure_xtrg_tt',
    'lptclosure_qcd', 'lptclosure_w', 'lptclosure_tt',
    'osssclosure_qcd'
]
systs = [(name, 'up') for name in systs_names] + [(name, 'down') for name in systs_names]
filling_variables = [
    't1_pt', 't1_decayMode', 'njets', 'vis_mass', 'mt', 'mu_pt', 'lep_dr', 'met',
    'el_pt', 'mjj', 'is_antiTauIso', 'cross_trigger'
]


def build_histogram(name):
    """Build 2d hist to fill with fake fraction."""
    return bh.Histogram(
        bh.axis.Variable([0, 50, 80, 100, 110, 120, 130, 150, 170, 200, 250, 1000]),  # vis_mass
        bh.axis.Variable([-0.5, 0.5, 1.5, 15])  # njets
    )


def encode_categories(data, channel):
    def cats(row):
        if row.njets == 0:
            return channel + '_0jet'
        elif row.njets > 1 and row.mjj > 300:
            return channel + '_vbf'
        else:
            return channel + '_boosted'

    return data.apply(cats, axis=1)


def get_weights(df, fake_weights, fractions, channel, doSyst=False):
    categories = encode_categories(df[['njets', 'mjj']], channel)
    if channel == 'et':
        pt_name = 'el_pt'
    else:
        pt_name = 'mu_pt'

    weights = {'fake_weight': np.empty(len(categories))}
    if doSyst:
        for syst in systs:
            weights[syst[0] + "_" + syst[1]] = np.empty(len(categories))

    xbins = fractions['frac_data']['{}_inclusive'.format(channel)].axes[0].index(df['vis_mass'].values)
    ybins = fractions['frac_data']['{}_inclusive'.format(channel)].axes[1].index(df['njets'].values)

    for i, cat in enumerate(categories.values):
        frac_w = fractions['frac_w'][cat].value(xbins[i], ybins[i])
        frac_tt = fractions['frac_tt'][cat].value(xbins[i], ybins[i])
        frac_qcd = fractions['frac_qcd'][cat].value(xbins[i], ybins[i])

        weights['fake_weight'][i] = fake_weights.get_ff(df.t1_pt.iloc[i], df.mt.iloc[i], df.vis_mass.iloc[i], df[pt_name].iloc[i], df.lep_dr.iloc[i],
                                                        df.met.iloc[i], df.njets.iloc[i], df.cross_trigger.iloc[i],
                                                        frac_tt, frac_qcd, frac_w)

        if doSyst:
            for syst in systs:
                weights[syst[0] + "_" + syst[1]][i] = fake_weights.get_ff(df.t1_pt.iloc[i], df.mt.iloc[i], df.vis_mass.iloc[i], df[pt_name].iloc[i],
                                                                          df.lep_dr.iloc[i], df.met.iloc[i], df.njets.iloc[i], df.cross_trigger.iloc[i],
                                                                          frac_tt, frac_qcd, frac_w, syst[0], syst[1])

    return weights


def main(args):
    start = time.time()
    dataframes = pd.HDFStore(args.input)
    channel = dataframes['metadata'].channel.values[0]
    year = dataframes['metadata'].year.values[0]
    categories = [channel + pref for pref in ['_inclusive', '_0jet', '_boosted', '_vbf']]

    inputs = {
        'frac_w': ['W', 'ZJ', 'VVJ', 'STJ'],
        'frac_tt': ['TTJ'],
        'frac_data': ['data_obs'],
        'frac_real': ['STL', 'VVL', 'TTL', 'ZL', 'STT', 'VVT', 'TTT', 'embed'],
    }

    fractions = {
        'frac_w': {cat: build_histogram('frac_w_{}'.format(cat)) for cat in categories},
        'frac_tt': {cat: build_histogram('frac_tt_{}'.format(cat)) for cat in categories},
        'frac_qcd': {cat: build_histogram('frac_qcd_{}'.format(cat)) for cat in categories},
        'frac_data': {cat: build_histogram('frac_data_{}'.format(cat)) for cat in categories},
        'frac_real': {cat: build_histogram('frac_real_{}'.format(cat)) for cat in categories},
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
            fractions[frac]['{}_inclusive'.format(channel)].fill(
                anti_iso_events.vis_mass.values, anti_iso_events.njets.values, weight=anti_iso_events.evtwt.values)
            fractions[frac]['{}_0jet'.format(channel)].fill(
                zero_jet_events.vis_mass.values, zero_jet_events.njets.values, weight=zero_jet_events.evtwt.values)
            fractions[frac]['{}_boosted'.format(channel)].fill(
                boosted_events.vis_mass.values, boosted_events.njets.values, weight=boosted_events.evtwt.values)
            fractions[frac]['{}_vbf'.format(channel)].fill(
                vbf_events.vis_mass.values, vbf_events.njets.values, weight=vbf_events.evtwt.values)

    for cat in categories:
        qcd_values = fractions['frac_data'][cat].copy(deep=True).to_numpy()[0]
        qcd_values -= fractions['frac_w'][cat].to_numpy()[0]
        qcd_values -= fractions['frac_tt'][cat].to_numpy()[0]
        qcd_values -= fractions['frac_real'][cat].to_numpy()[0]
        qcd_values = np.clip(qcd_values, 0, None)
        fractions['frac_qcd'][cat][...] = qcd_values

        denom = qcd_values + fractions['frac_w'][cat].to_numpy()[0] + fractions['frac_tt'][cat].to_numpy()[0]
        print('Category: {}'.format(cat))
        print('\tfrac_w: {}'.format(fractions["frac_w"][cat].sum() / sum(sum(denom))))
        print('\tfrac_tt: {}'.format(fractions["tfrac_tt"][cat].sum() / sum(sum(denom))))
        print('\tfrac_qcd: {}'.format(fractions["tfrac_qcd"][cat].sum() / sum(sum(denom))))
        print('\tfrac_real: {}'.format(fractions["tfrac_real"][cat].sum() / sum(sum(denom))))

        fractions['frac_w'][cat][...] = fractions['frac_w'][cat].to_numpy()[0] / denom
        fractions['frac_tt'][cat][...] = fractions['frac_tt'][cat].to_numpy()[0] / denom
        fractions['frac_qcd'][cat][...] = fractions['frac_qcd'][cat].to_numpy()[0] / denom

    to_process = inputs['frac_data'] + inputs['frac_real']
    processed = []
    for p in to_process:
        events = dataframes[p]
        anti_iso_events = events[
            (events['is_antiTauIso'] > 0) & (events['contamination'] == 0)
        ]

        weights = get_weights(anti_iso_events[filling_variables], anti_iso_events, fractions, channel, args.syst)
        for name, weight in weights.items():
            if p != 'data_obs':
                weight *= -1
            anti_iso_events[name] = weight

        processed.append(anti_iso_events)

    jet_fakes = pd.concat(processed)
    jet_fakes.to_hdf(args.input, 'jetFakes', complevel=9, complib='blosc:lz4')


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='path to input files')
    parser.add_argument('--syst', required=True, help='process jetFakes systematics')
    main(parser.parse_args())
