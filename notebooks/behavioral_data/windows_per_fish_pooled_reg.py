# Normoxia and hypoxia windows (minutes) per fish, as used in
#   notebooks/behavioral_data/model_fit_new_data_results_linear_kernels_pooled_null_final_allfish_windows_pooled_reg.ipynb  (state of 2026-11-12)
# For swim_burst_trials_v5.ipynb:  paste WIN over the line  WIN = {n: dict(WINDOWS) for n in fish_ids}  (WIN[fish][condition] = (start, end))
WINDOWS = {'normoxia': (5.0, 20.0), 'hypoxia': (25.0, 40.0)}      # the default of v5, used for a fish that is not in WIN

# --- (1) the fish that are in KEEP_FISH of the pooled_reg notebook now (7) ---
WIN = {
    '20220728-fish04':     {'normoxia': (5.0, 15.0),   'hypoxia': (25.7, 40.0)},     # detected
    '20220730-fish01-TS2': {'normoxia': (5.0, 15.0),   'hypoxia': (30.88, 40.0)},    # detected
    '20220912-fish02':     {'normoxia': (8.0, 20.0),   'hypoxia': (30.0, 40.0)},     # set by hand (gtACR2, 0-20 min normal water, 20-40 min low-O2 perfusion; the onset burst of swimming at 26-28 min is left out)
    '20221118-fish01':     {'normoxia': (5.0, 15.0),   'hypoxia': (30.2, 40.0)},     # detected
    'fish01-6dpf':         {'normoxia': (5.0, 20.0),   'hypoxia': (26.63, 40.0)},    # normoxia set by hand, hypoxia detected
    'fish01-7dpf-grating': {'normoxia': (5.0, 15.0),   'hypoxia': (26.73, 40.0)},    # detected
    'fish03-7dpf':         {'normoxia': (33.33, 43.33), 'hypoxia': (13.33, 23.33)},  # set by hand, in SWIM-RECORD minutes: the O2 record starts 1000 s earlier (see SWIM_OFFSET_S); normoxia = the re-normoxia after the hypoxia
}
# a fish with a swim offset: its swim record starts this many s AFTER the O2 record. Crop the O2 of such a fish AFTER the normalisation of v5
# (v5 data cell: O2[nm] = o_ / o_[SR * 60 * 5:SR * 60 * 10].mean()), e.g. after the loop:   for nm, off in SWIM_OFFSET_S.items(): O2[nm] = O2[nm][int(off * SR):]
SWIM_OFFSET_S = {'fish03-7dpf': 1000}

# --- (2) the data that the fits of the pooled_reg notebook actually use for a condition = the window plus its extension block (EXTEND_MIN = 2, merged with the window); no entry = the window only ---
WIN_USED = {
    '20220728-fish04':     {'normoxia': (5.0, 19.2),  'hypoxia': (25.7, 40.0)},
    '20220730-fish01-TS2': {'normoxia': (5.0, 17.0),  'hypoxia': (28.8, 40.0)},
    '20220912-fish02':     {'normoxia': (5.8, 20.0),  'hypoxia': (27.8, 40.0)},
    '20221118-fish01':     {'normoxia': (5.0, 17.0),  'hypoxia': (30.2, 40.0)},
    'fish01-6dpf':         {'normoxia': (5.0, 20.0),  'hypoxia': (26.63, 40.0)},
    'fish01-7dpf-grating': {'normoxia': (5.0, 20.0),  'hypoxia': (26.73, 40.0)},
    'fish03-7dpf':         {'normoxia': (33.33, 43.33), 'hypoxia': (13.33, 23.33)},
}

# --- (3) fish that were in the pooled_reg notebook earlier and are commented out of KEEP_FISH now (the windows they had) ---
WIN_COMMENTED_OUT = {
    '20220730-fish03-TS3': {'normoxia': (3.0, 20.0),   'hypoxia': (30.0, 40.0)},     # set by hand
    '20220804-fish04':     {'normoxia': (5.0, 13.82),  'hypoxia': (28.35, 40.0)},    # detected
}

# --- (4) the other fish of v5: the detected periods of new_model_fits_v20261103/model_fit_new_data_results_linear_kernels_pooled_null_all_fish.ipynb (10 min windows, not capped at 40 min; NOT part of the pooled_reg notebook) ---
WIN_OTHER = {
    '20220728-fish03':     {'normoxia': (5.0, 10.1),  'hypoxia': (30.1, 40.1)},      # only 5 min of normoxia, 24 swim bins
    '20220729-fish01':     {'normoxia': (5.0, 15.0),  'hypoxia': (23.9, 34.0)},      # swims in 99% of the bins
    '20220729-fish02':     {'normoxia': (5.0, 15.0),  'hypoxia': (25.8, 35.8)},      # 8 swim bins in normoxia
    '20220729-fish03-TS2': {'normoxia': (5.0, 15.0),  'hypoxia': (27.0, 37.0)},      # swims in 94% of the bins
    # '20220729-fish03-TS3': no hypoxia period (O2 stays above 50% of its 5-10 min level)
    '20220730-fish02':     {'normoxia': (5.0, 20.0),  'hypoxia': (25.5, 35.5)},      # normoxia set by hand
    '20220909-fish03':     {'normoxia': (5.0, 15.0),  'hypoxia': (25.4, 35.4)},
    'fish02-7dpf-0728':    {'normoxia': (5.0, 15.0),  'hypoxia': (26.1, 36.0)},      # 9 swim bins in normoxia
    'fish02-7dpf-1118':    {'normoxia': (5.0, 15.0),  'hypoxia': (23.1, 33.1)},      # reoxygenated from about 34 min
}
