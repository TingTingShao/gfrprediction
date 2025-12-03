#!/usr/bin/env python3
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from passingBablokRegression import PassingBablok
from pyCompare._plotBlandAltman import blandAltman
from metrics import Evaluation
import config

pbplot=PassingBablok()
eval=Evaluation()
# Load data once at module top‐level so make_plot can see it
RESULTS_CSV = config.ANALYSIS_PATH  + "/data25_gfrcal_pred.csv"
ORIG_CSV    = config.DATA_PATH      + "/data.csv"
OUT_DIR     = config.PLOT_PATH

data       = pd.read_csv(RESULTS_CSV).dropna()
data=data[data['dataset']=='ext_val'].rename(columns={'rf_src': 'rf_scr',
                                                     'rf_full_src': 'rf_full_scr'})
data_orig  = pd.read_csv(ORIG_CSV)
mGFR       = data['mGFR']
exclusion=['mGFR_cat', 'AGE_cat', 'SEX_M']


# pred_cols  = [c for c in data.columns if c not in data_orig.columns and c not in exclusion]
# pred_cols=['EKFCcombi', 'CKiDU25combi','rf_full_combi', 'rf_full_mean', 'rf_equations_full', 'rf_equations_forward', 'rf_equations_backward', 'rf_equations_manual_selected']
data=data.rename(columns={'rf_scr': 'RF2-25_SCr', 'rf_combi': 'RF2-25_combi', 'rf_cysc': 'RF2-25_CysC', 'rf_full_scr': 'RF_full_SCr', 'rf_full_cysc': 'RF_full_CysC', 'rf_full_combi': 'RF_full_combi'})

pred_cols=['RF2-25_SCr', 'RF_full_SCr',  'RF2-25_combi', 'RF_full_combi', 'RF2-25_CysC', 'RF_full_CysC', 'EKFC_Crea', 'FAScysc', 'FAScombiHt']

print(pred_cols)
def make_plot(pred_col):
    preds = data[pred_col]
    # Passing–Bablok
    try:
        pbplot.plot(
            mGFR, preds,
            path=os.path.join(OUT_DIR, f"passingBablok_{pred_col}.png"),
            ylabel=pred_col,
            format='png'
        )
    except Exception as e:
        print(f"[ERROR] Passing-Bablok failed for {pred_col}: {e}")
    
    
    # Bland–Altman variants
    # for perc, prec, suffix in [
    #     (False, False, ""),
    #     (False, True,  "_pre"),
    #     (True,  False, "_per"),
    # ]:
    #     try:
    #         blandAltman(
    #             preds, mGFR,
    #             limitOfAgreement=1.96,
    #             confidenceInterval=95,
    #             percentage=perc,
    #             precision=prec,
    #             # dpi=1200,
    #             # figureFormat="pdf",
    #             savePath=os.path.join(OUT_DIR, f"blandAltman_{pred_col}{suffix}.png")
    #         )
    #     except Exception as e:
    #         print(f"[ERROR] Bland–Altman({perc=},{prec=}) failed for {pred_col}: {e}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    with Pool(processes=min(len(pred_cols), cpu_count())) as pool:
        pool.map(make_plot, pred_cols)

    
