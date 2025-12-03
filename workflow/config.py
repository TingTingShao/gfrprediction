
DATA_PATH="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/gfrprediction/dataset"
ANALYSIS_PATH="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/gfrprediction/analysis"
TEMP_PATH="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/gfrprediction/analysis/temp"
PLOT_PATH="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/gfrprediction/analysis/plots"

FEATURES_USED =['SEX_M', 'AGE', 'Scr', 'CYSC', 'Ht', 'Wt', 'BMI']
LABEL = 'mGFR'

FEATURES_ENGINEERED=['SEX_M', 'AGE', 'Scr', 'CYSC', 'Ht', 'Wt', 'BMI', 
       'CKiDU25Crea', 'CKiDU25CysC', 'FAScrea','FAScreaHt', 'FAScysc', 'FAScombi', 'FAScombiHt', 'EKFC_Crea', 'EKFC_CysC', 'EKFCcombi', 'CKiDU25combi']

FEATURES_ENGINEERED2=['SEX_M', 'AGE', 'Scr', 'CYSC', 'Ht', 'Wt', 'BMI', 
       'CKiDU25Crea', 'CKiDU25CysC', 'FAScrea','FAScreaHt', 'FAScysc',  'EKFC_Crea', 'EKFC_CysC']

FEATURES_ENGINEERED3=['SEX_M', 'AGE', 'Scr', 'CYSC', 'Ht', 'Wt', 'BMI', 
       'EKFC_Crea', 'EKFC_CysC', 'EKFCcombi']

FEATURES_ENGINEERED4=['SEX_M', 'AGE', 'Scr', 'CYSC', 'Ht', 'Wt', 'BMI', 
        'EKFCcombi']