import pandas as pd
import matplotlib.pyplot as plt

onto = pd.read_csv('Noneresults_nfolds5_nsimilar_5_onto_resnik.csv')
als_data = pd.read_csv('results_nfolds5_nsimilar_5_ALS.csv')
bpr_data = pd.read_csv('results_nfolds5_nsimilar_5_BPR.csv')
als_onto_data = pd.read_csv('Noneresults_nfolds5_nsimilar_5_als_onto_resnik_m1.csv')
bpr_onto_data = pd.read_csv('Noneresults_nfolds5_nsimilar_5_bpr_onto_resnik_m1.csv')




metrics_onto = onto.iloc[:6, 1:]  
metrics_als = als_data.iloc[:6, 1:]  
metrics_bpr = bpr_data.iloc[:6, 1:]
metrics_als_onto = als_onto_data.iloc[:6, 1:]
metrics_bpr_onto = bpr_onto_data.iloc[:6, 1:]


metrics_onto.columns = metrics_onto.columns.str.extract('(\d+)').astype(int).squeeze()
metrics_als.columns = metrics_als.columns.str.extract('(\d+)').astype(int).squeeze()
metrics_bpr.columns = metrics_bpr.columns.str.extract('(\d+)').astype(int).squeeze()
metrics_als_onto.columns = metrics_als_onto.columns.str.extract('(\d+)').astype(int).squeeze()
metrics_bpr_onto.columns = metrics_bpr_onto.columns.str.extract('(\d+)').astype(int).squeeze()

metrics_onto = metrics_onto.sort_index(axis=1)
metrics_als = metrics_als.sort_index(axis=1)
metrics_bpr = metrics_bpr.sort_index(axis=1)
metrics_als_onto = metrics_als_onto.sort_index(axis=1)
metrics_bpr_onto = metrics_bpr_onto.sort_index(axis=1)


metrics_labels = ['Precision', 'Recall', 'F-measure', 'MRR', 'DCG', 'nDCG']

for i, metric in enumerate(metrics_labels):
    plt.figure(figsize=(10, 4))  
    plt.plot(metrics_onto.columns, metrics_onto.iloc[i], marker='o', linestyle='-', color='blue', label=f'ONTO {metric}', alpha=0.8)
    plt.plot(metrics_als.columns, metrics_als.iloc[i], marker='^', linestyle=':', color='green', label=f'ALS {metric}', alpha=0.8)
    plt.plot(metrics_bpr.columns, metrics_bpr.iloc[i], marker='x', linestyle='-.', color='red', label=f'BPR {metric}', alpha=0.6)
    plt.plot(metrics_als_onto.columns, metrics_als_onto.iloc[i], marker='*', linestyle='--', color='purple', label=f'ALS_ONTO {metric}', alpha=0.9)
    plt.plot(metrics_bpr_onto.columns, metrics_bpr_onto.iloc[i], marker='s', linestyle='-', color='orange', label=f'BPR_ONTO {metric}', alpha=0.7)
    plt.title(f'{metric} Comparison')
    plt.xlabel('Top-N')
    plt.ylabel(f'{metric} Score')
    plt.legend()
    plt.tight_layout()
    plt.show()  
