from util_funs import read_files

RESULTS_PATH = "./results/"
EXPORTS_PATH = './exports/'
RUN_ID = "682517"

lineages = []
dataset_results_dict = read_files(RESULTS_PATH, df_name="tan", run_id=RUN_ID,lineages=lineages)

dataset_name = list(dataset_results_dict.keys())[0][0]
filename = dataset_name + '_export_run_' + RUN_ID + '.csv'

with open(EXPORTS_PATH + filename, 'w') as f:
    for i, data in enumerate(dataset_results_dict.values()):
        result_dict = data['result_dict']
        if (i == 0):
            header = []
            for i, lab in enumerate(data['labels']):
                header.append(data['labels'][i])
                header.append(data['labels'][i] + '_CI95_low')
                header.append(data['labels'][i] + '_CI95_up')
            f.write(','.join(header))
            f.write(',lineage_id')
            f.write('\n')
        f.write(','.join([str(item) for t in result_dict.values() for item in t]))
        f.write(',' + str(data['lineage_id']))
        f.write('\n')