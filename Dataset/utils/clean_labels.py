'''
Clean labels.
Get arround spelling error by string similarity check.
CZ
3/29/2022
'''

import pandas as pd
from difflib import SequenceMatcher


# your label.csv here
labels_path = 'data/labels/video.phase.trainingData.clean.StudentVersion.csv'

# extracted true names.csv
# this is extracted by: 
#   set(map(
#       lambda s: s.lower().rstrip('0123456789 '), 
#       names
#   ))
# and then mannually select out the (assumed) correct spelling
names = [
    'access',
    'acquiring suture', 
    'adhesiolysis', 
    'blurry', 
    'catheter insertion', 
    'direct hernia repair', 
    'mesh placement', 
    'mesh positioning', 
    'out of body', 
    'peritoneal closure', 
    'peritoneal scoring', 
    'perperitoneal dissection', 
    'primary hernia repair', 
    'reduction of hernia', 
    'stationary idle', 
    'suture positioning', 
    'transitionary idle'
]


def clean_labels(labels_df, names_df):
    ''' Clean labels in labels_df using name_df '''

    def similar(a, b):
        ''' String similarity measurement
            Arguments: 
                a - raw_label : str
                b - true_label : str
            Return: 
                a similarity score in [0, 1]
        '''
        return SequenceMatcher(None, a, b).ratio()

    # add a new column
    labels_df['PhaseNameClean'] = labels_df['PhaseName'].map(
        # compare raw label with true names
        lambda raw: names_df.loc[
            # select the name with highest similarity
            names_df['Name'].apply(
                lambda x: similar(x, raw)
            ).argmax()
        ].values[0]
    )

    # drop rows with PhaseName == 'Access'
    labels_df = labels_df.drop(labels_df[
        (labels_df['PhaseName'] == 'access')# | (labels_df['PhaseName'] == 'acquiring suture')
    ].index)

    return labels_df


# example of use
if __name__ == '__main__':
    labels_df = pd.read_csv(labels_path)
    names_df = pd.DataFrame({'Name': names})#pd.read_csv(names_path, names=['Name'])
    labels_df = clean_labels(labels_df, names_df)
    labels_df.to_csv('data/labels/phase_trainingData_clean.csv')

