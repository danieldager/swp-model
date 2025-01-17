import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import pandas as pd
from generate_ccv import vowels, ccv_combined_df, sonority_hierachy, calc_sonority_diff
from generate_vcc import vcc_combined_df

# Create a temporary list to store values
ccvcc_combinations=[]

# Iterate over values to pair CCV and VCC by matching vowels
for vowel in vowels:
    ccv_filtered = ccv_combined_df[ccv_combined_df['V'] == vowel]
    vcc_filtered = vcc_combined_df[vcc_combined_df['V'] == vowel]
    
    # Pair together
    
    for _, ccv_row in ccv_filtered.iterrows():
        for _, vcc_row in vcc_filtered.iterrows():

            # Check SSP Status for CCV and VCC
            ssp_status = 'Follow' if ccv_row['SSP Status'] == 'Follow' and vcc_row['SSP Status'] == 'Follow' else 'Violate'

            # Calculate Sonority Difference for CCV and VCC
            sonority_diff_ccv = calc_sonority_diff(ccv_row['C1 Manner'], ccv_row['C2 Manner'])
            sonority_diff_vcc = calc_sonority_diff(vcc_row['C1 Manner'], vcc_row['C2 Manner'])

            ccvcc_combinations.append({
                'CCVCC': ccv_row['C1'] + ccv_row['C2'] + vowel + vcc_row['C1'] + vcc_row['C2'],
                'C1': ccv_row['C1'],
                'C2': ccv_row['C2'],
                'V': vowel,
                'C3': vcc_row['C1'],
                'C4': vcc_row['C2'],
                'C1C2 SSP Status': ccv_row['SSP Status'],
                'C1C2 Sonority Difference':sonority_diff_ccv, 
                'C3C4 SSP Status': vcc_row['SSP Status'],
                'C3C4 Sonority Difference':sonority_diff_vcc,
                'SSP Status': ssp_status,
            })

# Convert to a dataframe
ccvcc_df=pd.DataFrame(ccvcc_combinations)

# Print Test
#print(ccvcc_df.head())

# Export to csv file
#ccvcc_df.to_csv('C:/Users/ali_a/Desktop/Single Word Processing Stage/SSP Human Error/Data/CCVCC_SSP_Data.csv', index=False)
