
import pandas as pd
from generate_ccv import consonant_data, consonant_manner_map, vowels, consonant_df, sonority_hierachy, calc_sonority_diff

# Generate all VCC Combinations that follow SSP
vcc_ssp = []
for C1 in consonant_df['Consonants']:
    for C2 in consonant_df['Consonants']:
        if C1 != C2:  # avoids repetitions
            # Extract manners
            M1 = consonant_manner_map[C1]
            M2 = consonant_manner_map[C2]

            # Extract index of manner (where it first appears) so we can quantify SSP order
            if consonant_df[consonant_df['Manner of Articulation'] == M1].index[0] < consonant_df[consonant_df['Manner of Articulation'] == M2].index[0]:
                # Checking why SSP has more values than non-SSP, can delete
                #print(f"Skipping {C1} and {C2} (M1: {M1}, M2: {M2}) due to SSP order")
                continue

            # Ensure manners are distinct
            #if M1 != M2:
            for V in vowels:
                vcc_ssp.append((V, C1, C2))
# Convert to a dataframe
vcc_ssp_df = pd.DataFrame(vcc_ssp, columns=['V', 'C1', 'C2'])

# Generate all VCC combinations that violate SSP
vcc_non_ssp = []

for C1 in consonant_df['Consonants']:
    for C2 in consonant_df['Consonants']:
        if C1 != C2:  # avoids repetitions
            # Extract manners
            M1 = consonant_manner_map[C1]
            M2 = consonant_manner_map[C2]

            # Extract index of manner (where it first appears) so we can quantify SSP order
            if consonant_df[consonant_df['Manner of Articulation'] == M1].index[0] > consonant_df[consonant_df['Manner of Articulation'] == M2].index[0]:
                # Checking why SSP has more values than non-SSP, can delete
                #print(f"Skipping {C1} and {C2} (M1: {M1}, M2: {M2}) due to SSP order")
                continue

            # Ensure manners are distinct
            #if M1 != M2:
            for V in vowels:
                vcc_non_ssp.append((V, C1, C2))

# Convert to a dataframe
vcc_non_ssp_df = pd.DataFrame(vcc_non_ssp, columns=['V', 'C1', 'C2'])

# Add SSP status to original dfs distinguish
vcc_ssp_df['SSP Status'] = 'Follow'
vcc_non_ssp_df['SSP Status'] = 'Violate'

# Add VCC column to original dfs to extract later
vcc_ssp_df['VCC'] = vcc_ssp_df['V'] + vcc_ssp_df['C1'] + vcc_ssp_df['C2']
vcc_non_ssp_df['VCC'] = vcc_non_ssp_df['V'] + vcc_non_ssp_df['C1'] + vcc_non_ssp_df['C2']

# Add manners using the corrected map
vcc_ssp_df['C1 Manner'] = vcc_ssp_df['C1'].map(consonant_manner_map)
vcc_ssp_df['C2 Manner'] = vcc_ssp_df['C2'].map(consonant_manner_map)

vcc_non_ssp_df['C1 Manner'] = vcc_non_ssp_df['C1'].map(consonant_manner_map)
vcc_non_ssp_df['C2 Manner'] = vcc_non_ssp_df['C2'].map(consonant_manner_map)

# All together now
vcc_combined_df = pd.concat([vcc_ssp_df[['VCC', 'V', 'C1', 'C1 Manner', 'C2', 'C2 Manner', 'SSP Status']], 
                             vcc_non_ssp_df[['VCC', 'V', 'C1', 'C1 Manner', 'C2', 'C2 Manner', 'SSP Status']]], ignore_index=True)


vcc_combined_df['Sonority Difference'] = vcc_combined_df.apply(
    lambda row: calc_sonority_diff(row['C1 Manner'], row['C2 Manner']),
    axis=1
)

# Print Test
#print(vcc_combined_df.head(50))

# Export to a csv file
vcc_combined_df.to_csv('C:/Users/ali_a/OneDrive/Single Word Processing Stage/SSP Human Error/Data/VCC_SSP_Data.csv', index=False)


# Print # of combinations for verification
#print(f"SSP combinations count: {len(vcc_ssp_df)}")
#print(f"Non-SSP combinations count: {len(vcc_non_ssp_df)}")

# Prevent from running when referenced
if __name__ == '__main__':
    print("This is a test print in SSP_VCC_combinations.py")
