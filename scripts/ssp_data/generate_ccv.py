
import pandas as pd

# Define the data for consonants with Manner and Place of Articulation
consonant_data = {
    'Consonants': [ # Changed from IPA to Arpabet
        'P', 'B', 'T', 'D', 'K', 'G', 'Q',        # Plosives
        'F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'H',  # Fricatives
        'CH', 'JH',                                # Affricates
        'M', 'N', 'NG',                             # Nasals
        'L', 'R',                                   # Liquids
        'Y', 'W', 'W'                           # Glides
    ],
    'Manner of Articulation': [
        'Plosive', 'Plosive', 'Plosive', 'Plosive', 'Plosive', 'Plosive', 'Plosive',
        'Fricative', 'Fricative', 'Fricative', 'Fricative', 'Fricative', 'Fricative', 
        'Fricative', 'Fricative', 'Fricative',
        'Affricate', 'Affricate',
        'Nasal', 'Nasal', 'Nasal',
        'Liquid', 'Liquid',                       # 'l' and 'ɹ' under Liquids
        'Glide', 'Glide', 'Glide'                 # Separate `(w)` for Bilabial and Velar
    ],
    'Place of Articulation': [
        'Bilabial', 'Bilabial', 'Alveolar', 'Alveolar', 'Velar', 'Velar', 'Glottal',  # Plosives
        'Labiodental', 'Labiodental', 'Dental', 'Dental', 'Alveolar', 'Alveolar', 
        'Alveopalatal', 'Alveopalatal', 'Glottal',  # Fricatives
        'Alveopalatal', 'Alveopalatal',  # Affricates
        'Bilabial', 'Alveolar', 'Velar',  # Nasals
        'Alveolar', 'Alveolar',           # Liquids
        'Palatal', 'Bilabial', 'Velar'    # `(w)` in both Bilabial and Velar
    ]
}

# Create a dataframe
consonant_df = pd.DataFrame(consonant_data)

# Remove Duplicates of (w) (IPA or W (ARPABET)
consonant_df = consonant_df.drop_duplicates(subset=['Consonants'])
# Pivot the dataframe to create a matrix
consonant_matrix = consonant_df.pivot_table(
    index='Manner of Articulation',
    columns='Place of Articulation',
    values='Consonants',
    aggfunc=lambda x: ', '.join(x)  # Combine multiple IPA letters into one cell
).fillna('-')  # Fill empty cells with a placeholder

# Display the matrix
#print(consonant_matrix)

# Now to generate combinations...

# Create a list of vowels
vowels = ['A','E','I','O','U']

# Extract consonants
consonants = consonant_df['Consonants'].tolist()

# Handle mapping of manners to avoid reindexing error
consonant_manner_map = consonant_df.groupby('Consonants').first()['Manner of Articulation']

'''
# Use a nested for loop to generate all CCV combinations following SSP
ccv_ssp = []

for C1 in range(len(consonants)):
    for C2 in range(len(consonants)):
        if C1 != C2:  # avoids repetitions
            # Extract manners
            M1 = consonant_manner_map[consonants[C1]]
            M2 = consonant_manner_map[consonants[C2]]

            # Ensure manners are distinct
            if M1 != M2:
                for V in vowels:
                    ccv_ssp.append((consonants[C1], consonants[C2], V))
'''
ccv_ssp = []
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
                ccv_ssp.append((C1, C2, V))
# Convert to a dataframe
ccv_ssp_df = pd.DataFrame(ccv_ssp, columns=['C1', 'C2', 'V'])

# Generate all CCV combinations that violate SSP
ccv_non_ssp = []

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
                ccv_non_ssp.append((C1, C2, V))

# Convert to a dataframe
ccv_non_ssp_df = pd.DataFrame(ccv_non_ssp, columns=['C1', 'C2', 'V'])

# Add SSP status to original dfs distinguish
ccv_ssp_df['SSP Status'] = 'Follow'
ccv_non_ssp_df['SSP Status'] = 'Violate'

# Add CCV column to original dfs to extract later
ccv_ssp_df['CCV'] = ccv_ssp_df['C1'] + ccv_ssp_df['C2'] + ccv_ssp_df['V']
ccv_non_ssp_df['CCV'] = ccv_non_ssp_df['C1'] + ccv_non_ssp_df['C2'] + ccv_non_ssp_df['V']

# Add manners using the corrected map
ccv_ssp_df['C1 Manner'] = ccv_ssp_df['C1'].map(consonant_manner_map)
ccv_ssp_df['C2 Manner'] = ccv_ssp_df['C2'].map(consonant_manner_map)

ccv_non_ssp_df['C1 Manner'] = ccv_non_ssp_df['C1'].map(consonant_manner_map)
ccv_non_ssp_df['C2 Manner'] = ccv_non_ssp_df['C2'].map(consonant_manner_map)

# All together now
ccv_combined_df = pd.concat([ccv_ssp_df[['CCV', 'C1', 'C1 Manner', 'C2', 'C2 Manner', 'V', 'SSP Status']], 
                             ccv_non_ssp_df[['CCV', 'C1', 'C1 Manner', 'C2', 'C2 Manner', 'V', 'SSP Status']]], ignore_index=True)

# Display the combined dataframe
#print(ccv_combined_df.head(20))


# Now we will add a column measuring the degree to which the syllables adhere/violate SSP

# First assign numerical values

sonority_hierachy = {
    'Plosive': 1,
    'Fricative': 2,
    'Affricate': 3,
    'Nasal': 4,
    'Liquid': 5,
    'Glide': 6
}

# Now a function to calculate difference

def calc_sonority_diff(m1,m2):
    return sonority_hierachy[m2]-sonority_hierachy[m1]

# Apply to dataframe

ccv_combined_df['Sonority Difference']=ccv_combined_df.apply(lambda row: calc_sonority_diff(row['C1 Manner'], row['C2 Manner']), axis=1) 

# Print Test
print(ccv_combined_df.head(50))

# Export to a csv file
#ccv_combined_df.to_csv('C:/Users/ali_a/OneDrive/Single Word Processing Stage/SSP Human Error/Data/CCV_SSP_Data.csv', index=False)

# Print # of combinations for verification
#print(f"SSP combinations count: {len(ccv_ssp_df)}")
#print(f"Non-SSP combinations count: {len(ccv_non_ssp_df)}")


if __name__ == '__main__':
    print("This is a test print in SSP_CCV_combinations.py")
