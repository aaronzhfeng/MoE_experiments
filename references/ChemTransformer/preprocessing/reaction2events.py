import pandas as pd
import selfies as sf
from utils import get_position_indices, SPECIAL_TOKENS, SimpleSmilesTokenizer

SMILE_tokenizer = SimpleSmilesTokenizer()
def tokenize_smiles(smiles, approach):
    if approach == 'selfies':
        selfies_str = sf.encoder(smiles)  # Convert SMILES to SELFIES
        return selfies_str.split('][')  # Tokenize SELFIES representation
    elif approach == 'smile':
        return SMILE_tokenizer.tokenize(smiles)

# Numeric tokenization function (using subword tokenization approach for flexibility)
def tokenize_numeric(value):
    return list(value)

# Build input sequence for each reaction
def build_input_sequence(reactants, catalysts, solvents, reagents, temperature,
                          compound_approach, add_amount=False, add_temperature=False):
    sequence = []

    for compound in reactants.keys():
        sequence.append(SPECIAL_TOKENS['reactant'])    
        sequence.extend(tokenize_smiles(compound, approach=compound_approach))

        # if add_amount:
        #     amount = reactants[compound]
        #     amount_value, unit = amount['value'].strip(), amount['unit']
        #     sequence.append(SPECIAL_TOKENS[unit])
        #     sequence.extend(tokenize_numeric(amount_value))

    for compound in catalysts.keys():
        sequence.append(SPECIAL_TOKENS['catalyst'])    
        sequence.extend(tokenize_smiles(compound, approach=compound_approach))
    
    for compound in solvents.keys():
        sequence.append(SPECIAL_TOKENS['solvent'])    
        sequence.extend(tokenize_smiles(compound, approach=compound_approach))
    
    for compound in reagents.keys():
        sequence.append(SPECIAL_TOKENS['reagent'])    
        sequence.extend(tokenize_smiles(compound, approach=compound_approach))

    # Add temperature with special token
    if add_temperature:
        sequence.append(SPECIAL_TOKENS['CELSIUS'])
        sequence.extend(tokenize_numeric(str(temperature)))
    return sequence

def build_product_sequence(product_smiles, approach):
    tokens = ['<BOS>']
    product_cnt = len(product_smiles.keys())
    for idx, product in enumerate(product_smiles.keys()):
        tokens.extend(tokenize_smiles(product, approach))
        if product_cnt > 1 and idx+1 != product_cnt:
            tokens.append('<SEP>')
    tokens.append('<EOS>')
    return tokens

# Process the dataset, convert to non-padded and tokenized sequences
def process_dataset(file_path, approach, use_amount=False, use_temperature=False):
    print('Loading CSV file')
    data = pd.read_csv(file_path)
    processed_data = []

    print('Start Processing')
    for idx, row in data.iterrows():
        reactants = eval(row['Reactants'])
        catalysts = eval(row['Catalysts'])
        solvents = eval(row['Solvents'])
        reagents = eval(row['Reagents'])
        temperature = row['Conditions']
        product_smiles = eval(row['Products'])

        # Build sequences
        input_sequence = build_input_sequence(reactants, catalysts, solvents, reagents, 
                                              temperature, approach, add_amount=use_amount,
                                              add_temperature=use_temperature)
        target_sequence = build_product_sequence(product_smiles, approach)

        # Get Segment Positions
        input_pos_indices = get_position_indices(input_sequence)
        target_pos_indices = get_position_indices(target_sequence)
        # check < 2 reactants and no product
        if ((len(input_sequence) == 0) or (max(input_pos_indices) < 1) or
                (len(target_sequence) == 0) or (max(target_pos_indices) < 0)): 
            
            print(f'Row Index {idx} is empty. Seq_len is {len(input_sequence)}. Tgt_len is {len(target_sequence)}')
            continue

        # Append to the processed data list
        processed_data.append({
            'x': input_sequence,
            'y': target_sequence,
            'x_position': input_pos_indices,
            'y_position': target_pos_indices
        })

        if (idx+1) % 100000 == 0:
            print('[preparing data]', idx+1)

    return processed_data

if __name__ == '__main__':
    file_path = '../data/reactions_unique.csv'
    approach = 'smile' # choose from selfies or smile
    processed_data = process_dataset(file_path, approach=approach)

    # save data
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_parquet(f'./data/USPTO_480K/{approach}_events.parquet', index=False)
    print(len(processed_df), 'Rows Tokenized')
    print('Input Sequence Length Average:', processed_df['x'].str.len().mean())
    print('Input Sequence Length Median:', processed_df['x'].str.len().median())
    print('Input Sequence Max Length:', processed_df['x'].str.len().max())
    # processed_df.to_csv('./temp.csv')

    # for reaction in processed_data[:10]:
    #     print("Input Sequence:", reaction['x'])
    #     print("Product Sequence:", reaction['y'])
    #     print()
