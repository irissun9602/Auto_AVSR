import os

def split_filelist(input_file, num_splits=28):
    # Create directory if it does not exist
    if not os.path.exists('filelist_0.9'):
        os.makedirs('filelist_0.9')

    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    total_files = len(lines)
    chunk_size = total_files // num_splits + (1 if total_files % num_splits != 0 else 0)
    
    for i in range(num_splits):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = lines[start_idx:end_idx]
        output_file = os.path.join('filelist_0.9', f'filelist{i:03d}.txt')
        with open(output_file, 'w') as out_f:
            out_f.writelines(chunk)
        print(f'Created {output_file} with {len(chunk)} entries')

# Example usage
input_file = 'score_range_0.9_1.0.txt'
split_filelist(input_file)
