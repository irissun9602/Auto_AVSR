def modify_scores(input_file, output_file):
    # Read the prediction file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Process each line and modify the score
    with open(output_file, 'w') as out_file:
        for line in lines:
            filename, score = line.strip().split(';')
            score = float(score)
            if 0.1 < score < 0.5:
                score = 0.4
            elif 0.75 < score <0.9:
                score = 1
            elif 0.5 < score < 0.75:
                score = 0.6
            out_file.write(f'{filename};{score:.6f}\n')

if __name__ == "__main__":
    # Define the input and output file names
    input_file = 'prediction.txt'
    output_file = 'make_mixed_prediction.txt'

    # Modify the scores based on the specified criteria
    modify_scores(input_file, output_file)