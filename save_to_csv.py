def save_to_csv(filename, result):
    filepath = './csv/' + filename
    result.to_csv(filepath, index=False)
    print(f'Saved results to {filepath}!')