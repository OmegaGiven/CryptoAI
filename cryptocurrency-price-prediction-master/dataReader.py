import csv

def parse_csv_file(csv_fp):
    """
    Takes a csv file specified by the path csv_fp and
    converts it into an array of examples, each of which
    is a dictionary of key-value pairs where keys are
    column names and the values are column attributes.
    """
    examples = []
    with open(csv_fp) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        key_names = None
        for row in csv_reader:
            if len(row) == 0:
                continue
            if line_count == 0:
                key_names = row
                for i in range(len(key_names)):
                    ## strip whitespace on both ends.
                    row[i] = row[i].strip()
                    line_count += 1
            else:
                ex = {}
                for i, k in enumerate(key_names):
                    ## strip white spaces on both ends.
                    ex[k] = row[i].strip()
                examples.append(ex)
        return examples, key_names