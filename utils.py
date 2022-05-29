import csv

from os import makedirs, path

def read_csv(path):
    rows = []
    with open(path, encoding="latin-1") as csvfile:
        file_reader = csv.reader(csvfile)
        for row in file_reader:
            rows.append(list(row))
    return rows


def write_csv(dir, filename, r):
    makedirs(dir, exist_ok=True)
    f = open(path.join(dir, filename), "w", newline="")
    writer = csv.writer(f)
    writer.writerows(r)
    f.close()
