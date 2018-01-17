import pandas as pd


def parseNDBC(path):
    with open(path, 'r') as f:
        data = [line.rstrip() for line in f.readlines()]
    columns = [value for value in data[0].split(' ') if len(value) > 0]
    units = [value for value in data[1].split(' ') if len(value) > 0]
    columns = [f'{c} ({u})' for c, u in zip(columns, units)]
    values = []
    for row in data[2:]:
        row = [float(value) for value in row.split(' ') if len(value) > 0]
        values.append(row)
    dt = []
    for row in values:
        dt.append(pd.datetime(
            year=int(row[0]),
            month=int(row[1]),
            day=int(row[2]),
            hour=int(row[3]),
            minute=int(row[4])
        ))
    data = pd.DataFrame(data=values, columns=columns, index=dt)
    for col in columns[:5]:
        del data[col]
    return data
