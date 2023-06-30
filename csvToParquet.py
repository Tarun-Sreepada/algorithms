import pandas as pd
import numpy as np

def csvToParquet(csv_file, sep, version="Transactional"):

    """
    This function converts a csv file to a parquet file.

    Parameters
    ----------
    csv_file : str
        The path to the csv file.
    sep : str
        The separator used in the csv file.
    version : str
        The version of the csv file. Choose from: Transactional, Temporal

    Returns
    ------- 
    None

    Outputs
    -------
    parquet_file : csv
        Parquet file with the same name as the csv file as the same directory.
    dict_file
        Dictionary file with the same name as the csv file as the same directory.
        Used for converting the parquet file back to the csv file, or the ints to string.


    
    """


    conversion = {}
    conNum = 1

    if version == "Transactional":
        parquet_file = csv_file.replace(".csv", ".parquet")
        file = []
        for line in open(csv_file):
            # file.append([int(i) for i in line.strip().split(sep)])
            line = line.strip().split(sep)
            temp = []
            for i in line:
                if i not in conversion:
                    conversion[i] = conNum
                    conNum += 1
                temp.append(conversion[i])
            file.append(temp)

        avg = sum([len(i) for i in file]) // len(file) + 1

        averaged_file = []
        index =  1
        indexes = []
        for line in file:
            if len(line) > avg:
                for i in range(0, len(line), avg):
                    indexes.append(index)
                    temp = line[i:i+avg]
                    if len(temp) < avg:
                        temp.extend([-1] * (avg - len(temp)))
                    averaged_file.append(temp)
            else:
                if len(line) < avg:
                    line.extend([-1] * (avg - len(line)))
                indexes.append(index)
                averaged_file.append(line)
            index += 1

        columns = [str(i) for i in range(1, avg+1)]
        df = pd.DataFrame(averaged_file, columns=columns, index=indexes)

        df.to_parquet(parquet_file, engine='pyarrow')

        df = pd.read_parquet(parquet_file, engine='pyarrow')

    elif version == "Temporal":
        parquet_file = csv_file.replace(".csv", ".parquet")
        file = []
        indexes = []
        for line in open(csv_file):
            # first item is the index
            line = line.strip().split(sep)
            indexes.append(int(line[0]))
            # file.append([int(i) for i in line[1:]])
            temp = []
            for i in line[1:]:
                if i not in conversion:
                    conversion[i] = conNum
                    conNum += 1
                temp.append(conversion[i])
            file.append(temp)


        avg = sum([len(i) for i in file]) // len(file) + 1

        averaged_file = []
        nindexes = []
        for i in range(len(file)):
            if len(file[i]) > avg:
                for j in range(0, len(file[i]), avg):
                    nindexes.append(indexes[i])
                    temp = file[i][j:j+avg]
                    if len(temp) < avg:
                        temp.extend([-1] * (avg - len(temp)))
                    averaged_file.append(temp)
            else:
                if len(file[i]) < avg:
                    file[i].extend([-1] * (avg - len(file[i])))
                nindexes.append(indexes[i])
                averaged_file.append(file[i])
        columns = [str(i) for i in range(1, avg+1)]
        df = pd.DataFrame(averaged_file, columns=columns, index=nindexes)

        df.to_parquet(parquet_file, engine='pyarrow')

        df = pd.read_parquet(parquet_file, engine='pyarrow')

    else:
        print("(Invalid version)\tChoose from: Transactional, Temporal")

    # print conversion dictionary
    with open(parquet_file.replace(".parquet", ".dict"), 'w') as f:
        f.write("Int\tItem\n")
        for k, v in conversion.items():
            f.write(str(v) + ":->:" + str(k) + "\n")
        

if __name__ == "__main__":
    csvToParquet("/Users/tarunsreepada/Documents/Code/Data/Temporal/Temporal_T10I4D100K.csv", '\t', version="Temporal")