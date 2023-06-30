# Critical imports
import os
import cudf
import time
import psutil
import cupy as cp
import numpy as np
import pandas as pd
os.environ['LIBCUDF_CUFILE_POLICY'] = 'ALWAYS'
os.environ['LIBCUDF_NVCOMP_POLICY'] = 'ALWAYS'

cp.cuda.Device(1).use()

class gdseclat:

    support_kernel = cp.RawKernel(r'''
            extern "C"
            #define uint64_t unsigned long long int
            __global__ void support(uint64_t *bitArray,                 // containing transactions
                                    uint64_t *support,                  // for support
                                    uint64_t *thingsToCompare,          // for things to compare
                                    uint64_t thingsToCompareIndex,     // for things to compare index
                                    uint64_t numberOfThingsToCompare,   // for number of things to compare
                                    uint64_t numberOfElements,
                                    uint64_t minimumSupport) 
            {
                // Calculate thread ID
                uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

                // Return if the thread ID is greater than the number of elements
                if (tid > numberOfThingsToCompare - 1) return;

                uint64_t holder = 0;
                uint64_t supportCounter = 0;
                
                // Iterate through elements
                for (int i = 0; i < numberOfElements; i++) {
                    // Initialize holder with all bits set to 1
                    holder = 0xFFFFFFFFFFFFFFFF;

                    // Calculate the intersection
                    for (int j = thingsToCompareIndex * tid; j < thingsToCompareIndex * (tid + 1); j++) {
                        holder = holder & bitArray[thingsToCompare[j] * numberOfElements + i];
                    }

                    // Count the number of set bits in the intersection
                    supportCounter += __popcll(holder);
                }

                // Store the support count in the support array
                if (supportCounter >= minimumSupport)
                    support[tid] = supportCounter;
                return;
            }

            ''', 'support')

    def __init__(self, file_name, minimum_support):
        self.file_name = file_name
        self.minimum_support = minimum_support
        self.Patterns = {}

    def csv_to_parquet(csv_file, delimiter):
        file = []
        with open(csv_file, "r") as f:
            for line in f:
                file.append([int(x) for x in line.strip().split(delimiter)])
        # if sizes of all rows are not equal, then make them equal by adding -1
        max_len = max([len(x) for x in file])
        for row in file:
            if len(row) < max_len:
                row.extend([np.NAN]*(max_len-len(row)))

        df = pd.DataFrame(file)

        # rename columns to have string names
        df.rename(columns={i: "col"+str(i) for i in range(len(df.columns))}, inplace=True)

        parquet_file = csv_file.replace(".csv", ".parquet")
        df.to_parquet(parquet_file)

        return parquet_file

    def read_file(self, file_name, minimum_support):
        # Read the parquet file into a cuDF DataFrame
        data_frame = cudf.read_parquet(file_name)

        # Fill missing values with -1 and convert the DataFrame to a CuPy array
        data_array = data_frame.fillna(-1).to_cupy(dtype=cp.int64)

        # data_array = cp.array(data_array)

        # Get unique item IDs, excluding the missing value indicator (-1)
        unique_items = cp.unique(data_array)[1:]

        # Get the locations (indices) of each item in the data array and their occurrence count
        item_locations = []
        occurrence_counts = []
        for item in unique_items:
            locs = cp.where(data_array == item)[0]
            item_locations.append(locs)
            occurrence_counts.append(int(len(locs)))

        # Combine item ID, occurrence count, and locations into a single list
        item_info = list(zip(unique_items.get().tolist(), occurrence_counts, item_locations))

        # Sort the item_info list by occurrence count in descending order
        item_info.sort(reverse=True, key=lambda x: x[1])

        # Filter the item_info list based on the minimum_support threshold
        item_info = [x for x in item_info if x[1] >= minimum_support]
        for i in range(len(item_info)):
            self.Patterns[item_info[i][0]] = item_info[i][1]

        # Initialize the cumulative locations list and the where list
        cumulative_locations = [0, item_info[0][1]]
        where = item_info[0][2]

        # Calculate the cumulative locations and update the where list
        for i in range(1, len(item_info)):
            cumulative_locations.append(cumulative_locations[-1] + item_info[i][1])
            where = cp.append(where, item_info[i][2])

        # Convert lists to CuPy arrays
        cumulative_locations = cp.array(cumulative_locations, dtype=cp.uint64)

        # Initialize the bit representation matrix
        arraySizePerItem = data_frame.index[-1]//64 + 1
        bit_representation = cp.zeros((len(item_info), arraySizePerItem), dtype=cp.uint64)
        
        # Define the parallel bit conversion kernel
        pbc = cp.RawKernel(r'''
        extern "C"
        #define uint64_t unsigned long long int 
        __global__ void pb(uint64_t items, uint64_t* where, uint64_t* cumulative_locations, uint64_t* bit_representation, uint64_t arraySizePerItem)
        {
            uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid > items) return;

            uint64_t start = cumulative_locations[tid];
            uint64_t end = cumulative_locations[tid + 1];

            for (uint64_t i = start; i < end; i++)
            {
                uint64_t row = where[i];
                uint64_t col = row / 64;
                uint64_t bit = row % 64;
                bit_representation[tid * arraySizePerItem + col] |= 1ULL << (63 - bit);
            }
        return;
        }
        ''', 'pb')

        # Execute the kernel
        pbc((len(item_info) // 512 + 1,), (512,), (len(item_info), where, cumulative_locations, bit_representation, arraySizePerItem))

        index2item = {i: item_info[i][0] for i in range(len(item_info))}

        return bit_representation, index2item, arraySizePerItem

    def miner(self, bit_representation, keys, minimum_support, arraySizePerItem, index2item):
        print("Number of Keys: " + str(len(keys)))
        newKeys = []
        for i in range(len(keys)):
            # print("i: " + str(i), end="\r")
            for j in range(i+1, len(keys)):
                if keys[i][:-1] == keys[j][:-1] and keys[i][-1] != keys[j][-1]:
                    newCan = keys[i] + [keys[j][-1]]
                    newKeys.append(newCan)
                else:
                    break
        # print("")

        if len(newKeys):
            size = len(newKeys[0])
            numkeys = len(newKeys)
            newKeys = cp.array(newKeys, dtype=cp.int64)
            newKeys = newKeys.flatten()
            # print(cp.max(newKeys))
            support = cp.zeros(len(newKeys), dtype=cp.uint64)

            
            
            self.support_kernel((len(newKeys) // 512 + 1,), (512,), (bit_representation, support, newKeys, size, numkeys, arraySizePerItem, minimum_support))
            cp.cuda.Device().synchronize()

            keys = newKeys.get()
            # locations = locations.get()
            indexes = cp.where(support > 0)[0].get()

            support = support.get()

            newKeys = []
            for i in range(len(indexes)):
                key = keys[indexes[i] * size:size * (indexes[i]+1)]
                nkey = sorted([index2item[key[i]] for i in range(len(key))])
                self.Patterns[tuple(nkey)] = support[indexes[i]]
                newKeys.append(list(key))

            keys = newKeys
            if len(keys) > 1:
                self.miner(bit_representation, keys, minimum_support, arraySizePerItem, index2item)

    def startMine(self):
        start = time.time()

        ps = psutil.Process(os.getpid())

        bit_representation, index2item, arraySizePerItem = self.read_file(self.file_name, self.minimum_support)
        keys = []
        for i in range(len(index2item)):
            keys.append([i])

        if len(keys) > 1:
            self.miner(bit_representation, keys, self.minimum_support, arraySizePerItem, index2item)

        end = time.time()
        
        self.runtime = end - start
        self.patterns = self.Patterns
        self.memoryUSS = ps.memory_full_info().uss

    def getRuntime(self):
        return self.runtime
    
    def getPatterns(self):
        return self.patterns
    
    def getMemoryUSS(self):
        return self.memoryUSS
    

if __name__ == "__main__":
    file = "Transactional_kosarak.parquet"
    minimum_support = 50000
    obj = gdseclat(file, minimum_support)
    obj.startMine()
    print("Run Time: " + str(obj.getRuntime()))
    print("Memory Usage: " + str(obj.getMemoryUSS()))
    print("Number of Patterns: " + str(len(obj.getPatterns())))

    # from PAMI.frequentPattern.basic.FPGrowth import FPGrowth
    # obj = FPGrowth("datasets\\transactional\\transactional_pumsb_star.csv", minimum_support, "\t")
    # obj.startMine()
    # print("Run Time: " + str(obj.getRuntime()))
    # print("Memory Usage: " + str(obj.getMemoryUSS()))
    # print("Number of Patterns: " + str(len(obj.getPatterns())))