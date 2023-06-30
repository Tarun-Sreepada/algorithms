#  Copyright (C)  2021 Rage Uday Kiran
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#      (at your option) any later version.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.



import os
import time
import mmap
import psutil
import cupy as cp
import numpy as np
import numba as nb


searchGPU = cp.RawKernel(r'''

#define uint32_t unsigned int

extern "C" __global__
void searchGPU(uint32_t *items, uint32_t *utils, uint32_t *indexesStart, uint32_t *indexesEnd, uint32_t numTransactions,
                uint32_t *candidates, uint32_t candidateSize, uint32_t numCandidates,
                uint32_t *candidateCost, uint32_t *candidateLocalUtil, uint32_t *candidateSubtreeUtil,
                uint32_t *secondaryReference, uint32_t *secondaries, uint32_t numSecondaries)
{

    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numTransactions) return;
    uint32_t *cands = new uint32_t[candidateSize];

    uint32_t start = indexesStart[tid];
    uint32_t end = indexesEnd[tid];

    for (uint32_t i = 0; i < numCandidates; i++) {
        for (uint32_t j = 0; j < candidateSize; j++) {
            cands[j] = candidates[i * candidateSize + j];
        }

        uint32_t found = 0;
        uint32_t foundCost = 0;
        uint32_t foundLoc = 0;

        for (uint32_t j = start; j < end; j++)
        {
            if (items[j] == cands[found])
            {
                found++;
                foundCost += utils[j];
                foundLoc = j;
            }
        }

        if (found != candidateSize) continue;

        atomicAdd(&candidateCost[i], foundCost);

        for (uint32_t j = foundLoc + 1; j < end; j++)
        {
            if (secondaries[secondaryReference[i] * numSecondaries + items[j]])
            {
                foundCost += utils[j];
            }
        }

        uint32_t temp = 0;
        for (uint32_t j = foundLoc + 1; j < end; j++)
        {
            if (secondaries[secondaryReference[i] * numSecondaries + items[j]])
            {
                atomicAdd(&candidateLocalUtil[i * numSecondaries + items[j]], foundCost);
                atomicAdd(&candidateSubtreeUtil[i * numSecondaries + items[j]], foundCost - temp);
                temp += utils[j];
            }
        }
    }

    delete[] cands;

}

''', 'searchGPU')


searchGPUSharedMem = cp.RawKernel(r'''
#define uint32_t unsigned int

extern __shared__ uint32_t shared[];

extern "C" __global__
void searchGPUSharedMem(uint32_t *items, uint32_t *utils, uint32_t *indexesStart, uint32_t *indexesEnd, uint32_t numTransactions,
                uint32_t *candidates, uint32_t candidateSize, uint32_t numCandidates,
                uint32_t *candidateCost, uint32_t *candidateLocalUtil, uint32_t *candidateSubtreeUtil,
                uint32_t *secondaryReference, uint32_t *secondaries, uint32_t numSecondaries)
{
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numTransactions) return;

    // Calculate the starting index in shared memory for the current transaction
    uint32_t sharedLoc = tid % 32;
    uint32_t sharedStart = 0;
    for (uint32_t i = tid - sharedLoc; i < tid; i++)
    {
        sharedStart += 2 * (indexesEnd[i] - indexesStart[i]) + candidateSize;
    }

    // Copy items and utils to shared memory
    uint32_t start = indexesStart[tid];
    uint32_t end = indexesEnd[tid];
    uint32_t size = end - start;
    
    for (uint32_t i = start; i < end; i++)
    {
        shared[sharedStart + i - start + candidateSize] = items[i];
    }

    for (uint32_t i = start; i < end; i++)
    {
        shared[sharedStart + i - start + candidateSize + size] = utils[i];
    }
    
    // Calculate the starting indices for items and utils in shared memory
    uint32_t sharedItemStart = sharedStart + candidateSize;
    uint32_t sharedUtilStart = sharedStart + candidateSize + size;

    // Iterate over the candidates
    for (uint32_t i = 0; i < numCandidates; i++) {
        // Copy the current candidate to shared memory
        for (uint32_t j = 0; j < candidateSize; j++) {
            shared[sharedStart + j] = candidates[i * candidateSize + j];
        }

        uint32_t found = 0;
        uint32_t foundCost = 0;
        uint32_t foundLoc = 0;

        // Search for the current candidate in the shared items
        for (uint32_t j = sharedItemStart; j < sharedItemStart + size; j++)
        {
            if (shared[j] == shared[sharedStart + found])
            {
                found++;
                foundCost += shared[sharedUtilStart + j - sharedItemStart];
                foundLoc = j;
            }
        }

        if (found != candidateSize) continue;

        // Update the candidate cost
        atomicAdd(&candidateCost[i], foundCost);

        // Update candidateLocalUtil and candidateSubtreeUtil
        for (uint32_t j = foundLoc + 1; j < sharedItemStart + size; j++)
        {
            if (secondaries[secondaryReference[i] * numSecondaries + shared[j]])
            {
                foundCost += shared[sharedUtilStart + j - sharedItemStart];
            }
        }

        uint32_t temp = 0;
        for (uint32_t j = foundLoc + 1; j < sharedItemStart + size; j++)
        {
            if (secondaries[secondaryReference[i] * numSecondaries + shared[j]])
            {
                atomicAdd(&candidateLocalUtil[i * numSecondaries + shared[j]], foundCost);
                atomicAdd(&candidateSubtreeUtil[i * numSecondaries + shared[j]], foundCost - temp);
                temp += shared[sharedUtilStart + j - sharedItemStart];
            }
        }
    }
}


''', 'searchGPUSharedMem')

class GPUEFIM:

    """
    EFIM is one of the fastest algorithm to mine High Utility ItemSets from transactional databases.
    
    Reference:
    ---------
        Zida, S., Fournier-Viger, P., Lin, J.CW. et al. EFIM: a fast and memory efficient algorithm for
        high-utility itemset mining. Knowl Inf Syst 51, 595–625 (2017). https://doi.org/10.1007/s10115-016-0986-0
    
    Attributes:
    ----------
        inputFile (str): The input file path.
        minUtil (int): The minimum utility threshold.
        sep (str): The separator used in the input file.
        threads (int): The number of threads to use.
        Patterns (dict): A dictionary containing the discovered patterns.
        rename (dict): A dictionary containing the mapping between the item IDs and their names.
        runtime (float): The runtime of the algorithm in seconds.
        memoryRSS (int): The Resident Set Size (RSS) memory usage of the algorithm in bytes.
        memoryUSS (int): The Unique Set Size (USS) memory usage of the algorithm in bytes.

    Methods:
    -------
        read_file(): Read the input file and return the filtered transactions, primary items, and secondary items.
        search(collections): Search for high utility itemsets in the given collections.
        startMine(): Start the EFIM algorithm.
        savePatterns(outputFile): Save the patterns discovered by the algorithm to an output file.
        getPatterns(): Get the patterns discovered by the algorithm.
        getRuntime(): Get the runtime of the algorithm.
        getMemoryRSS(): Get the Resident Set Size (RSS) memory usage of the algorithm.
        getMemoryUSS(): Get the Unique Set Size (USS) memory usage of the algorithm.
        printResults(): Print the results of the algorithm.

    """


    def __init__(self, inputFile, minUtil, sep = '\t'):
        self.inputFile = inputFile
        self.minUtil = minUtil
        self.sep = sep
        self.Patterns = {}
        self.rename = {}


    # Read input file
    def read_file(self):
        file_data = []
        twu = {}

        with open(self.inputFile, 'r') as f:
            fd = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

            for line in iter(fd.readline, b""):
                line = line.decode('utf-8').strip().split(":")
                
                # Parse and process the line
                line = [x.split(self.sep) for x in line]
                weight = int(line[1][0])

                # Update file data with the parsed items
                file_data.append([line[0], [int(x) for x in line[2]]])

                for k in line[0]:
                    if k not in twu:
                        twu[k] = weight
                    else:
                        twu[k] += weight

        # Filter TWU dictionary based on minUtil (minimum utility threshold)
        twu = {k: v for k, v in twu.items() if v >= self.minUtil}

        # Sort TWU items by utility
        twu = {k: v for k, v in sorted(twu.items(), key=lambda item: item[1], reverse=True)}

        strToInt = {}
        t = len(twu)
        for k in twu.keys():
            strToInt[k] = t
            self.rename[t] = k
            t -= 1

        secondary = set(self.rename.keys())

        # Filter and sort transactions
        subtree = {}
        filtered_transactions = {}

        for col in file_data:
            zipped = zip(col[0], col[1])
            transaction = [(strToInt[x], y) for x, y in zipped if x in strToInt]
            transaction = sorted(transaction, key=lambda x: x[0])
            if len(transaction) > 0:
                val = [x[1] for x in transaction]
                key = [x[0] for x in transaction]
                
                fs = frozenset(key)

                if fs not in filtered_transactions:
                    filtered_transactions[fs] = [key, val, 0]
                else:
                    filtered_transactions[fs][1] = [x + y for x, y in zip(filtered_transactions[fs][1], val)]

                subUtil = sum([x[1] for x in transaction])
                temp = 0

                for i in range(len(transaction)):
                    item = key[i]
                    if item not in subtree:
                        subtree[item] = subUtil - temp
                    else:
                        subtree[item] += subUtil - temp
                    temp += val[i]

        indexesStart = [0]
        indexesEnd = []
        items = []
        utils = []

        # sort filtered transactions by length
        filtered_transactions = {k: v for k, v in sorted(filtered_transactions.items(), key=lambda item: len(item[1][0]), reverse=True)}

        # get the sum of len of top 32 transactions
        sumLen = 0
        for k, v in list(filtered_transactions.items())[:32]:
            sumLen += len(v[0])
        self.memoryConsumption = 32 * sumLen * 2
        self.fourtyeightkb = 384000

        # for k,v in filtered_transactions.items():
        #     print(k, ":", v)

        for key in filtered_transactions.keys():
            # print(indexesStart[-1], end="|")
            indexesEnd.append(indexesStart[-1] + len(filtered_transactions[key][0]))
            items.extend(filtered_transactions[key][0])
            utils.extend(filtered_transactions[key][1])
            # for i in range(len(filtered_transactions[key][0])):
            #     print(str(filtered_transactions[key][0][i]) + ":" + str(filtered_transactions[key][1][i]), end=" ")
            # print("|", indexesEnd[-1])
            indexesStart.append(indexesEnd[-1])

        indexesStart.pop()

        self.items = cp.array(items, dtype=np.uint32)
        self.utils = cp.array(utils, dtype=np.uint32)
        self.indexesStart = cp.array(indexesStart, dtype=np.uint32)
        self.indexesEnd = cp.array(indexesEnd, dtype=np.uint32)

        primary = [key for key in subtree.keys() if subtree[key] >= self.minUtil]

        # secondary is from 0 to len(secondary) - 1
        secondary = [i for i in range(len(secondary) + 1)]

        self.secondaryLen = len(secondary)
        self.numTransactions = len(filtered_transactions)
        
        return primary, secondary
    
    @staticmethod
    @nb.njit
    def filter_subtree_util(subtree_util, min_util):
        new_subtree_util = []
        for j in range(len(subtree_util)):
            if subtree_util[j] >= min_util:
                new_subtree_util.append(j)
        return new_subtree_util
    
    def search(self, collection):

        while len(collection) > 0:
            print("Collections: ", len(collection), "Patterns: ", len(self.Patterns))
            candidates = []
            secondaryReference = []
            secondaries = []

            temp = 0
            for item in collection:
                primaries = item[1]
                candidates.extend([item[0] + [primary] for primary in primaries])
                secondaryReference.extend([temp] * len(primaries))
                secondaries.extend(item[2])
                temp += 1



            candidateSize = len(collection[0][0]) + 1
            numCandidates = len(candidates)
            # print("Candidates: ", numCandidates)

            # flatten candidates
            candidates = [item for sublist in candidates for item in sublist]
            candidates = cp.array(candidates, dtype=np.uint32)

            # print(secondaries)
            secondaries = cp.array(secondaries, dtype=np.uint32)
            secondaryReference = cp.array(secondaryReference, dtype=np.uint32)

            costs = cp.zeros(numCandidates, dtype=np.uint32)
            localUtil = cp.zeros(numCandidates * self.secondaryLen, dtype=np.uint32)
            subtreeUtil = cp.zeros(numCandidates * self.secondaryLen, dtype=np.uint32)

            # items, utils, indexesStart, indexesEnd, numTransactions
            # candidates, candidateSize, numCandidates,
            # candidateCost, candidateLocalUtil, candidateSubtreeUtil, 
            # secondaryReference, secondaries, numSecondaries

            numOfThreads = 32
            numOfBlocks = self.numTransactions // numOfThreads + 1

            additionalMemConsump = 0

            if numCandidates > 32:
                additionalMemConsump = 32 * candidateSize * 32 * 1.5
            else:
                additionalMemConsump = numCandidates * candidateSize * 32 * 1.5
            # print("additionalMemConsump + self.memoryConsumption / self.fourtyeightkb:", additionalMemConsump, "+", self.memoryConsumption, "/", self.fourtyeightkb)

            if additionalMemConsump + self.memoryConsumption > self.fourtyeightkb or numCandidates < 10:
                searchGPU((numOfBlocks,), (numOfThreads,), 
                        (self.items, self.utils, self.indexesStart, self.indexesEnd, self.numTransactions,
                            candidates, candidateSize, numCandidates,  
                            costs, localUtil, subtreeUtil,
                            secondaryReference, secondaries, self.secondaryLen))
            else:
                searchGPUSharedMem((numOfBlocks,), (numOfThreads,), 
                        (self.items, self.utils, self.indexesStart, self.indexesEnd, self.numTransactions,
                            candidates, candidateSize, numCandidates,  
                            costs, localUtil, subtreeUtil,
                            secondaryReference, secondaries, self.secondaryLen), shared_mem=(additionalMemConsump + self.memoryConsumption) / 8)

                    
            cp.cuda.runtime.deviceSynchronize()

            # subtreeUtil = cp.where(subtreeUtil == 0, cp.iinfo(np.uint32).max, subtreeUtil)

            # get results from GPU
            candidates = candidates.get()
            costs = costs.get()

            # localUtil set number to 0 if less than minUtil else set to 1
            localUtil = cp.where(localUtil < self.minUtil, 0, 1)

            localUtil = localUtil.get()
            subtreeUtil = subtreeUtil.get()

            # resize candidates
            candidates = np.resize(candidates, (numCandidates, candidateSize))
            localUtil = np.resize(localUtil, (numCandidates, self.secondaryLen))
            subtreeUtil = np.resize(subtreeUtil, (numCandidates, self.secondaryLen))

            newCollections = []
            #  collection = [[[], primary, secondary]]  

            for i in range(numCandidates):
                # print(candidates[i], costs[i], subtreeUtil[i], localUtil[i])
                if costs[i] >= self.minUtil:
                    # self.Patterns[tuple(candidates[i])] = costs[i]
                    self.Patterns[tuple([self.rename[item] for item in candidates[i]])] = costs[i]
                
                # newSubtreeUtil = []
                # for j in range(len(subtreeUtil[i])):
                #     if subtreeUtil[i][j] >= self.minUtil:
                #         newSubtreeUtil.append(j)
                newSubtreeUtil = self.filter_subtree_util(subtreeUtil[i], self.minUtil)
                if len(newSubtreeUtil) > 0 and cp.sum(localUtil[i]) > 0:
                    # print(candidates[i], newSubtreeUtil, newLocalUtil)
                    newCollections.append([list(candidates[i]), newSubtreeUtil, localUtil[i]])
                    # print()
                
            collection = newCollections

    def savePatterns(self, outputFile):
        with open(outputFile, 'w') as f:
            for key, value in self.Patterns.items():
                joined = " ".join(key) + " #UTIL: " + str(value) + "\n"
                f.write(joined)

    def startMine(self):
        """
        Start the EFIM algorithm.

        Returns:
            None
        """

        ps = psutil.Process(os.getpid())

        self.start = time.time()

        primary, secondary = self.read_file()

        collection = [[[], primary, secondary]]

        self.search(collection)

        self.memoryRSS = ps.memory_info().rss
        self.memoryUSS = ps.memory_full_info().uss

        end = time.time()
        self.runtime = end - self.start

        # newPatterns = {}
        # for key, value in self.Patterns.items():
        #     newKey = tuple([self.rename[x] for x in key])
        #     newPatterns[newKey] = value
        
        # self.Patterns = newPatterns


    def getPatterns(self):
        """
        Get the patterns discovered by the algorithm.

        Returns:
            dict: A dictionary containing the discovered patterns.
        """
        return self.Patterns

    def getRuntime(self):
        """
        Get the runtime of the algorithm.

        Returns:
            float: The runtime in seconds.
        """
        return self.runtime

    def getMemoryRSS(self):
        """
        Get the Resident Set Size (RSS) memory usage of the algorithm.

        Returns:
            int: The RSS memory usage in bytes.
        """
        return self.memoryRSS

    def getMemoryUSS(self):
        """
        Get the Unique Set Size (USS) memory usage of the algorithm.

        Returns:
            int: The USS memory usage in bytes.
        """
        return self.memoryUSS

    
    def printResults(self):
        print("Total number of High Utility Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in seconds:", self.getRuntime())


if __name__ == "__main__":

    # inputFile = 'EFIM/accidents_utility_spmf.txt'
    # minUtil = 28000000

    inputFile = 'EFIM/chainstore.txt'
    minUtil = 3600000

    # inputFile = 'EFIM/test.txt'
    # minUtil = 5

    # inputFile = "EFIM/BMS_utility_spmf.txt"
    # minUtil = 2020000

    # inputFile = "EFIM/Utility_pumsb.csv"
    # minUtil = 11000000

    sep = " "
    f = GPUEFIM(inputFile, minUtil, sep)
    f.startMine()
    f.savePatterns("output.txt")
    print("# of patterns: " + str(len(f.getPatterns())))
    print("Time taken: " + str(f.getRuntime()))

