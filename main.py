# Author: Aarav Varshney
# Date: 20/11/2022

import numpy as np
import time
import json
from tabulate import tabulate

class ChannelEncoding:
    def __init__(self, p = 0.01, DEBUG = 1, numSimulations = 1000):
        self.p = p
        self.DEBUG = DEBUG
        self.numSimulations = numSimulations

    # define constants:
    G = np.array([[1, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
    H = np.array([[1, 0, 1, 0], [1, 1, 0, 1]], dtype=int)
    k = 2
    n = 4

    # standard array: 
    COSET_LEADERS = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1,0,0,0], [0,0,1,0]], dtype=int)
    SYNDROMES = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=int)
   
    ''' UTILS '''
    # to get the index of a 2d array given a 1d array
    def getIndex(self, test: np.ndarray(shape=(2, 4)), array: np.ndarray(shape=1)):
        for idx, val in enumerate(test):
            if np.array_equal(val,array):
                return idx     

        # should never come here
        return -1       

    # if flip the bits of the message
    def ifFlipBits(self, p: float):
        a =  np.random.random()
        if a < p:
            return True
        return False

    ''' MAIN FUNCTIONS '''
    # recieve a 2 bit message from source encoder
    def encoding(self, msg: np.ndarray(shape=k, dtype=int)):
        encodedMsg:np.ndarray(shape = self.n, dtype=int) = []

        # multiply the message with G
        encodedMsg.append(np.dot(self.G, msg) % 2)
        return encodedMsg

    # introduce a channel error with probability p
    def simulateChannel(self, encodedMsg: np.ndarray(shape=n, dtype=int), p: float):
        newEncodedMsg = np.ndarray(encodedMsg.shape, dtype=int)
        for i in range(len(encodedMsg)):
            newEncodedMsg[i] = 1 - encodedMsg[i] if self.ifFlipBits(p) else encodedMsg[i]      

        return newEncodedMsg

    # decode the message
    def decoding(self, newEncodedMsg: np.ndarray(shape=n, dtype=int)):
        # multiply the message with H
        syndrome = np.dot(self.H, newEncodedMsg) % 2
        # find the syndrome in the syndromes array
        syndromeIndex = self.getIndex(self.SYNDROMES, syndrome)
        # find the coset leader in the coset leaders array
        cosetLeader = self.COSET_LEADERS[syndromeIndex]

        if self.DEBUG > 1:
            print("Syndrome: ", syndrome)
            print("Syndrome Index: ", syndromeIndex)
            print("Coset Leader: ", cosetLeader)

        # xor the coset leader with the encoded message
        decodedMsg = np.bitwise_xor(cosetLeader, newEncodedMsg)
        return decodedMsg

    # the difference in bits of the original message and the decoded message
    def calculateSER(self, p = None):
        # run the simulation numSimulations times
        totalSER = 0
        if p is None:
            p = self.p
        
        for i in range(self.numSimulations):
            result = self.runSim(p = p)
            totalSER += result["symbolErrorRate"]
            if self.DEBUG >= 1:
                print("Simulation: ", i, " of ", self.numSimulations, " with p = ", p, " and SER = ", result["symbolErrorRate"])
        
        res = float(totalSER / self.numSimulations)
        return res

    def calculateWER(self, p = None):
        # run the simulation numSimulations times
        totalWER = 0
        if p is None:
            p = self.p

        for i in range(self.numSimulations):
            result = self.runSim(p = p)
            totalWER += result["wordErrorRate"]
        return totalWER / self.numSimulations

    def runSim(self, p = None):

        if p is None:
            p = self.p

        # generate a 2 bit message
        msg = np.random.randint(2, size=2)
        encodedMsg = self.encoding(msg)[0]
        newEncodedMsg: np.ndarray(shape=self.n, dtype=int) = self.simulateChannel(encodedMsg, p)
        newDecodedMsg = self.decoding(newEncodedMsg)
        decodedMsg = newDecodedMsg[0:self.k]

        symbolErrorRate = np.sum(np.bitwise_xor(msg, decodedMsg)) / self.k
        wordErrorRate = 1 if symbolErrorRate > 0 else 0

        if self.DEBUG > 1:
            print("Message: ", msg)
            print("Encoded Message: ", encodedMsg)
            print("New Encoded Message: ", newEncodedMsg)
            print("Decoded Message: ", newDecodedMsg)
            print("New Decoded Message: ", decodedMsg)
        
        # return word error and bit error
        result = {
            "wordErrorRate": wordErrorRate,
            "symbolErrorRate": symbolErrorRate
        }

        # TODO: cache the result by saving in json file
        return result 

    ''' TESTS '''
    def testEncoding(self, msg, expected):
        encodedMsg = self.encoding(msg)[0]
        assert np.array_equal(encodedMsg, expected)

    def testSimulateChannel(self, encodedMsg, expected, p):
        newEncodedMsg = self.simulateChannel(encodedMsg, p)
        assert np.array_equal(newEncodedMsg, expected)

    def testDecoding(self, newEncodedMsg, expected):
        newDecodedMsg = self.decoding(newEncodedMsg)
        assert np.array_equal(newDecodedMsg, expected)

    def testCalculateSER(self, expected, p):
        ser = self.calculateSER(p=p)
        assert ser == expected
    
    def testCalculateWER(self, expected, p):
        wer = self.calculateWER(p = p)
        assert wer == expected

    def test(self):
        # test the encoding function
        msgs = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
        encodedMsgs = np.array([[1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 0]])
        for i in range(len(msgs)):
            self.testEncoding(msgs[i], encodedMsgs[i])

        probs = np.array([0, 1])
        expected = np.array([[[1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 0]],
                             [[0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1]]])

        # test the simulate channel function
        for i in range(len(probs)):
            for j in range(len(encodedMsgs)):
                self.testSimulateChannel(encodedMsgs[j], expected[i][j], probs[i])

        # test the decoding function
        expectedNewDecodedMsgs = np.array([[[1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 0]], 
                                           [[0, 0, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]]])

        for i in range(len(probs)):
            for j in range(len(encodedMsgs)):
                self.testDecoding(expected[i][j], expectedNewDecodedMsgs[i][j])

        # test the calculate SER function
        expectedSER = [0, 1]
        for i in range(len(probs)):
            self.testCalculateSER(expectedSER[i], probs[i])


def main():
    # 1000000
    numSimulations = [100, 1000, 10000, 100000, 1000000]
    data = []
    for numSimulation in numSimulations:
        channel = ChannelEncoding(p=0.01, DEBUG=0, numSimulations=numSimulation)
        # channel.test()
        start = time.time()
        ser = channel.calculateSER()
        end = time.time()
        wer = channel.calculateWER()
        elapsed = str(end - start).split(".")[0]
        currentData = [ser, wer, numSimulation, elapsed]
        print("SER: ", ser, " WER: ", wer, " Num Simulations: ", numSimulation, " Elapsed Time: ", elapsed)
        data.append(currentData)
    
    print(tabulate(data, headers=["SER", "WER", "Num Sims", "time"]))
if __name__ == '__main__':
    main()
