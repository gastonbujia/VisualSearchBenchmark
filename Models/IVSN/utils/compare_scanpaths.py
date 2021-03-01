import json
from os import mkdir, listdir, path

resultsDir = '../results_python_matlab/'

def main():
    jsonPythonFile = open(resultsDir + 'Scanpaths_Pytorch with VGG16.json', 'r')
    jsonMatlabFile = open(resultsDir + 'Scanpaths_Original model.json', 'r')
    jsonPythonStructs = json.load(jsonPythonFile)
    jsonMatlabStructs = json.load(jsonMatlabFile)
    jsonMatlabFile.close()
    jsonPythonFile.close()
    differences = []
    differentLengthScanpathsBy3OrMore = 0
    differentLengthScanpathsBy2OrLess = 0
    sameLengthDifferentScanpaths = 0
    shorterScanpathsPython = 0
    shorterScanpathsMatlab = 0
    for pythonStruct in jsonPythonStructs:
        for matlabStruct in jsonMatlabStructs:
            if pythonStruct['image'][3:6] == matlabStruct['image'][0:3]: # Me fijo que estoy comparando scanpaths de la misma imagen
                if type(matlabStruct['X']) != type(pythonStruct['X']): # Hay algunos scanpaths que en matlab se guardaron como int en lugar de una lista con un solo int
                    matlabStruct['X'] = [matlabStruct['X']]
                    matlabStruct['Y'] = [matlabStruct['Y']]
                    
                lengthDifference = abs(len(pythonStruct['X']) - len(matlabStruct['X'])) # Comparo las longitudes, si son distintas ya hay algún problema
                xData = compareStructs(pythonStruct, matlabStruct, 'X', 'X', lengthDifference > 0)
                yData = compareStructs(pythonStruct, matlabStruct, 'Y', 'Y', lengthDifference > 0)
                if lengthDifference > 0:
                    if (len(pythonStruct['X']) > len(matlabStruct['X'])):
                        shorterScanpathsMatlab += 1
                    else:
                        shorterScanpathsPython += 1
                    if lengthDifference > 3:
                        differentLengthScanpathsBy3OrMore += 1
                    else:                        
                        differentLengthScanpathsBy2OrLess += 1                        
                else:
                    sameLengthDifferentScanpaths += xData['different paths?']
                differences.append({ "image" : pythonStruct['image'], "length difference" :lengthDifference, "X Python" : xData['python coords'], "X Matlab" : xData['matlab coords'], "Y Python" : yData['python coords'], "Y Matlab" : yData['matlab coords'], "fixations X distance" : xData['coords distance'], "fixations Y distance" : yData['coords distance']})
    print("There are " + str(differentLengthScanpathsBy2OrLess + differentLengthScanpathsBy3OrMore) + " scanpaths with different lengths (" + str(differentLengthScanpathsBy2OrLess) + " with a difference of two fixations or less, " + \
        str(differentLengthScanpathsBy3OrMore) + " with a difference of 3 fixations or more)")
    print("Python version has " + str(shorterScanpathsPython) + " scanpaths with less fixations than its counterpart in the MATLAB version")
    print("MATLAB version has " + str(shorterScanpathsMatlab) + " scanpaths with less fixations than its counterpart in the Python version")
    differences.append({"scanpaths with different lengths by 3 or more fixations" : differentLengthScanpathsBy3OrMore, "scanpaths with different lengths by 2 or less fixations" : differentLengthScanpathsBy2OrLess, "scanpaths with same length but different paths" : sameLengthDifferentScanpaths})
    jsonDifferencesFile = open(resultsDir + 'scanpathsDifferences_pytorch.json', 'w')
    json.dump(differences, jsonDifferencesFile, indent = 4)
    jsonDifferencesFile.close()




def compareStructs(firstStruct, secondStruct, fieldFirstStruct, fieldSecondStruct, isNotSameLength):
    inPython = list(map(int, firstStruct[fieldFirstStruct])) # Estaban almacenados como strings, los paso a ints
    inMatlab = list(map(int, secondStruct[fieldSecondStruct]))
    if isNotSameLength:
        coordsDistance = "the scanpaths don't have the same length" # Si los scanpaths no tienen la misma longitud, no me importa el camino que hicieron
        differentPaths = False
    else:
        # Si los scanpaths tienen la misma longitud, me fijo cuanto difieren ambos recorridos (medida en valor absoluto)
        coordsDistance = list(map(abs, [x - y for x, y in zip(inPython, inMatlab)]))
        differentPaths = bool(sum(list(map(lambda x: int(x > 100), coordsDistance))))
        # map() no devuelve una lista, por eso uso list()
    return{"python coords" :inPython, "matlab coords" : inMatlab, "coords distance" : coordsDistance, "different paths?" : differentPaths}
    

main()
