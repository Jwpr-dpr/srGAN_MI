from trainer import *
import fire
import sys
def main(pathToData:str, pathToTrainFile:str, pathToTestFile:str, pathToValFile:str,
          bathSize:int, process:str,learningRate:float, epochs:int, generator_path:str,
          discriminator_path):

    if process == 'training' or 'tunning':
        SR_Trainer.train(pathToData, pathToTrainFile, pathToTestFile, pathToValFile, bathSize,
               process,learningRate, epochs, generator_path)
    elif process == 'test':
        SR_Trainer.test(pathToData, pathToTestFile, pathToValFile, bathSize, learningRate, generator_path, discriminator_path)
    else:
        print("Please select a valid process")
        sys.exit(1)
    return

if __name__ == '__main__':
    fire.Fire(main)
