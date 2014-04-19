package cs189hw6;

import java.util.*;

public class Classifier {
	
	public static void main(String[] str) {
		//For CrossValidation
		String inputFileNameX = "/Users/bonghyunkim/Desktop/Xtrain.txt";
		String inputFileNameY = "/Users/bonghyunkim/Desktop/Ytrain.txt";
		crossValidationSingleNeuralNetwork(inputFileNameX, inputFileNameY);
	}
	
	static void crossValidationSingleNeuralNetwork(String inputFileNameX, String inputFileNameY) {
		System.out.println("CrossValidation SingleNeuralNetwork");
		ArrayList<Sample> allSamples = Parser.parse(inputFileNameX, inputFileNameY);
		double[] learningRates = {0.1};
		int[] numEpochs = {100};
		int numTestingSamples = (int) allSamples.size();
		
		for (double learningRate : learningRates) {
			for (int numEpoch : numEpochs) {
				Collections.shuffle(allSamples);
				ArrayList<Double> errorRates = new ArrayList<Double>();
				for (int i = 0; i < 10; i++) {
					ArrayList<Sample> trainSample = new ArrayList<Sample>();
					ArrayList<Sample> testSample = new ArrayList<Sample>();
					for (int j = 0; j < numTestingSamples; j++) {
						testSample.add(allSamples.get(j));
					}
					for (int k = numTestingSamples; k < allSamples.size(); k++) {
						trainSample.add(allSamples.get(k));
					}
					SingleNeuralNetwork neuralNetwork = new SingleNeuralNetwork(trainSample, learningRate, numEpoch);
					neuralNetwork.classifyAll(testSample);
					double errorRate = getErrorRate(testSample);
					errorRates.add(errorRate);
					for (int m = 0; m < numTestingSamples; m++) {
						allSamples.add(allSamples.remove(0));
					}
				}
				System.out.println(learningRate + ", " + numEpoch + ", " + getAvgErrorRate(errorRates));
			}
		}
	}
	
	private static double getErrorRate(ArrayList<Sample> samples) {
		int wrongClassification = 0;
		for (Sample sample : samples) {
			if (sample.getActualClass() != sample.getPredictedClass()) {
				wrongClassification ++;
			}
		}
 		return (double) wrongClassification / samples.size();
	}
	
	private static double getAvgErrorRate(ArrayList<Double> errorRates) {
		Double totalError = 0.0;
		for (Double err : errorRates) {
			totalError = totalError + err;
		}
		return (double) (totalError / errorRates.size());
	}
	
}
