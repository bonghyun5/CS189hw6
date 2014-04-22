package cs189hw6;

import java.util.*;

public class Classifier {
	final static String MEAN_SQUARE_ERROR = "meanSquareError";
	final static String CROSS_ENTROPY_ERROR = "crossEntropyError";
	
	public static void main(String[] str) {
		//For CrossValidation
		String inputFileNameX = "/Users/bonghyunkim/Desktop/data/data/Xtrain_1000.txt";
		String inputFileNameY = "/Users/bonghyunkim/Desktop/data/data/Ytrain_1000.txt";
		crossValidationSingleNeuralNetwork(inputFileNameX, inputFileNameY);
	}
	
	static void crossValidationSingleNeuralNetwork(String inputFileNameX, String inputFileNameY) {
		System.out.println("CrossValidation SingleNeuralNetwork");
		ArrayList<Sample> allSamples = Parser.parse(inputFileNameX, inputFileNameY);
		double[] learningRates = {0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
		int[] numEpochs = {100};
		int numTestingSamples = (int) (allSamples.size() / 10);
		System.out.println(allSamples.get(0).getPixels());
		allSamples = normalizeData(allSamples);
		System.out.println(allSamples.get(0).getPixels());
		System.out.println(numTestingSamples);
		for (double learningRate : learningRates) {
			for (int numEpoch : numEpochs) {
				Collections.shuffle(allSamples);
				ArrayList<Double> errorRates = new ArrayList<Double>();
				//for (int i = 0; i < 10; i++) {
					ArrayList<Sample> trainSample = new ArrayList<Sample>();
					ArrayList<Sample> testSample = new ArrayList<Sample>();
					for (int j = 0; j < numTestingSamples; j++) {
						testSample.add(allSamples.get(j));
					}
					for (int k = numTestingSamples; k < allSamples.size(); k++) {
						trainSample.add(allSamples.get(k));
					}
					SingleNeuralNetwork neuralNetwork = new SingleNeuralNetwork(trainSample, learningRate, numEpoch, MEAN_SQUARE_ERROR);
					neuralNetwork.classifyAll(testSample);
					double errorRate = getErrorRate(testSample);
					errorRates.add(errorRate);
					for (int m = 0; m < numTestingSamples; m++) {
						allSamples.add(allSamples.remove(0));
					}
				//}
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
 		return (double) wrongClassification / (double) samples.size();
	}
	
	private static double getAvgErrorRate(ArrayList<Double> errorRates) {
		Double totalError = 0.0;
		for (Double err : errorRates) {
			totalError = totalError + err;
		}
		return (double) ((double) totalError / (double) errorRates.size());
	}
	

	static ArrayList<Sample> normalizeData(ArrayList<Sample> samples) {
		ArrayList<Sample> normalizedSamples = new ArrayList<Sample>();
		for (Sample sample : samples) {
			ArrayList<Double> pixels = sample.getPixels();
			pixels = normalize(pixels);
			sample.setPixels(pixels);
			normalizedSamples.add(sample);
		}
		return normalizedSamples;
	}
	
	static private ArrayList<Double> normalize(ArrayList<Double> data) {
		double mean = getMean(data);
		double stdDev = getStdDev(data);
		ArrayList<Double> normalizedData = new ArrayList<Double>();
		for (double d : data) {
			normalizedData.add(((double) (d - mean)) / (stdDev));
		}
		return normalizedData;
	}
	
	static private double getMean(ArrayList<Double> data) {
		double sum = 0.0;
		for (double i : data) {
			sum = sum + i;
		}
		return (double) sum / (double) data.size();
	}
	
   static  private double getVariance(ArrayList<Double> data) {
        double mean = getMean(data);
        double temp = 0;
        for(double a :data) {
            temp += (mean-a)*(mean-a);
        }
        return temp / (double) data.size();
    }

    static private double getStdDev(ArrayList<Double> data) {
        return Math.sqrt(getVariance(data));
    }
}
