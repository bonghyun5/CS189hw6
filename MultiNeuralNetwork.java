package cs189hw6;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MultiNeuralNetwork {
	final static int numLayers = 3;
	final static int pixelsSize = 28 * 28;
	final static int nin = pixelsSize;
	final static int nhid1 = 300;
	final static int nhid2 = 100;
	final static int nout = 10;
	final static int batchSize = 200;
	
	final static String MEAN_SQUARE_ERROR = "meanSquareError";
	final static String CROSS_ENTROPY_ERROR = "crossEntropyError";
	
	double learningRate;
	int numEpochs;
	String errorEquation;
	
	double[][] weightHid1Matrix = new double[nin][nhid1];
	double[][] weightHid2Matrix = new double[nhid1][nhid2];
	double[][] weightOutputMatrix = new double[nhid2][nout];
	double[] biasHid1 = new double[nhid1];
	double[] biasHid2 = new double[nhid2];
	double[] biasOutput = new double[nout];
	ArrayList<ArrayList<Double>> activations;
	ArrayList<ArrayList<Double>> errors;
	
	MultiNeuralNetwork(ArrayList<Sample> samples, double learningRate, int numEpochs, String errorEquation) {
		this.learningRate = learningRate;
		this.numEpochs = numEpochs;
		this.errorEquation = errorEquation;
		initializeWeightMatrix();
		initializeBias();
		train(samples);
	}
	
	void initializeWeightMatrix() {
		double smallWeight = 0.01;
		for (int i = 0 ; i < nin; i++) {
			for (int j = 0; j < nhid1; j++) {
				Random r = new Random();
				weightHid1Matrix[i][j] = (r.nextDouble() - 0.5) * smallWeight;
			}
		}
		for (int i = 0 ; i < nhid1; i++) {
			for (int j = 0; j < nhid2; j++) {
				Random r = new Random();
				weightHid2Matrix[i][j] = (r.nextDouble() - 0.5) * smallWeight;
			}
		}
		for (int i = 0 ; i < nhid2; i++) {
			for (int j = 0; j < nout; j++) {
				Random r = new Random();
				weightOutputMatrix[i][j] = (r.nextDouble() - 0.5) * smallWeight;
			}
		}
	}
	
	void initializeBias() {
		Random r = new Random();
		double smallWeight = 0.01;
		for (int i = 0; i < nhid1; i++) {
			biasHid1[i] = (r.nextDouble() - 0.5) * smallWeight;
		}
		for (int i = 0; i < nhid2; i++) {
			biasHid2[i] = (r.nextDouble() - 0.5) * smallWeight;
		}
		for (int i = 0; i < nout; i++) {
			biasOutput[i] = (r.nextDouble() - 0.5) * smallWeight;
		}
	}
	
	void train(ArrayList<Sample> samples) {
		for (int epoch = 0; epoch < numEpochs; epoch ++) {
			//Shuffle and Split into mini-batches
			ArrayList<ArrayList<Sample>> miniBatches = shuffleAndGetMiniBatches(samples, batchSize);
			//System.out.println(miniBatches);
			//For each minimatches
			for (ArrayList<Sample> miniBatch : miniBatches) {
				//Compute gradient and update accordingly
				updateWeightBiasMatrix(miniBatch, learningRate);
			}
		}
		printWeightMatrix(weightHid1Matrix);
		System.out.println("--");
		printWeightMatrix(weightHid2Matrix);
		System.out.println("--");
		printWeightMatrix(weightOutputMatrix);
	}
	
	void classify(Sample sample) {
		int predictedClass = 0;
		double bestActivationVal = 0;
		for (int i = 0; i < nout; i ++) {
			double activationVal = getActivationVal(sample, i);
			if (activationVal > bestActivationVal) {
				predictedClass = i;
				bestActivationVal = activationVal;
			}
		}
		System.out.println("predictedClass: " + predictedClass);
		System.out.println("activationVal: " + bestActivationVal);
		sample.setPredictedClass(predictedClass);
	}
	
	void classifyAll(ArrayList<Sample> samples) {
		for (Sample sample : samples) {
			classify(sample);
		}
	}
	
	private double getActivationVal(Sample sample, int c) {
		double totalActivationVal = 0.0;
		for (int i = 0; i < nhid2; i ++) {
			double totalActivationi = 0.0;
			for (int j = 0; j < nhid1; j++) {
				double totalActivationj = 0.0;
				for (int k = 0; k < nin; k++) {
					totalActivationj = totalActivationj + (weightHid1Matrix[k][j] * sample.getPixels().get(k));
				}
				totalActivationj = totalActivationj + biasHid1[j];
				//System.out.println("totalActivationj:" + totalActivationj);
				totalActivationj = activationFunction(totalActivationj);
				totalActivationi = totalActivationi + (weightHid2Matrix[j][i] * totalActivationj);
			}
			totalActivationi = totalActivationi + biasHid2[i];
			//System.out.println("totalActivationi:" + totalActivationi);
			totalActivationi = activationFunction(totalActivationi);
			totalActivationVal = totalActivationVal + (weightOutputMatrix[i][c] * totalActivationi);
		}
		//System.out.println("totalActivationVal:" + totalActivationVal);
		totalActivationVal = totalActivationVal + biasOutput[c];
		return activationFunction(totalActivationVal);
	}
	
	private ArrayList<ArrayList<Sample>> shuffleAndGetMiniBatches(List<Sample> samples, int batchSize) {
		ArrayList<ArrayList<Sample>> miniBatches = new ArrayList<ArrayList<Sample>>();
		Collections.shuffle(samples);
		int numBatches = (int) Math.ceil((double) samples.size() / (double) batchSize);
		for (int i = 0; i < numBatches; i ++) {
			int fromIndex = i * batchSize;
			int toIndex = fromIndex + batchSize;
			toIndex = (toIndex > samples.size()) ? samples.size() : toIndex;
			ArrayList<Sample> batch = new ArrayList<Sample>(samples.subList(fromIndex, toIndex));
			miniBatches.add(batch);
		}
		return miniBatches;
	}
	
	private ArrayList<ArrayList<Double>> initializeActivations() {
		ArrayList<ArrayList<Double>> activations = new ArrayList<ArrayList<Double>>();
		activations.add(new ArrayList<Double>(nhid1));
		activations.add(new ArrayList<Double>(nhid2));
		activations.add(new ArrayList<Double>(nout));
		return activations;
	}
	
	private ArrayList<ArrayList<Double>> initializeErrors() {
		ArrayList<ArrayList<Double>> errors = new ArrayList<ArrayList<Double>>();
		errors.add(new ArrayList<Double>(nhid1));
		errors.add(new ArrayList<Double>(nhid2));
		errors.add(new ArrayList<Double>(nout));
		return errors;
	}
	
	
	private double getActivationValue(int layer, int i, ArrayList<Double> xs) {
		double summed = 0.0;
		if (layer == nhid1) {
			for (int j = 0; j < nin; j++) {
				summed = (weightHid1Matrix[j][i] * xs.get(j)) + summed;
			}
		} else if (layer == nhid2) {
			for (int j = 0; j < nhid1; j++) {
				summed = (weightHid2Matrix[j][i] * xs.get(j));
			}
		} else {
			for (int j = 0; j < nout; j++) {
				summed = (weightOutputMatrix[j][j] * xs.get(j));
			}
		}
		return activationFunction(summed);
	}
	
	private double activationFunction(double n) {
		return Math.tanh(n);
	}
	
	private double activationFunctionPrime(double n) {
		return 1 - (n * n);
	}
	
	private double getSummedError(int layer, int i) {
		double summed = 0.0;
		if (layer == nout) {
			for (int j = 0; j < nout; j ++) {
				double t = (i == j) ? 1 : 0;
				double y = activations.get(2).get(j);
				double error = t - y;
				summed = summed + (error * weightOutputMatrix[i][j]);
			}
		} else if (layer == nhid2) {
			for (int j = 0; j < nout; j ++) {
				double error = errors.get(2).get(j);
				summed = summed + (error * weightHid2Matrix[i][j]);
			}
		} else {
			for (int j = 0; j < nhid2; j ++) {
				double error = errors.get(1).get(j);
				summed = summed + (error * weightHid1Matrix[i][j]);
			}
		}
		return summed;
	}
	
	private void updateWeightBiasMatrix(ArrayList<Sample> miniBatch, double learningRate) {
		activations = initializeActivations();
		errors = initializeErrors();
		
		double[][] upWeightHid1Matrix = new double[nin][nhid1];
		double[][] upWeightHid2Matrix = new double[nhid1][nhid2];
		double[][] upWeightOutputMatrix = new double[nhid2][nout];
		double[] upBiasHid1 = new double[nhid1];
		double[] upBiasHid2 = new double[nhid2];
		double[] upBiasOutput = new double[nout];	
		
		for (Sample sample : miniBatch) {
			for (int i = 0; i < nhid1; i++) {
				double activation = getActivationValue(nhid1, i, sample.getPixels());
				activations.get(0).add(activation);
			}
			for (int i = 0; i < nhid2; i++) {
				double activation = getActivationValue(nhid2, i, activations.get(0));
				activations.get(1).add(activation);
			}
			for (int i = 0; i < nout; i++) {
				double activation = getActivationValue(nout, i, activations.get(1));
				activations.get(2).add(activation);
			}
			
			
			for (int j = 0; j < nout; j++) {
				double primed = activationFunctionPrime(activations.get(2).get(j));
				double summedError = getSummedError(nout, j);
				double errorValue = primed * summedError;
				errors.get(2).add(errorValue);
			}
			for (int j = 0; j < nhid2; j++) {
				double primed = activationFunctionPrime(activations.get(1).get(j));
				double summedError = getSummedError(nhid2, j);
				double errorValue = primed * summedError;
				errors.get(1).add(errorValue);
			}
			for (int j = 0; j < nhid1; j++) {
				double primed = activationFunctionPrime(activations.get(0).get(j));
				double summedError = getSummedError(nhid1, j);
				double errorValue = primed * summedError;
				errors.get(0).add(errorValue);
			}
			
			for (int j = 0; j < nhid1; j++) {
				double error = errors.get(0).get(j);
				upBiasHid1[j] = upBiasHid1[j] + error;
				for (int i = 0; i < nin; i++) {
					double x = sample.getPixels().get(i);
					upWeightHid1Matrix[i][j] = upWeightHid1Matrix[i][j] + (x * error);
				}
			}
			
			for (int j = 0; j < nhid2; j++) {
				double error = errors.get(1).get(j);
				upBiasHid2[j] = upBiasHid2[j] + error;
				for (int i = 0; i < nhid1; i++) {
					double x = sample.getPixels().get(i);
					upWeightHid2Matrix[i][j] = upWeightHid2Matrix[i][j] + (x * error);
				}
			}
			
			for (int j = 0; j < nout; j++) {
				double error = errors.get(2).get(j);
				upBiasOutput[j] = upBiasOutput[j] + error;
				for (int i = 0; i < nhid2; i++) {
					double x = sample.getPixels().get(i);
					upWeightOutputMatrix[i][j] = upWeightOutputMatrix[i][j] + (x * error);
				}
			}
		}	
		
		for (int j = 0; j < nout; j++) {
			biasOutput[j] = biasOutput[j] - (learningRate * upBiasOutput[j]);
			for (int i = 0; i < nhid2; i++) {
				weightOutputMatrix[i][j] = weightOutputMatrix[i][j] - (learningRate * upWeightOutputMatrix[i][j]);
			}
		}
		for (int j = 0; j < nhid2; j++) {
			biasHid2[j] = biasHid2[j] - (learningRate * upBiasHid2[j]);
			for (int i = 0; i < nhid1; i++) {
				weightHid2Matrix[i][j] = weightHid2Matrix[i][j] - (learningRate * upWeightHid2Matrix[i][j]);
			}
		}
		for (int j = 0; j < nhid1; j++) {
			biasHid1[j] = biasHid1[j] - (learningRate * upBiasHid1[j]);
			for (int i = 0; i < nin; i++) {
				weightHid1Matrix[i][j] = weightHid1Matrix[i][j] - (learningRate * upWeightHid1Matrix[i][j]);
			}
		}
	}
	
	public void printWeightMatrix(double[][] grid) {
		System.out.println("Weight Matrix");
		for(int i = 0; i < grid.length; i++) {
			for(int j = 0; j < grid[0].length; j++) {
				System.out.print(grid[i][j]);
				System.out.print(" ");
			}
			System.out.println();
		}
	}
	
	public void printBias(double[] grid) {
		System.out.println("Bias");
		for (int i = 0; i < nout; i ++) {
			System.out.print(grid[i]);
		}
		System.out.println();
	}
	
}
