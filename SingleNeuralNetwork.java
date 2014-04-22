package cs189hw6;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class SingleNeuralNetwork {
	final static int pixelsSize = 28 * 28;
	final static int nin = pixelsSize;
	final static int nout = 10;
	final static int batchSize = 200;
	final static String MEAN_SQUARE_ERROR = "meanSquareError";
	final static String CROSS_ENTROPY_ERROR = "crossEntropyError";
	
	double learningRate;
	int numEpochs;
	String errorEquation;
	
	double[][] weightMatrix = new double[nin][nout];
	double[] bias = new double[nout];
	
	SingleNeuralNetwork(ArrayList<Sample> samples, double learningRate, int numEpochs, String errorEquation) {
		this.learningRate = learningRate;
		this.numEpochs = numEpochs;
		this.errorEquation = errorEquation;
		initializeWeightMatrix();
		initializeBias();
		train(samples, learningRate, numEpochs, errorEquation);
		//printWeightMatrix(weightMatrix);
		//printBias(bias);
	}
	
	void initializeWeightMatrix() {
		for (int i = 0 ; i < nin; i++) {
			for (int j = 0; j < nout; j++) {
				Random r = new Random();
				weightMatrix[i][j] = r.nextDouble() - 0.5;
			}
		}
	}
	
	void initializeBias() {
		
	}
	
	void train(ArrayList<Sample> samples, double learningRate, int numEpochs, String errorEquation) {
		for (int epoch = 0; epoch < numEpochs; epoch ++) {
			//Shuffle and Split into mini-batches
			ArrayList<ArrayList<Sample>> miniBatches = shuffleAndGetMiniBatches(samples, batchSize);
			//System.out.println(miniBatches);
			//For each minimatches
			for (ArrayList<Sample> miniBatch : miniBatches) {
				//Compute gradient and update accordingly
				updateWeightBiasMatrix(miniBatch, learningRate, errorEquation);
			}
		}
	}
	
	void classify(Sample sample) {
		int predictedClass = 0;
		double bestActivationVal = 0;
		for (int i = 0; i < nout; i ++) {
			double activationVal = getActivationVal(sample, i);
			//System.out.println(activationVal);
			//System.out.println("activationVal:" + activationVal);
			if (activationVal > bestActivationVal) {
				predictedClass = i;
				bestActivationVal = activationVal;
			}
		}
		sample.setPredictedClass(predictedClass);
	}
	
	void classifyAll(ArrayList<Sample> samples) {
		for (Sample sample : samples) {
			classify(sample);
		}
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
	
	private void updateWeightBiasMatrix(ArrayList<Sample> miniBatch, double learningRate, String errorEquation) {
		double[][] updatedWeightMatrix = new double[nin][nout];
		double[] updatedBias = new double[nout];
		for (Sample sample : miniBatch) {
			for (int j = 0; j < nout; j ++) {
				double y = getPredictVal(sample, j);
				double t = getTrueVal(sample, j);
				double deltaBias = getDeltaValBias(y, t, errorEquation);
				updatedBias[j] = updatedBias[j] + deltaBias;
				for (int i = 0; i < nin; i++) {
					double x = getXiVal(sample, i);
					//System.out.println("X:" + x);
					double delta = getDeltaValWeight(y, t, x, errorEquation);
					//System.out.println(delta);
					updatedWeightMatrix[i][j] = updatedWeightMatrix[i][j] + delta;
				}	
			}
		}
		for (int j = 0; j < nout; j++) {
			bias[j] = bias[j] - (learningRate * updatedBias[j]);
			for (int i = 0; i < nin; i++) {
				weightMatrix[i][j] = weightMatrix[i][j] - (learningRate * updatedWeightMatrix[i][j]);
			}
		}
	}
	
	private double getActivationVal(Sample sample, int j) {
		double totalActivationVal = 0.0;
		ArrayList<Double> pixels = sample.getPixels();
		for (int i = 0; i < pixels.size(); i++) {
			totalActivationVal = totalActivationVal + (weightMatrix[i][j] * pixels.get(i));
		}
		totalActivationVal = totalActivationVal + bias[j];
		return sigmoidFunction(totalActivationVal);
	}
	
	private double getPredictVal(Sample sample, int supposed) {
		double totalActivationVal = 0.0;
		ArrayList<Double> pixels = sample.getPixels();
		for (int i = 0; i < pixels.size(); i++) {
			totalActivationVal = totalActivationVal + (weightMatrix[i][supposed] * pixels.get(i));
		}
		totalActivationVal = totalActivationVal + bias[supposed];
		//System.out.println(totalActivationVal);
		return sigmoidFunction(totalActivationVal);
	}
	
	private double getTrueVal(Sample sample, int supposed) {
		if (sample.getActualClass() == supposed) {
			return 1.0;
		} else {
			return 0.0;
		}
	}
	
	private double getXiVal(Sample sample, int i) {
		return (double) sample.getPixels().get(i);
	}
	
	private double getDeltaValBias(double y, double t, String errorEquation) {
		if (errorEquation == MEAN_SQUARE_ERROR) {
			double delta = (y - t) * (y) * (1 - y);
			return delta;
		} else {
			double smallWeight = 0.0001;
			double delta = (-1) * (((t) / (smallWeight + y)) - ((1 - t) / (1 - y + smallWeight))) * (y) * (1.0 - y);
			return delta;
		}
	}
	
	private double getDeltaValWeight(double y, double t, double x, String errorEquation) {
		if (errorEquation == MEAN_SQUARE_ERROR) {
			double delta = (y - t) * (y) * (1 - y) * x;
			return delta;
		} else {
			double smallWeight = 0.0001;
			double delta = (-1) * (((t) / (smallWeight + y)) - ((1 - t) / (1 - y + smallWeight))) * (y) * (1.0 - y) * x;
			return delta;
		}
	}
	
	private double sigmoidFunction(double n) {
		double a = Math.exp(n * -1);
		double b = 1 + a;
		return 1 / b;
	}

	public void printWeightMatrix(double[][] grid) {
		System.out.println("Weight Matrix");
		for(int i = 0; i < nin; i++) {
			for(int j = 0; j < nout; j++) {
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
