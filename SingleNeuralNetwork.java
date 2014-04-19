package cs189hw6;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SingleNeuralNetwork {
	final static int pixelsSize = 28 * 28;
	final static int nin = pixelsSize;
	final static int nout = 10;
	final static int batchSize = 200;
	
	double learningRate;
	int numEpochs;
	
	double[][] weightMatrix = new double[nin][nout];
	double[] bias = new double[nout];
	
	SingleNeuralNetwork(ArrayList<Sample> samples, double learningRate, int numEpochs) {
		this.learningRate = learningRate;
		this.numEpochs = numEpochs;
		train(samples, learningRate, numEpochs);
	}
	
	void train(ArrayList<Sample> samples, double learningRate, int numEpochs) {
		for (int epoch = 0; epoch < numEpochs; epoch ++) {
			//Shuffle and Split into mini-batches
			ArrayList<ArrayList<Sample>> miniBatches = shuffleAndGetMiniBatches(samples, batchSize);
			//For each minimatches
			for (ArrayList<Sample> miniBatch : miniBatches) {
				//Compute gradient of Loss function
				double[][]  gradientLoss = computeGradientLoss();
				//Update according to gradient
				weightMatrix = updateWeightMatrix(weightMatrix, gradientLoss, learningRate);
			}
		}
	}
	
	void classify(Sample sample) {
		//Classify samples and put into samples.predictedClass
	}
	
	void classifyAll(ArrayList<Sample> samples) {
		//Classify all samples
	}
	
	private ArrayList<ArrayList<Sample>> shuffleAndGetMiniBatches(List<Sample> samples, int batchSize) {
		ArrayList<ArrayList<Sample>> miniBatches = new ArrayList<ArrayList<Sample>>();
		Collections.shuffle(samples);
		int numBatches = (int) Math.ceil(samples.size() / batchSize);
		for (int i = 0; i < numBatches; i ++) {
			int fromIndex = i * batchSize;
			int toIndex = fromIndex + batchSize;
			toIndex = (toIndex > samples.size()) ? samples.size() : toIndex;
			ArrayList<Sample> batch = (ArrayList<Sample>) samples.subList(fromIndex, toIndex);
			miniBatches.add(batch);
		}
		return miniBatches;
	}
	
	private double[][] computeGradientLoss() {
		//Compute Gradient Loss
		return null;
	}
	
	private double[][] updateWeightMatrix(double[][] old, double[][] gradientLoss, double learningRate) {
		double[][] updatedWeightMatrix = new double[nin][nout];
		for (int i = 0; i < nin; i ++) {
			for (int j = 0; j < nout; j++) {
				updatedWeightMatrix[i][j] = old[i][j] + (learningRate * gradientLoss[i][j]);
			}
		}
		return updatedWeightMatrix;
	}
		
}
