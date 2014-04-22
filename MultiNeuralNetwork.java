package cs189hw6;

import java.util.ArrayList;

public class MultiNeuralNetwork {
	final static int pixelsSize = 28 * 28;
	final static int nin = pixelsSize;
	final static int nhid1 = 300;
	final static int nhid2 = 100;
	final static int nout = 10;
	final static int batchSize = 200;
	
	double learningRate;
	int numEpochs;
	
	double[][] weightHid1Matrix = new double[nin][nhid1];
	double[][] weightHid2Matrix = new double[nhid1][nhid2];
	double[][] weightOutputMatrix = new double[nhid2][nout];
	double[] biasHid1 = new double[nhid1];
	double[] biasHid2 = new double[nhid2];
	double[] biasOutput = new double[nout];
	
	MultiNeuralNetwork(ArrayList<Sample> samples, double learningRate, int numEpochs) {
		initializeWeightMatrix();
		initializeBias();
		this.learningRate = learningRate;
		this.numEpochs = numEpochs;
		train(samples, learningRate, numEpochs);
	}
	
	void initializeWeightMatrix() {
		
	}
	
	void initializeBias() {
		
	}
	
	void train(ArrayList<Sample> samples, double learningRate, int numEpochs) {
		
	}
	
	void classify(Sample sample) {
		
	}
	
	void classifyAll(ArrayList<Sample> samples) {
		
	}
	
	
}
