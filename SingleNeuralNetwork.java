package cs189hw6;

public class SingleNeuralNetwork {
	final static int pixelsSize = 28 * 28;
	final static int nin = pixelsSize;
	final static int nout = 10;
	
	private double[][] weightMatrix = new double[nin][nout];
	private double[] bias = new double[nout];
	
	
}
