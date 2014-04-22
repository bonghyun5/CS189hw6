package cs189hw6;

import java.util.ArrayList;

public class Sample {
	final static int pixelsSize = 28 * 28;
	
	private ArrayList<Double> pixels = new ArrayList<Double>();
	private int actualClass;
	private int predictedClass;
	
	ArrayList<Double> getPixels() {
		return pixels;
	}
	
	void setPixels(ArrayList<Double> pixels) {
		this.pixels = pixels;
	}
	
	int getActualClass() {
		return actualClass;
	}
	
	void setActualClass(int actualClass) {
		this.actualClass = actualClass;
	}
	
	int getPredictedClass() {
		return predictedClass;
	}
	
	void setPredictedClass(int predictedClass) {
		this.predictedClass = predictedClass;
	}
	
}
