package cs189hw6;

import java.util.ArrayList;

public class Sample {
	final static int pixelsSize = 28 * 28;
	
	private ArrayList<Integer> pixels = new ArrayList<Integer>();
	private int actualClass;
	private int predictedClass;
	
	ArrayList<Integer> getPixels() {
		return pixels;
	}
	
	void setPixels(ArrayList<Integer> pixels) {
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
