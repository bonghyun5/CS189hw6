package cs189hw6;

import java.io.*;
import java.util.*;

/*
 * Class to parse the data file in to ArrayList of Samples 
 */
public class Parser {
	
	/*
	 * Returns an ArrayList of ArrayLists containing all the training data
	 */
	protected static ArrayList<Sample> parse(String xFileLocation) {
		BufferedReader br = null;
		ArrayList<Sample> samples = new ArrayList<Sample>();

		try {
			String sCurrentLine;
			br = new BufferedReader(new FileReader(xFileLocation));
			while ((sCurrentLine = br.readLine()) != null) {
				String[] arr = sCurrentLine.split(",");
				ArrayList<Double> pixels = new ArrayList<Double>();
				for (int i = 0; i < arr.length; i++) {
					pixels.add(Double.parseDouble(arr[i]));
				}
				Sample s = new Sample();
				s.setPixels(pixels);
				samples.add(s);
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if (br != null)br.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
		return samples;
	}
	
	/*
	 * Returns an ArrayList of ArrayLists containing all the training data with actual spam value
	 */
	protected static ArrayList<Sample> parse(String xFileLocation, String yFileLocation) {
		BufferedReader br = null;
		BufferedReader br2 = null;
		//ArrayList of ArrayLists, each represents the length 57 vector of spam attribute values
		//	and the also the y value. 3450 ArrayLists of size 58 ArrayLists.
		ArrayList<Sample> samples = new ArrayList<Sample>();
		
		try {
			String sCurrentLine;
			br = new BufferedReader(new FileReader(xFileLocation));
			while ((sCurrentLine = br.readLine()) != null) {
				String[] arr = sCurrentLine.split(",");
				ArrayList<Double> pixels = new ArrayList<Double>();
				for (int i = 0; i < arr.length; i++) {
					pixels.add(Double.parseDouble(arr[i]));
				}
				Sample s = new Sample();
				s.setPixels(pixels);
				samples.add(s);
			}
			
			br2 = new BufferedReader(new FileReader(yFileLocation));
			
			int i = 0;
			while ((sCurrentLine = br2.readLine()) != null) {
				samples.get(i).setActualClass(Integer.parseInt(sCurrentLine));
				i++;
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if (br != null)br.close();
				if (br2 != null)br2.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
		return samples;
	}
	
}
