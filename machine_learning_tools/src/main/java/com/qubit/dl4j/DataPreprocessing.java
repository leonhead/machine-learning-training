package com.qubit.dl4j;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

public class DataPreprocessing {
	
	private DataSet trainingData;
	private DataSet testData;
	private DataNormalization normalizer;
	

	public DataPreprocessing(){
		normalizer = new NormalizerStandardize();
	}
	
	public SplitTestAndTrain splitTrainAndTest(DataSet allData, double percentOfTrain){

		allData.shuffle();
		SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(percentOfTrain);
		
		return testAndTrain;
	
	}
	
	public void normilizeData(){
		
		normalizer.fit(trainingData); 
        normalizer.transform(trainingData);
        normalizer.transform(testData); 
	}


	public DataSet getTrainingData() {
		return trainingData;
	}



	public void setTrainingData(DataSet trainingData) {
		this.trainingData = trainingData;
	}



	public DataSet getTestData() {
		return testData;
	}



	public void setTestData(DataSet testData) {
		this.testData = testData;
	}
	
	public DataNormalization getNormalizer() {
		return normalizer;
	}

}
