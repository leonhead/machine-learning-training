package com.qubit.dl4j.housing;

import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import com.qubit.dl4j.DataController;
import com.qubit.dl4j.DataPlot;
import com.qubit.dl4j.DataPreprocessing;
import com.qubit.dl4j.DatasetLoader;


public class HousingTestMain {

	public static final String DATASET_PATH = "com/qubit/datasets/housing.data";
	public static final int LINE_TO_SKIP = 0;
	public static final int BATCH_SIZE  = 100;
	public static final String DELIMITER  = ",";
	public static final int TRAGET_REGRESSION_INDEX = 13;
	public static final double PERCENT_OF_TRAIN = 0.65;
	

	public static final int INPUT_LAYER = 13;
	public static final int HIDDEN_LAYER = 10;
	public static final int OUTPUT_LAYER = 1;

	public static void main(String[] args) {


		DataPlot view = new DataPlot();
		DatasetLoader dataLoader;
		try {
			dataLoader = new DatasetLoader(DATASET_PATH,BATCH_SIZE,LINE_TO_SKIP,DELIMITER,TRAGET_REGRESSION_INDEX);

			DataPreprocessing preprocessingHandler = new DataPreprocessing();
			DataController controller = new DataController(view, preprocessingHandler,dataLoader);

			controller.createTestAndTrainDataSet(PERCENT_OF_TRAIN);
			controller.normilizeData();
			controller.createNeuronalNetwork(INPUT_LAYER,HIDDEN_LAYER,OUTPUT_LAYER);
			controller.trainNeuronalNetwork();
			controller.testNeuronalNetwork();


			// 0.09744	0	5.96	0	0.499	5.841	61.4	3.3779	5	279	19.2	377.56	11.41	20
			// 0.01311	90	1.22	0	0.403	7.249	21.9	8.6966	5	226	17.9	395.93	4.81	35.4
			//final INDArray input = Nd4j.create(new double[] {0.09744,0,5.96,0,0.499,5.841,61.4,3.3779,5,279,19.2,377.56,11.41}, new int[] { 1, 13 });
			//final INDArray expect = Nd4j.create(new double[] {20}, new int[] { 1, 1 });
			final INDArray input = Nd4j.create(new double[] {0.09744,0,5.96,0,0.499,5.841,61.4,3.3779,5,279,19.2,377.56,11.41}, new int[] { 1, 13 });
			final INDArray expect = Nd4j.create(new double[] {20}, new int[] { 1, 1 });
			DataSet inputSet = new DataSet(input, expect);
			controller.getNormilizer().transform(inputSet);

			INDArray predictInput = inputSet.getFeatures();	
			controller.predictNeuronalNetwork(predictInput);

		} catch (IOException | InterruptedException e) {
			e.printStackTrace();
		}


	}

}
