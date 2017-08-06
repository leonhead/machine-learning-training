package com.qubit.dl4j.bitcoin;

import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import com.qubit.dl4j.DataController;
import com.qubit.dl4j.DataPlot;
import com.qubit.dl4j.DataPreprocessing;
import com.qubit.dl4j.DatasetLoader;


public class BitcoinMain {

	public static final String DATASET_PATH = "com/qubit/datasets/btc.csv";
	public static final int LINE_TO_SKIP = 1;
	public static final int BATCH_SIZE  = 100;
	public static final String DELIMITER  = ",";
	public static final int TRAGET_REGRESSION_INDEX = 10;
	public static final double PERCENT_OF_TRAIN = 0.65;

	public static final int INPUT_LAYER = 10;
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

			// 3213	1	50	1498680256079	52615.4253779639	4104.9213721832	234.0809635049	0.2517	9256.2101516481	||205246.068609161||	6	85677426.3714731
			final INDArray input = Nd4j.create(new double[] {3213,1,50,1498680256,52615.4253779639,4104.9213721832,234.0809635049,0.2517,9256.2101516481,85677426.3714731}, new int[] { controller.getOutputlayerSize(), controller.getInputLayerSize() });		
			final INDArray expect = Nd4j.create(new double[]{205246.068609161});
			DataSet inputSet = new DataSet(input, expect);
			controller.getNormilizer().transform(inputSet);

			INDArray predictInput = inputSet.getFeatures();	
			controller.predictNeuronalNetwork(predictInput);

		} catch (IOException | InterruptedException e) {
			e.printStackTrace();
		}
	}

}
