package com.qubit.dl4j.housing;

import java.io.IOException;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

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
	
	//Random number generator seed, for reproducability
		public static final int seed = 12345;
		//Number of iterations per minibatch
		public static final int iterations = 1;
		//Number of epochs (full passes of the data)
		public static final int nEpochs = 100;
		//Network learning rate
		public static final double learningRate = 0.01;

	public static void main(String[] args) {


		DataPlot view = new DataPlot();
		DatasetLoader dataLoader;
		try {
			dataLoader = new DatasetLoader(DATASET_PATH,BATCH_SIZE,LINE_TO_SKIP,DELIMITER,TRAGET_REGRESSION_INDEX);

			DataPreprocessing preprocessingHandler = new DataPreprocessing();
			DataController controller = new DataController(view, preprocessingHandler,dataLoader);
			
			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.seed(seed) //include a random seed for reproducibility
					// use stochastic gradient descent as an optimization algorithm
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
					.iterations(iterations)
					.learningRate(learningRate) //specify the learning rate
					.updater(Updater.NESTEROVS).momentum(0.9) //specify the rate of change of the learning rate.
					.regularization(true).l2(1e-4) // regulation of weights	
					.list()
					.layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
							.nIn(INPUT_LAYER)
							.nOut(HIDDEN_LAYER)
							.activation(Activation.TANH)
							.weightInit(WeightInit.XAVIER)
							.build())
					.layer(1, new OutputLayer.Builder(LossFunction.MSE) //create hidden layer
							.nIn(HIDDEN_LAYER)
							.nOut(OUTPUT_LAYER)
							.activation(Activation.IDENTITY)
							.weightInit(WeightInit.XAVIER)
							.build())
					.pretrain(false).backprop(true) //use backpropagation to adjust weights
					.build();

			controller.createTestAndTrainDataSet(PERCENT_OF_TRAIN);
			controller.normilizeData();
			controller.createNeuronalNetwork(conf);
			
			
			
			controller.trainNeuronalNetwork(nEpochs);
			controller.testNeuronalNetwork();


			// 0.09744	0	5.96	0	0.499	5.841	61.4	3.3779	5	279	19.2	377.56	11.41	20
			// 0.01311	90	1.22	0	0.403	7.249	21.9	8.6966	5	226	17.9	395.93	4.81	35.4
			//final INDArray input = Nd4j.create(new double[] {0.09744,0,5.96,0,0.499,5.841,61.4,3.3779,5,279,19.2,377.56,11.41}, new int[] { 1, 13 });
			//final INDArray expect = Nd4j.create(new double[] {20}, new int[] { 1, 1 });
			final INDArray input = Nd4j.create(new double[] {0.09744,0,5.96,0,0.499,5.841,61.4,3.3779,5,279,19.2,377.56,11.41}, new int[] { OUTPUT_LAYER, INPUT_LAYER });
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
