package com.qubit.dl4j.bitcoin;

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
			dataLoader = new DatasetLoader();
			dataLoader.loadCSVRecordData(DATASET_PATH,BATCH_SIZE,LINE_TO_SKIP,DELIMITER,TRAGET_REGRESSION_INDEX,true);

			DataPreprocessing preprocessingHandler = new DataPreprocessing();
			DataController controller = new DataController(view, preprocessingHandler,dataLoader);

			controller.createTestAndTrainDataSet(PERCENT_OF_TRAIN);
			controller.normilizeData();
			
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
			
			
			controller.createNeuronalNetwork(conf);
			controller.enableUI();
			controller.trainNeuronalNetwork(nEpochs);
			controller.testRegressionNeuronalNetwork();

			// 3213	1	50	1498680256079	52615.4253779639	4104.9213721832	234.0809635049	0.2517	9256.2101516481	||205246.068609161||	6	85677426.3714731
			final INDArray input = Nd4j.create(new double[] {3213,1,50,1498680256,52615.4253779639,4104.9213721832,234.0809635049,0.2517,9256.2101516481,85677426.3714731}, new int[] { OUTPUT_LAYER, INPUT_LAYER });		
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
