package com.qubit.dl4j.moon;

import java.io.IOException;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import com.qubit.dl4j.DataController;
import com.qubit.dl4j.DataPlot;
import com.qubit.dl4j.DataPreprocessing;
import com.qubit.dl4j.DatasetLoader;

public class MoonTestMain {

	public static final String TRAIN_DATASET_PATH = "com/qubit/datasets/moon_data_train.csv";
	public static final String TEST_DATASET_PATH = "com/qubit/datasets/moon_data_eval.csv";
	public static final int LINE_TO_SKIP = 0;
	public static final int BATCH_SIZE  = 50;
	public static final String DELIMITER  = ",";

	public static final int INPUT_LAYER = 2;
	public static final int HIDDEN_LAYER = 20;
	public static final int OUTPUT_LAYER = 2;

	//Random number generator seed, for reproducability
	public static final int seed = 123;
	//Number of iterations per minibatch
	public static final int iterations = 1;
	//Number of epochs (full passes of the data)
	public static final int nEpochs = 100;
	//Network learning rate
	public static final double learningRate = 0.005;

	public static void main(String[] args) {

		DataPlot view = new DataPlot();
		DatasetLoader dataLoader;
		try {
			dataLoader = new DatasetLoader();
			dataLoader.loadCSVRecordData(TRAIN_DATASET_PATH,BATCH_SIZE,LINE_TO_SKIP,DELIMITER,OUTPUT_LAYER);
			
			DataPreprocessing preprocessingHandler = new DataPreprocessing();
			preprocessingHandler.setTrainingData(dataLoader.getAllData());
			dataLoader.loadCSVRecordData(TEST_DATASET_PATH,BATCH_SIZE,LINE_TO_SKIP,DELIMITER,OUTPUT_LAYER);
			preprocessingHandler.setTestData(dataLoader.getAllData());
			
			DataController controller = new DataController(view, preprocessingHandler,dataLoader);


			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.seed(seed)
					.iterations(1)
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
					.learningRate(learningRate)
					.updater(Updater.NESTEROVS).momentum(0.9)
					.list()
					.layer(0, new DenseLayer.Builder().nIn(INPUT_LAYER).nOut(HIDDEN_LAYER)
							.weightInit(WeightInit.XAVIER)
							.activation(Activation.RELU)
							.build())
					.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
							.weightInit(WeightInit.XAVIER)
							.activation(Activation.SOFTMAX)
							.nIn(HIDDEN_LAYER).nOut(OUTPUT_LAYER).build())
					.pretrain(false).backprop(true).build();

			//controller.normilizeData();
			controller.createNeuronalNetwork(conf);

			controller.trainNeuronalNetwork(nEpochs);
			controller.testClassificationNeuronalNetwork(OUTPUT_LAYER);
			controller.plot();

		}
		catch (IOException | InterruptedException e) {
			e.printStackTrace();
		}

	}
}
