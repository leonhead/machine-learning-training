package com.qubit.dl4j;

import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class NeuronalNetwork {

	//Random number generator seed, for reproducability
	public static final int seed = 12345;
	//Number of iterations per minibatch
	public static final int iterations = 1;
	//Number of epochs (full passes of the data)
	public static final int nEpochs = 100;
	//Network learning rate
	public static final double learningRate = 0.01;

	private MultiLayerNetwork model;

	private int numInput;
	private int numOutputs;


	private int nHidden;

	public NeuronalNetwork(int numInput, int nHidden, int numOutputs) {
		this.numInput = numInput;
		this.nHidden = nHidden;
		this.numOutputs = numOutputs;
	}

	private MultiLayerConfiguration createNetworkConfiguration(){

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
						.nIn(numInput)
						.nOut(nHidden)
						.activation(Activation.TANH)
						.weightInit(WeightInit.XAVIER)
						.build())
				.layer(1, new OutputLayer.Builder(LossFunction.MSE) //create hidden layer
						.nIn(nHidden)
						.nOut(numOutputs)
						.activation(Activation.IDENTITY)
						.weightInit(WeightInit.XAVIER)
						.build())
				.pretrain(false).backprop(true) //use backpropagation to adjust weights
				.build();
		return conf;
	}


	public void train(DataSet trainingData) {

		MultiLayerConfiguration conf = createNetworkConfiguration();
		model = new MultiLayerNetwork(conf);
		model.init();
		//print the score with every 1 iteration
		model.setListeners(new ScoreIterationListener(1));

		for( int i=0; i<nEpochs; i++ ){
			model.fit(trainingData);
		}
	}

	public String eval(DataSet testData) {

		RegressionEvaluation evaluation = new RegressionEvaluation(1);
		INDArray features = testData.getFeatureMatrix();

		INDArray lables = testData.getLabels();
		INDArray predicted = model.output(features, false);

		evaluation.eval(lables, predicted);
		return evaluation.stats();

	}

	public INDArray predict(INDArray input, boolean b) {
		return model.output(input, false);
	}

	public int getNumInput() {
		return numInput;
	}

	public int getNumOutputs() {
		return numOutputs;
	}

}
