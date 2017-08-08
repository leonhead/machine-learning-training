package com.qubit.dl4j;

import org.apache.log4j.Logger;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;



public class DataController {

	final static Logger logger = Logger.getLogger(DataController.class);

	private DataPreprocessing preprocessingHandler;
	private DataPlot view;	
	private DatasetLoader dataLoader;
	private NeuronalNetwork neuronalNetwork;

	public DataController(DataPlot view,DataPreprocessing preprocessingHandler,DatasetLoader dataLoader){
		this.dataLoader = dataLoader;
		this.preprocessingHandler = preprocessingHandler;
		this.view = view;

	}

	public void plot(){
		view.plot();
	}
	
	public void enableUI(){
		neuronalNetwork.enableUI();
	}

	public void createTestAndTrainDataSet(double percentOfTrain){
		SplitTestAndTrain trainAndTest = preprocessingHandler.splitTrainAndTest(dataLoader.getAllData(), percentOfTrain);
		preprocessingHandler.setTestData(trainAndTest.getTest());
		preprocessingHandler.setTrainingData(trainAndTest.getTrain());
	}

	public void normilizeData(){
		preprocessingHandler.normilizeData();
	}

	public void createNeuronalNetwork(MultiLayerConfiguration conf){
		logger.debug("Build model...");
		neuronalNetwork = new NeuronalNetwork();
		neuronalNetwork.createNetwork(conf);

	}

	public void trainNeuronalNetwork(int nEpochs){
		logger.debug("Train model...");
		neuronalNetwork.train(preprocessingHandler.getTrainingData(),nEpochs);	

	}

	public void testNeuronalNetwork(){
		logger.debug("Evaluate model...");
		String evaluationResults = neuronalNetwork.eval(preprocessingHandler.getTestData());
		logger.info("\n"+evaluationResults);
	}

	public void predictNeuronalNetwork(INDArray input){

		INDArray out = neuronalNetwork.predict(input,false);
		logger.debug("predict: "+out);
	}

	public DataNormalization getNormilizer(){
		return preprocessingHandler.getNormalizer();
	}

}
