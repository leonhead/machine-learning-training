package com.qubit.dl4j;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.factory.Nd4j;



public class DataController {

	final static Logger logger =  LogManager.getLogger(DataController.class);

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
		INDArray features = dataLoader.getAllData().getFeatures();
		INDArray labels = dataLoader.getAllData().getLabels();
		
		  //Plot the data:
        double xMin = 0;
        double xMax = 1.0;
        double yMin = -0.2;
        double yMax = 0.8;

        //Let's evaluate the predictions at every point in the x/y input space
        int nPointsPerAxis = 100;
        double[][] evalPoints = new double[nPointsPerAxis*nPointsPerAxis][2];
        int count = 0;
        for( int i=0; i<nPointsPerAxis; i++ ){
            for( int j=0; j<nPointsPerAxis; j++ ){
                double x = i * (xMax-xMin)/(nPointsPerAxis-1) + xMin;
                double y = j * (yMax-yMin)/(nPointsPerAxis-1) + yMin;

                evalPoints[count][0] = x;
                evalPoints[count][1] = y;

                count++;
            }
        }

		
		
		INDArray backgroundIn = Nd4j.create(evalPoints);
        INDArray backgroundOut = neuronalNetwork.getModel().output(backgroundIn);
		
		view.plotData(features, labels, backgroundIn, backgroundOut, nPointsPerAxis);
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

	public void testRegressionNeuronalNetwork(){
		logger.debug("Evaluate model...");
		String evaluationResults = neuronalNetwork.evalRegression(preprocessingHandler.getTestData());
		logger.info("\n"+evaluationResults);
	}
	public void testClassificationNeuronalNetwork(int outputLayer){
		logger.debug("Evaluate model...");
		String evaluationResults = neuronalNetwork.evalClassification(preprocessingHandler.getTestData(),outputLayer);
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
