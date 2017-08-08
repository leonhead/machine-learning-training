package com.qubit.dl4j;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

public class NeuronalNetwork {


	private MultiLayerNetwork model;

	public NeuronalNetwork() {
	}
	
	public void createNetwork(MultiLayerConfiguration conf){

		model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(1));
	}


	public void train(DataSet trainingData, int nEpochs) {

		for( int i=0; i<nEpochs; i++ ){
			model.fit(trainingData);
		}	
	}
	
	public void enableUI() {
		//Initialize the user interface backend
	    UIServer uiServer = UIServer.getInstance();
	    //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
	    StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later   
	    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
	    uiServer.attach(statsStorage);
	    //Then add the StatsListener to collect this information from the network, as it trains
	    model.setListeners(new StatsListener(statsStorage));	
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

}
