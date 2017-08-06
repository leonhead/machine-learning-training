package com.qubit.dl4j;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class DatasetLoader {

	private DataSet allData;

	public DatasetLoader(String filename, int batchSize, int numLinesToSkip, String delimiter, int targetRegressionIndex) throws FileNotFoundException, IOException, InterruptedException {
		loadCSVRecordData(filename, batchSize, numLinesToSkip, delimiter,targetRegressionIndex);
	}


	private DataSet loadCSVRecordData(String filename, int batchSize, int numLinesToSkip, String delimiter, int targetRegressionIndex) throws FileNotFoundException, IOException, InterruptedException{

		RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);	
		recordReader.initialize(new FileSplit(new ClassPathResource(filename).getFile()));

		DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, targetRegressionIndex, targetRegressionIndex, true);
		allData = iterator.next();

		return allData;
	}


	public DataSet getAllData() {
		return allData;
	}

}
