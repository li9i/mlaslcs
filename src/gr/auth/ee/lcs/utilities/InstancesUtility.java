/*
 *	Copyright (C) 2011 by Allamanis Miltiadis
 *
 *	Permission is hereby granted, free of charge, to any person obtaining a copy
 *	of this software and associated documentation files (the "Software"), to deal
 *	in the Software without restriction, including without limitation the rights
 *	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the Software is
 *	furnished to do so, subject to the following conditions:
 *
 *	The above copyright notice and this permission notice shall be included in
 *	all copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *	THE SOFTWARE.
 */
/**
 * 
 */
package gr.auth.ee.lcs.utilities;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Vector;

import weka.core.Instance;
import weka.core.Instances;

/**
 * A utility class for converting a Weka Instance to a double array
 * 
 * @author Miltiadis Allamanis
 * 
 */

public final class InstancesUtility {
	
	public static Vector<Instances[]> testInstances = new Vector<Instances[]>();
	public static Vector<Instances[]> trainInstances = new Vector<Instances[]>();

	/**
	 * Perform the conversion.
	 * 
	 * @param set
	 *            the set containing the instances
	 * @return a double[][] containing the instances and their respective
	 *         attributes
	 */
	public static double[][] convertIntancesToDouble(final Instances set) {
		if (set == null)
			return null;

		final double[][] result = new double[set.numInstances()][set
				.numAttributes()];
		for (int i = 0; i < set.numInstances(); i++) {

			for (int j = 0; j < set.numAttributes(); j++) {
				result[i][j] = set.instance(i).value(j);
			}
		}

		return result;

	}

	/**
	 * Opens an file and creates an instance
	 * 
	 * @param filename
	 * @return the Weka Instances opened by the file
	 * @throws IOException
	 */
	public static Instances openInstance(final String filename)
			throws IOException {
		final FileReader reader = new FileReader(filename);
		return new Instances(reader);
	};

	/**
	 * Private Constructor to avoid instantiation.
	 */
	private InstancesUtility() {
	}
	

	
	/**
	 * Returns the label cardinality of the specified set.
	 * 
	 */
	public static double getLabelCardinality (final Instances set) { 
		if (set == null) return -1;
		
		int numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels", 1);
		double sumOfLabels = 0;

		for (int i = 0; i < set.numInstances(); i++) {
			for (int j = set.numAttributes() - numberOfLabels; j < set.numAttributes(); j++) {
				sumOfLabels += set.instance(i).value(j);
			}
		}
		
		//System.out.println("sumOfLabels:" + sumOfLabels);
		//System.out.println("instances.length:" + set.numInstances());

		
		if (set.numInstances()!= 0) {
			//System.out.println("labelCardinality:" + (double) (sumOfLabels / set.numInstances()));

			return (double) (sumOfLabels / set.numInstances()); 
		}
		return 0;
	}
	
	/**
	 * Ta instances einai pollaplasia tou ari9mou ton folds.
	 * Apo ena sunolo apo instances mou epistrefei to kommati mikous instances.numInstances / numberOfFolds me index index. 
	 * O index ksekinaei apo to miden.
	 * Stin ousia to xrisimopoio otan xorizo ena partition apo intances se test set kai train set.
	 * 9a paro ena kommati gia test kai ta upoloipa gia train. Ego apla dino to index tou test, kai ta upoloipa 9a ginoun train.
	 * see splitPartitionIntoFolds
	 * 
	 * _____
	 * |_6_| index = 0
	 * |_6_|		 1
	 * |_6_|		 2 
	 * |_6_|		 3
	 * |_6_|		 4	
	 * |_6_|		 5
	 * |_6_|		 6
	 * |_6_|		 7		
	 * |_6_|		 8
	 * |_6_|		 9
	 * 
	 * */
	public static Instances getPartitionSegment(Instances instances, int index, int numberOfFolds) {
		
		// an exei ginei la9os, epistrepse null
		if (instances.numInstances() % numberOfFolds != 0) {
			System.out.println("Number of instances not a multiple of " + numberOfFolds);
			return null;
		}
		
		int numberOfInstancesToGet = instances.numInstances() / numberOfFolds;
		Instances segment = new Instances(instances, numberOfInstancesToGet);
		
		for (int i = index * numberOfInstancesToGet; i < (index + 1) * numberOfInstancesToGet; i++) {
			segment.add(instances.instance(i));
		}
		return segment;
	}
	
	
	
	
	
	/**
	 * Splits the .arff input dataset to |number-of-distinct-label-combinations| Instances which are stored in the partitions[] array. 
	 * Called by initializePopulation() as a preparatory step to clustering.
	 * @throws Exception 
	 * 
	 * */
	
	public static Instances[] partitionInstances (final AbstractLearningClassifierSystem lcs, 
													final String filename) 
																			throws Exception {

		// Open .arff
		final Instances set = InstancesUtility.openInstance(filename);
		if (set.classIndex() < 0) {
			set.setClassIndex(set.numAttributes() - 1);
		}
		//set.randomize(new Random());
		int numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels", 1);
		
		// the partitions vector holds the indices		
		String stringsArray[] = new String [lcs.instances.length];
		int indicesArray[] = new int [lcs.instances.length];

		
		// metatrepo to labelset gia ka9e deigma se string kai to apo9ikeuo ston pinaka stringsArray
		for (int i = 0; i < set.numInstances(); i++) {
			stringsArray[i] = "";
			indicesArray[i] = i; // isos kai na mi xreiazetai. an randomize() xreiazetai profanos

			for (int j = set.numAttributes() - numberOfLabels; j < set.numAttributes(); j++) {
				stringsArray[i] += (int) set.instance(i).value(j);
			}
		}

		// contains the indicesVector(s)
		Vector<Vector> mothershipVector = new Vector<Vector>();
		
		String baseString = "";
		for (int i = 0; i < set.numInstances(); i++) {
			
			baseString = stringsArray[i];
			if (baseString.equals("")) continue;
			Vector<Integer> indicesVector = new Vector<Integer>();
			
			for (int j = 0; j < set.numInstances(); j++) {
				if (baseString.equals(stringsArray[j])) {
					stringsArray[j] = "";
					indicesVector.add(j);
				}
			}
			mothershipVector.add(indicesVector);
		}
		
		Instances[] partitions = new Instances[mothershipVector.size()];
		
		for (int i = 0; i < mothershipVector.size(); i++) {
			partitions[i] =  new Instances(set, mothershipVector.elementAt(i).size());
			for (int j = 0; j < mothershipVector.elementAt(i).size(); j++) {
				Instance instanceToAdd = set.instance((Integer) mothershipVector.elementAt(i).elementAt(j));
				partitions[i].add(instanceToAdd);
			}
		}	
		/*
		 * mexri edo exei sximastistei o pinakas partitions, pou periexei stis diafores 9eseis tou, xorismeno to dataset ana sunduasmous klaseon.
		 * periexei kai ta attributes kai ta labels, alla gia to clustering i eisodos prepei na einai mono ta attributes,
		 * synepos prepei na diagrapsoume ta labels. to analambanei i initializePopulation()
		 */
		return partitions;
	}
	
	
	
	public static Instances[] partitionInstances (final AbstractLearningClassifierSystem lcs, 
													final Instances trainSet) 
									throws Exception {

		// Open .arff
		final Instances set = trainSet;
		if (set.classIndex() < 0) {
			set.setClassIndex(set.numAttributes() - 1);
		}
		//set.randomize(new Random());
		int numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels", 1);
		
		
		
		
		
		
		
		
		
		
		// the partitions vector holds the indices		
		String stringsArray[] = new String[trainSet.numInstances()];//new String [lcs.instances.length];
		int indicesArray[] = new int [trainSet.numInstances()];//new int [lcs.instances.length];
		
		
		
		
		
		
		
		
		
		
		
		
		// metatrepo to labelset gia ka9e deigma se string kai to apo9ikeuo ston pinaka stringsArray
		for (int i = 0; i < set.numInstances(); i++) {
			stringsArray[i] = "";
			indicesArray[i] = i; // isos kai na mi xreiazetai. an randomize() xreiazetai profanos
		
			for (int j = set.numAttributes() - numberOfLabels; j < set.numAttributes(); j++) {
				stringsArray[i] += (int) set.instance(i).value(j);
			}
		}
		
		// contains the indicesVector(s)
		Vector<Vector> mothershipVector = new Vector<Vector>();
		
		String baseString = "";
		for (int i = 0; i < set.numInstances(); i++) {
		
			baseString = stringsArray[i];
			if (baseString.equals("")) continue;
			Vector<Integer> indicesVector = new Vector<Integer>();
			
			for (int j = 0; j < set.numInstances(); j++) {
				if (baseString.equals(stringsArray[j])) {
					stringsArray[j] = "";
					indicesVector.add(j);
				}
			}
			mothershipVector.add(indicesVector);
		}
		
		Instances[] partitions = new Instances[mothershipVector.size()];
		
		for (int i = 0; i < mothershipVector.size(); i++) {
			partitions[i] =  new Instances(set, mothershipVector.elementAt(i).size());
			for (int j = 0; j < mothershipVector.elementAt(i).size(); j++) {
				Instance instanceToAdd = set.instance((Integer) mothershipVector.elementAt(i).elementAt(j));
				partitions[i].add(instanceToAdd);
			}
		}	
		/*
		* mexri edo exei sximastistei o pinakas partitions, pou periexei stis diafores 9eseis tou, xorismeno to dataset ana sunduasmous klaseon.
		* periexei kai ta attributes kai ta labels, alla gia to clustering i eisodos prepei na einai mono ta attributes,
		* synepos prepei na diagrapsoume ta labels. to analambanei i initializePopulation()
		*/
		return partitions;
		}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	public static void splitDatasetIntoFolds (final AbstractLearningClassifierSystem lcs, 
												final Instances dataset,
												final int numberOfFolds) throws Exception {

		Instances[] partitions = InstancesUtility.partitionInstances(lcs, dataset);
		
		testInstances.setSize(partitions.length);
		trainInstances.setSize(partitions.length);
		
		
		int lowerBound = (int) Math.floor((double) dataset.numInstances() / (double) numberOfFolds);
		int upperBound = (int) Math.ceil((double) dataset.numInstances() / (double) numberOfFolds);
		
		// 9elo lowerBound <= numberOfTestInstancesPerFold[i] <= upperBound
		int [] numberOfTestInstancesPerFold = new int[numberOfFolds];

		
		// esto oti X partitions exoun partitions[i].numInstances() > numberOfFolds. 
		// tote, ta vector testInstances kai trainInstances 9a periexoun meta apo to splitPartitionIntoFolds X arrays dld X stoixeia
		// de 9a exoun aparaitita partitions.length stoixeia. an ego saroso gia partitions.length omos borei na paro nullPointerException i arrayOutOfBounds
		
		Vector<Integer> vectorOfPartitionIndices = new Vector<Integer>();
		


		
		//System.out.println("\nVector of test instances only with bulk:\n");
		for (int i = 0; i < partitions.length; i++) {
			
			if (partitions[i].numInstances() > numberOfFolds) {
				InstancesUtility.splitPartitionIntoFolds(partitions[i], numberOfFolds, i);
				vectorOfPartitionIndices.add(i);
			}	
			else {
				
				
				Instances[] emptyArrayTest = new Instances[numberOfFolds];
				Instances[] emptyArrayTrain = new Instances[numberOfFolds];

				for (int j = 0; j < numberOfFolds; j++) {
					emptyArrayTest[j] = new Instances (partitions[0], partitions[i].numInstances());
					emptyArrayTrain[j] = new Instances (partitions[0], partitions[i].numInstances());

				}
				//placeholders
				InstancesUtility.testInstances.add(i, emptyArrayTest);
				InstancesUtility.trainInstances.add(i, emptyArrayTrain);
			}	
		}
		

		// se auto to simeio ola ta partitions me numInstances > numFolds exoun xoristei katallila.
		// auto pou menei einai na xorisoume ta leftovers, 1on apo ta parapano partitions, kai 2on apo auta pou eksarxis eixan numInstances < numFolds
		
		//System.out.println("testInstances: " + testInstances.size()); // partitions.length + vectorOfPartitionIndices logo metatopisis arxika (bulk)
																	  // meta, 2 * partitions.length

		
		
		for (int i = 0; i < numberOfFolds; i++) {
			int instancesSum = 0;
			for (int j = 0; j < vectorOfPartitionIndices.size(); j++) {
				instancesSum += InstancesUtility.testInstances.elementAt(vectorOfPartitionIndices.elementAt(j))[i].numInstances();	
			}
			
			// auto einai o arxikos ari9mos apo intances sto test gia ka9e fold
			numberOfTestInstancesPerFold[i] = instancesSum;
			//System.out.println("numberOfTestInstancesPerFold[" + i + "] = " + numberOfTestInstancesPerFold[i]);
		}
		
/*		for (int i = 0; i < partitions.length; i++) {
			System.out.println();
			System.out.print("i = " + i + " |");
			for (int j = 0; j < numberOfFolds; j++) {
				System.out.print("_" + testInstances.elementAt(i)[j].numInstances() + "|");
			}
		}
		System.out.println("\n");*/

		
		/*
		 * 
		 *  i = 0 |_0|_0|_0|_0|_0|_0|_0|_0|_0|_0|
			i = 1 |_0|_0|_0|_0|_0|_0|_0|_0|_0|_0|
			i = 2 |_0|_0|_0|_0|_0|_0|_0|_0|_0|_0|
			i = 3 |_0|_0|_0|_0|_0|_0|_0|_0|_0|_0|
			i = 4 |_0|_0|_0|_0|_0|_0|_0|_0|_0|_0|
			i = 5 |_1|_1|_1|_1|_1|_1|_1|_1|_1|_1|
			i = 6 |_3|_3|_3|_3|_3|_3|_3|_3|_3|_3|
			i = 7 |_6|_6|_6|_6|_6|_6|_6|_6|_6|_6|
		 * 
		 * 
		 * */

		
		
		for (int i = 0; i < partitions.length; i++) {

			int numberOfLeftoverInstances = partitions[i].numInstances() % numberOfFolds; // eg 64 % 10 = 4
			Instances leftoverInstances = new Instances (partitions[i], numberOfLeftoverInstances);

			if (numberOfLeftoverInstances > 0) {
				
				
				// 9a ksekiniso apo to telos. etsi ki allios ta {numberOfLeftoverInstances} teleutaia se ka9e partitions 
				// afisa sta partitions gia ta opoia kalesa tin splitPartitionIntoFolds.
				for (int k = partitions[i].numInstances() - 1; k >= partitions[i].numInstances() - numberOfLeftoverInstances; k--) {
					leftoverInstances.add(partitions[i].instance(k));
				}
				
	/*		    ArrayList<Integer> numbers = new ArrayList<Integer>();
	
			    for (int k = 0; k < numberOfLeftoverInstances; k++) {
			      numbers.add(k);
			    }
			    Collections.shuffle(numbers);*/
			    
				// gia ka9e partition kano randomize ta folds. 9a epilekso na balo ta leftover instances sta prota {numberOfLeftoverInstances} folds
				// ta opoia omos einai randomly katanamimena. an den itan randomly ta prota folds 9a upirxe anisokatanomi. 
				// panta sta prota 9a upirxan instances tou protou partition kai paei legontas
				
			    ArrayList<Integer> folds = new ArrayList<Integer>();
				
			    for (int k = 0; k < numberOfFolds; k++) {
			    	folds.add(k);
			    }
			 
			    Collections.shuffle(folds);  
			    
			    
			    
				int j = 0;
				while (leftoverInstances.numInstances() > 0) {
				    int foldIndex = folds.get(j);
					//System.out.println(foldIndex);

					if (numberOfTestInstancesPerFold[foldIndex] < upperBound) {
	
						Instance toBeAdded = leftoverInstances.instance(0);
						
						
						// bale to proto instance ton leftovers se test
						testInstances.elementAt(i)[foldIndex].add(toBeAdded);
						
/*						for (int k = 0; k < numberOfFolds; k++) {
							System.out.println("i = " + i + " fold = " + k);
							System.out.println(testInstances.elementAt(i)[foldIndex]);
						}*/
						
						//System.out.println("added " + toBeAdded);
						numberOfTestInstancesPerFold[foldIndex]++;
						
						// auto pou topo9eti9ike se test gia to trexon fold prepei na bei se ola ta alla folds os train, ektos apo to trexon fold
						for (int k = 0; k < numberOfFolds; k++) {
							if (k != foldIndex) {
								trainInstances.elementAt(i)[k].add(toBeAdded);
							}
						}
						
						// afairese to instance pou balame se test
						leftoverInstances.delete(0);
						
					}
					j++;
					// if j hits the roof reset it. 
					//borei na uparxoun folds pou akoma den exoun ftasei to ano orio tous kai na ta paraleipsoume
					if (j == numberOfFolds)
						j = 0;
				}
			}
			
/*			System.out.print("i = " + i + " |");
			for (int j = 0; j < numberOfFolds; j++) {
				System.out.print("_" + testInstances.elementAt(i)[j].numInstances() + "|");
			}
			System.out.println();*/	
		}
		
		
/*		for (int i = 0; i < numberOfFolds; i++) {
			System.out.println(numberOfTestInstancesPerFold[i]);
		}


		System.out.println("tests:");
		for (int i = 0; i < partitions.length; i++) {
			System.out.print("i = " + i + " |");
			for (int j = 0; j < numberOfFolds; j++) {
				System.out.print("_" + testInstances.elementAt(i)[j].numInstances() + "|");
			}
			System.out.println();
		}
		
		System.out.println("trains:");
		for (int i = 0; i < partitions.length; i++) {
			System.out.print("i = " + i + " |");
			for (int j = 0; j < numberOfFolds; j++) {
				System.out.print("_" + trainInstances.elementAt(i)[j].numInstances() + "|");
			}
			System.out.println();
		}	*/
	
	}
	
	
	
	
	/**
	 * spaei ena partition (syllogi apo instances pou anikoun ston idio syndyasmo labels) se train set kai test set, afinontas leftovers.
	 * proupo9etei oti partition.numInstances > numberOfFolds.
	 * 
	 * ta leftovers 9a prepei na katanemi9oun etsi oste ka9e test set na exei 
	 * 
	 * floor(totalNumInstances / numberOfFolds) <= testSetNumInstances <= ceil(totalNumInstances / numberOfFolds)
	 */
	public static void splitPartitionIntoFolds (Instances partition, int numberOfFolds, int partitionIndex) {
		
		int numberOfTestInstancesPerFold = partition.numInstances() / numberOfFolds; // eg 64 / 10 = 6
		int numberOfLeftoverInstances = partition.numInstances() % numberOfFolds; // eg 64 % 10 = 4
		int numberOfTrainInstancesPerFold = partition.numInstances() - numberOfTestInstancesPerFold - numberOfLeftoverInstances; // eg 64 - 6 - 4 = 54
		
		Instances[] testArrayPerPartition = new Instances[numberOfFolds];
		Instances[] trainArrayPerPartition = new Instances[numberOfFolds];
		
		Instances bulk = new Instances(partition, partition.numInstances() - numberOfLeftoverInstances);
		// 9a xoriso ta 64 sunolika instances se 6 test, 54 train kai ta upoloipa 4 9a ta afiso stin akri
		// 6 + 54 = 60, pollaplasio tou 10.
		// ta prota 60 9a pane temporarily ston pinaka roundArray gia eukolia
		for (int i = 0; i < partition.numInstances() - numberOfLeftoverInstances; i++) {
			bulk.add(partition.instance(i));
		}
		
		
		for (int i = 0; i < numberOfFolds; i++) {
			testArrayPerPartition[i] = InstancesUtility.getPartitionSegment(bulk, i, numberOfFolds);
			trainArrayPerPartition[i] = new Instances(bulk, numberOfFolds);

			for (int j = 0; j < numberOfFolds; j++) {
				if (j != i) {
					for(int k = 0; k < numberOfTestInstancesPerFold; k++) {
						Instance kthInstance = InstancesUtility.getPartitionSegment(bulk, j, numberOfFolds).instance(k);
						trainArrayPerPartition[i].add(kthInstance);
					}
				}
			}	
		}
		
		// synolika 9a ginoun partitions.length additions
		// topo9etise ton ka9e pinaka sti 9esi pou tou analogei aka analoga me to index tou partition 
		InstancesUtility.testInstances.add(partitionIndex, testArrayPerPartition);
		InstancesUtility.trainInstances.add(partitionIndex, trainArrayPerPartition);
	}
	

	
	
			
	
}
		

