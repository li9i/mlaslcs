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
package gr.auth.ee.lcs.classifiers.populationcontrol;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.IPopulationControlStrategy;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.geneticalgorithm.IRuleSelector;

/**
 * A fixed size control strategy. Classifiers are deleted based on the selector
 * tournaments
 * 
 * @stereotype ConcreteStrategy
 * 
 * @author Miltos Allamanis
 * 
 */
public class FixedSizeSetWorstFitnessDeletion implements
		IPopulationControlStrategy {

	private AbstractLearningClassifierSystem myLcs;
	
	/**
	 * The Natural Selector used to select the the classifier to be deleted.
	 * @uml.property  name="mySelector"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private final IRuleSelector mySelector;

	/**
	 * The fixed population size of the controlled set.
	 * @uml.property  name="populationSize"
	 */
	private final int populationSize;
	
	private int numberOfDeletions;
	
	private long deletionTime;

	/**
	 * Removes all zero coverage rules
	 * @uml.property  name="zeroCoverageRemoval"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private InadequeteClassifierDeletionStrategy zeroCoverageRemoval;
	
	private AbstractUpdateStrategy updateStrategy;

	/**
	 * Constructor of deletion strategy.
	 * 
	 * @param maxPopulationSize
	 *            the size that the population will have
	 * @param selector
	 *            the selector used for deleting
	 */
	public FixedSizeSetWorstFitnessDeletion(
											 final AbstractLearningClassifierSystem lcs,
											 final int maxPopulationSize, 
											 final IRuleSelector selector) {
		
		this.populationSize = maxPopulationSize;
		mySelector = selector; // roulette wheel gia ton GMlASLCS3
		zeroCoverageRemoval = new InadequeteClassifierDeletionStrategy(lcs);
		myLcs = lcs;
		updateStrategy = lcs.getUpdateStrategy();
	}

	/**
	 * @param aSet
	 *            the set to control
	 * @see gr.auth.ee.lcs.classifiers.IPopulationControlStrategy#controlPopulation(gr.auth.ee.lcs.classifiers.ClassifierSet)
	 * 
	 * 
	 * ekteleitai otan kano addClassifier ston population. diladi otan kano cover i ga. (sto cover einai me false to thorough)
	 * diagrapse prota autous pou exoun zero coverage. 
	 * sti sunexeia, an akoma eimaste pano apo to ano orio tou pli9ismou, diagrapse me rouleta osous kanones prepei oste na pesoume kato apo to ano orio
	 */
	@Override
	public final void controlPopulation(final ClassifierSet aSet) {

		final ClassifierSet toBeDeleted = new ClassifierSet(null);
		
//		not necessary anymore. deletion of zero coverage rules occurs right after the formation of the match set
 
/*		if (aSet.getTotalNumerosity() > populationSize) 
			zeroCoverageRemoval.controlPopulation(aSet);*/

		numberOfDeletions = 0;
		deletionTime = 0;
		
		while (aSet.getTotalNumerosity() > populationSize) {
			long time1 = - System.currentTimeMillis();
			
			numberOfDeletions++;
			// se auto to simeio upologizei maxPopulation + 1 pi9anotites, ka9os gia na kli9ei i controlPopulation, prepei na exei uperbei to ano orio tou pli9usmou
			
			//if (numberOfDeletions == 1) 
			updateStrategy.computeDeletionProbabilities(aSet);
			
			mySelector.select(1, aSet, toBeDeleted); // me rouleta
			Classifier cl = toBeDeleted.getClassifier(0);

			
			if (cl.formulaForD == 1)
				aSet.firstDeletionFormula++;
			else if (cl.formulaForD == 0) 
				aSet.secondDeletionFormula++;
			
			if (cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_COVER || (cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_INIT))
				aSet.coveredDeleted++;
			else if (cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_GA)
				aSet.gaedDeleted++;
			
			// monitor deletions
			monitorDeletions(aSet, cl);
						
			aSet.deleteClassifier(cl);
			toBeDeleted.deleteClassifier(0);

			time1 += System.currentTimeMillis();
			
			deletionTime += time1;
		}
		
	}
	
	
	
	@Override
	public final void controlPopulationSmp(final ClassifierSet aSet) {

		final ClassifierSet toBeDeleted = new ClassifierSet(null);
		
//		not necessary anymore. deletion of zero coverage rules occurs right after the formation of the match set
 
/*		if (aSet.getTotalNumerosity() > populationSize) 
			zeroCoverageRemoval.controlPopulation(aSet);*/

		numberOfDeletions = 0;
		deletionTime = 0;
		
		while (aSet.getTotalNumerosity() > populationSize) {
			long time1 = - System.currentTimeMillis();
			
			numberOfDeletions++;
			// se auto to simeio upologizei maxPopulation + 1 pi9anotites, ka9os gia na kli9ei i controlPopulation, prepei na exei uperbei to ano orio tou pli9usmou
			
			//if (numberOfDeletions == 1) 
			updateStrategy.computeDeletionProbabilitiesSmp(aSet);
			
			mySelector.selectSmp2(1, aSet, toBeDeleted); // me rouleta			
			
			Classifier cl = toBeDeleted.getClassifier(0);

			
			if (cl.formulaForD == 1)
				aSet.firstDeletionFormula++;
			else if (cl.formulaForD == 0) 
				aSet.secondDeletionFormula++;
			
			if (cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_COVER || (cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_INIT))
				aSet.coveredDeleted++;
			else if (cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_GA)
				aSet.gaedDeleted++;
			
			// monitor deletions
			monitorDeletions(aSet, cl);
						
			aSet.deleteClassifier(cl);
			toBeDeleted.deleteClassifier(0);
			
			time1 += System.currentTimeMillis();
			
			deletionTime += time1;

		}
	}
	
	
	
	
	public final int getNumberOfDeletionsConducted(){
		return numberOfDeletions;
	}
	
	public final long getDeletionTime(){
		return deletionTime;
	}

	
	
	/**
	 * record the progress of the deletion process
	 * */
	public void monitorDeletions(ClassifierSet aSet, Classifier cl) {
		
		//Macroclassifier macro = aSet.getActualMacroclassifier(cl);
		
		//double acc = cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_ACCURACY);
		double acc = cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION);

		/*double relativeExperience = (double) cl.cummulativeInstanceCreated / 
			(myLcs.getCummulativeCurrentInstanceIndex() == 0 ? 1 : myLcs.getCummulativeCurrentInstanceIndex());*/
		
		double qualityIndex = -0.1;
		
		if (cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_COVER|| cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_INIT) {
			if (cl.objectiveCoverage > 0) // to cl.objectiveCoverage apokta timi otan gia proti fora o kanonas dei olo to dataset
				qualityIndex =/* acc * relativeExperience * macro.numerosity*/ cl.objectiveCoverage;
/*			else if (cl.objectiveCoverage == -1) { 
				// osoi diagrafontai edo, den exoun dei olo to dataset oute mia fora.
				qualityIndex = -0.1;
			}*/
			// den ginetai cl.objectiveCoverage == 0 giati 9a diagrafotan logo zero coverage
			myLcs.qualityIndexOfClassifiersCoveredDeleted.add((float) qualityIndex);
			myLcs.qualityIndexOfClassifiersGaedDeleted.add((float) -0.2);
			myLcs.originOfDeleted.add(0);
			myLcs.accuracyOfCoveredDeletion.add((float) acc);
			myLcs.accuracyOfGaedDeletion.add((float) -0.1);

		}
		
		else if (cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_GA) {
			if (cl.objectiveCoverage > 0) // to cl.objectiveCoverage apokta timi otan gia proti fora o kanonas dei olo to dataset
				qualityIndex = /*acc * relativeExperience * macro.numerosity*/ cl.objectiveCoverage;
/*			else if (cl.objectiveCoverage == -1) { 
				// osoi diagrafontai edo, den exoun dei olo to dataset oute mia fora.
				qualityIndex = -0.1;
			}*/
			myLcs.qualityIndexOfClassifiersGaedDeleted.add((float) qualityIndex);
			myLcs.qualityIndexOfClassifiersCoveredDeleted.add((float) -0.2);
			myLcs.originOfDeleted.add(1);
			myLcs.accuracyOfGaedDeletion.add((float) acc);
			myLcs.accuracyOfCoveredDeletion.add((float) -0.1);


			// den ginetai cl.objectiveCoverage == 0 giati 9a diagrafotan logo zero coverage
		}

		
		
		myLcs.qualityIndexOfDeleted.add((float) qualityIndex);
		myLcs.accuracyOfDeleted.add((float) acc);
		myLcs.iteration.add(myLcs.totalRepetition);

	}
}
