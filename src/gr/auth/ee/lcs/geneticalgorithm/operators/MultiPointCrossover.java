package gr.auth.ee.lcs.geneticalgorithm.operators;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.geneticalgorithm.IBinaryGeneticOperator;
import gr.auth.ee.lcs.utilities.ExtendedBitSet;
import gr.auth.ee.lcs.utilities.SettingsLoader;

public class MultiPointCrossover implements IBinaryGeneticOperator {

	
	/**
	 * The LCS instance being used.
	 */
	final AbstractLearningClassifierSystem myLcs;
	
	
	public final int numberOfLabels; 

	
	/**
	 * Constructor.
	 */
	public MultiPointCrossover(AbstractLearningClassifierSystem lcs) {
		myLcs = lcs;
		numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels", 1);

	}
	
	
	@Override
	public Classifier operate(Classifier classifierA, 
							   Classifier classifierB, 
							   int label,
							   int mutationPoint) {
		
		final Classifier child;
		
		child = myLcs.getNewClassifier(performCrossover(classifierA, classifierB, mutationPoint, label, numberOfLabels));

		return child;
	}
	
	
	/**
	 * A protected function that performs a multi point crossover.
	 * 
	 * @param chromosomeA
	 *            the first chromosome to crossover
	 * @param chromosomeB
	 *            the second chromosome to crossover
	 * @param position
	 *            the position (bit) to perform the crossover
	 * @return the new cross-overed (child) chromosome
	 */
	protected final ExtendedBitSet performCrossover(final ExtendedBitSet chromosomeA, 
													  final ExtendedBitSet chromosomeB,
													  final int position,
													  final int label, 
													  final int numberOfLabels) {
		
		/*
		 *  __________________________________________________
		 * | 0-1 | 2-3 | 4-5 | 6-7 | => | 8-9 | 10-11 | 12-13 |
		 * 
		 * for label 1 (0-1-2)
		 * antecedentBoundLeft = 0
		 * antecedentBoundRight = 14 - 2 * 3 - 1 = 7
		 * labelBoundLeft = 7 + 2 * 1 +1 = 10
		 * labelBoundRight = 10 + 1 = 11
		 * 
		 * */
		
		final int antecedentBoundLeft = 0;
		final int antecedentBoundRight = chromosomeA.size() - 2 * numberOfLabels - 1;
		final int labelBoundLeft = antecedentBoundRight + 2 * label + 1;
		final int labelBoundRight = labelBoundLeft + 1;
		
		final ExtendedBitSet child = (ExtendedBitSet) chromosomeA.clone();
		
		if (position <= antecedentBoundRight) {
			child.setSubSet(position, chromosomeB.getSubSet(position, antecedentBoundRight - position + 1)); 
			child.setSubSet(labelBoundLeft, chromosomeB.getSubSet(labelBoundLeft, 2));
		}
		else {
			int chromosomeSize = chromosomeA.size() - 2 * (numberOfLabels - 1); 
			child.setSubSet(position + 2 * label, chromosomeB.getSubSet(position + 2 * label, chromosomeSize - position));
		}
		
		return child;
	}
	

}
