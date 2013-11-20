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
package gr.auth.ee.lcs.data.updateAlgorithms;

import edu.rit.pj.IntegerForLoop;
import edu.rit.pj.ParallelRegion;
import edu.rit.pj.ParallelSection;
import edu.rit.pj.ParallelTeam;
import edu.rit.util.Random;
import edu.rit.util.Range;
import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.IPopulationControlStrategy;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.classifiers.statistics.MeanCoverageStatistic;
import gr.auth.ee.lcs.classifiers.statistics.MeanFitnessStatistic;
import gr.auth.ee.lcs.classifiers.statistics.MeanNicheSizeStatistic;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.data.ClassifierTransformBridge;
import gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy;
import gr.auth.ee.lcs.geneticalgorithm.algorithms.SteadyStateGeneticAlgorithm;
import gr.auth.ee.lcs.geneticalgorithm.algorithms.SteadyStateGeneticAlgorithm.EvolutionOutcome;
import gr.auth.ee.lcs.geneticalgorithm.algorithms.SteadyStateGeneticAlgorithmNew;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Vector;

/**
 * An alternative MlASLCS update algorithm.
 * 
 * @author Miltiadis Allamanis
 * 
 */
public class MlASLCS4UpdateAlgorithm extends AbstractUpdateStrategy {

	public class EvolutionTimeMeasurements {
		public long timeA;
		public long timeB;
		public long timeC;
		public long timeD;
	}
	
	public static Vector<EvolutionTimeMeasurements> measurements0 = 
		new Vector<EvolutionTimeMeasurements>();
	
	public static Vector<EvolutionTimeMeasurements> measurements1 = 
		new Vector<EvolutionTimeMeasurements>();
	
	/**
	 * A data object for the MlASLCS3 update algorithms.
	 * 
	 * @author Miltos Allamanis
	 * 
	 */
	final static class MlASLCSClassifierData implements Serializable {

		/**
		 * 
		 */
		private static final long serialVersionUID = 2584696442026755144L;

		/**
		 * d refers to the paper's d parameter in deletion possibility
		 */
		
		public double d = 0;
		
		/**
		 * The classifier's fitness
		 */
		public double fitness = 1;//Double.MIN_NORMAL; //.5;

		/**
		 * niche set size estimation.
		 */
		public double ns = 1; //20;

		/**
		 * Match Set Appearances.
		 */
		public double msa = 0;

		/**
		 * true positives.
		 */
		public double tp = 0;
		
		/**
		 * totalFitness = numerosity * fitness
		 */
		
		public double totalFitness = 1;
		
		
		// k for fitness sharing
		public double k = 0;
		
		public int minCurrentNs = 0;
		
		
		public String toString(){
			return 	 "d = " + d 
					+ " fitness = " + fitness
					+ " ns = " + ns
					+ " msa= " + msa
					+ "tp = " + tp
					+ " minCurrentNs = " + minCurrentNs;
		} 
						
	}
	
	
	/**
	 * The way to differentiate the choice of the fitness calculation formula.
	 * 
	 * Simple = (acc)^n
	 * 
	 * Complex = F + β(k - F)
	 * 
	 * Sharing = F + β((k*num)/(Σ k*num) - F)
	 * 
	 * */
	public static final int FITNESS_MODE_SIMPLE 	= 0;
	public static final int FITNESS_MODE_COMPLEX 	= 1;
	public static final int FITNESS_MODE_SHARING 	= 2;
	
	
	public static final int DELETION_MODE_KOVACS = 0;
	public static final int DELETION_MODE_KOVACS_POWERED = 1;
	public static final int DELETION_MODE_MILTOS = 2;
	public static final int DELETION_MODE_E_KOVACS = 3;

	
	
	public static double ACC_0 = (double) SettingsLoader.getNumericSetting("ASLCS_Acc0", .99);
	
	public static double a = (double) SettingsLoader.getNumericSetting("ASLCS_Alpha", .1);
	

	public static final int labelParallelMode = 1; 

	/**
	 * The deletion mechanism. 0 for (cl.myClassifier.experience > THETA_DEL) && (data.fitness < DELTA * meanPopulationFitness)
	 * 						   1 for (cl.myClassifier.experience > THETA_DEL) && (Math.pow(data.fitness,n) < DELTA * meanPopulationFitness)	
	 * 
	 * 0 as default
	 * */
		
	public final int DELETION_MODE = (int) SettingsLoader.getNumericSetting("DELETION_MODE", 0);

	/**
	 * The delta (δ) parameter used in determining the formula of possibility of deletion
	 */
	
	public static double DELTA = (double) SettingsLoader.getNumericSetting("ASLCS_DELTA", .1);
	
	/**
	 * The fitness mode, 0 for simple, 1 for complex. 0 As default.
	 */
	public final int FITNESS_MODE = (int) SettingsLoader.getNumericSetting("FITNESS_MODE", 0);
	
	
	/**
	 * The update mode, 0 for adding offsprings and controlling per offspring, 1 for adding all offsprings and controlling once. 0 As default.
	 */
	public final int UPDATE_MODE = (int) SettingsLoader.getNumericSetting("UPDATE_MODE", 0);
	
	
	/**
	 * do classifiers that don't decide clearly for the label, participate in the correct sets?
	 * */
	public final boolean wildCardsParticipateInCorrectSets = SettingsLoader.getStringSetting("wildCardsParticipateInCorrectSets", "false").equals("true");
	
	
	/** 
	if wildCardsParticipateInCorrectSets is true, and balanceCorrectSets is also true, control the population of the correct sets 
	by examining the numerosity of a correct set comprising only with wildcards against that of a correct set without them.
	if [C#only] <= wildCardParticipationRatio * [C!#], the correct set consists of wildcards AND non-wildcard rules 
	*/

	public final boolean balanceCorrectSets = SettingsLoader.getStringSetting("balanceCorrectSets", "false").equals("true");
	
	public final double wildCardParticipationRatio = SettingsLoader.getNumericSetting("wildCardParticipationRatio", 1);
	
	/**
	 * The learning rate.
	 */
	private final double LEARNING_RATE = SettingsLoader.getNumericSetting("LearningRate", 0.2);
	
	
	/**
	 * The theta_del parameter.
	 */
	public static int THETA_DEL = (int) SettingsLoader.getNumericSetting("ASLCS_THETA_DEL", 20);
	
	
	/**
	 * The MLUCS omega parameter.
	 */	
	private final double OMEGA = SettingsLoader.getNumericSetting("ASLCS_OMEGA", 0.9);
	
	/**
	 * The MLUCS phi parameter.
	 */	
	private final double PHI =  SettingsLoader.getNumericSetting("ASLCS_PHI", 1);


	/**
	 * The LCS instance being used.
	 */
	private final AbstractLearningClassifierSystem myLcs;

	/**
	 * Genetic Algorithm.
	 */
	public final IGeneticAlgorithmStrategy ga;

	/**
	 * The fitness threshold for subsumption.
	 */
	private final double subsumptionFitnessThreshold;

	/**
	 * The experience threshold for subsumption.
	 */
	private final int subsumptionExperienceThreshold;

	/**
	 * Number of labels used.
	 */
	private final int numberOfLabels;

	/**
	 * The n dumping factor for acc.
	 */
	private final double n;
		
	
	public int numberOfEvolutionsConducted;
	
	public int numberOfDeletionsConducted;
	
	public int numberOfSubsumptionsConducted;
	
	public int numberOfNewClassifiers;
	
	public long evolutionTime;
	
	public long subsumptionTime;
	
	public long deletionTime;
	
	public long generateCorrectSetTime;
	
	public long updateParametersTime;
	
	public long selectionTime;
	
	static Vector<Integer> indicesToSubsumeSmp;
	
	static ClassifierSet newClassifiersSetSmp;
	
	static Vector<Integer> labelsToEvolveSmp;
	
	static ClassifierSet[] labelCorrectSetsSmp;
	
	static ClassifierSet populationSmp;
	
	static long seedSmp1;
	
	static long seedSmp2;
	
	ParallelTeam ptEvolve;
	
	static IGeneticAlgorithmStrategy gaSmp;
	
	static long subsumptionTimeSmp;
	
	static long subsumptionTimeMax;
	
	static int numOfProcessors;
	
	static int div;
	
	static int mod;
	
	private ParallelTeam ptGenerateCorrectSet;
	
	private ParallelTeam ptGenerateCorrectSetNew;
	
	private ParallelTeam ptUpdateParameters;
	
	private ParallelTeam ptComputeDeletionParameters;
	
	private ParallelTeam ptComputeFitnessSum;
	
	static ClassifierSet matchSetSmp;
	
	static ClassifierSet matchSetNewSmp;
	
	static ClassifierSet correctSetNewSmp;
	static ClassifierSet correctSetOnlyWildcardsNewSmp;
	static ClassifierSet correctSetWithoutWildcardsNewSmp;
	
	static int instanceIndexSmp;
	
	static int instanceIndexNewSmp;
	
	static int labelIndexNewSmp;
	
	static int numberOfLabelsSmp;
	
	static Vector<Integer> firstToGenerate;
	static Vector<Integer> lastToGenerate;
	
	static double meanPopulationFitnessSmp;
	static double fitnessSumSmp;
	static ClassifierSet aSetSmp;
	
	static boolean wildCardsParticipateInCorrectSetsSmp = SettingsLoader.getStringSetting("wildCardsParticipateInCorrectSets", "false").equals("true");
	
	static boolean balanceCorrectSetsSmp = SettingsLoader.getStringSetting("balanceCorrectSets", "false").equals("true");
	
	static double wildCardParticipationRatioSmp = SettingsLoader.getNumericSetting("wildCardParticipationRatio", 1);
	/**
	 *  holds the classifiers' indices in the match set with the lowest coverage. used when deleting from [M]
	 * */
	private ArrayList <Integer> lowestCoverageIndices;
	
	private ArrayList <Integer> lowestFitnessIndices;
	
	private ArrayList <Float> distinctCoverage;
	
	private boolean commencedDeletions = false;
	
	public ClassifierTransformBridge bridge;
	
	public int numberOfAttributes;
	


	

	/**
	 * Constructor.
	 * 
	 * @param lcs
	 *            the LCS being used.
	 * @param labels
	 *            the number of labels
	 * @param geneticAlgorithm
	 *            the GA used
	 * @param nParameter
	 *            the ASLCS dubbing factor
	 * @param fitnessThreshold
	 *            the subsumption fitness threshold to be used.
	 * @param experienceThreshold
	 *            the subsumption experience threshold to be used
	 */
	public MlASLCS4UpdateAlgorithm(final double nParameter,
									final double fitnessThreshold, 
									final int experienceThreshold,
									IGeneticAlgorithmStrategy geneticAlgorithm, 
									int labels,
									AbstractLearningClassifierSystem lcs) {
		
		this.subsumptionFitnessThreshold = fitnessThreshold;
		this.subsumptionExperienceThreshold = experienceThreshold;
		myLcs = lcs;
		numberOfLabels = labels;
		n = nParameter;
		ga = geneticAlgorithm;
			
		ptEvolve = new ParallelTeam();
		
		ptGenerateCorrectSet = new ParallelTeam();
		
		ptUpdateParameters = new ParallelTeam();
		
		ptComputeDeletionParameters = new ParallelTeam();
		
		ptGenerateCorrectSetNew = new ParallelTeam();
		
		ptComputeFitnessSum = new ParallelTeam();
		
		firstToGenerate = new Vector<Integer>();
		lastToGenerate = new Vector<Integer>();
		
		Runtime runtime = Runtime.getRuntime(); 
		
		numOfProcessors = runtime.availableProcessors();  
		
		lowestCoverageIndices = new ArrayList <Integer>();
		lowestFitnessIndices = new ArrayList <Integer>();
		distinctCoverage = new ArrayList<Float>();
	
		
		/*DELETION_MODE = (int) SettingsLoader.getNumericSetting("DELETION_MODE", 0);
		FITNESS_MODE = (int) SettingsLoader.getNumericSetting("FITNESS_MODE", 0);
		wildCardsParticipateInCorrectSets = SettingsLoader.getStringSetting("wildCardsParticipateInCorrectSets", "true").equals("true");*/
		
		System.out.println("Update algorithm states: ");
		System.out.println("fitness mode: 	" + FITNESS_MODE);
		System.out.println("deletion mode: 	" + DELETION_MODE);
		System.out.println("update mode: 	" + UPDATE_MODE);
		System.out.println("update algorithm: " + 4);
		System.out.print("# => [C] " + wildCardsParticipateInCorrectSets);
		if (wildCardsParticipateInCorrectSets) 
			System.out.println(", balance [C]: " + balanceCorrectSets + "\n");
		else
			System.out.println("\n");

	}

	/**
	 * This method provides a centralized point in MlASLCS4UpdateAlgorithm for computing each classifier's deletion probability
	 * because it is being called by two separate methods, computeDeletionProbabilities and computeDeletionProbabilitiesSmp
	 * 
	 * */
	
	private void computeCoreDeletionProbabilities (final Macroclassifier cl, 
													final MlASLCSClassifierData data,
													final double meanFitness) {
		
		commencedDeletions = true;

		if (DELETION_MODE == DELETION_MODE_KOVACS) {
			data.d = data.ns * ((cl.myClassifier.experience > THETA_DEL) && (data.fitness < DELTA * meanFitness) ? 
			meanFitness / data.fitness : 1);	
		
			/* mark the formula responsible for deleting this classifier 
			* (if exp > theta_del and fitness < delta * <f>) ==> formula = 1, else 0. */
			
			cl.myClassifier.formulaForD = ((cl.myClassifier.experience > THETA_DEL) && (data.fitness < DELTA * meanFitness)) ? 1 : 0;
		}
		
		else if (DELETION_MODE == DELETION_MODE_KOVACS_POWERED) {
		
			data.d = data.ns * ((cl.myClassifier.experience > THETA_DEL) && (Math.pow(data.fitness,n) < DELTA * meanFitness) ? 
			meanFitness / Math.pow(data.fitness,n) : 1);	
		
			/* mark the formula responsible for deleting this classifier 
			* (if exp > theta_del and fitness ^ n < delta * <f>) ==> formula = 1, else 0. */
			
			cl.myClassifier.formulaForD = ((cl.myClassifier.experience > THETA_DEL) && (Math.pow(data.fitness,n) < DELTA * meanFitness)) ? 1 : 0;
		}
		
		else if (DELETION_MODE == DELETION_MODE_MILTOS) {
		
			// miltos original
			/*				data.d = 1 / (data.fitness * ((cl.myClassifier.experience < THETA_DEL) ? 100.
			: Math.exp(-data.ns + 1)));
			
			cl.myClassifier.formulaForD = (cl.myClassifier.experience < THETA_DEL) ? 1 : 0;*/
			
/*			if (cl.myClassifier.experience < THETA_DEL){
				data.d = Math.exp(1 / data.fitness);
				cl.myClassifier.formulaForD = 1;
			}
			else {
				//data.d = Math.exp(data.ns - 1) / data.fitness;
				//data.d = Math.exp(10 / data.fitness) * Math.pow(data.ns, 10 / data.fitness);
				//data.d = Math.pow(10, data.ns / data.fitness) * Math.exp(data.ns - 1); 
				cl.myClassifier.formulaForD = 0;
			}*/
			
			
/*			MeanNicheSizeStatistic meanNs = new MeanNicheSizeStatistic();
			double meanPopulationNs = meanNs.getMetric(myLcs);*/
			
/*			if (cl.myClassifier.objectiveCoverage < 0 || cl.myClassifier.experience < 10) {
				data.d = 0;
			}
			else*/
			if (cl.myClassifier.experience < THETA_DEL) {
/*				if (data.fitness < DELTA * meanFitness) 
					data.d = Math.exp(data.ns * meanFitness / data.fitness);
				else*/
					data.d = /*(data.ns - 1) **/ Math.exp(1 / data.fitness) ;
				//data.d = 1 / (100 * data.fitness);
				//data.d = data.ns * Math.exp(Math.sqrt(meanPopulationNs) / data.fitness);
				//data.d = Math.exp(Math.sqrt(meanPopulationNs) / data.fitness);
				//data.d = (data.ns - 1) * Math.exp(10 / data.fitness);
				cl.myClassifier.formulaForD = 1;
			}
			else {
				
/*				if (data.fitness < DELTA * meanFitness) 
					data.d = Math.exp(data.ns * meanFitness / data.fitness);
				else {*/
					//data.d = Math.exp(data.ns) / data.fitness;
					data.d = Math.exp(data.ns - 1) / data.fitness;
					cl.myClassifier.formulaForD = 0;
				//}
			}
			
			//data.d = cl.myClassifier.experience > THETA_DEL ? Math.exp((data.ns - 1) / data.fitness) : Math.exp(data.ns - 1) / data.fitness;
		
		}
		
		else if (DELETION_MODE == DELETION_MODE_E_KOVACS) {
		
			if ((cl.myClassifier.experience > THETA_DEL) && (data.fitness < DELTA * meanFitness)) 
				data.d = Math.exp(data.ns * meanFitness / data.fitness) ;
			else
				data.d = Math.exp(data.ns);
			
			
			cl.myClassifier.formulaForD = (cl.myClassifier.experience > THETA_DEL) && (data.fitness < DELTA * meanFitness) ? 1 : 0;
		}

	}
	
	
	/**
	 * 
	 * For every classifier, compute its deletion probability.
	 * 
	 * @param aSet
	 * 			the classifierset of which the classifiers' deletion probabilities we will compute
	 * */
	@Override
	public void computeDeletionProbabilities (ClassifierSet aSet) {

		
		final int numOfMacroclassifiers = aSet.getNumberOfMacroclassifiers();
		
/*		// calculate the mean fitness of the population, used in the deletion mechanism
		double fitnessSum = 0;
		double meanPopulationFitness = 0;
		
		for (int j = 0; j < numOfMacroclassifiers; j++) {
			fitnessSum += aSet.getClassifierNumerosity(j)
					* aSet.getClassifier(j).getComparisonValue(COMPARISON_MODE_EXPLORATION); 
		}

		meanPopulationFitness = (double) (fitnessSum / aSet.getTotalNumerosity());*/
		
		MeanFitnessStatistic meanFit = new MeanFitnessStatistic(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION);
		double meanPopulationFitness = meanFit.getMetric(myLcs);
		
/*		MeanCoverageStatistic meanCov = new MeanCoverageStatistic();
		double meanPopulationCoverage = meanCov.getMetric(myLcs);
		
		MeanNicheSizeStatistic meanNs = new MeanNicheSizeStatistic();
		double meanPopulationNs = meanNs.getMetric(myLcs);*/
		
		

		
		/* update the d parameter, employed in the deletion mechanism, for each classifier in the match set, {currently population-wise} due to the change in 
		 * the classifiers's numerosities, niches' sizes, fitnesses and the mean fitness of the population
		 */
		for (int i = 0; i < numOfMacroclassifiers; i++) {
			//final Macroclassifier cl = matchSet.getMacroclassifier(i);
			final Macroclassifier cl = aSet.getMacroclassifier(i);
			//final Macroclassifier actualMacro = aSet.getActualMacroclassifier(i);
			final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
			
			computeCoreDeletionProbabilities(cl, data, meanPopulationFitness);


		/*		
				if (cl.myClassifier.experience < THETA_DEL)
					data.d = 1 / (100.0 * data.fitness);
				else
					data.d = Math.exp(data.ns / data.fitness - 1) / (cl.myClassifier.objectiveCoverage == -1 ? 100 : cl.myClassifier.objectiveCoverage);
				
				cl.myClassifier.formulaForD = (cl.myClassifier.experience < THETA_DEL) ? 1 : 0;
				
				
				
				if (cl.myClassifier.objectiveCoverage < 0)
					data.d = 1 / data.fitness; // gia na xorisoume autous pou arxika prepei na diagrafoun alla oloi den exoun megalo exp
				else
					data.d = Math.exp(10 / (data.fitness));
				
							
				
				if (cl.myClassifier.experience < THETA_DEL) 
					data.d = 0; // protect the new classifiers
				
				else if (data.fitness >= 0.1) 
					data.d = Math.exp(data.ns);
				
				else 
					data.d = Math.exp(data.ns * meanPopulationFitness / data.fitness);
				
				double nsRatio = data.ns / meanPopulationNs;
				// 9elo na exei dei toulaxiston mia fora olo to dataset
				double covRatio = (cl.myClassifier.getCoverage() > 0) ? meanPopulationCoverage / cl.myClassifier.getCoverage() : 0; 
				double fitRatio = Double.isNaN(data.fitness) ? 0 :meanPopulationFitness / data.fitness;
				
				
				if (cl.myClassifier.experience < THETA_DEL) 
					data.d = 0;
				
				else if (cl.myClassifier.getCoverage() > 0) {
					
					if (data.ns > meanPopulationNs && cl.myClassifier.getCoverage() < meanPopulationCoverage)
						data.d = Math.exp(data.ns / meanPopulationNs * meanPopulationCoverage / cl.myClassifier.getCoverage() / data.fitness);
					
					else if (data.ns > meanPopulationNs && cl.myClassifier.getCoverage() > meanPopulationCoverage)
						data.d = Math.exp (data.ns / meanPopulationNs * cl.myClassifier.getCoverage() / meanPopulationCoverage / data.fitness);
						
				}
						
				
				
				//data.d = Math.pow(data.ns * DELTA, data.ns + 1);
				
				//data.d = 1 / (1 + Math.exp(-data.d));
				
				
				if ((cl.myClassifier.experience > THETA_DEL) && (data.fitness < DELTA * meanPopulationFitness)) 
					data.d = Math.exp(data.ns * meanPopulationFitness / data.fitness) ;
				else
					data.d = Math.exp(data.ns);
				
				cl.myClassifier.formulaForD = ((cl.myClassifier.experience > THETA_DEL) 
						&& (data.fitness < DELTA * meanPopulationFitness)) ? 1 : 0;
				
			}
			*/

			
		}	
	}
	

	/**
	 * 
	 * For every classifier, compute its deletion probability.
	 * Smp version.
	 * 
	 * @param aSet
	 * 			the classifierset of which the classifiers' deletion probabilities we will compute
	 * */
	@Override
	public final void computeDeletionProbabilitiesSmp(ClassifierSet aSet) {

		
//		final int numOfMacroclassifiers = aSet.getNumberOfMacroclassifiers();
		
//		// calculate the mean fitness of the population, used in the deletion mechanism
//		double fitnessSum = 0;
//		double meanPopulationFitness = 0;
//		
//		for (int j = 0; j < numOfMacroclassifiers; j++) {
//			fitnessSum += aSet.getClassifierNumerosity(j)
//					* aSet.getClassifier(j).getComparisonValue(COMPARISON_MODE_EXPLORATION); 
//		}
//
//		meanPopulationFitness = (double) (fitnessSum / aSet.getTotalNumerosity());
		
		aSetSmp = aSet;
		
		try{
			ptComputeFitnessSum.execute( new ParallelRegion() {
			
				public void start()
				{
					fitnessSumSmp = 0;
				}
			
				public void run() throws Exception
				{
					execute(0, aSetSmp.getNumberOfMacroclassifiers()-1, new IntegerForLoop() {
						
						double fitnessSum_thread;
						
						public void start()
						{
							fitnessSum_thread = 0;
						}						
						public void run(int first, int last)
						{
							for ( int i = first; i <= last ; ++i )
							{
								fitnessSum_thread += aSetSmp.getClassifierNumerosity(i)
												* aSetSmp.getClassifier(i).getComparisonValue(COMPARISON_MODE_EXPLORATION);
							}
						}
						
						public void finish() throws Exception
						{
							region().critical( new ParallelSection() {
								public void run()
								{
									fitnessSumSmp += fitnessSum_thread;
								}
							});
						}
						
					});
				}	
			});
		}
		catch ( Exception e)
		{
			e.printStackTrace();
		}
		
		/* update the d parameter, employed in the deletion mechanism, for each classifier in the match set, {currently population-wise} due to the change in 
		 * the classifiers's numerosities, niches' sizes, fitnesses and the mean fitness of the population
		 */
		
		meanPopulationFitnessSmp = fitnessSumSmp / aSetSmp.getTotalNumerosity();
		
				
		try {
			ptComputeDeletionParameters.execute( new ParallelRegion() {
			
				public void run() throws Exception
				{
					execute(0, aSetSmp.getNumberOfMacroclassifiers() - 1, new IntegerForLoop() {
						
						public void run(int first, int last)
						{
							for ( int i = first ; i <= last ; i++ )
							{
								final Macroclassifier cl = aSetSmp.getMacroclassifier(i);
								final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();

								computeCoreDeletionProbabilities(cl, data, meanPopulationFitnessSmp);
							}
						}
						
					});
				}
			
			});
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		
	}
	
	
	
	
	
	
	
	
	
	
	/**
	 * Delete classifiers from every match set formed.
	 * 
	 */
	private void controlPopulationInMatchSet (final ClassifierSet population, final ClassifierSet matchSet) {
		controlPopulationInMatchSet1(population, matchSet);
	}
	
	
	/**
	 * Delete classifiers from every correct set formed.
	 * 
	 */
	private void  controlPopulationInCorrectSet (final ClassifierSet population, 
												  final ClassifierSet matchSet,
												  final ClassifierSet correctSet) {
		
		controlPopulationInCorrectSet1(population, matchSet, correctSet);
	}
	
	
	
	
	
	
	
	private void controlPopulationInCorrectSet1 (final ClassifierSet population, 
													final ClassifierSet matchSet,
													final ClassifierSet correctSet) {
		
		double lowestCoverage = Double.MAX_VALUE;
		int toBeDeleted = -1;

		for (int i = 0; i < correctSet.getNumberOfMacroclassifiers(); i++) {
			
			final Classifier cl = correctSet.getClassifier(i);
			if (cl.objectiveCoverage > 0 && cl.objectiveCoverage <= lowestCoverage) { // CL.getcoverage?
				
				if (cl.objectiveCoverage < lowestCoverage) {
					lowestCoverageIndices.clear();
				}
				
				lowestCoverage = cl.objectiveCoverage;
				lowestCoverageIndices.add(i);
			}
		}
		
		if (lowestCoverageIndices.size() > 1) {
			
			double lowestFitness = Double.MAX_VALUE;
			
			for (int i = 0; i < lowestCoverageIndices.size(); i++) {
				
				final Macroclassifier macro = correctSet.getMacroclassifier(lowestCoverageIndices.get(i));
				final Classifier cl = macro.myClassifier;

				if (cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS) <= lowestFitness) {

					lowestFitness = cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS);
					toBeDeleted = lowestCoverageIndices.get(i);
				}
			}
			
			if (toBeDeleted >= 0) {
				myLcs.numberOfClassifiersDeletedInMatchSets++;
				population.deleteClassifier(correctSet.getMacroclassifier(toBeDeleted).myClassifier);
				matchSet.deleteClassifier(correctSet.getMacroclassifier(toBeDeleted).myClassifier);
				correctSet.deleteClassifier(toBeDeleted);
			}
		}
		
		lowestCoverageIndices.clear();

				
	}
	
	
	private void controlPopulationInCorrectSet2 (final ClassifierSet population, 
													final ClassifierSet matchSet,
													final ClassifierSet correctSet) {

		for (int i = 0; i < correctSet.getNumberOfMacroclassifiers(); i++) {
		
			final Classifier cl = correctSet.getClassifier(i);
			
			if (cl.objectiveCoverage > 0 && !distinctCoverage.contains((float) cl.objectiveCoverage)) 
				distinctCoverage.add((float) cl.objectiveCoverage);	
		}
		
		for (int c = 0; c < distinctCoverage.size(); c++) {
		
			double lowestFitness = Double.MAX_VALUE;
			
			int toBeDeleted = -1;
			int howManyInBlock = 0;
			
			for (int m = 0; m < correctSet.getNumberOfMacroclassifiers(); m++) {
				
				final Classifier cl = correctSet.getClassifier(m);
				final double fitness = cl.getComparisonValue(COMPARISON_MODE_PURE_FITNESS);
				
				if (cl.objectiveCoverage == distinctCoverage.get(c))
					howManyInBlock++;
				
				if (cl.objectiveCoverage == distinctCoverage.get(c) && fitness <= lowestFitness) {
				
					lowestFitness = fitness;
					toBeDeleted = m;
				}
			}
			
			if (howManyInBlock > 1) {
				myLcs.numberOfClassifiersDeletedInMatchSets++;
				population.deleteClassifier(correctSet.getClassifier(toBeDeleted));
				matchSet.deleteClassifier(correctSet.getClassifier(toBeDeleted));
				correctSet.deleteClassifier(toBeDeleted);
			}
		}
		
		distinctCoverage.clear();


}

	
	private void controlPopulationInCorrectSet3 (final ClassifierSet population, final ClassifierSet matchSet, final ClassifierSet correctSet) {
		
		
		ArrayList <ClassifierSet> classifiersPerCoverage = new ArrayList<ClassifierSet>();
		// get the distinct coverages
		for (int i = 0; i < correctSet.getNumberOfMacroclassifiers(); i++) {
			
			final Macroclassifier macro = correctSet.getMacroclassifier(i);
			final Classifier cl = correctSet.getClassifier(i);
			
			if (cl.objectiveCoverage > 0 && !distinctCoverage.contains((float) cl.objectiveCoverage)) {
				distinctCoverage.add((float) cl.objectiveCoverage);	
				classifiersPerCoverage.add(new ClassifierSet(null));
			}
			
			for (int j = 0; j < distinctCoverage.size(); j++) {
				if (cl.objectiveCoverage == distinctCoverage.get(j))
					classifiersPerCoverage.get(j).addClassifier(macro, false);
			}
		}
		
		
		// edo exo brei ta distinct coverages kai exo spasei tous kanones tou M se classifier sets me basi to coverage tous 
		
		
		
		for (int c = 0; c < distinctCoverage.size(); c++) {
			
			// brisko to mean fitness tou ka9e classifier set
			
			double meanFitness = 0;
			
			for (int i = 0; i < classifiersPerCoverage.get(c).getNumberOfMacroclassifiers(); i++) {
				final Macroclassifier cl = classifiersPerCoverage.get(c).getMacroclassifier(i);
				meanFitness += cl.myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS);
			}

			meanFitness /= classifiersPerCoverage.get(c).getNumberOfMacroclassifiers();
			
			for (int i = 0; i < classifiersPerCoverage.get(c).getNumberOfMacroclassifiers(); i++) {
				final Macroclassifier cl = classifiersPerCoverage.get(c).getMacroclassifier(i);
				if (cl.myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS) < 0.1 * meanFitness) {
					classifiersPerCoverage.get(c).deleteClassifier(cl.myClassifier);
					population.deleteClassifier(cl.myClassifier);
					matchSet.deleteClassifier(cl.myClassifier);
					correctSet.deleteClassifier(cl.myClassifier);
					
				}
			}
			
			

		}
		
		distinctCoverage.clear();
		classifiersPerCoverage.clear();
	}
	
	
	private void controlPopulationInMatchSet1(final ClassifierSet population, final ClassifierSet matchSet) {
		
		//System.out.println(matchSet);
		
		double lowestCoverage = Double.MAX_VALUE;

		for (int i = 0; i < matchSet.getNumberOfMacroclassifiers(); i++) {
			
			final Classifier cl = matchSet.getClassifier(i);
			if (cl.objectiveCoverage > 0 && cl.objectiveCoverage <= lowestCoverage) { // CL.getcoverage?
				
				if (cl.objectiveCoverage < lowestCoverage) {
					lowestCoverageIndices.clear();
				}
				
				lowestCoverage = cl.objectiveCoverage;
				lowestCoverageIndices.add(i);
			}
		}
		
		if (lowestCoverageIndices.size() > 1) {
			
			double lowestFitness = Double.MAX_VALUE;
			int toBeDeleted = -1;

			for (int i = 0; i < lowestCoverageIndices.size(); i++) {
				
				final Macroclassifier macro = matchSet.getMacroclassifier(lowestCoverageIndices.get(i));
				final Classifier cl = macro.myClassifier;

				if (cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS) <= lowestFitness) {

					lowestFitness = cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS);
					toBeDeleted = lowestCoverageIndices.get(i);
				}
			}

			if (toBeDeleted >= 0) {
				myLcs.numberOfClassifiersDeletedInMatchSets++;
				population.deleteClassifier(matchSet.getMacroclassifier(toBeDeleted).myClassifier);
				matchSet.deleteClassifier(toBeDeleted);
			}
		}
		lowestCoverageIndices.clear();
				
	}
	
	
	private void controlPopulationInMatchSet2(final ClassifierSet population, final ClassifierSet matchSet) {
		
/*		bridge = myLcs.getClassifierTransformBridge();
		numberOfAttributes = bridge.getNumberOfAttributes();*/
		
		double lowestCoverage = Double.MAX_VALUE;

		for (int i = 0; i < matchSet.getNumberOfMacroclassifiers(); i++) {
			
			final Classifier cl = matchSet.getClassifier(i);
			if (cl.objectiveCoverage > 0 && cl.objectiveCoverage <= lowestCoverage) { // CL.getcoverage?
				
				if (cl.objectiveCoverage < lowestCoverage) 
					lowestCoverageIndices.clear();
				
				
				lowestCoverage = cl.objectiveCoverage;
				lowestCoverageIndices.add(i);
			}
		}
		// edo exoume parei autous pou exoun to mikrotero coverage
		
		
		if (lowestCoverageIndices.size() > 1) {
			


			double lowestFitness = Double.MAX_VALUE;
			int toBeDeleted = -1;
			
			for (int i = 0; i < lowestCoverageIndices.size(); i++) {
				
				final Macroclassifier macro = matchSet.getMacroclassifier(lowestCoverageIndices.get(i));
				final Classifier cl = macro.myClassifier;
				final double fitness = cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS);
				
/*				int specificLabels = 0;
				for (int l = numberOfAttributes; l < numberOfAttributes + numberOfLabels; l++) {
					if (bridge.isAttributeSpecific(cl, l)) {
						specificLabels++;;
					}
				}*/
				


				if (fitness <= lowestFitness) {
					
					if (fitness< lowestFitness) 
						lowestFitnessIndices.clear();
					
					
					lowestFitness = fitness;
					lowestFitnessIndices.add(lowestCoverageIndices.get(i));

					toBeDeleted = lowestCoverageIndices.get(i);
				}
			}
			
			
			if (toBeDeleted > 0) {
				population.deleteClassifier(matchSet.getMacroclassifier(toBeDeleted).myClassifier);
				matchSet.deleteClassifier(toBeDeleted);
				myLcs.numberOfClassifiersDeletedInMatchSets++;
				
			}
			// edo exoume parei auton pou exei to xamilotero fitness. omos, borei na uparxoun parapano apo enas me to xamilotero fitness
			// 9a broume sti sunexeia autous pou exoun tis perissoteres adiafories sta labels
			
			//if (lowestFitnessIndices.size() > 1) {
				
/*				int lowestSpecificity = Integer.MAX_VALUE;
				bridge = myLcs.getClassifierTransformBridge();
				numberOfAttributes = bridge.getNumberOfAttributes();
				int indexToBeDeleted = -1;
				
				for (int i = 0; i < indicesInMatchSetWithLowestFitness.size(); i++) {
						
	delete largest ns
 * 
 * 					final Macroclassifier macro = matchSet.getMacroclassifier(indicesInMatchSetWithLowestFitness.elementAt(i));
						final Classifier cl = macro.myClassifier;
						final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.getUpdateDataObject();
						
						if (data.ns >= largestNs) {
							toBeDeletedIndex = indicesInMatchSetWithLowestFitness.elementAt(i);
							largestNs = data.ns;
						}
					}
					if (largestNs > 0) {
						population.deleteClassifier(matchSet.getMacroclassifier(toBeDeletedIndex).myClassifier);
						myLcs.numberOfClassifiersDeletedInMatchSets++;
					}
				
				
						final Macroclassifier macro = matchSet.getMacroclassifier(indicesInMatchSetWithLowestFitness.elementAt(i));
						final Classifier cl = macro.myClassifier;
						
						int specificLabels = 0;
						for (int l = numberOfAttributes; l < numberOfAttributes + numberOfLabels; l++) {
							if (bridge.isAttributeSpecific(cl, l)) {
								specificLabels++;;
							}
						}
						
						if (specificLabels <= lowestSpecificity) {
							
							lowestSpecificity = specificLabels;
							indexToBeDeleted = indicesInMatchSetWithLowestFitness.elementAt(i);						
						}

					}	
				
				if (indexToBeDeleted > 0)	{
					myLcs.numberOfClassifiersDeletedInMatchSets++;
					population.deleteClassifier(matchSet.getMacroclassifier(indexToBeDeleted).myClassifier);
				}*/
					
				
					//int indexToDelete = (int) (Math.random() * (indicesInMatchSetWithLowestFitness.size() - 1));
					//population.deleteClassifier(matchSet.getMacroclassifier(indicesInMatchSetWithLowestFitness.elementAt(indexToDelete)).myClassifier);
					/*population.deleteClassifier(matchSet.getMacroclassifier(lowestFitnessIndices.lastElement()).myClassifier);
					myLcs.numberOfClassifiersDeletedInMatchSets++;
				

			}*/
			
/*			else if (lowestFitnessIndices.size() == 1) {
				
				lowestCoverageIndices.clear();
				lowestFitnessIndices.clear();
				
				double lowestFitness2 = Double.MAX_VALUE;

				for (int i = 0; i < matchSet.getNumberOfMacroclassifiers(); i++) {
					
					final Classifier cl = matchSet.getClassifier(i);
					final double fitness = cl.getComparisonValue(COMPARISON_MODE_PURE_FITNESS);
					if (fitness <= lowestFitness2) { 
						
						if (fitness != lowestFitness2) {
							lowestFitnessIndices.clear();
						}
						
						lowestFitness2 = fitness;
						lowestFitnessIndices.add(i);
					}
				}
				
				
				
				if (lowestFitnessIndices.size() > 1) {

					double lowestCov = Double.MAX_VALUE;
					
					for (int i = 0; i < lowestFitnessIndices.size(); i++) {
						
						final Macroclassifier macro = matchSet.getMacroclassifier(lowestFitnessIndices.elementAt(i));
						final Classifier cl = macro.myClassifier;
						final double cov = cl.objectiveCoverage;

						if (cov <= lowestCov) {
							
							if (cov != lowestCov) {
								lowestCoverageIndices.clear();
							}
							
							lowestCov = cov;
							lowestCoverageIndices.add(lowestFitnessIndices.elementAt(i));

							//toBeDeleted = indicesInMatchSetWithLowestCoverage.elementAt(i)
						}
					}
					// edo exoume parei auton pou exei to xamilotero fitness. omos, borei na uparxoun parapano apo enas me to xamilotero fitness
					// 9a broume sti sunexeia autous pou exoun tis perissoteres adiafories sta labels
					
					if (lowestCoverageIndices.size() > 1) {
						population.deleteClassifier(matchSet.getMacroclassifier(lowestCoverageIndices.lastElement()).myClassifier);
						myLcs.numberOfClassifiersDeletedInMatchSets++;
					}
				}
	
			}*/
		}
/*		else if (indicesInMatchSetWithLowestCoverage.size() == 1) {
			population.deleteClassifier(matchSet.getMacroclassifier(indicesInMatchSetWithLowestCoverage.elementAt(0)).myClassifier);
			myLcs.numberOfClassifiersDeletedInMatchSets++;
		}*/
		
		
		
		
		
/*		if (indicesInMatchSetWithLowestFitness.size() == 1) { //an einai mono enas, antistrepse ti diadikasia, prota fit meta cov.
			
			
			indicesInMatchSetWithLowestCoverage.clear();
			indicesInMatchSetWithLowestFitness.clear();
			
			double lowestFitness = Double.MAX_VALUE;

			for (int i = 0; i < matchSet.getNumberOfMacroclassifiers(); i++) {
				
				final Classifier cl = matchSet.getClassifier(i);
				final double fitness = cl.getComparisonValue(COMPARISON_MODE_PURE_FITNESS);
				if (fitness <= lowestFitness) { 
					
					if (fitness != lowestFitness) {
						indicesInMatchSetWithLowestFitness.clear();
					}
					
					lowestFitness = fitness;
					indicesInMatchSetWithLowestFitness.add(i);
				}
			}
			
			
			
			if (indicesInMatchSetWithLowestFitness.size() > 1) {

				double lowestCov = Double.MAX_VALUE;
				
				for (int i = 0; i < indicesInMatchSetWithLowestFitness.size(); i++) {
					
					final Macroclassifier macro = matchSet.getMacroclassifier(indicesInMatchSetWithLowestFitness.elementAt(i));
					final Classifier cl = macro.myClassifier;
					final double cov = cl.objectiveCoverage;

					if (cov <= lowestCov) {
						
						if (cov != lowestCov) {
							indicesInMatchSetWithLowestCoverage.clear();
						}
						
						lowestCov = cov;
						indicesInMatchSetWithLowestCoverage.add(indicesInMatchSetWithLowestFitness.elementAt(i));

						//toBeDeleted = indicesInMatchSetWithLowestCoverage.elementAt(i)
					}
				}
				// edo exoume parei auton pou exei to xamilotero fitness. omos, borei na uparxoun parapano apo enas me to xamilotero fitness
				// 9a broume sti sunexeia autous pou exoun tis perissoteres adiafories sta labels
				
				if (indicesInMatchSetWithLowestCoverage.size() > 1) {
					population.deleteClassifier(matchSet.getMacroclassifier(indicesInMatchSetWithLowestCoverage.lastElement()).myClassifier);
					myLcs.numberOfClassifiersDeletedInMatchSets++;
				}
			}
	
		}*/
		
		lowestCoverageIndices.clear();
		lowestFitnessIndices.clear();
	}
	

	/**
	 * Quantization per coverage, delete the classifier with the lowest fitness within every block.
	 * 
	 * */
	private void controlPopulationInMatchSet3 (final ClassifierSet population, final ClassifierSet matchSet) {
		
/*		System.out.println("==================================================================================================================================");
		System.out.println("matchset:");
		System.out.println(matchSet);
*/
		
		for (int i = 0; i < matchSet.getNumberOfMacroclassifiers(); i++) {
			
			final Classifier cl = matchSet.getClassifier(i);
			
			if (cl.objectiveCoverage > 0 && !distinctCoverage.contains((float) cl.objectiveCoverage)) 
				distinctCoverage.add((float) cl.objectiveCoverage);	
		}
		
/*		System.out.println("coverages:");
		for (int c = 0; c < distinctCoverage.size(); c++) {
			System.out.println(distinctCoverage.get(c));
			for (int m = 0; m < matchSet.getNumberOfMacroclassifiers(); m++) {
				
				final Classifier cl = matchSet.getClassifier(m);
				if (cl.objectiveCoverage == distinctCoverage.get(c)) {
					System.out.println(cl + ", fit: " + cl.getComparisonValue(COMPARISON_MODE_PURE_FITNESS));
				}
			}
			
		}
		System.out.println();
*/
		

		
		
		for (int c = 0; c < distinctCoverage.size(); c++) {
			
			double lowestFitness = 1;//Double.MAX_VALUE;

			int toBeDeleted = -1;
			int howManyInBlock = 0;
			for (int m = 0; m < matchSet.getNumberOfMacroclassifiers(); m++) {
				
				final Classifier cl = matchSet.getClassifier(m);
				final double fitness = cl.getComparisonValue(COMPARISON_MODE_PURE_FITNESS);
				
				if (cl.objectiveCoverage == distinctCoverage.get(c))
					howManyInBlock++;
				
				if (cl.objectiveCoverage == distinctCoverage.get(c) && fitness < lowestFitness) {
					
/*					if (fitness < lowestFitness)
						lowestFitnessIndices.clear();
*/
					
					
					lowestFitness = fitness;
					lowestFitnessIndices.add(m);
					toBeDeleted = m;
				}
			}
			
			if (howManyInBlock > 1) {
				//System.out.println("deleted");
				//System.out.println(matchSet.getClassifier(toBeDeleted));
				population.deleteClassifier(matchSet.getClassifier(toBeDeleted));
				matchSet.deleteClassifier(toBeDeleted);

			}
			
			
			lowestFitnessIndices.clear();

		}
		
		distinctCoverage.clear();

	}
	
	
	/**
	 * Quantization per coverage, delete in the classifiers with the lowest fitness within the blocks 
	 * that have a mean coverage lower than the population's (match set's maybe?) mean coverage.
	 * 
	 * */
	private void controlPopulationInMatchSet4 (final ClassifierSet population, final ClassifierSet matchSet) {
		
/*		System.out.println("==================================================================================================================================");
		System.out.println("matchset:");
		System.out.println(matchSet);
*/	

		double coverageSum = 0;

		final int numOfMacroclassifiers = matchSet.getNumberOfMacroclassifiers();
		for (int i = 0; i < numOfMacroclassifiers; i++)
			coverageSum += matchSet.getClassifierNumerosity(i)
					* matchSet.getClassifier(i).getCoverage();
		
		double meanPopulationCoverage = coverageSum / numOfMacroclassifiers;
		
		
		for (int i = 0; i < matchSet.getNumberOfMacroclassifiers(); i++) {
			
			final Classifier cl = matchSet.getClassifier(i);
			
			if (cl.objectiveCoverage > 0 && !distinctCoverage.contains((float) cl.objectiveCoverage) && cl.objectiveCoverage < meanPopulationCoverage) 
				distinctCoverage.add((float) cl.objectiveCoverage);	
		}
		
/*		System.out.println("coverages:");
		for (int c = 0; c < distinctCoverage.size(); c++) {
			System.out.println(distinctCoverage.get(c));
			for (int m = 0; m < matchSet.getNumberOfMacroclassifiers(); m++) {
				
				final Classifier cl = matchSet.getClassifier(m);
				if (cl.objectiveCoverage == distinctCoverage.get(c)) {
					System.out.println(cl + ", fit: " + cl.getComparisonValue(COMPARISON_MODE_PURE_FITNESS));
				}
			}
			
		}
		System.out.println();
*/
		

		
		
		for (int c = 0; c < distinctCoverage.size(); c++) {
			
			double lowestFitness = Double.MAX_VALUE;

			int toBeDeleted = -1;
			int howManyInBlock = 0;
			for (int m = 0; m < matchSet.getNumberOfMacroclassifiers(); m++) {
				
				final Classifier cl = matchSet.getClassifier(m);
				final double fitness = cl.getComparisonValue(COMPARISON_MODE_PURE_FITNESS);
				
				if (cl.objectiveCoverage == distinctCoverage.get(c))
					howManyInBlock++;
				
				if (cl.objectiveCoverage == distinctCoverage.get(c) && fitness <= lowestFitness) {
					
/*					if (fitness < lowestFitness)
						lowestFitnessIndices.clear();
*/
					
					
					lowestFitness = fitness;
					lowestFitnessIndices.add(m);
					toBeDeleted = m;
				}
			}
			
			if (howManyInBlock > 1) {
				//System.out.println("deleted");
				//System.out.println(matchSet.getClassifier(toBeDeleted));
				population.deleteClassifier(matchSet.getClassifier(toBeDeleted));
				matchSet.deleteClassifier(toBeDeleted);

				myLcs.numberOfClassifiersDeletedInMatchSets++;

			}
			
			
			lowestFitnessIndices.clear();

		}
		
		distinctCoverage.clear();

	}
	
	
	/**
	 * Quantize per coverage, delete every classifier with fitness < DELTA * < F > in every quantum
	 * 
	 * */
	private void controlPopulationInMatchSet5 (final ClassifierSet population, final ClassifierSet matchSet) {
		
		
		ArrayList <ClassifierSet> classifiersPerCoverage = new ArrayList<ClassifierSet>();
		// get the distinct coverages
		for (int i = 0; i < matchSet.getNumberOfMacroclassifiers(); i++) {
			
			final Macroclassifier macro = matchSet.getMacroclassifier(i);
			final Classifier cl = matchSet.getClassifier(i);
			
			if (cl.objectiveCoverage > 0 && !distinctCoverage.contains((float) cl.objectiveCoverage)) {
				distinctCoverage.add((float) cl.objectiveCoverage);	
				classifiersPerCoverage.add(new ClassifierSet(null));
			}
			
			for (int j = 0; j < distinctCoverage.size(); j++) {
				if (cl.objectiveCoverage == distinctCoverage.get(j))
					classifiersPerCoverage.get(j).addClassifier(macro, false);
			}
		}

		
		
		// edo exo brei ta distinct coverages kai exo spasei tous kanones tou M se classifier sets me basi to coverage tous 
		
		
		
		for (int c = 0; c < distinctCoverage.size(); c++) {
			
			// brisko to mean fitness tou ka9e classifier set
			
			double meanFitness = 0;
			
			for (int i = 0; i < classifiersPerCoverage.get(c).getNumberOfMacroclassifiers(); i++) {
				final Macroclassifier cl = classifiersPerCoverage.get(c).getMacroclassifier(i);
				meanFitness += cl.myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS);
			}

			meanFitness /= classifiersPerCoverage.get(c).getNumberOfMacroclassifiers();
			
			//if (classifiersPerCoverage.get(c).getNumberOfMacroclassifiers() > 1) {
				for (int i = 0; i < classifiersPerCoverage.get(c).getNumberOfMacroclassifiers(); i++) {
					final Macroclassifier cl = classifiersPerCoverage.get(c).getMacroclassifier(i);
					if (cl.myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS) < DELTA * meanFitness) {
						
						classifiersPerCoverage.get(c).deleteClassifier(cl.myClassifier);
						population.deleteClassifier(cl.myClassifier);
						matchSet.deleteClassifier(cl.myClassifier);
						
					}
				}
			
			//}

		}
		
		distinctCoverage.clear();
		classifiersPerCoverage.clear();
	}
	
	
	
	
	/**
	 * In the lowest coverage block of classifiers, delete the one that its fitness is the lowest and also lower than the mean fitness 
	 * of the block's classifiers.
	 * */
	private void controlPopulationInMatchSet6 (final ClassifierSet population, final ClassifierSet matchSet) {
		
		double lowestCoverage = Double.MAX_VALUE;

		for (int i = 0; i < matchSet.getNumberOfMacroclassifiers(); i++) {
			
			final Classifier cl = matchSet.getClassifier(i);
			if (cl.objectiveCoverage > 0 && cl.objectiveCoverage <= lowestCoverage) { // CL.getcoverage?
				
				if (cl.objectiveCoverage < lowestCoverage) {
					lowestCoverageIndices.clear();
				}
				
				lowestCoverage = cl.objectiveCoverage;
				lowestCoverageIndices.add(i);
			}
		}
		
		if (lowestCoverageIndices.size() > 1) {
			
			
			/* get the mean fitness of the block */
			double meanFitness = 0;
			for (int i = 0; i < lowestCoverageIndices.size(); i++) {
		
				final Macroclassifier macro = matchSet.getMacroclassifier(lowestCoverageIndices.get(i));
				final Classifier cl = macro.myClassifier;

				meanFitness += cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS);
			}
			
			meanFitness /= lowestCoverageIndices.size();

			
			double lowestFitness = Double.MAX_VALUE;
			int toBeDeleted = -1;

			for (int i = 0; i < lowestCoverageIndices.size(); i++) {
				
				final Macroclassifier macro = matchSet.getMacroclassifier(lowestCoverageIndices.get(i));
				final Classifier cl = macro.myClassifier;

				if (cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS) <= lowestFitness) {						
					lowestFitness = cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS);
					toBeDeleted = lowestCoverageIndices.get(i);
					
				}
			}
			
			if (lowestFitness < meanFitness) {
				if (toBeDeleted >= 0) { // not necessary here
					myLcs.numberOfClassifiersDeletedInMatchSets++;
					population.deleteClassifier(matchSet.getMacroclassifier(toBeDeleted).myClassifier);
					matchSet.deleteClassifier(toBeDeleted);
				}
			}
		}
		
		lowestCoverageIndices.clear();
				
	}

	
	private void controlPopulationInIncorrectSet1 (final ClassifierSet population, 
			final ClassifierSet matchSet,
			final ClassifierSet incorrectSet) {

		double highestCoverage = Double.MIN_VALUE;
		int toBeDeleted = -1;
		
		for (int i = 0; i < incorrectSet.getNumberOfMacroclassifiers(); i++) {
		
		final Classifier cl = incorrectSet.getClassifier(i);
		if (cl.objectiveCoverage > 0 && cl.objectiveCoverage >= highestCoverage) { // CL.getcoverage?
		
			if (cl.objectiveCoverage > highestCoverage) {
				lowestCoverageIndices.clear();
			}
			
			highestCoverage = cl.objectiveCoverage;
			lowestCoverageIndices.add(i);
			}
		}
		
		if (lowestCoverageIndices.size() > 1) {
		
			double lowestFitness = Double.MAX_VALUE;
			
			for (int i = 0; i < lowestCoverageIndices.size(); i++) {
			
				final Macroclassifier macro = incorrectSet.getMacroclassifier(lowestCoverageIndices.get(i));
				final Classifier cl = macro.myClassifier;
				
				if (cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS) <= lowestFitness) {
				
					lowestFitness = cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS);
					toBeDeleted = lowestCoverageIndices.get(i);
				}
			}
			
			if (toBeDeleted >= 0) {
				myLcs.numberOfClassifiersDeletedInMatchSets++;
				population.deleteClassifier(incorrectSet.getMacroclassifier(toBeDeleted).myClassifier);
				matchSet.deleteClassifier(incorrectSet.getMacroclassifier(toBeDeleted).myClassifier);
				incorrectSet.deleteClassifier(toBeDeleted);
			}
		}
		
		lowestCoverageIndices.clear();
	
	
	}
	
	
	
	
	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#cover(gr.auth.ee.lcs.classifiers
	 * .ClassifierSet, int)
	 */
	@Override
	public void cover(ClassifierSet population, 
					    int instanceIndex) {
		
		final Classifier coveringClassifier = myLcs.getClassifierTransformBridge()
											  .createRandomCoveringClassifier(myLcs.instances[instanceIndex]);
		
		coveringClassifier.created = myLcs.totalRepetition;//ga.getTimestamp();
		
		coveringClassifier.cummulativeInstanceCreated = myLcs.getCummulativeCurrentInstanceIndex();
		
		coveringClassifier.setClassifierOrigin(Classifier.CLASSIFIER_ORIGIN_COVER);
		myLcs.numberOfCoversOccured ++ ;
		population.addClassifier(new Macroclassifier(coveringClassifier, 1), false);
	}
	
	
	@Override
	public void coverSmp(ClassifierSet population, 
					    int instanceIndex) {
		
		final Classifier coveringClassifier = myLcs
											  .getClassifierTransformBridge()
											  .createRandomCoveringClassifier(myLcs.instances[instanceIndex]);
		coveringClassifier.created = myLcs.totalRepetition;//ga.getTimestamp();
		coveringClassifier.setClassifierOrigin(Classifier.CLASSIFIER_ORIGIN_COVER); // o classifier proekupse apo cover
		myLcs.numberOfCoversOccured ++ ;
		population.addClassifierSmp(new Macroclassifier(coveringClassifier, 1), false, null);
	}
	
	
	private Macroclassifier coverNew( int instanceIndex ) {
		
		final Classifier coveringClassifier = myLcs.getClassifierTransformBridge()
		  									  .createRandomCoveringClassifier(myLcs.instances[instanceIndex]);

		coveringClassifier.created = myLcs.totalRepetition;//ga.getTimestamp();
		
		coveringClassifier.cummulativeInstanceCreated = myLcs.getCummulativeCurrentInstanceIndex();
		
		coveringClassifier.setClassifierOrigin(Classifier.CLASSIFIER_ORIGIN_COVER);
		myLcs.numberOfCoversOccured ++ ;
		return new Macroclassifier(coveringClassifier, 1);
	}
	

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#createStateClassifierObject()
	 * */
	@Override				

	public Serializable createStateClassifierObject() {
		return new MlASLCSClassifierData();
	}
	
	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#createStateClassifierObjectArray()
	 * */
	@Override	
	public Serializable[] createClassifierObjectArray() {
		
		MlASLCSClassifierData classifierObjectArray[] = new MlASLCSClassifierData[(int) SettingsLoader.getNumericSetting("numberOfLabels", 1)];
		for (int i = 0; i < numberOfLabels; i++) {
			classifierObjectArray[i] = new MlASLCSClassifierData();
		}
		return classifierObjectArray;
	}
	
	
	
	/**
	 * Generates the correct set.
	 * 
	 * @param matchSet
	 *            the match set
	 * @param instanceIndex
	 *            the global instance index
	 * @param labelIndex
	 *            the label index
	 * @return the correct set
	 */
	private ClassifierSet generateLabelCorrectSet(final ClassifierSet matchSet,
												   final int instanceIndex, 
												   final int labelIndex) {
		
		final ClassifierSet correctSet = new ClassifierSet(null);
		final ClassifierSet correctSetOnlyWildcards = new ClassifierSet(null);
		final ClassifierSet correctSetWithoutWildcards = new ClassifierSet(null);

		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();
		
		for (int i = 0; i < matchSetSize; i++) {
			
			final Macroclassifier cl = matchSet.getMacroclassifier(i);
			
			if (wildCardsParticipateInCorrectSets) {
				
				if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) >= 0) // change: (=) means # => [C]
					correctSet.addClassifier(cl, false);
				
				if (balanceCorrectSets) {
					
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) == 0) 
						correctSetOnlyWildcards.addClassifier(cl, false);
					
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) > 0)
						correctSetWithoutWildcards.addClassifier(cl, false);
				}
			}
			else 
				if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) > 0)
					correctSet.addClassifier(cl, false);

		}
		
		if (wildCardsParticipateInCorrectSets && balanceCorrectSets) {
			int correctSetWithoutWildcardsNumerosity = correctSetWithoutWildcards.getNumberOfMacroclassifiers();
			int correctSetOnlyWildcardsNumerosity = correctSetOnlyWildcards.getNumberOfMacroclassifiers();
	
			if (correctSetOnlyWildcardsNumerosity <= wildCardParticipationRatio * correctSetWithoutWildcardsNumerosity)
				return correctSet;
			else	
				return correctSetWithoutWildcards;
		}
		
		else return correctSet;
	}
	
	
	
	/**
	 * Generates the incorrect set.
	 * 
	 * @param matchSet
	 *            the match set
	 * @param instanceIndex
	 *            the global instance index
	 * @param labelIndex
	 *            the label index
	 * @return the incorrect set
	 */
	private ClassifierSet generateLabelIncorrectSet(final ClassifierSet matchSet,
												   final int instanceIndex, 
												   final int labelIndex) {
		
		final ClassifierSet incorrectSet = new ClassifierSet(null);
		final ClassifierSet incorrectSetOnlyWildcards = new ClassifierSet(null);
		final ClassifierSet incorrectSetWithoutWildcards = new ClassifierSet(null);

		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();
		
		for (int i = 0; i < matchSetSize; i++) {
			
			final Macroclassifier cl = matchSet.getMacroclassifier(i);
			
			if (wildCardsParticipateInCorrectSets) {
				
				if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) < 0) // change: (=) means # => [C]
					incorrectSet.addClassifier(cl, false);
				
				if (balanceCorrectSets) {
					
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) == 0) 
						incorrectSetOnlyWildcards.addClassifier(cl, false);
					
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) < 0)
						incorrectSetWithoutWildcards.addClassifier(cl, false);
				}
			}
			else 
				if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) <= 0)
					incorrectSet.addClassifier(cl, false);

		}
		
		if (wildCardsParticipateInCorrectSets && balanceCorrectSets) {
			int incorrectSetWithoutWildcardsNumerosity = incorrectSetWithoutWildcards.getNumberOfMacroclassifiers();
			int incorrectSetOnlyWildcardsNumerosity = incorrectSetOnlyWildcards.getNumberOfMacroclassifiers();
	
			if (incorrectSetOnlyWildcardsNumerosity <= wildCardParticipationRatio * incorrectSetWithoutWildcardsNumerosity)
				return incorrectSet;
			else	
				return incorrectSetWithoutWildcards;
		}
		
		else return incorrectSet;
	}
	

	private ClassifierSet generateLabelCorrectSetSmp(final ClassifierSet matchSet,
													   final int instanceIndex, 
													   final int labelIndex,
													   final int threadId) {
		
		final ClassifierSet correctSet = new ClassifierSet(null);
		final ClassifierSet correctSetOnlyWildcards = new ClassifierSet(null);
		final ClassifierSet correctSetWithoutWildcards = new ClassifierSet(null);

		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();
		
		if (threadId == 0)
		{	
			for (int i = 0; i < matchSetSize; i++) {
				
				final Macroclassifier cl = matchSet.getMacroclassifier(i);
				
				if (wildCardsParticipateInCorrectSetsSmp) {

					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) >= 0) // change: (=) means # => [C]
							correctSet.addClassifier(cl, false);

					if (balanceCorrectSetsSmp) {
						
						if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) == 0) 
							correctSetOnlyWildcards.addClassifier(cl, false);
						
						if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) > 0)
							correctSetWithoutWildcards.addClassifier(cl, false);
					}
				}
				else 
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) > 0)
					correctSet.addClassifier(cl, false);

			}
		}
		else
		{
			for (int i = 0; i < matchSetSize; i++) {
				
				final Macroclassifier cl = matchSet.getMacroclassifier(i);
				
				if (wildCardsParticipateInCorrectSetsSmp) {

					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) >= 0) // change: (=) means # => [C]
							correctSet.addClassifier(cl, false);

					if (balanceCorrectSetsSmp) {
						
						if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) == 0) 
							correctSetOnlyWildcards.addClassifier(cl, false);
						
						if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) > 0)
							correctSetWithoutWildcards.addClassifier(cl, false);
					}
				}
				else 
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) > 0)
					correctSet.addClassifier(cl, false);

			}
		}
		
		if (wildCardsParticipateInCorrectSetsSmp && balanceCorrectSetsSmp) {
			int correctSetWithoutWildcardsNumerosity = correctSetWithoutWildcards.getNumberOfMacroclassifiers();
			int correctSetOnlyWildcardsNumerosity = correctSetOnlyWildcards.getNumberOfMacroclassifiers();
	
			if (correctSetOnlyWildcardsNumerosity <= wildCardParticipationRatioSmp * correctSetWithoutWildcardsNumerosity)
				return correctSet;
			else	
				return correctSetWithoutWildcards;
		}
		
		else return correctSet;
		
	}
	
	
	private ClassifierSet generateLabelIncorrectSetSmp (final ClassifierSet matchSet,
														   final int instanceIndex, 
														   final int labelIndex,
														   final int threadId) {

		final ClassifierSet incorrectSet = new ClassifierSet(null);
		final ClassifierSet incorrectSetOnlyWildcards = new ClassifierSet(null);
		final ClassifierSet incorrectSetWithoutWildcards = new ClassifierSet(null);
		
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();
		
		if (threadId == 0)
		{	
			for (int i = 0; i < matchSetSize; i++) {
			
				final Macroclassifier cl = matchSet.getMacroclassifier(i);
				
				if (wildCardsParticipateInCorrectSetsSmp) {
				
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) < 0) // change: (=) means # => [C]
						incorrectSet.addClassifier(cl, false);
					
					if (balanceCorrectSetsSmp) {
						
						if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) == 0) 
							incorrectSetOnlyWildcards.addClassifier(cl, false);
						
						if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) < 0)
							incorrectSetWithoutWildcards.addClassifier(cl, false);
					}
				}
				else 
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) <= 0)
						incorrectSet.addClassifier(cl, false);
			
			}
		}
		else
		{
			
			for (int i = 0; i < matchSetSize; i++) {
			
				final Macroclassifier cl = matchSet.getMacroclassifier(i);
				
				if (wildCardsParticipateInCorrectSetsSmp) {
				
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) < 0) // change: (=) means # => [C]
						incorrectSet.addClassifier(cl, false);
					
					if (balanceCorrectSetsSmp) {
					
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) == 0) 
						incorrectSetOnlyWildcards.addClassifier(cl, false);
					
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) < 0)
						incorrectSetWithoutWildcards.addClassifier(cl, false);
					}
				}
				else 
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) <= 0)
						incorrectSet.addClassifier(cl, false);
			
			}
		}
		
		if (wildCardsParticipateInCorrectSetsSmp && balanceCorrectSetsSmp) {
			int correctSetWithoutWildcardsNumerosity = incorrectSetWithoutWildcards.getNumberOfMacroclassifiers();
			int correctSetOnlyWildcardsNumerosity = incorrectSetOnlyWildcards.getNumberOfMacroclassifiers();
			
			if (correctSetOnlyWildcardsNumerosity <= wildCardParticipationRatioSmp * correctSetWithoutWildcardsNumerosity)
				return incorrectSet;
			else	
				return incorrectSetWithoutWildcards;
		}
		
		else return incorrectSet;
		
		}
	
	
	private ClassifierSet generateLabelCorrectSetNewSmp(final ClassifierSet matchSet,
			   											final int instanceIndex, 
			   											final int labelIndex) {
		
		matchSetNewSmp = matchSet;
		instanceIndexNewSmp = instanceIndex;
		labelIndexNewSmp = labelIndex;
		
		correctSetNewSmp = new ClassifierSet(null);
		correctSetOnlyWildcardsNewSmp = new ClassifierSet(null);
		correctSetWithoutWildcardsNewSmp = new ClassifierSet(null);
		
		try{
			ptGenerateCorrectSetNew.execute( new ParallelRegion() {
			
				public void run() throws Exception
				{
					execute(0,matchSetSmp.getNumberOfMacroclassifiers()-1, new IntegerForLoop() {
						
						public void run( int first, int last )
						{
							for ( int i = first; i <= last ; ++i )
							{
								final Macroclassifier cl = matchSet.getMacroclassifier(i);
								
								if (wildCardsParticipateInCorrectSets) {
									
									if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) >= 0) // change: (=) means # => [C]
										correctSetNewSmp.addClassifier(cl, false);
									
									if (balanceCorrectSets) {
										
										if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) == 0) 
											correctSetOnlyWildcardsNewSmp.addClassifier(cl, false);
										
										if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) > 0)
											correctSetWithoutWildcardsNewSmp.addClassifier(cl, false);
									}
								}
								else 
									if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) > 0)
									correctSetNewSmp.addClassifier(cl, false);
							}
						}
						
					});
				}
			
			});
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		
		if (wildCardsParticipateInCorrectSets && balanceCorrectSets) {
			int correctSetWithoutWildcardsNumerosity = correctSetWithoutWildcardsNewSmp.getNumberOfMacroclassifiers();
			int correctSetOnlyWildcardsNumerosity = correctSetOnlyWildcardsNewSmp.getNumberOfMacroclassifiers();
	
			if (correctSetOnlyWildcardsNumerosity <= wildCardParticipationRatio * correctSetWithoutWildcardsNumerosity)
				return correctSetNewSmp;
			else	
				return correctSetWithoutWildcardsNewSmp;
		}
		
		else return correctSetNewSmp;
		
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#getComparisonValue(gr.auth
	 * .ee.lcs.classifiers.Classifier, int)
	 */
	@Override
	public double getComparisonValue(Classifier aClassifier, int mode) {
				
		final MlASLCSClassifierData data = (MlASLCSClassifierData) aClassifier.getUpdateDataObject();
		
		switch (mode) {
		case COMPARISON_MODE_EXPLORATION:
			//return aClassifier.objectiveCoverage < 0 ? 0 : data.fitness;
			//return aClassifier.objectiveCoverage < 0 ? 0 : (aClassifier.experience < aClassifier.objectiveCoverage * myLcs.instances.length * 2 ? data.fitness / 10 : data.fitness);
			//return data.d != 0 ? (aClassifier.objectiveCoverage < 0 ? /*Math.pow(data.fitness, 2)*/0 : data.fitness) : (aClassifier.experience < THETA_DEL ? 0 : data.fitness);
			//return data.d == 0 ? (aClassifier.objectiveCoverage < 0 ? /*Math.pow(data.fitness, 2)*/0 : data.fitness) : (aClassifier.experience < THETA_DEL ? 0 : data.fitness);
			//return data.d != 0 ? (aClassifier.objectiveCoverage < 0 ? 0 : (aClassifier.experience < 0.5 * THETA_DEL ? 0: data.fitness)) : (aClassifier.experience < THETA_DEL ? 0 : data.fitness);
			return aClassifier.experience < THETA_DEL ? 0 : data.fitness;
			//return aClassifier.experience < 10 ? 0 : data.fitness;

			//return data.d != 0 ? (aClassifier.experience < 0.5 * THETA_DEL ? 0: data.fitness) : (aClassifier.experience < THETA_DEL ? 0 : data.fitness);

		case COMPARISON_MODE_DELETION:
			return data.d;
		
		case COMPARISON_MODE_EXPLOITATION:
			return Double.isNaN(data.tp / data.msa) ? 0 : data.tp / data.msa;
			
		case COMPARISON_MODE_PURE_FITNESS:
			return data.fitness;
			
		case COMPARISON_MODE_PURE_ACCURACY:
			return Double.isNaN(data.tp / data.msa) ? 0 : data.tp / data.msa;
		
		case COMPARISON_MODE_ACCURACY:
			return (aClassifier.objectiveCoverage < 0) ? 2.0 : data.tp / data.msa;

		default:
		}
		return 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#getData(gr.auth.ee.lcs.classifiers
	 * .Classifier)
	 */
	@Override
	public String getData(Classifier aClassifier) {
		
		final MlASLCSClassifierData data = ((MlASLCSClassifierData) aClassifier.getUpdateDataObject());
		
        DecimalFormat df = new DecimalFormat("#.####");

		return  /* " internalFitness: " + df.format(data.fitness) 
				+ */"tp:|" + df.format(data.tp)  + "|"
				+ "msa:|" + df.format(data.msa)  + "|"
				+ "ns:|" + df.format(data.ns) + "|"
				+ "d:|" + df.format(data.d) + "|"
				/*+ " total fitness: " + df.format(data.totalFitness) 
				+ " alt fitness: " + df.format(data.alternateFitness) */ ;
	}

	
	public double getNs (Classifier aClassifier) {
		final MlASLCSClassifierData data = (MlASLCSClassifierData) aClassifier.getUpdateDataObject();
		return data.ns;
	}
	
	public double getAccuracy (Classifier aClassifier) {
		final MlASLCSClassifierData data = (MlASLCSClassifierData) aClassifier.getUpdateDataObject();
		return (Double.isNaN(data.tp / data.msa) ? 0.0 : data.tp / data.msa);
	}
	
	
	
	@Override
	public void inheritParentParameters(Classifier parentA, 
										 Classifier parentB,
										 Classifier child) {
		
		final MlASLCSClassifierData childData = ((MlASLCSClassifierData) child.getUpdateDataObject());
		
/*		final MlASLCSClassifierData parentAData = ((MlASLCSClassifierData) parentA
				.getUpdateDataObject());
		final MlASLCSClassifierData parentBData = ((MlASLCSClassifierData) parentB
				.getUpdateDataObject());
		childData.ns = (parentAData.ns + parentBData.ns) / 2;*/
		
		childData.ns = 1;
		child.setComparisonValue(COMPARISON_MODE_EXPLORATION, 1);
	}
	
	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#performUpdate(gr.auth.ee.lcs
	 * .classifiers.ClassifierSet, gr.auth.ee.lcs.classifiers.ClassifierSet)
	 */
	@Override
	public void performUpdate(ClassifierSet matchSet, ClassifierSet correctSet) {
		// Nothing here!
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#setComparisonValue(gr.auth
	 * .ee.lcs.classifiers.Classifier, int, double)
	 */
	@Override
	public void setComparisonValue(Classifier aClassifier, 
									int mode,
									double comparisonValue) {
		
		final MlASLCSClassifierData data = ((MlASLCSClassifierData) aClassifier.getUpdateDataObject());
		data.fitness = comparisonValue;
	}
	
	
	
	
	/**
	 * Share a the fitness among a set.
	 * 
	 * @param matchSet
	 * 			the match set
	 * 
	 * @param labelCorrectSet
	 *           a correct set in which we share fitness
	 *            
	 * @param l
	 * 			 the index of the label for which the labelCorrectSet is formed
	 * 
	 * @param instanceIndex
	 * 			the index of the instance           
	 * 
	 * @author alexandros philotheou
	 * 
	 */
	private void shareFitness(final ClassifierSet matchSet, 
								final ClassifierSet labelCorrectSet,
								final int l,
								int instanceIndex) {
		
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();

		double relativeAccuracy = 0;
		
		for (int i = 0; i < matchSetSize; i++) { // gia ka9e macroclassifier
			
			final Macroclassifier cl = matchSet.getMacroclassifier(i); 
			final MlASLCSClassifierData dataArray[] = (MlASLCSClassifierData[]) cl.myClassifier.getUpdateDataArray();
			final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();

			// Get classification ability for label l. an anikei sto labelCorrectSet me alla logia.
			final float classificationAbility = cl.myClassifier.classifyLabelCorrectly(instanceIndex, l);
			final int labelNs = labelCorrectSet.getTotalNumerosity();
			
			// update true positives, msa and niche set size
			if (classificationAbility == 0) {// an proekupse apo adiaforia

				dataArray[l].tp += OMEGA;
				dataArray[l].msa += PHI;
				
				data.tp += OMEGA;
				data.msa += PHI;
				
				if (wildCardsParticipateInCorrectSets) {
					
					dataArray[l].minCurrentNs = Integer.MAX_VALUE;

					if (dataArray[l].minCurrentNs > labelNs) 
						dataArray[l].minCurrentNs = labelNs;

					if ((dataArray[l].tp / dataArray[l].msa) > ACC_0) {
						dataArray[l].k = 1;
					}
					else {
						dataArray[l].k = a * Math.pow(((dataArray[l].tp / dataArray[l].msa) / ACC_0), n);
						}
				}
				else
					dataArray[l].k = 0;
					
				
			}
			else if (classificationAbility > 0) { // an proekupse apo 9etiki apofasi 
				dataArray[l].minCurrentNs = Integer.MAX_VALUE;

				dataArray[l].tp += 1;
				data.tp += 1;
				
				if (dataArray[l].minCurrentNs > labelNs) 
					dataArray[l].minCurrentNs = labelNs;
				
				if ((dataArray[l].tp / dataArray[l].msa) > ACC_0) {
					dataArray[l].k = 1;
				}
				else {
					dataArray[l].k = a * Math.pow(((dataArray[l].tp / dataArray[l].msa) / ACC_0), n);
				}	
			}
			else dataArray[l].k = 0;
			
			
			// update msa for positive or negative decision (not updated above)
			if (classificationAbility != 0) {
				dataArray[l].msa += 1;
				data.msa += 1;
			}
			
			 relativeAccuracy += cl.numerosity * dataArray[l].k;
		} // kleinei to for gia ka9e macroclassifier
		
		if (relativeAccuracy == 0) relativeAccuracy = 1;

		for (int i = 0; i < matchSetSize; i++) {
			final Macroclassifier cl = matchSet.getMacroclassifier(i); 
			final MlASLCSClassifierData dataArray[] = (MlASLCSClassifierData[]) cl.myClassifier.getUpdateDataArray();
			dataArray[l].fitness += LEARNING_RATE * (cl.numerosity * dataArray[l].k / relativeAccuracy - dataArray[l].fitness);
			
			//dataArray[l].fitness = Math.pow(dataArray[l].tp / dataArray[l].msa, n); //==> GOOD although too compact
		}
	}
	
	

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.data.AbstractUpdateStrategy#updateSet(gr.auth.ee.lcs.
	 * classifiers.ClassifierSet, gr.auth.ee.lcs.classifiers.ClassifierSet, int,
	 * boolean)
	 */
	@Override
	public void updateSet(ClassifierSet population, 
						   ClassifierSet matchSet,
						   int instanceIndex, 
						   boolean evolve) {

		// Create all label correct sets
		final ClassifierSet[] labelCorrectSets = new ClassifierSet[numberOfLabels];
		
		if(commencedDeletions)
			controlPopulationInMatchSet(population, matchSet);
		
		generateCorrectSetTime = -System.currentTimeMillis(); 
		
		for (int i = 0; i < numberOfLabels; i++) { // gia ka9e label parago to correctSet pou antistoixei se auto
			
			labelCorrectSets[i] = generateLabelCorrectSet(matchSet, instanceIndex, i); // periexei tous kanones pou apofasizoun gia to label 9etika.
			
/*			System.out.println("label: " + i);
			System.out.print("instance: ");
			for (int k = 0; k < myLcs.instances[0].length / 2; k++) {
				System.out.print((int)myLcs.instances[instanceIndex][k]);
			}
			System.out.print("=>");
			for (int k = myLcs.instances[0].length / 2; k < myLcs.instances[0].length; k++) {
				System.out.print((int)myLcs.instances[instanceIndex][k]);
			}
			System.out.println(labelCorrectSets[i]);*/
		
		}		
		
		generateCorrectSetTime += System.currentTimeMillis();

		
		int CorrectSetsPopulation = 0;
		for (int i = 0; i < numberOfLabels; i++) {
			CorrectSetsPopulation += labelCorrectSets[i].getNumberOfMacroclassifiers() ;
		}
		
		myLcs.meanCorrectSetNumerosity = CorrectSetsPopulation / numberOfLabels;

		
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();

		updateParametersTime = -System.currentTimeMillis();
		
		if (FITNESS_MODE == FITNESS_MODE_SIMPLE || FITNESS_MODE == FITNESS_MODE_COMPLEX) {
			// For each classifier in the matchset
			for (int i = 0; i < matchSetSize; i++) { // gia ka9e macroclassifier

				final Macroclassifier cl = matchSet.getMacroclassifier(i); // getMacroclassifier => fernei to copy, oxi ton idio ton macroclassifier
				

				int minCurrentNs = Integer.MAX_VALUE;
				final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
	
				for (int l = 0; l < numberOfLabels; l++) {
					// Get classification ability for label l. an anikei sto labelCorrectSet me alla logia.
					final float classificationAbility = cl.myClassifier.classifyLabelCorrectly(instanceIndex, l);
					final int labelNs = labelCorrectSets[l].getTotalNumerosity();

					if (classificationAbility == 0) {// an proekupse apo adiaforia
						data.tp += OMEGA;
						data.msa += PHI;
						
						if (wildCardsParticipateInCorrectSets) {
							if (minCurrentNs > labelNs) { 
								minCurrentNs = labelNs;
							}
						}
					}
					else if (classificationAbility > 0) { // an proekupse apo 9etiki apofasi (yper)
						data.tp += 1;
						
						if (minCurrentNs > labelNs) { // bainei edo mono otan exei prokupsei apo 9etiki apofasi
							minCurrentNs = labelNs;
						}
					}
					if (classificationAbility != 0) 
						data.msa += 1;
				} // kleinei to for gia ka9e label
	
				cl.myClassifier.experience++;
				
	
				
				/* einai emmesos tropos na elegkso oti o kanonas anikei sto (se ena toulaxiston ennoo) labelCorrectSet
				 * giati to minCurrentNs allazei mono an classificationAbility > 0 dld o kanonas apofasizei, den adiaforei
				 */
				if (minCurrentNs != Integer.MAX_VALUE) {
					//data.ns += .1 * (minCurrentNs - data.ns);
					data.ns += LEARNING_RATE * (minCurrentNs - data.ns);
				}
				
				switch (FITNESS_MODE) {
				
				case FITNESS_MODE_SIMPLE:
					data.fitness = Math.pow ((data.tp) / (data.msa), n);
					break;

				case FITNESS_MODE_COMPLEX:
					data.fitness += LEARNING_RATE * (Math.pow((data.tp) / (data.msa), n) - data.fitness);
					
/*					  data.fitness += LEARNING_RATE * (cl.numerosity * Math.pow((data.tp) / (data.msa), n) - data.fitness);
					  data.fitness /= cl.numerosity;*/
					 
					break;
				}
				updateSubsumption(cl.myClassifier);
			} // kleinei to for gia ka9e macroclassifier
		}
		
		
		
		else if (FITNESS_MODE == FITNESS_MODE_SHARING) {
			
			for (int l = 0; l < numberOfLabels; l++) {
				shareFitness(matchSet, labelCorrectSets[l], l, instanceIndex);
			} 
			
			for (int i = 0; i < matchSetSize; i++) { 
				final Macroclassifier cl = matchSet.getMacroclassifier(i);	
				cl.myClassifier.experience++; 
				final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
				final MlASLCSClassifierData dataArray[] = (MlASLCSClassifierData[]) cl.myClassifier.getUpdateDataArray();
				
				double fitnessSum = 0;
				double ns = 0;
				
				for (int l = 0; l < numberOfLabels; l++) {
					fitnessSum += dataArray[l].fitness;	
					ns += dataArray[l].minCurrentNs;
				}
				ns /= numberOfLabels;
				data.fitness = (fitnessSum / cl.numerosity) / numberOfLabels;

				if (ns != Integer.MAX_VALUE) {
					//data.ns += .1 * (minCurrentNs - data.ns);
					data.ns += LEARNING_RATE * (ns - data.ns);
				}
					
				if (Math.pow(data.tp / data.msa, n) > ACC_0) {
					if (cl.myClassifier.experience >= this.subsumptionExperienceThreshold)
						cl.myClassifier.setSubsumptionAbility(true);
				}
				else {
					cl.myClassifier.setSubsumptionAbility(false);
				}
				
			} 
		}
		
		updateParametersTime += System.currentTimeMillis();
		
		evolutionTime = 0;
		
		numberOfEvolutionsConducted = 0;
		numberOfSubsumptionsConducted = 0;
		numberOfDeletionsConducted = 0;
		numberOfNewClassifiers = 0;
		subsumptionTime = 0;
		deletionTime = 0;
			
		if (evolve) {
			evolutionTime = -System.currentTimeMillis();

			for (int l = 0; l < numberOfLabels; l++) {
				if (labelCorrectSets[l].getNumberOfMacroclassifiers() > 0) {
					
					ga.evolveSet(labelCorrectSets[l], population, l);
					population.totalGAInvocations = ga.getTimestamp();
					
					numberOfEvolutionsConducted += ga.evolutionConducted();
					numberOfSubsumptionsConducted += ga.getNumberOfSubsumptionsConducted();
					numberOfNewClassifiers += ga.getNumberOfNewClassifiers();
					subsumptionTime += ga.getSubsumptionTime();
					deletionTime += ga.getDeletionTime();
					numberOfDeletionsConducted += ga.getNumberOfDeletionsConducted();
				} else {
					
					this.cover(population, instanceIndex);
					numberOfNewClassifiers++;
					IPopulationControlStrategy theControlStrategy = population.getPopulationControlStrategy();
					numberOfDeletionsConducted += theControlStrategy.getNumberOfDeletionsConducted(); 
					deletionTime += theControlStrategy.getDeletionTime();
				}
			}
			evolutionTime += System.currentTimeMillis();
		}
	}	
	
	
	public void updateSetSmp(ClassifierSet population, 
							   ClassifierSet matchSet,
							   int instanceIndex, 
							   boolean evolve) {
		

		final ClassifierSet[] labelCorrectSets;
		
		labelCorrectSetsSmp = new ClassifierSet[numberOfLabels];
		matchSetSmp = matchSet;
		instanceIndexSmp = instanceIndex;
		numberOfLabelsSmp = numberOfLabels;
		
		generateCorrectSetTime = -System.currentTimeMillis();
		
		try{
			ptGenerateCorrectSet.execute( new ParallelRegion() {				
				public void run() throws Exception
				{
					execute(0,numberOfLabelsSmp-1, new IntegerForLoop () {
						
						ClassifierSet[] labelCorrectSets_thread;
						
						Vector<Integer> labelCorrectSetsIndices;
						
						long p0,p1,p2,p3,p4,p5,p6,p7;
						long p8,p9,pa,pb,pc,pd,pe,pf;
						
						public void run(int first, int last)
						{
							labelCorrectSets_thread = new ClassifierSet[last-first+1];
							labelCorrectSetsIndices = new Vector<Integer>(); 
							for ( int i = first; i <= last ; i++ )
							{
								final ClassifierSet correctSet = new ClassifierSet(null);
								final int matchSetSize = matchSetSmp.getNumberOfMacroclassifiers();
								for (int j = 0; j < matchSetSize; j++) 
								{
									final Macroclassifier cl = matchSetSmp.getMacroclassifier(j);
									if (cl.myClassifier.classifyLabelCorrectly(instanceIndexSmp, i) > 0)
										correctSet.addClassifier(cl, false);
								}
								labelCorrectSets_thread[i-first] = correctSet;
								labelCorrectSetsIndices.add(i); 
							}							
						}
						
						public void finish() throws Exception
						{
							region().critical(new ParallelSection() {
								public void run()
								{
									for (int i = 0 ; i < labelCorrectSetsIndices.size() ; i++)
									{
										labelCorrectSetsSmp[labelCorrectSetsIndices.elementAt(i)] =
											labelCorrectSets_thread[i];
									}
								}
							});
						}
						
					});
				}			
			});
		}
		catch( Exception e )
		{
			e.printStackTrace();
		}
		
		generateCorrectSetTime += System.currentTimeMillis();
		
		labelCorrectSets = labelCorrectSetsSmp;

		
		int CorrectSetsPopulation = 0;
		for (int i = 0; i < numberOfLabels; i++) {
			CorrectSetsPopulation += labelCorrectSets[i].getNumberOfMacroclassifiers() ;
		}
		myLcs.meanCorrectSetNumerosity = CorrectSetsPopulation / numberOfLabels;

		updateParametersTime = -System.currentTimeMillis();
		
		if (FITNESS_MODE == FITNESS_MODE_SIMPLE || FITNESS_MODE == FITNESS_MODE_COMPLEX) {
			
			try {
			ptUpdateParameters.execute( new ParallelRegion() {
				public void run() throws Exception
				{
					execute(0,matchSetSmp.getNumberOfMacroclassifiers()-1,new IntegerForLoop() {
						
						public void run(int first,int last)
						{
							for ( int i = first; i <= last ; ++i )
							{
								final Macroclassifier cl = matchSetSmp.getMacroclassifier(i);
								int minCurrentNs = Integer.MAX_VALUE;
								final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
								
								for (int l = 0; l < numberOfLabelsSmp; l++) {
									// Get classification ability for label l. an anikei sto labelCorrectSet me alla logia.
									final float classificationAbility = cl.myClassifier.classifyLabelCorrectly(instanceIndexSmp, l);
									final int labelNs = labelCorrectSetsSmp[l].getTotalNumerosity();

									if (classificationAbility == 0) {// an proekupse apo adiaforia
										data.tp += OMEGA;
										data.msa += PHI;
										
										if (wildCardsParticipateInCorrectSets) {
											if (minCurrentNs > labelNs) { 
												minCurrentNs = labelNs;
											}
										}
									}
									else if (classificationAbility > 0) { // an proekupse apo 9etiki apofasi (yper)
										data.tp += 1;
										
										if (minCurrentNs > labelNs) { // bainei edo mono otan exei prokupsei apo 9etiki apofasi
											minCurrentNs = labelNs;
										}
									}
									if (classificationAbility != 0) data.msa += 1;
								} // kleinei to for gia ka9e label
					
								cl.myClassifier.experience++;
								
								/* einai emmesos tropos na elegkso oti o kanonas anikei sto (se ena toulaxiston ennoo) labelCorrectSet
								 * giati to minCurrentNs allazei mono an classificationAbility > 0 dld o kanonas apofasizei, den adiaforei
								 */
								if (minCurrentNs != Integer.MAX_VALUE) {
									//data.ns += .1 * (minCurrentNs - data.ns);
									data.ns += LEARNING_RATE * (minCurrentNs - data.ns);
								}
								
								switch (FITNESS_MODE) {
								
								case FITNESS_MODE_SIMPLE:
									data.fitness = Math.pow((data.tp) / (data.msa), n);
									break;
								case FITNESS_MODE_COMPLEX:
									data.fitness += LEARNING_RATE * (Math.pow((data.tp) / (data.msa), n) - data.fitness);
									
				                     /*data.fitness += LEARNING_RATE * (cl.numerosity * Math.pow((data.tp) / (data.msa), n) - data.fitness);
									  data.fitness /= cl.numerosity;*/
									 
									break;
								}
								updateSubsumption(cl.myClassifier);
								
							}
							
						}
						
					});
				}
			});
			}
			catch ( Exception e)
			{
				e.printStackTrace();
			}
		}	
		else if (FITNESS_MODE == FITNESS_MODE_SHARING) {
			
			for (int l = 0; l < numberOfLabels; l++) {
				shareFitness(matchSet, labelCorrectSets[l], l, instanceIndex);
			}
			
			try {
				ptUpdateParameters.execute( new ParallelRegion() {
					public void run() throws Exception
					{
						execute(0,matchSetSmp.getNumberOfMacroclassifiers()-1,new IntegerForLoop() {
							
							public void run(int first,int last)
							{
								for ( int i = first; i <= last ; ++i )
								{
									final Macroclassifier cl = matchSetSmp.getMacroclassifier(i);	
									cl.myClassifier.experience++; 
									final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
									final MlASLCSClassifierData dataArray[] = (MlASLCSClassifierData[]) cl.myClassifier.getUpdateDataArray();
									
									double fitnessSum = 0;
									double ns = 0;
									
									for (int l = 0; l < numberOfLabelsSmp; l++) {
										fitnessSum += dataArray[l].fitness;	
										ns += dataArray[l].minCurrentNs;
									}
									ns /= numberOfLabelsSmp;
									data.fitness = (fitnessSum / cl.numerosity) / numberOfLabels;

									if (ns != Integer.MAX_VALUE) {
										//data.ns += .1 * (minCurrentNs - data.ns);
										data.ns += LEARNING_RATE * (ns - data.ns);
									}
										
									if (Math.pow(data.tp / data.msa, n) > ACC_0) {
										if (cl.myClassifier.experience >= subsumptionExperienceThreshold)
											cl.myClassifier.setSubsumptionAbility(true);
									}
									else {
										cl.myClassifier.setSubsumptionAbility(false);
									}
								}
								
							}
							
						});
					}
				});
				}
				catch ( Exception e)
				{
					e.printStackTrace();
				}
		}
		
		updateParametersTime += System.currentTimeMillis();
		
		evolutionTime = 0;
		
		numberOfEvolutionsConducted = 0;
		numberOfSubsumptionsConducted = 0;
		numberOfDeletionsConducted = 0;
		numberOfNewClassifiers = 0;
		subsumptionTime = 0;
		deletionTime = 0;
			
		if (evolve) {
			evolutionTime = -System.currentTimeMillis();
			for (int l = 0; l < numberOfLabels; l++) {
				if (labelCorrectSets[l].getNumberOfMacroclassifiers() > 0) {
					ga.evolveSetSmp(labelCorrectSets[l], population, l);
					population.totalGAInvocations = ga.getTimestamp();
					
					numberOfEvolutionsConducted += ga.evolutionConducted();
					numberOfSubsumptionsConducted += ga.getNumberOfSubsumptionsConducted();
					numberOfNewClassifiers += ga.getNumberOfNewClassifiers();
					subsumptionTime += ga.getSubsumptionTime();
					deletionTime += ga.getDeletionTime();
					numberOfDeletionsConducted += ga.getNumberOfDeletionsConducted();
				} else {
					this.coverSmp(population, instanceIndex);
					numberOfNewClassifiers++;
					IPopulationControlStrategy theControlStrategy = population.getPopulationControlStrategy();
					numberOfDeletionsConducted += theControlStrategy.getNumberOfDeletionsConducted(); 
					deletionTime += theControlStrategy.getDeletionTime();
				}
			}
			evolutionTime += System.currentTimeMillis();
		}
		
		
	}
	
	
	
	
	@Override
	public void updateSetNew(ClassifierSet population, 
							   ClassifierSet matchSet,
							   int instanceIndex, 
							   boolean evolve) {
		
		
		/*
		 * If "&& evolve" is enabled in the condition below,
		 * rules will be deleted from match sets only during the
		 * training period (iterations), not during the update period that follows it.
		 * */
		
		if (commencedDeletions /* && evolve */) {
				controlPopulationInMatchSet(population, matchSet);
		}

		
		// Create all label correct sets
		final ClassifierSet[] labelCorrectSets = new ClassifierSet[numberOfLabels];
		//final ClassifierSet[] labelIncorrectSets = new ClassifierSet[numberOfLabels];


		
		generateCorrectSetTime = -System.currentTimeMillis();

		for (int i = 0; i < numberOfLabels; i++) { // gia ka9e label parago to correctSet pou antistoixei se auto
			
			labelCorrectSets[i] = generateLabelCorrectSet(matchSet, instanceIndex, i); // periexei tous kanones pou apofasizoun gia to label 9etika.
			//labelIncorrectSets[i] = generateLabelIncorrectSet(matchSet, instanceIndex, i);
			

			//controlPopulationInCorrectSet(population, matchSet, labelCorrectSets[i]);
			//controlPopulationInIncorrectSet1(population, matchSet, labelIncorrectSets[i]);
			
			
/*			System.out.println("label: " + i);
			System.out.print("instance: ");
			for (int k = 0; k < myLcs.instances[0].length / 2; k++) {
				System.out.print((int)myLcs.instances[instanceIndex][k]);
			}
			System.out.print("=>");
			for (int k = myLcs.instances[0].length / 2; k < myLcs.instances[0].length; k++) {
				System.out.print((int)myLcs.instances[instanceIndex][k]);
			}
			System.out.println(labelCorrectSets[i]);*/
		
		}
		
		generateCorrectSetTime += System.currentTimeMillis();

		
		int CorrectSetsPopulation = 0;
		for (int i = 0; i < numberOfLabels; i++) {
			CorrectSetsPopulation += labelCorrectSets[i].getNumberOfMacroclassifiers() ;
		}
		myLcs.meanCorrectSetNumerosity = CorrectSetsPopulation / numberOfLabels;

		
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();

		updateParametersTime = -System.currentTimeMillis();
		
		if (FITNESS_MODE == FITNESS_MODE_SIMPLE || FITNESS_MODE == FITNESS_MODE_COMPLEX) {
			// For each classifier in the matchset
			for (int i = 0; i < matchSetSize; i++) { // gia ka9e macroclassifier
				
				final Macroclassifier cl = matchSet.getMacroclassifier(i); // getMacroclassifier => fernei to copy, oxi ton idio ton macroclassifier
				
				int minCurrentNs = Integer.MAX_VALUE;
				final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
	
				for (int l = 0; l < numberOfLabels; l++) {
					// Get classification ability for label l. an anikei sto labelCorrectSet me alla logia.
					final float classificationAbility = cl.myClassifier.classifyLabelCorrectly(instanceIndex, l);
					final int labelNs = labelCorrectSets[l].getTotalNumerosity();

					if (classificationAbility == 0) {// an proekupse apo adiaforia
						data.tp += OMEGA;
						data.msa += PHI;
						
						if (wildCardsParticipateInCorrectSets) {
							if (minCurrentNs > labelNs) { 
								minCurrentNs = labelNs;
							}
						}
					}
					else if (classificationAbility > 0) { // an proekupse apo 9etiki apofasi (yper)
						data.tp += 1;
						
						if (minCurrentNs > labelNs) { // bainei edo mono otan exei prokupsei apo 9etiki apofasi
							minCurrentNs = labelNs;
						}
					}
					
					if (classificationAbility != 0) 
						data.msa += 1;
				} // kleinei to for gia ka9e label
	
				cl.myClassifier.experience++;
				
	
				
				/* einai emmesos tropos na elegkso oti o kanonas anikei sto (se ena toulaxiston ennoo) labelCorrectSet
				 * giati to minCurrentNs allazei mono an classificationAbility > 0 dld o kanonas apofasizei, den adiaforei
				 */
				if (minCurrentNs != Integer.MAX_VALUE) {
					//data.ns += .1 * (minCurrentNs - data.ns);
					data.ns += LEARNING_RATE * (minCurrentNs - data.ns);
				}
				
				switch (FITNESS_MODE) {
				
				case FITNESS_MODE_SIMPLE:
					data.fitness = Math.pow((data.tp) / (data.msa), n);
					break;
				case FITNESS_MODE_COMPLEX:
					data.fitness += LEARNING_RATE * (Math.pow((data.tp) / (data.msa), n) - data.fitness);
					
/*					  data.fitness += LEARNING_RATE * (cl.numerosity * Math.pow((data.tp) / (data.msa), n) - data.fitness);
					  data.fitness /= cl.numerosity;*/
					 
					break;
				}
				updateSubsumption(cl.myClassifier);

			} // kleinei to for gia ka9e macroclassifier
		}
		
		
		
		else if (FITNESS_MODE == FITNESS_MODE_SHARING) {
			
			for (int l = 0; l < numberOfLabels; l++) {
				shareFitness(matchSet, labelCorrectSets[l], l, instanceIndex);
			} 
			
			for (int i = 0; i < matchSetSize; i++) { 
				final Macroclassifier cl = matchSet.getMacroclassifier(i);	
				cl.myClassifier.experience++; 
				final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
				final MlASLCSClassifierData dataArray[] = (MlASLCSClassifierData[]) cl.myClassifier.getUpdateDataArray();
				
				double fitnessSum = 0;
				double ns = 0;
				
				for (int l = 0; l < numberOfLabels; l++) {
					fitnessSum += dataArray[l].fitness;	
					ns += dataArray[l].minCurrentNs;
				}
				ns /= numberOfLabels;
				data.fitness = (fitnessSum / cl.numerosity) / numberOfLabels;

				if (ns != Integer.MAX_VALUE) {
					//data.ns += .1 * (minCurrentNs - data.ns);
					data.ns += LEARNING_RATE * (ns - data.ns);
				}
					
				if (Math.pow(data.tp / data.msa, n) > ACC_0) {
					if (cl.myClassifier.experience >= this.subsumptionExperienceThreshold)
						cl.myClassifier.setSubsumptionAbility(true);
				}
				else {
					cl.myClassifier.setSubsumptionAbility(false);
				}
				
			} 
		}
		
		updateParametersTime += System.currentTimeMillis(); 
		
		numberOfEvolutionsConducted = 0;
		numberOfSubsumptionsConducted = 0;
		numberOfDeletionsConducted = 0;
		numberOfNewClassifiers = 0;
		evolutionTime = 0;
		subsumptionTime = 0;
		deletionTime = 0;
		
		if (evolve) {
			
			evolutionTime = -System.currentTimeMillis();
						
			Vector<Integer> labelsToEvolve = new Vector<Integer>();
			Vector<Integer> labelsToCover = new Vector<Integer>();
			
			for (int l = 0; l < numberOfLabels; l++) {
				if (labelCorrectSets[l].getNumberOfMacroclassifiers() > 0) {
					
					ga.increaseTimestamp();
					int meanAge = ga.getMeanAge(labelCorrectSets[l]);
					if ( !( ga.getTimestamp() - meanAge < ga.getActivationAge()) )
					{
						labelsToEvolve.add(l);
						for ( int i = 0; i < labelCorrectSets[l].getNumberOfMacroclassifiers(); i++ )
						{
							labelCorrectSets[l].getClassifier(i).timestamp = ga.getTimestamp();
						}
					}					
				} else {
					labelsToCover.add(l);
				}
			}
			
			numberOfEvolutionsConducted = labelsToEvolve.size();
			
			Vector<Integer> indicesToSubsume = new Vector<Integer>();
			
			ClassifierSet newClassifiersSet = new ClassifierSet(null);
			
			for ( int i = 0; i < labelsToEvolve.size(); i++ )
			{
				ga.evolveSetNew(labelCorrectSets[labelsToEvolve.elementAt(i)], population, labelsToEvolve.get(i));
				indicesToSubsume.addAll(ga.getIndicesToSubsume());
				newClassifiersSet.merge(ga.getNewClassifiersSet());
				
				subsumptionTime += ga.getSubsumptionTime();				
			}
			
			for ( int i = 0; i < labelsToCover.size(); i++ )
			{
				newClassifiersSet.addClassifier(this.coverNew(instanceIndex),false);
			}
			
			population.totalGAInvocations = ga.getTimestamp();

			
			numberOfSubsumptionsConducted = indicesToSubsume.size();
			numberOfNewClassifiers        = newClassifiersSet.getNumberOfMacroclassifiers();
			
			for ( int i = 0; i < indicesToSubsume.size() ; i++ )
			{
				population.getMacroclassifiersVector().get(indicesToSubsume.elementAt(i)).numerosity++; 
				population.getMacroclassifiersVector().get(indicesToSubsume.elementAt(i)).numberOfSubsumptions++; 
				population.totalNumerosity++;
			}
			
			population.mergeWithoutControl(newClassifiersSet);
			
			deletionTime = -System.currentTimeMillis();
			final IPopulationControlStrategy theControlStrategy = population.getPopulationControlStrategy();
			theControlStrategy.controlPopulation(population);
			deletionTime += System.currentTimeMillis();
			
			numberOfDeletionsConducted = theControlStrategy.getNumberOfDeletionsConducted();
			
			evolutionTime += System.currentTimeMillis();
			
		}
		
	}


	@Override
	public void updateSetNewSmp(ClassifierSet population, 
							   ClassifierSet matchSet,
							   int instanceIndex, 
							   boolean evolve) {
		
		/*
		 * patenta gia moirasmo sta threads xwris IntegerForLoop
		 * 
		 * */
		
		
//		try{
//			ptGenerateCorrectSet.execute( new ParallelRegion() {
//				
////				//public ClassifierSet[] labelCorrectSets_thread = new ClassifierSet[numberOfLabelsSmp];
//				public void run() throws Exception
//				{
//					final int first = firstToGenerate.elementAt(getThreadIndex());
//					final int last = lastToGenerate.elementAt(getThreadIndex());
//					ClassifierSet[] labelCorrectSets_thread = new ClassifierSet[last-first+1];
//					
//					long p0,p1,p2,p3,p4,p5,p6,p7;
//					long p8,p9,pa,pb,pc,pd,pe,pf;
//					
//					for ( int i = first ; i <= last; ++i )
//					{
////								labelCorrectSets_thread[i-first] = generateLabelCorrectSet(matchSetSmp,instanceIndexSmp,i);
//						final ClassifierSet correctSet = new ClassifierSet(null);
//						final int matchSetSize = matchSetSmp.getNumberOfMacroclassifiers();
//						for (int j = 0; j < matchSetSize; j++) 
//						{
//							final Macroclassifier cl = matchSetSmp.getMacroclassifier(j);
//							if (cl.myClassifier.classifyLabelCorrectly(instanceIndexSmp, i) > 0)
//								correctSet.addClassifier(cl, false);
//						}
//						labelCorrectSets_thread[i-first] = correctSet;					
//					}
//					
//					final ClassifierSet[] labelCorrectSets_thread_final = labelCorrectSets_thread;
//							
//					region().critical(new ParallelSection() {
//						public void run()
//						{
//							for ( int i = first ; i <= last ; i++ )
//							{
//								labelCorrectSetsSmp[i] = labelCorrectSets_thread_final[i-first];
//							}
//						}
//					});
//				}			
//			});
//		}
//		catch( Exception e )
//		{
//			e.printStackTrace();
//		}
		
		labelCorrectSetsSmp = new ClassifierSet[numberOfLabels];
		
		matchSetSmp = matchSet;
		instanceIndexSmp = instanceIndex;
		numberOfLabelsSmp = numberOfLabels;
		
		generateCorrectSetTime = - System.currentTimeMillis();
		
		try{
			ptGenerateCorrectSet.execute( new ParallelRegion() {				
				public void run() throws Exception
				{
					execute(0, numberOfLabelsSmp - 1, new IntegerForLoop () {
						
						ClassifierSet[] labelCorrectSets_thread;
						
						Vector<Integer> labelCorrectSetsIndices;
						
						long p0,p1,p2,p3,p4,p5,p6,p7;
						long p8,p9,pa,pb,pc,pd,pe,pf;
						
						public void run(int first, int last)
						{
							int threadId = getThreadIndex();
							labelCorrectSets_thread = new ClassifierSet[last-first+1];
							labelCorrectSetsIndices = new Vector<Integer>(); 
							for ( int i = first; i <= last ; i++ )
							{
								labelCorrectSets_thread[i-first] = generateLabelCorrectSetSmp(matchSetSmp,instanceIndexSmp,i, threadId);
								labelCorrectSetsIndices.add(i); 
							}							
						}
						
						public void finish() throws Exception
						{
							region().critical(new ParallelSection() {
								public void run()
								{
									for (int i = 0 ; i < labelCorrectSetsIndices.size() ; i++)
									{
										labelCorrectSetsSmp[labelCorrectSetsIndices.elementAt(i)] =
											labelCorrectSets_thread[i];
									}
								}
							});
						}
						
					});
				}			
			});
		}
		catch( Exception e )
		{
			e.printStackTrace();
		}
		
		generateCorrectSetTime += System.currentTimeMillis();
		
		//controlPopulationInMatchSet(population, matchSet);

		
		int CorrectSetsPopulation = 0;
		for (int i = 0; i < numberOfLabels; i++) {
			CorrectSetsPopulation += labelCorrectSetsSmp[i].getNumberOfMacroclassifiers() ;
		}
		myLcs.meanCorrectSetNumerosity = CorrectSetsPopulation / numberOfLabels;
		

		updateParametersTime = -System.currentTimeMillis();
		
		if (FITNESS_MODE == FITNESS_MODE_SIMPLE || FITNESS_MODE == FITNESS_MODE_COMPLEX) {
			
			try {
			ptUpdateParameters.execute( new ParallelRegion() {
				public void run() throws Exception
				{
					execute(0,matchSetSmp.getNumberOfMacroclassifiers()-1,new IntegerForLoop() {
						
						public void run(int first,int last)
						{
							for ( int i = first; i <= last ; ++i )
							{
								final Macroclassifier cl = matchSetSmp.getMacroclassifier(i);
								int minCurrentNs = Integer.MAX_VALUE;
								final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
								
								for (int l = 0; l < numberOfLabelsSmp; l++) {
									// Get classification ability for label l. an anikei sto labelCorrectSet me alla logia.
									final float classificationAbility = cl.myClassifier.classifyLabelCorrectly(instanceIndexSmp, l);
									final int labelNs = labelCorrectSetsSmp[l].getTotalNumerosity();

									if (classificationAbility == 0) {// an proekupse apo adiaforia
										data.tp += OMEGA;
										data.msa += PHI;
										
										if (wildCardsParticipateInCorrectSets) {
											if (minCurrentNs > labelNs) { 
												minCurrentNs = labelNs;
											}
										}
									}
									else if (classificationAbility > 0) { // an proekupse apo 9etiki apofasi (yper)
										data.tp += 1;
										
										if (minCurrentNs > labelNs) { // bainei edo mono otan exei prokupsei apo 9etiki apofasi
											minCurrentNs = labelNs;
										}
									}
									if (classificationAbility != 0) 
										data.msa += 1;
								} // kleinei to for gia ka9e label
					
								cl.myClassifier.experience++;
								
								/* einai emmesos tropos na elegkso oti o kanonas anikei sto (se ena toulaxiston ennoo) labelCorrectSet
								 * giati to minCurrentNs allazei mono an classificationAbility > 0 dld o kanonas apofasizei, den adiaforei
								 */
								if (minCurrentNs != Integer.MAX_VALUE) {
									//data.ns += .1 * (minCurrentNs - data.ns);
									data.ns += LEARNING_RATE * (minCurrentNs - data.ns);
								}
								
								switch (FITNESS_MODE) {
								
								case FITNESS_MODE_SIMPLE:
									data.fitness = Math.pow((data.tp) / (data.msa), n);
									break;
								case FITNESS_MODE_COMPLEX:
									data.fitness += LEARNING_RATE * (Math.pow((data.tp) / (data.msa), n) - data.fitness);
									
				                     /*data.fitness += LEARNING_RATE * (cl.numerosity * Math.pow((data.tp) / (data.msa), n) - data.fitness);
									  data.fitness /= cl.numerosity;*/
									 
									break;
								}
								updateSubsumption(cl.myClassifier);
								
							}
							
						}
						
					});
				}
			});
			}
			catch ( Exception e)
			{
				e.printStackTrace();
			}
		}	
		else if (FITNESS_MODE == FITNESS_MODE_SHARING) {
			
			for (int l = 0; l < numberOfLabels; l++) {
				shareFitness(matchSet, labelCorrectSetsSmp[l], l, instanceIndex);
			}
			
			try {
				ptUpdateParameters.execute( new ParallelRegion() {
					public void run() throws Exception
					{
						execute(0,matchSetSmp.getNumberOfMacroclassifiers()-1,new IntegerForLoop() {
							
							public void run(int first,int last)
							{
								for ( int i = first; i <= last ; ++i )
								{
									final Macroclassifier cl = matchSetSmp.getMacroclassifier(i);	
									cl.myClassifier.experience++; 
									final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
									final MlASLCSClassifierData dataArray[] = (MlASLCSClassifierData[]) cl.myClassifier.getUpdateDataArray();
									
									double fitnessSum = 0;
									double ns = 0;
									
									for (int l = 0; l < numberOfLabelsSmp; l++) {
										fitnessSum += dataArray[l].fitness;	
										ns += dataArray[l].minCurrentNs;
									}
									ns /= numberOfLabelsSmp;
									data.fitness = (fitnessSum / cl.numerosity) / numberOfLabels;

									if (ns != Integer.MAX_VALUE) {
										//data.ns += .1 * (minCurrentNs - data.ns);
										data.ns += LEARNING_RATE * (ns - data.ns);
									}
										
									if (Math.pow(data.tp / data.msa, n) > ACC_0) {
										if (cl.myClassifier.experience >= subsumptionExperienceThreshold)
											cl.myClassifier.setSubsumptionAbility(true);
									}
									else {
										cl.myClassifier.setSubsumptionAbility(false);
									}
								}
								
							}
							
						});
					}
				});
				}
				catch ( Exception e)
				{
					e.printStackTrace();
				}
		}
		
		updateParametersTime += System.currentTimeMillis();

		
		numberOfEvolutionsConducted = 0;
		numberOfSubsumptionsConducted = 0;
		numberOfDeletionsConducted = 0;
		numberOfNewClassifiers = 0;
		evolutionTime = 0;
		subsumptionTime = 0;
		selectionTime = 0;
		deletionTime = 0;
			
		if (evolve) {
			
			evolutionTime = -System.currentTimeMillis();
			
			Vector<Integer> labelsToEvolve = new Vector<Integer>();
			
			Vector<Integer> labelsToCover = new Vector<Integer>();
			
			for (int l = 0; l < numberOfLabels; l++) {
				if (labelCorrectSetsSmp[l].getNumberOfMacroclassifiers() > 0) {
					
					ga.increaseTimestamp();
					int meanAge = ga.getMeanAge(labelCorrectSetsSmp[l]);
					if ( !( ga.getTimestamp() - meanAge < ga.getActivationAge()) )
					{
						labelsToEvolve.add(l);
						for ( int i = 0; i < labelCorrectSetsSmp[l].getNumberOfMacroclassifiers(); i++ )
						{
							labelCorrectSetsSmp[l].getClassifier(i).timestamp = ga.getTimestamp();
						}
					}					
				} else {
					labelsToCover.add(l);
				}
			}
			
			numberOfEvolutionsConducted = labelsToEvolve.size();
			population.totalGAInvocations = ga.getTimestamp();
			
			labelsToEvolveSmp = labelsToEvolve;
			seedSmp1 = (long)(100 * myLcs.instances.length * myLcs.iterations * Math.random());
						
			div = labelsToEvolveSmp.size() / numOfProcessors;
			mod = labelsToEvolveSmp.size() % numOfProcessors;
			
			indicesToSubsumeSmp = new Vector<Integer>();
			newClassifiersSetSmp = new ClassifierSet(null);
			subsumptionTimeSmp = 0;
			subsumptionTimeMax = 0;
			populationSmp = population;	
			
			
			if ( labelParallelMode == 1 )
			{
			
			/*
			 * YLOPOIHSH DIAMOIRASMOU gia GENIKI PERIPTWSI N LABELS
			 */
			
			if ( mod > 0 )
			{
				for ( int i = div * numOfProcessors; i < div * numOfProcessors + mod ; i++ )
				{
					IGeneticAlgorithmStrategy.EvolutionOutcome evolutionOutcome = 
						ga.evolveSetNewOneLabelSmp(
												labelCorrectSetsSmp
												[labelsToEvolveSmp.elementAt(i)], 
												population,
												labelsToEvolveSmp.elementAt(i));
					
					indicesToSubsumeSmp.addAll(evolutionOutcome.indicesToSubsume);
					newClassifiersSetSmp.merge(evolutionOutcome.newClassifierSet);
					subsumptionTimeSmp += evolutionOutcome.subsumptionTime;
				}
			}				
			if ( div > 0 )
			{
				try{
					ptEvolve.execute( new ParallelRegion() {
				
						public void run() throws Exception
						{
							execute(0, div * numOfProcessors - 1, new IntegerForLoop() {
							
								Vector<Integer> indicesToSubsume_thread;
								ClassifierSet newClassifiersSet_thread;
								long subsumptionTime_thread;
								Random prng_thread;
						
								public void start()
								{
									indicesToSubsume_thread = new Vector<Integer>();
									newClassifiersSet_thread = new ClassifierSet(null);
									subsumptionTime_thread = 0;
									prng_thread = Random.getInstance(seedSmp1);
									prng_thread.setSeed(seedSmp1);
								}
						
								public void run(int first, int last)
								{
									for ( int i = first; i <= last ; ++i )
									{
										prng_thread.skip(2 * first * (3 + labelCorrectSetsSmp[labelsToEvolveSmp.elementAt(i)].getClassifier(0).size()));
										IGeneticAlgorithmStrategy.EvolutionOutcome evolutionOutcome 
											= ga.evolveSetNewSmp(labelCorrectSetsSmp[labelsToEvolveSmp.elementAt(i)], 
																 populationSmp, 
																 prng_thread,
																 labelsToEvolveSmp.elementAt(i));
										indicesToSubsume_thread.addAll(evolutionOutcome.indicesToSubsume);
										newClassifiersSet_thread.merge(evolutionOutcome.newClassifierSet);
										subsumptionTime_thread += evolutionOutcome.subsumptionTime;
									}
								}
							
								public void finish() throws Exception {
									region().critical( new ParallelSection() {
										public void run()
										{
											indicesToSubsumeSmp.addAll(indicesToSubsume_thread);
											newClassifiersSetSmp.merge(newClassifiersSet_thread);
											if (subsumptionTime_thread > subsumptionTimeMax)
											{
												subsumptionTimeMax = subsumptionTime_thread;
											}
										}
									});
								}
						
							});
						}
					
					});
				}
				catch( Exception e)
				{
					e.printStackTrace();
				}
				subsumptionTimeSmp += subsumptionTimeMax;
			}
			
			}			
			else if (labelParallelMode == 0)
			{
				for ( int i = 0; i < labelsToEvolveSmp.size() ; i++ )
				{
					IGeneticAlgorithmStrategy.EvolutionOutcome evolutionOutcome = 
						ga.evolveSetNewOneLabelSmp(
													labelCorrectSetsSmp
													[labelsToEvolveSmp.elementAt(i)], 
													population,
													labelsToEvolveSmp.elementAt(i));
					indicesToSubsumeSmp.addAll(evolutionOutcome.indicesToSubsume);
					newClassifiersSetSmp.merge(evolutionOutcome.newClassifierSet);
					subsumptionTimeSmp += evolutionOutcome.subsumptionTime;
				}
			}
			
			for (int i = 0 ; i < labelsToCover.size(); i++)
			{
				newClassifiersSetSmp.addClassifier(this.coverNew(instanceIndex), false);
			}
			
			subsumptionTime = subsumptionTimeSmp;
			numberOfSubsumptionsConducted = indicesToSubsumeSmp.size();
			numberOfNewClassifiers        = newClassifiersSetSmp.getNumberOfMacroclassifiers();
			
			for ( int i = 0; i < indicesToSubsumeSmp.size() ; i++ )
			{
				population.getMacroclassifiersVector().get(indicesToSubsumeSmp.elementAt(i)).numerosity++; 
				population.getMacroclassifiersVector().get(indicesToSubsumeSmp.elementAt(i)).numberOfSubsumptions++; 
				population.totalNumerosity++;
			}
			
			population.mergeWithoutControl(newClassifiersSetSmp);
			
			deletionTime = -System.currentTimeMillis();
			final IPopulationControlStrategy theControlStrategy = population.getPopulationControlStrategy();
			theControlStrategy.controlPopulationSmp(population);
			deletionTime += System.currentTimeMillis();
			
			numberOfDeletionsConducted = theControlStrategy.getNumberOfDeletionsConducted();
			
			evolutionTime += System.currentTimeMillis();
		}
			
			/*
			 * OTI AKOLOUTHEI EINAI YLOPOIHSH GIA XWRISMO OTAN EXOUME 2 LABELS MONO
			 * otan kanoume evolve se 2 labels, xwrismo twn labels se thread
			 */
			
//			if (labelsToEvolveSmp.size() == 2)
//			{
//				try{
//				ptEvolve.execute( new ParallelRegion() {
//			
//					public void run() throws Exception
//					{
//						
//						Vector<Integer> indicesToSubsume_thread;
//						ClassifierSet newClassifiersSet_thread;
//						long subsumptionTime_thread = 0;
//						Random prng_thread;
//						
//						indicesToSubsume_thread = new Vector<Integer>();
//						newClassifiersSet_thread = new ClassifierSet(null);
//						prng_thread = Random.getInstance(seedSmp1);
//						prng_thread.setSeed(seedSmp1);
//						
//						SteadyStateGeneticAlgorithm.EvolutionOutcome evolutionOutcome;
//						
//						
//						if ( getThreadIndex() == 0 )
//						{
////							evolutionOutcome 
////							= gaSmp.evolveSetNewSmp(labelCorrectSetsSmp[labelsToEvolveSmp.elementAt(0)], 
////												 populationSmp, 
////												 prng_thread);
////							
////							EvolutionTimeMeasurements etm = new EvolutionTimeMeasurements();
////							etm.timeA = evolutionOutcome.timeA;
////							etm.timeB = evolutionOutcome.timeB;
////							etm.timeC = evolutionOutcome.timeC;
////							etm.timeD = evolutionOutcome.timeD;
////							
////							measurements0.add(etm);
//							
//							evolutionOutcome
//							= ga.evolveSetNewSmp(labelCorrectSetsSmp[labelsToEvolveSmp.elementAt(0)], 
//							 populationSmp, 
//							 prng_thread);
//						}
//						else
//						{
//							prng_thread.skip(2*(3+labelCorrectSetsSmp[labelsToEvolveSmp.elementAt(1)].getClassifier(0).size()));
//							evolutionOutcome 
//							= ga.evolveSetNewSmp(labelCorrectSetsSmp[labelsToEvolveSmp.elementAt(1)], 
//												 populationSmp, 
//												 prng_thread);
//							
////							EvolutionTimeMeasurements etm = new EvolutionTimeMeasurements();
////							etm.timeA = evolutionOutcome.timeA;
////							etm.timeB = evolutionOutcome.timeB;
////							etm.timeC = evolutionOutcome.timeC;
////							etm.timeD = evolutionOutcome.timeD;
////							
////							measurements1.add(etm);
//			
//						}
////						
////						indicesToSubsume_thread.addAll(evolutionOutcome.indicesToSubsume);
////						newClassifiersSet_thread.merge(evolutionOutcome.newClassifierSet);
////						subsumptionTime_thread += evolutionOutcome.subsumptionTime;
////						
////						final Vector<Integer> indicesToSubsume_final = indicesToSubsume_thread;
////						final ClassifierSet newClassifiersSet_final = newClassifiersSet_thread;
////						final long subsumptionTime_final = subsumptionTime_thread;
////						
////						
////						region().critical( new ParallelSection() {
////							public void run()
////							{
////								indicesToSubsumeSmp.addAll(indicesToSubsume_final);
////								newClassifiersSetSmp.merge(newClassifiersSet_final);
////								subsumptionTimeSmp += subsumptionTime_final;
////							}
////						});
//						
//						
////						execute(0,labelsToEvolveSmp.size()-1, new IntegerForLoop() {
////						
////							Vector<Integer> indicesToSubsume_thread;
////							ClassifierSet newClassifiersSet_thread;
////							long subsumptionTime_thread = 0;
////							Random prng_thread;
////					
////							public void start()
////							{
////								indicesToSubsume_thread = new Vector<Integer>();
////								newClassifiersSet_thread = new ClassifierSet(null);
////								prng_thread = Random.getInstance(seedSmp1);
////								prng_thread.setSeed(seedSmp1);
////							}
////					
////							public void run(int first, int last)
////							{
////								int threadId = getThreadIndex();
////								for ( int i = first; i <= last ; ++i )
////								{
////									SteadyStateGeneticAlgorithm.EvolutionOutcome evolutionOutcome;
////									prng_thread.skip(2*first*(3+labelCorrectSetsSmp[labelsToEvolveSmp.elementAt(i)].getClassifier(0).size()));
////									if ( threadId == 0 )
////									{
////										evolutionOutcome 
////										= ga.evolveSetNewSmp(labelCorrectSetsSmp[labelsToEvolveSmp.elementAt(i)], 
////															 populationSmp, 
////															 prng_thread);
////									}
////									else
////									{
////										evolutionOutcome 
////										= ga2.evolveSetNewSmp(labelCorrectSetsSmp[labelsToEvolveSmp.elementAt(i)], 
////															 populationSmp, 
////															 prng_thread);
////									}								
//////									SteadyStateGeneticAlgorithm.EvolutionOutcome evolutionOutcome 
//////										= gaSmp.evolveSetNewSmp(labelCorrectSetsSmp[labelsToEvolveSmp.elementAt(i)], 
//////															 populationSmp, 
//////															 prng_thread);
////									indicesToSubsume_thread.addAll(evolutionOutcome.indicesToSubsume);
////									newClassifiersSet_thread.merge(evolutionOutcome.newClassifierSet);
////									subsumptionTime_thread += evolutionOutcome.subsumptionTime;
////								}
////							}
////						
////							public void finish() throws Exception {
////								region().critical( new ParallelSection() {
////									public void run()
////									{
////										indicesToSubsumeSmp.addAll(indicesToSubsume_thread);
////										newClassifiersSetSmp.merge(newClassifiersSet_thread);
////										subsumptionTimeSmp += subsumptionTime_thread;
////									}
////								});
////							}
////					
////						});
//			
//					}
//				
//				});
//			}
//			catch( Exception e)
//			{
//				e.printStackTrace();
//			}
//			}
//			else {	
			
				/*
				 * an den kanoume evolve se 2 labels evolve me OneLabel gia ola 
				 */
				
//				for ( int i = 0; i < labelsToEvolveSmp.size() ; i++ )
//				{
//					SteadyStateGeneticAlgorithm.EvolutionOutcome evolutionOutcome = 
//						ga.evolveSetNewOneLabelSmp(
//													labelCorrectSetsSmp
//													[labelsToEvolveSmp.elementAt(i)], 
//													population);
//					indicesToSubsumeSmp.addAll(evolutionOutcome.indicesToSubsume);
//					newClassifiersSetSmp.merge(evolutionOutcome.newClassifierSet);
//					subsumptionTimeSmp += evolutionOutcome.subsumptionTime;
//				}
//				
//			}	
	

	}


	/**
	 * Implementation of the subsumption strength.
	 * 
	 * @param aClassifier
	 *            the classifier, whose subsumption ability is to be updated
	 */
	protected void updateSubsumption(final Classifier aClassifier) {
		aClassifier.setSubsumptionAbility(
				(aClassifier.getComparisonValue(COMPARISON_MODE_EXPLOITATION) > subsumptionFitnessThreshold)
						&& (aClassifier.experience > subsumptionExperienceThreshold));
	}


}