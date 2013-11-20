package gr.auth.ee.lcs.implementations.global;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.IOException;
import java.util.Vector;

public class RepetitiveExec {
	
	
	/**
	 * A vector holding the {number of executions of newAllMlTypes.main()} evaluation arrays, coming from the ArffTrainTestLoader.
	 */
	public static Vector<double[]> evals = new Vector<double[]>();
	
	/**
	 * The mean evaluations across the  {number of executions of newAllMlTypes.main()} runs.
	 */
	private static double[] meanEvals;


	private static int numberOfRepetitions = 1;
	
	/**
	 * A class for conducting a series of runs, repetitively.
	 * @throws Exception 
	 * 
	 */
	public static void main(String[] args) throws   Exception {
		
		
		final String[] names = { 
				 "Accuracy(pcut)", 
				 "Recall(pcut)",
				 "HammingLoss(pcut)", 
				 "ExactMatch(pcut)", 
				 "Accuracy(ival)",
				 "Recall(ival)", 
				 "HammingLoss(ival)", 
				 "ExactMatch(ival)",
				 "Accuracy(best)", 
				 "Recall(best)", 
				 "HammingLoss(best)",
				 "ExactMatch(best)" };
		

		meanEvals = new double[names.length];


				
		for (int fitness = 1; fitness < 3; fitness++) {
			Object fitnessObject = SettingsLoader.setSetting("FITNESS_MODE", String.valueOf(fitness % 3));
			
			for (int deletion = 0; deletion < 2; deletion++) {
				Object deletionObject = SettingsLoader.setSetting("DELETION_MODE", String.valueOf(deletion % 2));
					
				for (int dontCare = 0; dontCare < numberOfRepetitions; dontCare++) {
					Object dontCareObject = SettingsLoader.setSetting("wildCardsParticipateInCorrectSets", "false"); 

					System.out.println("##############################");
					System.out.println("Repetitive execution states: ");
					System.out.println("fitness: " + (int) SettingsLoader.getNumericSetting("FITNESS_MODE", 0));
					System.out.println("deletion: " + (int) SettingsLoader.getNumericSetting("DELETION_MODE", 0));
					System.out.println("# in correctSets: " + SettingsLoader.getStringSetting("wildCardsParticipateInCorrectSets", "false") + "\n");
					AllMlTypes newAllMlTypes = (AllMlTypes) Class.forName("gr.auth.ee.lcs.implementations.global.AllMlTypes").newInstance();
					newAllMlTypes.main(args);
				}
				
				if (numberOfRepetitions > 1) {
					for (int cols = 0; cols < names.length; cols++) {
							
						double var = 0;
						for (int repetitions = 0; repetitions < numberOfRepetitions; repetitions++) {
							var += evals.elementAt(repetitions)[cols];
						}
						meanEvals[cols] = var / numberOfRepetitions;
					}
					
					System.out.println("Mean metric values for this run:\n");
					for (int i = 0; i < meanEvals.length; i++) {
						System.out.println(names[i] + ": " + meanEvals[i]);
						if ((i + 1) % 4 == 0) System.out.println();
					}
					
					evals.removeAllElements();
				}
			}
		}
	}
}

