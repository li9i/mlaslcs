/**
 * 
 */
package gr.auth.ee.lcs.distributed;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.data.ILCSMetric;
import gr.auth.ee.lcs.geneticalgorithm.IRuleSelector;

/**
 * An abstract class of an LCS callback that can be used for distributing rules
 * in a distributed LCS environment. Essentially this is a strategy for adding
 * received rules.
 * 
 * @author Miltiadis Allamanis
 * 
 */
public abstract class AbstractRuleDistributer implements ILCSMetric {

	/**
	 * The LCS used by this distributer.
	 * @uml.property  name="mLCS"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	final AbstractLearningClassifierSystem mLCS;

	/**
	 * A rule selector, to select the rules to be added into the LCS population.
	 * @uml.property  name="receiveSelector"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	final IRuleSelector receiveSelector;

	/**
	 * A rule selector, to select the rules to be send from the LCS.
	 * @uml.property  name="sendSelector"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	final IRuleSelector sendSelector;

	/**
	 * The local router interface.
	 * @uml.property  name="localRouter"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	IRuleRouter localRouter;

	/**
	 * @uml.property  name="newRules"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	ClassifierSet newRules = new ClassifierSet(null);

	public AbstractRuleDistributer(IRuleRouter router,
			AbstractLearningClassifierSystem lcs, IRuleSelector receiver,
			IRuleSelector sender) {
		localRouter = router;
		mLCS = lcs;
		sendSelector = sender;
		receiveSelector = receiver;
	}
	
	public void setRouter(IRuleRouter router) {
		localRouter = router;
	}

	/**
	 * Receive rules from other LCSs
	 * 
	 * @param ruleSet
	 *            the a set of rules being received
	 * @param metaData
	 *            any metadata on these rules (e.g. sender)
	 */
	public void receiveRules(ClassifierSet ruleSet) {
		synchronized (newRules) {
			receiveSelector.select(-1, ruleSet, newRules);
		}
	}

	@Override
	public double getMetric(AbstractLearningClassifierSystem lcs) {
		sendRules();
		return 0;
	}

	/**
	 * Send rules to the local router.
	 */
	public void sendRules() {
		ClassifierSet outRules = new ClassifierSet(null);
		sendSelector.select(5, mLCS.getRulePopulation(), outRules);
		
		localRouter.sendRules(outRules);
		synchronized (newRules) {
			for (int i = 0; i < newRules.getNumberOfMacroclassifiers(); i++) {
				final Classifier initial = newRules.getClassifier(i);
				final Classifier rule = (Classifier) initial.clone();
				mLCS.getRulePopulation().addClassifier(new Macroclassifier(rule, 1), true);
				
			}
			newRules = new ClassifierSet(null);
		}
	}

}
