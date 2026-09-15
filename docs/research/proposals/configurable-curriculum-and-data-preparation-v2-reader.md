# Which problem should the student practise next?

Adaptive curriculum · Version 2 reading report · Revised 14 September 2026

A training run can improve its average score while leaving whole skills untouched. It can also spend more and more time on problems the student already solves. Both failures point to the same question: as the student changes, how should its practice change?

We propose using the student's recent answers to choose its next problems. The sampler would favour problems that still produce a mixture of successful and unsuccessful attempts, keep trying unseen problems, and periodically revisit work it has set aside. Classes such as arithmetic and algebra would help it make an initial prediction about an unfamiliar problem; that problem's own results would then refine the prediction.

The hypothesis is that this would buy more learning from the same training budget. V2 specifies how to make those selections and track them. **It has not yet established that the policy improves real model training.** The examples below explain the proposal, and the simulation illustrates its behaviour under assumed learning rules.

## A balanced dataset can still give the wrong practice

Consider a collection of 10,000 math problems: 7,000 arithmetic problems and 1,000 each in algebra, geometry, and probability. Drawing uniformly from the collection spends 70% of the task budget on arithmetic.

Suppose arithmetic success rises from 60% to 95%, while the other subjects stay at 40%, 10%, and 30%. Overall success rises from 50% to 74.5%. The headline looks encouraging, but the student has made no progress in three of the four subjects.

An obvious first improvement is to balance the subjects. Giving each 25% of practice makes their coverage explicit. Measuring them separately also makes the remaining weaknesses visible. Equal shares are a useful baseline for this example; in a real project, the base shares should reflect the skills we want the student to acquire.

But fixed balancing only solves the initial allocation problem. Later, arithmetic might be reliably solved, algebra might still be improving, and geometry might produce repeated failures. Equal shares would keep spending the same amount on each. Within algebra, the sampler could also revisit a few familiar questions while leaving hundreds untouched.

We therefore need to choose at two levels: **which class deserves practice, and which problem within that class should come next?** To answer either, we first need to say what makes an attempt useful for the learning rule.

## For GRPO, different answers create a learning signal

The initial experiment uses ordinary GRPO. For each selected problem, the student generates several fresh answers. These attempts form a *group*. A checker scores each answer, and GRPO uses differences within that group to favour better attempts over worse ones.

With four attempts and binary success checks, three patterns matter:

| Problem | Four fresh attempts | What the group tells us |
| --- | --- | --- |
| Calculate 15% of 80 | Success, success, success, success | The student succeeded every time; these outcomes give no relative preference. |
| Solve 3x + 5 = 20 | Failure, success, failure, success | The successful attempts can receive positive relative credit and the failures negative credit. |
| Find a rectangle's area from its perimeter and length | Failure, failure, failure, failure | The student is struggling; these outcomes also give no relative preference. |

For the algebra group, write failure as 0 and success as 1. First calculate the mean reward, then subtract it from each attempt:

$$
\begin{gathered}
\text{Rewards}=[0,1,0,1]\\[6pt]
\text{Mean reward}=\frac{0+1+0+1}{4}=0.5\\[6pt]
\text{Differences}=[-0.5,+0.5,-0.5,+0.5]
\end{gathered}
$$

The signs already express which attempts should be favoured. GRPO then divides by the group's standard deviation to put those differences on a common scale.

Let $i$ name the problem, $j$ an attempt, and $G$ the number of attempts. The calculation we just performed becomes

$$
\begin{gathered}
\bar r_i=\frac{1}{G}\sum_{j=1}^{G}r_{ij}\\[8pt]
s_i^2=\frac{1}{G-1}\sum_{j=1}^{G}(r_{ij}-\bar r_i)^2\\[8pt]
A_{ij}=\frac{r_{ij}-\bar r_i}{s_i+\epsilon}
\end{gathered}
$$

Here $r_{ij}$ is the checked reward, $s_i$ is the sample standard deviation, and $\epsilon$ is a small stabilizer. The relative credit $A_{ij}$ is called the *advantage*. Applying the last two steps to our example gives

$$
\begin{gathered}
s_i=\sqrt{\frac{4(0.5)^2}{4-1}}=\sqrt{\frac13}\\[8pt]
[A_{i1},A_{i2},A_{i3},A_{i4}]\approx[-0.866,+0.866,-0.866,+0.866].
\end{gathered}
$$

These advantages enter GRPO's clipped policy objective; the complete objective is retained in the technical reference.

Now apply the same subtraction to four failures: every reward equals its group mean, so every numerator is zero. Four successes have the same property. This explains why simply choosing the hardest problems is insufficient. Four failures reveal a weakness, but they do not tell outcome-GRPO which attempt was better. Repeated success and repeated failure have different meanings for the student, even though both produce zero relative outcome advantage in that group. The reference-model penalty may still contribute to the update.

This gives us a possible clue for choosing practice: **reward contrast**, where fresh answers to the same problem receive different rewards. A mixed group shows that the student can produce a successful answer but does not yet do so consistently. It gives us a reason to consider that problem again.

Contrast is only a proxy for learning opportunity. A noisy checker could keep disagreeing forever without teaching the student anything. Four successes also do not prove mastery, and four failures do not prove that a problem is impossible. We keep success, contrast, and actual improvement separate throughout the proposal.

## Give useful problems more practice, while continuing to look

Return to the algebra problem above. Two attempts succeeded and two failed. After learning from that group, should the student move on to a new problem, or try this one again?

Always moving on would leave a possible learning opportunity unfinished. One update need not make the successful behaviour reliable. Another group at the same problem could give the updated student a further opportunity to produce successful answers and learn from the differences. The previous outcomes give us some evidence that this problem can supply a signal; an unseen problem might turn out to be consistently successful or consistently unsuccessful. That is the case for giving promising problems more than one round of practice.

Here, returning to a problem means generating a **fresh group of answers after a model update**. The earlier group informs the selection; the new group supplies the next training experience. If fresh groups continue to show contrast, further practice may be worthwhile. As answers become consistently successful, the reason to favour that problem weakens.

But following only the contrast we have already found creates another problem. A few familiar algebra questions could keep attracting practice while hundreds of untried questions remain invisible. Their lack of observations would tell us nothing about their value. The sampler could become increasingly confident about a narrow part of the collection without discovering what the student can learn elsewhere.

Exploration therefore needs a share of the budget even while familiar problems look promising. Trying unseen problems can uncover other sources of useful practice and test whether our current picture of a class is too narrow. Waiting until familiar problems stop producing contrast could delay that discovery indefinitely, especially when some of their variation comes from noise.

We need both: enough repetition to pursue an observed learning opportunity, and a continuing search for opportunities we have not yet observed. V2 turns this tradeoff into a policy that favours predicted contrast while reserving a minimum share of selections for unseen problems. The appropriate share is an experimental question.

To favour contrast, we need to predict whether the *next* group will contain different outcomes. Suppose a problem succeeds half the time. With four independent attempts, we can calculate the two ways to get a constant group:

$$
\begin{gathered}
\Pr(\text{all four succeed})=\left(\frac12\right)^4=\frac1{16}\\[8pt]
\Pr(\text{all four fail})=\left(\frac12\right)^4=\frac1{16}
\end{gathered}
$$

Every other outcome contains both success and failure. Subtract the two constant cases from the total probability:

$$
\begin{gathered}
\Pr(\text{mixed group})=1-\frac1{16}-\frac1{16}\\[8pt]
=\frac{14}{16}=\frac78=87.5\%.
\end{gathered}
$$

The same reasoning works for any success probability $p$ and group size $G$. At fixed student weights and generation conditions, assuming independent binary outcomes for this problem,

$$
\underbrace{m_G(p)}_{\text{chance of a mixed group}}
=1-\underbrace{p^G}_{\text{all succeed}}
-\underbrace{(1-p)^G}_{\text{all fail}}.
$$

For four attempts, a problem with $p=0.99$ produces a mixed group only about 3.94% of the time. So does a problem with $p=0.01$. This symmetry recovers the distinction we saw in the table: both offer little expected contrast, but one is usually solved and the other is usually failed. The formula predicts an opportunity for relative credit, not how much the student will learn from it. Correlated attempts or unreliable checks would also undermine this calculation.

<section class="calculation-example" aria-labelledby="work-through-the-effect-of-success-probability">

#### Work through the effect of success probability

Keep four attempts per group. Each row follows the same calculation: raise success and failure probabilities to the fourth power, then subtract both from one.

<table>
<caption>Mixed-group probability with four independent attempts. Values rounded to four decimals.</caption>
<thead><tr><th scope="col">Success probability</th><th scope="col">All succeed</th><th scope="col">All fail</th><th scope="col">Mixed group</th></tr></thead>
<tbody>
<tr><th scope="row">10%</th><td>0.0001</td><td>0.6561</td><td>0.3438</td></tr>
<tr><th scope="row">25%</th><td>0.0039</td><td>0.3164</td><td>0.6797</td></tr>
<tr><th scope="row">50%</th><td>0.0625</td><td>0.0625</td><td>0.8750</td></tr>
<tr><th scope="row">75%</th><td>0.3164</td><td>0.0039</td><td>0.6797</td></tr>
<tr><th scope="row">90%</th><td>0.6561</td><td>0.0001</td><td>0.3438</td></tr>
</tbody>
</table>

Compare 25% and 75% success: they predict the same amount of contrast, even though their reliability differs. At 50%, about 87.5 out of every 100 groups would be mixed *in expectation*. That is neither a guaranteed count nor a prediction of 87.5 useful model updates.

**Work it out:** if success rises from 50% to 80%, what happens to the expected mixed-group rate? Calculate the all-success and all-failure cases first.

<details>
<summary>Show the calculation for 80% success</summary>

$$
\begin{gathered}
\Pr(\text{all succeed})=0.8^4=0.4096\\[8pt]
\Pr(\text{all fail})=0.2^4=0.0016\\[8pt]
m_4(0.8)=1-0.4096-0.0016=0.5888
\end{gathered}
$$

Expected contrast falls from 87.5% to 58.88%, a drop of 28.62 percentage points, while success rises. A decline in contrast can accompany better performance; it is not automatically a training failure.

</details>
</section>

We do not know a real problem's $p$ in advance. For now, this gives us the quantity we want to estimate. The next section explains how class evidence and the problem's own answers supply that estimate.

For a concrete starting point, imagine the student has already tried part of the collection and a new request asks for ten problems, with four fresh attempts at each. A 20% discovery reserve would fill that request as follows.

**Reserve two places for unseen problems.** These are task identities never previously selected during this run. First choose a class using its current prediction and the coverage rule described below; then draw an unseen problem using the base weights within that class. Before seeing its answers, we cannot confidently rank one otherwise indistinguishable unseen problem above another.

**Use the other eight places for familiar problems.** Most of these choices favour problems expected to produce reward contrast. Some choices revisit problems that have gone longest without an attempt, so old conclusions can be checked. If there are too few familiar problems, select additional unseen ones. At startup, all ten selections can therefore be new.

The proposed discovery reserve is 20% of candidate groups. A separate 20% class-coverage component keeps eligible classes in consideration even when their forecasts are poor. These are candidate settings to test, rather than established optimal values.

The two percentages govern different decisions. Discovery reserves places for new task identities. Class coverage affects which class a draw comes from; for familiar tasks, that component also triggers reassessment. An unseen geometry problem can satisfy discovery and come from a coverage draw at the same time. The percentages therefore do not describe two disjoint portions of compute.

Within one optimizer step, the sampler avoids selecting the same problem twice while distinct eligible tasks remain. The four attempts *within* a group still share one problem, as GRPO requires. After the model update, a familiar problem becomes eligible again: its next group measures the updated student.

The loop is straightforward: choose problems, generate fresh answers, check them, record what happened, and update the model. The next selection uses the new evidence. The sampler changes what the student practises; ordinary GRPO still determines how it learns from those attempts.

## Use class evidence to choose which unseen problems to try

The difficult case is an unseen problem. It has no student history, so how can we estimate its chance of producing a mixed group? We can begin with the behaviour of other problems in its class, provided those observations cover the class reasonably well.

### Average the opportunities across problems

Suppose 60% of arithmetic problems always succeed, while the remaining 40% succeed half the time. The first part contributes no mixed groups; the second contributes them with the 87.5% probability we just derived. A randomly selected problem from this illustrative population therefore has predicted mixed-group rate

$$
\begin{gathered}
u_{\mathrm{arithmetic}}^{\mathrm{new}}=0.60\,m_4(1)+0.40\,m_4(0.5)\\[8pt]
=0.60(0)+0.40(0.875)\\[8pt]
=0.35.
\end{gathered}
$$

Why not just use the class's average success? See what happens when we average success first, then treat every problem as if it had that probability:

$$
\begin{gathered}
\text{Average success}=0.60(1)+0.40(0.5)=0.8\\[8pt]
m_4(0.8)=1-0.8^4-0.2^4=0.5888.
\end{gathered}
$$

That predicts considerably more mixed groups than 0.35. Averaging success first has erased the distinction between consistently solved problems and problems that still produce contrast.

We need to average the *mixed-group probabilities* instead. Let $\mathcal F_c$ describe our estimated distribution of success probabilities for an unseen problem in class $c$. The general form of the weighted calculation above is

$$
u_c^{\mathrm{new}}=\mathbb E_{p\sim\mathcal F_c}[m_G(p)].
$$

The expectation symbol means “average over the possibilities described by $\mathcal F_c$.” That distribution is a model fitted from representative grouped evidence, not something a class label tells us automatically. Repeatedly practising two algebra questions cannot establish it for all of algebra. Sparse or stale evidence calls for a weak prior or a return to the base selection rule.

### Turn those forecasts into selection probabilities

Suppose algebra's predicted mixed-group rate is 0.70, twice arithmetic's 0.35. With equal base class weights, allocating in proportion to those scores gives

$$
\begin{gathered}
\text{Arithmetic share}=\frac{0.35}{0.35+0.70}=\frac13\\[8pt]
\text{Algebra share}=\frac{0.70}{0.35+0.70}=\frac23.
\end{gathered}
$$

That rule alone would give a class no discovery at all if its forecast reached zero. Yet the forecast could be wrong. Preserve 20% of the class choice for the base mixture, and use the forecasts for the other 80%. Arithmetic's share becomes

$$
Q(\mathrm{arithmetic}\mid E_{\mathrm{new}})
=0.20\left(\frac12\right)
+0.80\left(\frac{0.35}{0.35+0.70}\right)
\approx0.367.
$$

Here $E_{\mathrm{new}}$ is the currently eligible unseen inventory. Algebra receives the remaining 63.3%. These shares distribute the reserved discovery places over repeated draws; two places in one request cannot reproduce the percentages exactly.

For unequal base weights, multiply each forecast by the class's base weight before normalizing. Let $E$ be the eligible pool for the current draw, $b_E(c)$ its normalized base class weights, and $a_c(E)$ the class's contrast score. For discovery, that score is $u_c^{\mathrm{new}}$. The same two operations—normalize the weighted forecasts, then mix in coverage—give

$$
\begin{gathered}
A_E(c)=\frac{b_E(c)a_c(E)}{\sum_d b_E(d)a_d(E)}\\[10pt]
Q(c\mid E)=\epsilon_c b_E(c)+(1-\epsilon_c)A_E(c).
\end{gathered}
$$

The sum includes only classes with eligible problems, and $\epsilon_c$ denotes the configured class-coverage fraction, here 0.20. If all scores are zero or cannot be compared reliably, use the base mixture. Otherwise, even a zero-scoring eligible class retains probability at least $\epsilon_c b_E(c)$. This is a chance of selection, not a guarantee that every batch contains that class. Eligibility is recalculated after every draw, so a class with no unseen problems cannot receive a discovery place.

The slider below changes the fraction of consistently solved arithmetic problems. Notice that arithmetic's discovery share falls as its predicted contrast falls, but coverage leaves it a chance of being selected.

<div id="reader-forecast"></div>

<section class="calculation-example" aria-labelledby="work-through-the-effect-of-class-exploration">

#### Work through the effect of class exploration

Now hold the forecasts fixed and change only the coverage fraction. Suppose arithmetic's predicted contrast has fallen to 0.0875 while algebra stays at 0.70. Both classes have unseen problems and equal base weights. Arithmetic's share under forecast-only selection is

$$
\frac{0.0875}{0.0875+0.70}=\frac19.
$$

For each row, take the coverage fraction times one half, then add the remaining fraction times one ninth. The final column is their sum.

<table>
<caption>Arithmetic's probability of receiving a discovery draw. The task-discovery reserve stays at 20%.</caption>
<thead><tr><th scope="col">Class coverage</th><th scope="col">Coverage term</th><th scope="col">Forecast term</th><th scope="col">Arithmetic share</th></tr></thead>
<tbody>
<tr><th scope="row">0%</th><td>0.0000</td><td>0.1111</td><td>11.1%</td></tr>
<tr><th scope="row">10%</th><td>0.0500</td><td>0.1000</td><td>15.0%</td></tr>
<tr><th scope="row">20%</th><td>0.1000</td><td>0.0889</td><td>18.9%</td></tr>
<tr><th scope="row">40%</th><td>0.2000</td><td>0.0667</td><td>26.7%</td></tr>
<tr><th scope="row">100%</th><td>0.5000</td><td>0.0000</td><td>50.0%</td></tr>
</tbody>
</table>

More coverage gives the low-scoring class more chances to contradict its forecast. It also shifts draws away from the class predicted to produce more contrast. At 100%, class selection returns to equal base shares. These are probabilities for individual discovery draws, not guaranteed batch shares or percentages of total compute.

**Work it out:** what share would arithmetic receive at 30% class coverage? Keep the two forecasts and the discovery reserve unchanged.

<details>
<summary>Show the calculation for 30% class coverage</summary>

$$
\begin{gathered}
\text{Coverage contribution}=0.30\times0.5=0.15\\[8pt]
\text{Forecast contribution}=0.70\times\frac19\approx0.0778\\[8pt]
\text{Arithmetic share}\approx0.15+0.0778=22.8\%
\end{gathered}
$$

Arithmetic receives about 22.8% of discovery draws and algebra 77.2%. The number of reserved discovery places has not changed; only their class allocation has.

</details>
</section>

### Let fresh answers revise the task's forecast

That remaining chance matters. Consider an arithmetic problem, A17, which inherits a low forecast but is selected anyway. Its four answers are failure, success, failure, success. How should those observations change our belief about its success probability?

Compare two possibilities for A17's success probability. For the particular sequence we observed, each possibility gives a likelihood:

$$
\begin{gathered}
\text{If }p=0.5:\quad 0.5^2(1-0.5)^2=0.0625\\[8pt]
\text{If }p=0.9:\quad 0.9^2(1-0.9)^2=0.0081.
\end{gathered}
$$

Divide those likelihoods to see how strongly the answers favour the first possibility over the second:

$$
\frac{0.0625}{0.0081}\approx7.7.
$$

The observed sequence is about 7.7 times as likely under the first possibility. We should therefore increase its weight relative to the second, while still taking our earlier beliefs into account.

Doing this for every possible $p$ gives a Bayesian update. If task $i$ has $n_i^+$ successes and $n_i^-$ failures, multiply its prior weights by the likelihood of those observations, then normalize them to sum or integrate to one:

$$
d\mathcal F_i(p)\ \propto\
\underbrace{p^{n_i^+}(1-p)^{n_i^-}}_{\text{likelihood of the answers}}
\underbrace{d\mathcal F_{c,-i}(p)}_{\text{prior from class evidence}}.
$$

The notation $c,-i$ means that the class prior excludes this task's evidence, or was frozen before that evidence arrived, so we do not count the same answers twice. Unlike the deliberately extreme toy population above, a usable prior must leave room for success probabilities inside $(0,1)$: otherwise contrary outcomes can become impossible under the model.

We now have an updated distribution $\mathcal F_i$ for this particular problem. Use the same averaging operation as before to predict its next group:

$$
u_i=\mathbb E_{p\sim\mathcal F_i}[m_G(p)].
$$

At the start, this is the class-based forecast. With sufficient compatible observations and a prior that allows them, the problem's own results increasingly determine it. Mixed outcomes can make A17 a stronger practice candidate; repeated success eventually reduces its expected contrast. A single surprising problem does not immediately justify promoting every unseen problem in arithmetic.

For familiar practice, the class score $a_c(E)$ is the fixed-weight average of these task scores over its eligible familiar problems. After an adaptive class choice, choose a familiar problem in proportion to its base task weight times $u_i$. For example, two equally weighted problems with scores 0.6 and 0.3 receive adaptive within-class probabilities $2/3$ and $1/3$. After a coverage class choice, instead reassess the eligible problem least recently attempted. For discovery, draw with base weights within the unseen class pool. The technical reference gives the resulting joint class-and-task law; these different routes must be recorded separately.

The update above assumes a stable success probability and appropriate independence within its evidence window. A learning student does not stay fixed forever. That brings us to the next requirement: deciding when the evidence behind these forecasts is too old to trust.

## Reassessment keeps yesterday's conclusions from becoming permanent

The student continues to change. A problem that was easy can become less reliable after training elsewhere. A geometry problem that once failed may become tractable after algebra improves. Recent evidence should therefore carry more weight than distant observations, and missing evidence should remain unknown.

The coverage component periodically chooses an eligible familiar problem that has gone longest without an attempt. This gives old all-success and all-failure problems another opportunity to change their forecasts. There is no permanent “learned” label that removes a problem from consideration.

The discovery reserve has a similar purpose for problems we have never selected. Its count carries across requests. At 20%, five successive requests for one group collectively owe one discovery place. Rounding each small request down to zero would quietly remove discovery from a run with small requests.

We can avoid that loss by rounding the cumulative target instead. After $N$ candidate selections, the whole-number target is $\lfloor\delta N\rfloor$, where $\delta$ is the discovery fraction and the floor symbol means round down. A request for $B$ more groups should reserve the increase in that target:

$$
D(N,B)=\lfloor\delta(N+B)\rfloor-\lfloor\delta N\rfloor.
$$

For eight initial groups followed by a request for two more, the calculation is

$$
\begin{gathered}
D(0,8)=\lfloor0.2(8)\rfloor-\lfloor0\rfloor=1\\[8pt]
D(8,2)=\lfloor0.2(10)\rfloor-\lfloor0.2(8)\rfloor=2-1=1\\[8pt]
\text{Total reserved places}=1+1=2.
\end{gathered}
$$

Together they reserve the same two places as one ten-group request. With enough unseen inventory, summing these differences preserves $\lfloor\delta N\rfloor$ reserved places across any request sizes. Extra discoveries at startup do not prepay later reservations. This is a guarantee about selections; a failed execution is still recorded separately from completed exposure.

Some training profiles also request replacement groups after discarding candidates. In that case, all requests within the optimizer step share the same selected-task set and cumulative discovery accounting. Rejected groups still supply evidence and still cost compute. The sampler must remember them before choosing replacements. Ordinary GRPO in the base comparison retains constant-reward groups; active retention belongs to a separately declared profile.

Finite inventories need explicit fallbacks. If no unseen problems remain, unfilled discovery places become reassessment and the shortfall is recorded. If too few distinct problems remain within a step, the policy permits a recorded repeat. These rules keep the run moving without claiming diversity it did not achieve.

Low contrast everywhere is not a completion test. It may mean the student is reliably successful, consistently unsuccessful, or poorly measured. Those possibilities require different decisions, which is why success and contrast remain separate.

## The sampler must earn its complexity through better learning

The strongest objection to V2 is that it could become very good at finding mixed rewards without improving the student. A class of noisy, unlearnable problems could attract practice indefinitely. A mistaken class forecast could steer discovery away from useful work. Reference measurements and controller overhead could cost more than the policy saves.

Training reward cannot resolve those objections by itself. Once the sampler changes which problems it selects, its average reward measures a different mixture. It could rise because the student improved, because the sampler chose easier problems, or both.

For example, suppose the student stays at 80% success on arithmetic and 20% on geometry. An equal mixture reports 50% success. Moving to 90% arithmetic raises the observed average to 74%, with no learning at all. To measure improvement, hold the evaluation weights fixed. If $\rho(i)$ is the fixed reference weight of problem $i$, the quantity we want is

$$
\begin{gathered}
\mu_\rho(t)=\sum_i\rho(i)p_i(t)\\[8pt]
\Delta_\rho=\mu_\rho(t_2)-\mu_\rho(t_1).
\end{gathered}
$$

Here $p_i(t)$ is the student's success probability at checkpoint $t$, and the weights sum to one. In practice, probes estimate these quantities with uncertainty; we cannot read the true probabilities directly. With the same weights at both checkpoints, changing the training mixture alone cannot create a gain in this measure. Comparable task, verifier, and generation conditions are also necessary.

We therefore need three distinct populations. Adaptive training supplies the updates. A declared controller reference population measures comparable performance over time and supports sampling decisions. A final held-out population assesses the result without being used to tune the controller. Reference probes have a budget that must be included in the comparison.

The first experiments should separate the proposal's claims:

| Comparison | Question it answers |
| --- | --- |
| Uniform problems versus fixed class balancing | Does balancing the classes help? |
| Fixed balancing versus adaptive class selection | Does changing class allocation help beyond balancing? |
| Class-only adaptation versus class-and-task adaptation | Do individual task histories add value when step uniqueness is controlled? |
| Fixed versus class-guided discovery, with the same reserve and uniqueness rules | Do class forecasts find useful unseen problems sooner? |
| Direct recent task statistics versus class-to-task prediction | Does the predictive model justify its additional complexity? |

Use the same final evaluation population and keep group size, generation settings, and the learning rule fixed within each comparison. Report performance by class as well as overall. Compare final performance at matched total cost, and cost to reach a predeclared target. Charge every generated attempt, rejected group, verification call, and reference probe to the method that used it.

More unique problems and more mixed groups would show that selection changed as intended. **Better held-out performance for the same total cost, or the same target reached more cheaply, would support the learning claim.** If the sampler improves contrast but not that outcome, the central hypothesis has not succeeded.

## Use the simulation to inspect decisions

The interactive example below makes the selection loop visible. It contains 64 synthetic tasks in four classes, ten groups per target batch, and four attempts per group. The controller sees sampled outcomes; it cannot read the learner's hidden success probabilities. No neural network or GRPO gradient is computed.

Start with ordinary learning and follow one selected task across model steps. Then try **Mostly solved class** to look for useful exceptions to the class forecast. Finally, try **Variance without learning**: probability tasks continue producing contrast but do not improve. That case exposes the gap between the sampler's proxy and the outcome we care about.

The three lanes show discovery, practice, and recheck selections. Recheck is an observed share, not a third configured quota. Task colour represents within-group variation; the purple bar represents observed mean reward. An all-success problem and an all-failure problem can have the same low-variation colour but very different bars. Select a task to inspect its history.

The success charts summarize recent sampled outcomes. Their movement can reflect changing selections as well as the simulation's assumed learning. Use the player to understand decisions and failure cases; real training comparisons must supply the evidence of benefit.

<details class="reader-demo">
<summary>Open the task-selection simulation</summary>
<div id="reader-simulation"></div>
</details>

## Teacher data and other learning rules are later experiments

The minimum curriculum needs runnable tasks, stable identities, class assignments, and meaningful outcome checks. A teacher is optional in v2. An immutable curriculum manifest would describe the shared inventory; each student's run would maintain its own observations and selection history.

When demonstrations are useful, one selected teacher can attempt tasks, an acceptance check can verify the results, and preparation can compile accepted traces into reusable training data. That data can support supervised fine-tuning before the online curriculum. Teacher outcomes describe the teacher; the actual student still needs its own calibration, including after supervised training. Saved teacher answers cannot serve as fresh student rollouts for GRPO.

Weighting updates is another separate experiment. Choosing algebra more often changes exposure. Multiplying the learning contribution of each algebra group changes the update even when the batch stays the same. The latter requires an explicitly identified algorithm variant and a comparison with sampling held fixed.

Efficiency rewards also introduce a new objective. A small raw bonus for lower cost can become a substantial signal after within-group normalization. Quality and resource use therefore need separate evaluation rather than an assumption that a small coefficient protects accuracy.

These extensions are worth testing after the base selection policy. For the first decision, the proposal is narrower: **can recent student evidence help us choose better practice than a fixed class mixture, once discovery, coverage, and every observation cost are counted?**

## Technical reference

The complete v2 specification is retained below for mathematical and implementation review. It contains the exact sampling law, predictive assumptions, discovery accounting, statistical intervals, recovery state, profile settings, and optional experiments. Its original section numbers are preserved so existing references remain usable.

The specification remains a research proposal. New public framework contracts require a canonical-baseline amendment before implementation. The original [Markdown specification](configurable-curriculum-and-data-preparation-v2.md) is unchanged by this editorial rewrite.

<div id="reader-reference"></div>
