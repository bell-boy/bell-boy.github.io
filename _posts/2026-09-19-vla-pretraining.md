---
layout: post
title: "Summary of Research on Pre-training for Robotic Manipulation"
usemathjax: true
margin_footnotes: true
---

_This is a post in a series of posts about that seek to answer interesting questions about robotics. I'll try to summarize existing research, and propose and execute new experiments when needed. Stay tuned!_

# Introduction

Recently, we've seen a wave of impressive generalist robotics demos for manipulation. Models can now complete complex manipulation tasks over long time horizons, and learn in context from long human demonstrations. 
Yet we're still at the point where the compute required to train these generalist models is still within reach for acadamia. 
The robotics race is starting to heat up, and industry pressures mean that robotics labs are publishing less and less. 
So before the frontier pulls away from open knowledge, I think now is a great time to try to answer the question: **What's a good starting point for learning a generalist policy for manipulation?**

I'm going to spend the rest of this blog post convincing you that the literature currently supports the following (simplfied) recipe:
1. Collect as much robot data as you can. It's ideal to have a ton of target embodiment data, but even data from other embodiments will help.
2. Take a pretrained VLM and attach an action expert. This action expert should also be a transformer, conditioning on the VLM hidden states via attention.
3. Train the architecture above with a Behavior Cloning objective on the dataset from Step 1.

It's worth noting that the strength of the support for each of these steps _varies_. Some steps (like 1 and 4) have a strong backing and seem unlikely to change soon, while we'll see that other steps (like 2) don't have nearly as much evidence.

We'll then try to get a sense for the policy quality this recipe produces using modern benchmarks. We'll go further and talk about the limits of policy generalization ability.

# Glossary

Definitions of the most common jargon you'll see throughout this post.

**Policy**: The "brain" of the robot. This is a function \\( \\pi(a | s) \\) that takes in some inputs \\( s \\) (e.g. camera data, a text prompt, the current position of the robot) and produces a probability distribution over actions.
The policy is said to produce an action distribution **conditioned** on it's input.

**Task Specific Policy**: Also referred to as narrow policies, these policies are trained to perform a single task. While they might be able to do this task robustly in many different environments, they aren't expected to perform any other tasks.

**Generalist Policy**: In contrast to narrow policies, these policies are trained to perform many different tasks across manipulation.

**Training Distribution**: The space of tasks, objects, and environments the policy was trained on. We'll often refer to tasks as being _in distribution_ or _out of distribution_. This typically refers to wheher or not the task was seen during training.
We might also claim to evaluate a policy under **Distribution Shift**. 
This just refers to when the conditions we evaluate under (like environment, object appearence, instruction text) are different than those that were seen during training.

**Fine-Tuning**: We'll often fine-tune generalist policies on datasets used to train narrow policies. This just means that we'll do some extra training on the generalist policy to improve performance on a specific task.

**Zero-shot Evaluation**: When we evaluate a generalist policy on a task without collecting data and finetuning on that task first, we'll often call this _zero-shot evaluation_. All this means is that we're measuring the performance of the policy out of the box.

# Robot Data

Data, like the rest of deep learning, is of central importance in Robot Learning. Our goals directly shape the type of data we want to collect and the format we want to collect it in.

For the sake of our discussion, let's assume we know what we want our robot to do, and we know the hardware we want the robot to do it on.
We'll refer to what we want the robot to do as the _target task_(s), and the robot body we want to deploy our policy on as the _target embodiment_.

Additionally, we'll also have some known way of "prompting" our policy to execute some given task. Policies that can be prompted this way are called *goal conditioned*.
There are various forms of goal conditioning, but the most common method you'll see is language.
We'll want this goal-conditioning (or a something similar) to be present in our data as well.

In [Data Scaling Laws In Imitation Learning For Robotic Manipulation](https://arxiv.org/pdf/2410.18647) _Lin et al._ show that if you scale target task data in diverse environments on the target embodiment[^1], you get a robust non-goal conditioned narrow policy that can execute the target task at high sucess rates (~90%) in completely unseen environments!

<figure>
  <video controls playsinline preload="metadata" style="width: 100%;">
    <source src="{{ '/assets/images/vla/datascalingvideo.mp4' | relative_url }}?v=faststart" type="video/mp4">
    Your browser does not support embedded video. <a href="{{ '/assets/images/vla/datascalingvideo.mp4' | relative_url }}">Watch the data scaling video</a>.
  </video>
  <figcaption style="font-size: 0.85em; color: #777;">Demo from <a href="https://arxiv.org/pdf/2410.18647" style="color: inherit;">Lin et al.</a></figcaption>
</figure>

This is fantastic! We've solved robotics! Sure these policies aren't generalists, but it seems likely that we can get there by scaling diverse data across many different tasks.

The only problem is that this is insanely expensive.

We're in _academia_! I'm an undergrad! Where am I going to find the time to record millions of demonstrations across the gamut of human ability!

Thankfully, academia has spent years collecting thousands of hours of data across various robots! Sure it's not all on our target tasks and it's not on our target embodiment, but it must be better than nothing right?

Right! There's a long line of evidence showing that training policies on large diverse[^2] robot datasts helps makes policies better across various axes!

[Open X-Embodiment](https://arxiv.org/pdf/2310.08864v5) was a huge effort across various academic institutions to create a large open source dataset for training generalist policies. 
They then tried to answer the question: _Does training on diverse robotics data across various embodiments improve task performance over training on narrow task specific datasets?_ 
For tasks with small scale datasets, the results are a resounding yes! Almost always, their generalist policy out performed methods trained on just the targeted data.
For tasks with larger datasets, the results were mixed. When the generalist policy was matched against a policy with the same architecture but trained only on the target task, the single task policy usually ended up on top. They posited that the architecture was too small to absorb the extra data and showed that a larger architecture trained on the the full dataset did end outperforming the smaller architecture trained on more targeted data.

[Octo](https://arxiv.org/pdf/2405.12213) Went further and trained on a larger split of OXE which they cleaned and filtered. Again, they find that the resulting policy outperforms policies trained from scratch. They also show that fine-tuning Octo to perform a new task on a new embodiment is often much more data-efficent than training a policy from scratch! 

[\\(\pi_0 \\)](https://arxiv.org/pdf/2410.24164), a much larger model later developed by many of the same people behind Octo, would train on an even larger dataset of diverse manipulation data (~ten thousand hours from online reports! Compare to the ~one thousand hours for octo). Again they show that the resulting policy performs better than policies trained from scratch, and that it's much more efficent to finetune pretrained policies. 

![\\(\pi_0 \\) task progress after fine-tuning, compared with training from scratch and zero-shot evaluation.]({{ '/assets/images/vla/pi0-complex-tasks.png' | relative_url }})

To investigate these claims, the Large Behavior Model team at the Toyota Research Institute performed a [A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation](https://arxiv.org/pdf/2507.05331v1). They seek to compare the performance of generlist policies to those trained from scratch. 
The paper is approachable and easy to read, and their claims are clear and well supported by the experiment results. At the same time, the experiments are done with a level of rigour not often seen in the literature! 
I highly reccomend reading this paper yourself, but some of their key findings are that:
1. Large Behavior Models (their name for generalist policies) perform similar to narrow polices out of the box when evaluated on tasks seen during pre-training.
2. Finetuning generalist policies does produce a small performance benefit compared to training from scratch when in distribution.
3. Finetuned generalist policies are much more robust to distribution shift (i.e. new environments and objects) than polices trained on smaller target data!

![Results from Toyota Research Institute: success rates for single-task policies, Large Behavior Models, and fine-tuned Large Behavior Models, without distribution shift (top) and with distribution shift (bottom). Focus on the 'aggregate' columns to get a clear picture of the results.]({{ '/assets/images/vla/tri-results.png' | relative_url }})

Now that we're training on multiple embodiments, we need an action represention that makes experience across these embodiments transferable. [Demystifying Action Space Design for Robotic Manipulation Policies](https://arxiv.org/pdf/2602.23408) compares four approaches to action represention: absolute joint angle, absolute end effector frame, delta joint angle, delta end effector frame.
- **Absolute joint angle** asks the policy to output absolute joint angles, meaning we precisely specify the pose the robot should be in.
- **Absolute end effector frame** asks the policy to output the position and rotation of the end effector (usually the gripper in manipulation setups) in 3D space. Inverse kinematics is used to figure out how to move the joints to said space
- **Delta joint angle** asks for the change in joint angle relative to the current pose. We specify how the current pose should change over the next time step
- **Delta end effector frame** asks the policy for the change in the position of the end effector relative to it's current position. Again, inverse kinematics can be used to figure out how to move the joints to the correct pose.

![Absolute commands specify target joint angles or end-effector poses; delta commands specify changes from the current state.]({{ '/assets/images/vla/action-representations-blog.png' | relative_url }})

They find that while delta joint angle produces the best policies when the data comes from a single embodiment, delta end effector frame produces policies that perform much better on diverse embodiments, and adapt well to new ones.

Finally, a common trend amongst robotic manipulation papers that employ large scale training is to train on a re-weighted (meaning some samples have a higher likelihood be being repeated) subset of these large datasets.
While more data helps, we can't avoid the fundamental issue of garbage in, garbage out.

Though most of the dataset weighting reported in papers like \\(\pi_0 \\) and Octo was done by hand, [Re-Mix](https://arxiv.org/pdf/2408.14037) is a great step in understanding how to systematically re-weight our data, using a min-max based objective to choose dataset weights. They see sizable gains in performance in policies trained on the dataset they evaluated on, though more work is needed.


So to summarize our findings on data:
- Modern methods can extract high performance on manipulation tasks from demonstration data
- In the absence of a large amount target data, we can leverage non-target robot data to imrpove performance.
- There are various ways we can represent actions, and the method we choose has significant impact on our policy's performance.
- Naively training on all the data out there won't necessarily produce the best policy. Various methods have been proposed for data selection, but the best such method is still an open area of research.

# Behavior Cloning, Action Chunking, and Flow Matching

The primary objective that pre-training methods use is **behavior cloning**.

Under behavior cloning, we're given some dataset of state-action[^5] pairs \\(S = \\{(s_{i}, a_{i})\\}\\) sampled from some reference policy 
(meaning \\( a_{i} \\sim \pi ( \cdot | s_{i}) \\)  where
\\( \pi \\) is our reference policy) and our goal is to match the reference policy as best as possible.

Behavior cloning is a natural objective to use because it allows us to learn from demonstration data, but there are two cruicial modern pieces for it's success.

## Action Chunking

The paper [ALOHA](https://arxiv.org/pdf/2304.13705)'s impact can be felt throughout the robot learning space. 
While it has two major contributions, the relevent one for our purposes is the demonstration that _action chunking_ combined with behavior cloning produces better policies than behavior cloning alone.

Under standard behavior cloning, policies consume one observation then predict one action. 
Under _action chunking_, policies consume one observation then predict _multiple actions_.
Naively executing these policies results in jerky motions, so the paper proposes a fix for this called _temporal ensembling_. 
Under temporal ensembling, at each timestep $t$ we sample the next $K$ actions. Instead of executing the current action at each step, we instead execute an expoential moving average of the past $K$ predictions for this timestep.

![Image from Zhao et. al describing the action chunking inference scheme]({{ '/assets/images/vla/act.png' | relative_url }})

It's kind of mysterous as to why action chunking works so well. 
It was originally thought that by training policies to predict chunks, we reduce the horizion at which errors can accumlate over. 
But, it was later noticed that this alone wasn't enough to explain the superior performance of the technique.
The [best investigation](https://arxiv.org/pdf/2608.02547) into this question that I know of argues that instead, it's the _ensembling_ of actions predicted using multiple observations that helps reduce compounding, often _more so_ than the chunking! 

## Diffusion Policy

[Diffusion Policy](https://arxiv.org/pdf/2303.04137) is another ground breaking paper that solved a central challange at the time: What action should my policy pick when there are multiple good options?

Usually, we model actions as continous $d$ dimensional vectors, so our policy must generate a continous distribution over these $d$ dimensional vectors. 
This is non-trivial, since the space of possible outputs is infinite!

The standard solution for a while was to predict a single action with a regression loss[^3], or model the action distribution with a single Gaussian. A Gaussian can change its mean and variance, but it always has just one peak.

This works well when good actions cluster around one choice. But when there are multiple distinct choices for the _same observation_, squared-error regression learns their _average_, which can be an invalid action. Imagine reaching around an obstacle: going left or right might work, but averaging the two sends the gripper straight into it! Even a Gaussian with learned variance can't represent both peaks, it spreads probability across the gap between them.

<figure>
  <img src="{{ '/assets/images/vla/bc-gaussian-vs-multimodal.svg' | relative_url }}" alt="Two panels compare demonstration distributions with their best single-Gaussian fits. For one cluster the mean sits at its peak; for two clusters the mean sits in the low-density gap." loading="lazy" width="1040" height="510" style="width: 100%; height: auto;">
  <figcaption class="image-caption">A one-dimensional toy example, holding the observation fixed in each panel. Blue shows the demonstration distribution; orange shows the Gaussian that matches its mean and variance. The diamond marks the mean, also the optimal squared-error prediction.</figcaption>
</figure>

Instead Diffusion Policy leverages the fact that diffusion models[^4] can learn complex continous distributions given only data sampled from that distribution. 
So instead of trying to always pick the single correct action, we _sample_ one of the good actions from our policy's action distribution!

[Flow matching](https://arxiv.org/abs/2210.02747)[^7], similar to diffusion, also learns distributions from data. Flow matching starts with a sample of standard Gaussian noise, then uses the model to repeatedly push them to the target distribution. As those samples move, their distribution can develop multiple peaks. At the end, we have a sample from the learned action distribution.

<figure>
  <img src="{{ '/assets/images/vla/bc-flow-evolution.svg' | relative_url }}" alt="Four density snapshots show standard Gaussian noise becoming a two-peaked action distribution. Below, a density map and individual sample paths show noise samples moving toward the two action modes as sampling time increases." loading="lazy" width="1040" height="710" style="width: 100%; height: auto;">
  <figcaption class="image-caption">An idealized flow toward the same two-peaked toy distribution. The curves and sample paths are computed from an analytic velocity field, rather than a trained policy. Sampling time τ runs from noise to actions; it is separate from robot execution time.</figcaption>
</figure>

So in summary:
- Behavior Cloning is the natural objective when we want to learn policies from demonstration data.
- Naively predicting the next action based on the current obseravtion results in brittle policies.
- There isn't strong consensus about why action chunking works, but emperically it provides strong gains.
- Action distributions can be _multimodal_. Squared-error regression averages distinct choices, potentially producing invalid actions. **Diffusion and flow matching** let us model multiple modes and sample actions from the learned distribution.

# Vision Language Action Models

Vision Langauge Action models, VLAs, are the dominant paradigm in generalist robotics. 
For a while, they seemed to be the _only_ relevant architecture in generalist robotics.
As such, there isn't much work rigourously comparing them to other model classes.
So I can't claim that VLAs are the _best_ architecture we have for generalist robot learning, just that they're the _default_.

Recent research has proposed a new class of models -- World Action Models (WAMs) -- as an alternative. We'll talk about these later.

To motivate VLAs, let's take a look at the architecture of Octo, one of the earliest generalist policies and one of the last major generalist non-VLA policies.
At it's core, Octo is a transformer that takes in a description of the current goal (via an image description of the desired end state or a text prompt telling the model what to do) and a sequence of observations (images in octo's case), both encoded as tokens.

To turn images and text into tokens, Octo uses two seperate encoders.
A pretrained language encoder produces lanugage tokens, and a learned image encoder produces image tokens.

![Visualization of the Octo architecture]({{ '/assets/images/vla/octo.png' | relative_url }})

Now what if instead of using disjointed encoders for language and vision so that the _policy_ has to learn to connect them (i.e. understand that when we refer to an apple, we mean 🍎), we use a pretrained encoder which has _already_ connected language and vision!

Vision Language Models (VLMs) are Large Langauge Models trained to also predict text based on images as well. They fit our encoder description almost perfectly.

This is the key idea behind VLAs.

\\(\pi_0 \\) (the cannonical VLA) has two major components: a pretrained Vision Language Model (PaliGemma-3b) and an action head[^9].

<figure>
  <img src="{{ '/assets/images/vla/vla-attention-gradients.svg' | relative_url }}" alt="Action-transformer queries attend to keys and values from VLM hidden states at successive layers. Dashed gold arrows trace gradients from the flow-matching behavior-cloning loss back through the action transformer and attention into the VLM." loading="lazy" width="1000" height="660" style="width: 100%; height: auto;">
  <figcaption class="image-caption">Gray arrows show forward computation; dashed gold arrows show a training-gradient path. The action expert reads the VLM's hidden states, and action supervision trains both. This is a schematic of the information flow: \(\pi_0 \) implements these connections with shared masked attention.</figcaption>
</figure>

The VLM takes in a task prompt and images corresponding to the current observations. During inference, it does not generate text. Instead, the action transformer attends to the VLM's intermediate hidden states, while also conditioning on proprioception. The action head learns the action distribution via flow matching.

The flow-matching loss is the behavior-cloning objective. Its gradients propagate through the action head and attention connections into the VLM, encouraging its representations to be more useful for predicting actions.

## Training VLAs

Since the release of \\(\pi_0 \\) in 2024, we've learned a bit more about how to best train VLAs, and I want to mention a few of the most impactful parts of the updated recipe.

### Co-training

The size of available robot data is many orders of magnitude smaller than pretraining datasets used when training LLMs. It's also costly to spend time teleoperating and collecting more data.
As a result, there's been a substantial amount of effort to figure out how to learn from alternative sources of data that are either 1) easier to collect 2) more naturally abundant. 

Training on auxilary data to improve robot performance is known as _co-training_.

There's strong evidence to believe that your co-training data should include a mix of standard vision langauge data and densely annotated human and robot video, and growing evidence that retargeted human ego-centric data is of value.

The Toyota Research Institute has a great paper studying this exact question. Like the other TRI paper mentioned, [A Systematic Study of Data Modalities and Strategies for Co-training Large Behavior Models for Robot Manipulation](https://arxiv.org/pdf/2602.01067v1) is a _fantastic_ paper. In it, multiple co-training methods from the literature are tried. They only find strong support for the following forms of data
1. Standard vision-language data: Data consisting of an image, some question about it's contents, and an answer.
2. Vision Language Model annotated Human & Robot video. A language model is given a video and asked to describe in detail what's happening in it. This can be see as standard vision-langauge data with the question "describe this video".

Notebly, both of these forms of data _don't_ improve in distribution manipulation performance! Their benefit instead comes from improving performance _out of distribution_!

<!-- maybe an example of both forms -->
<!-- plot from paper -->

We can get some validation of their findings from the [\\( \pi_{0.5} \\)](https://arxiv.org/pdf/2504.16054) report. Physical intellegence finds that a mix of data that the TRI would call "standard vision-langauge data" (they refer to it as multi-modal web data) greatly improves out of distribution success rates.

<!-- footnote about how TRI and pi0.5 disagree -->

While there's less rigourous empirical support, there's been a substatital wave of interest in utilizing human ego-centric data to improve policy performance. Typically in ego-centric pipelines, huamns wear a some camera rig that records what they see from a first person point of view. We later apply hand pose estimation models to get "actions" from these human videos. [EgoVerse](https://arxiv.org/pdf/2604.07607) evaluates this setup and find that when the human videos match the finetuning data, it leads to moderate imrpovement over just training on robot actions. They find almost no benefit to training on diverse human video data, but increased benefit when diverse data is paired with human data that better "matches" the robot training data.

<!-- images explaining the above paragraph -->

[EgoScale](https://arxiv.org/pdf/2602.16710) in contrast trains on _thousands_ of hours of human video, and finds that at this scale, there's a clear benefit to scaling diverse human data -- with a few caveats.
1. Like EgoVerse, they apply hand pose estimation to start extracting actions from human video, but in addition, they _retarget_ the human hand poses to the robot hand space, trying to find the robot pose that best "matches" the human pose. It's this retargeted hand pose they train on.
2. They include EgoDex, a dataset captured with an Apple Vision Pro, which has built in sensors that can label joints with better precision than images alone. They never ablate this addition. This is about ~4% of their dataset.

Additionally, similar to EgoVerse, they find that adding human data that matched robot data substantially improved perforamce. With the caveat being that the matched human data they collected didn't use the retargeting data pipeline the pretraining data used, instead they used hardware to exactly measure joint positions and had the camera setup match exactly the setup of their target embodiment.

<!-- ego scale plots -->

Their results are promising and scary at the same time. On the one hand, human pretraining greatly acclerates learning from finetuning, on the other hand on the scale needed to even get to 50% success rates is mind-boggling.



### Gradient Management

Multiple groups have independantly noticed that naively allowing action gradients to flow through cross attention from the uninitialized action head results in worse performance than more restrictive forms of information flow, like having the action head take in the last token's hidden state, and letting gradients only flow through that. There have been many solutions proposed.

Knowledge insulation, TRI co-training paper, VLM training mixtures

## World Action Models
I'd be remiss to not at least mention the primary alternative to VLAs -- World Action Models.

I don't really like this term. It doesn't point to a narrow class of models like VLA does, but it's the name that stuck.

In summary:
- The default architecture for most generalist policies is the **Vision Langauage Action** Model
- Since VLMs are pretrained to understand both vision and langauge, VLAs use them as a backbone to leverage this understanding to take actions
- There are a few subtle details to make VLA training go well and get the most our of your data. 
- There's not many of comparisions between VLAs and alternative architectures, in large part because there are few alternatives!

# Limitations of The Recipe

While we have abundant evidence that it's possible to pre-train robotic polices for improved performance over narrow polices, we don't have much evidence for various forms of _generalisation_.

In this section, we try to find out what we can say about generalist policy performance *out of distribution*.

## Environment & Object Generalization

Enviornment and object generalization -- the ability to perform the same task in a new environment or with different objects -- is what people commonly refer to when they mention generalization. 

This is probably also the best studied. As mentioned earlier, the Toyota Research Institute has a [great paper]() studying the effects of generalist training on this type of generalization. To reiterate: Generalist policies are more robust to environment and generalization shifts then narrow policies. 

<!-- maybe re-present plots? -->

Though the generalization is improved compared to the baseline, it's not of high enough quality to where we can deploy generalist policies zero-shot in new environments without further retraining.

We have evidence that by scaling diverse data (via [Lin et al.]()), zero-shot generalization to new objects can be improved, but strong generalization remains to be seen in any open policy.

## Compositional Generalisation

One desridata that we might have for a generalist policy is that it may learn how to _compose_ atomic skills learned in pretraining or finetuning.

If we know how to open drawers and place things in open drawers, we should be able to open drawers *and then* place things in open drawers, without ever seeing this combination in training.


[RoboHiMan](https://arxiv.org/pdf/2510.13149) sets out to test this question by constructing a hiarchy of 4 datasets to train on. 
1. The first dataset is simply all of the atomic skills that plan to evaluate. All collected in roughly the same setting
2. This dataset is all of the atomic skills but with _pertubations_ (meaning different lighting conditions, objects, environments) that match the pertubations that will be applied during evaluation in addition to the first dataset.
3. This dataset adds some of the composite skills we evaluate on, collected in roughly the same setting.
4. This final dataset adds those same composite skills but collected under pertrubed conditons.

They then evaluate success rates on atomic skills and composite skills

The paper finds that polices trained under our recipe (\\(\pi_0 \\)) using datasets L1 or L2 can't learn to compose skills like those demonstrated in L3 and L4.

[plot of \\(\pi_0 \\) and \\(\pi_{0.5} \\) results]

You'll notice that it also doesn't seem like the policies learned the composite skills _even when explicitly trained on them_. This begs the question, how much of their results were due to poor policy performance vs a failure to compose?

[Atom-Bench](https://arxiv.org/pdf/2606.16826) sets up similarly to RoboHiMan: They have a set of atomic skills and composite skills built up from those atomic skills. They collect demonstrations on the atomic skills and attempt to see if the model can complete the composite skills zero-shot. 

[plot]

Similar to RoboHiMan, they find that the models can't compose skills zero-shot. Interestingly, the come up with a metric to analyze what fraction of the failure is due to poor learning of the composite skills and what fraction is due to failure to compose.

They give each atomic skill a scoring rubric, to reward the policy for partial progress. For example, a simple task like "pick up the fork and place it in the plate" might give one point for picking up the fork, and another fork for picking up the plate.

Since all composite skills they test are made up of atomic skills, they can also combine these atomic rubrics for a composite rubric.

Now, in theory, if the policy had no issues composing skills, then the average score on a composite task should be the sum of the scores we get when we execute each of the atomic subskills in isolation. 

To measure the amount of failure that can be attributed to compositional failures, they introduce the metric _Compositional Failure Share_ (CFS). CFS is 0% when the average composite score is identical to the sums of the average atomic scores, and 100% when the atomic scores are executed perfectly but we still see a non-zero amount of failure.

As you see in the plot, the policies trained under our recipie have pretty high compositional failure shares, and \\(\pi_{0.5} \\)'s compositional failure share is close to 100%!

## Langauge Generalisation

We now want to ask the question: What happens when I _rephrase_ a task seen during training?

There aren't many comprehensive studies of language generalisation alone, however, a couple papers try to include language generalization in a suite of benchmarks testing policy robustness.

[LIBERO-Plus]() has a few particularly clean experiments, showing that 1) Current manipulation policies are shockingly robust to language instructions, often being able to execute the correct task despite being given an empty string. 2) Current manipulation policies fail to generalize tasks to new objects pointed at via text. 
They then claim that this suggests that modern generalist policies aren't really paying much attention to the goal conditioning string.
<!-- explain he meaning of this, plus the setup some more. also this is libero so... overfitting? -->
<!-- how does this square with the earlier results? didn't I just say strong object generalisation -->

% TODO plots son
A similar observation was noticed by TRI when performing their study:

> some quote here


# Conclusions & Further Areas of Research

Thank you so much for reading this far! Despite being such a long read, there's still so much I haven't covered (namely, WAMs). Please feel free to reach out with any questions or feedback you have on the post, and stay tuned for an upcoming post on posttraining and RL!

[^1]: They use Universial Manipulation Interface to collect data, so they aren't _technically_ on the target embodiment, but UMI data looks enough like target data that we'll give it a pass.

[^2]: Emphasis on _diverse_. A key finding of Lin et al. is that smaller more diverse datasets can outperform larger more uniform ones.

[^3]: This is done by trying to minimize the \\(\text{L}_2\\) loss $\lVert\mathbf{a}_t - \hat{\mathbf{a}}_t\rVert^2$

[^4]: I won't cover what diffusion models are or why they can do this, but it's extremely relevant from an implementation point of view. Please refer to this amazing work at [MIT](https://diffusion.csail.mit.edu/2026/) for an introduction.

[^5]: State $s_t$ is usually a collection of images taken from cameras attached to the robot + some info about the robot's current state. Currently in robot learning it's not common to (explicitly) predict actions based on a history of observations.


[^7]: Flow matching and diffusion are deeply related. While diffusion was popularized first, flow matching is a lot simpler conceptually and implementation wise. Most modern generalist robotics policies are trained with some variant of Flow Matching loss.


[^9]: while most VLAs are some variant of the \\(\pi_0 \\) architecture, \\(\pi_0 \\) wasn't the first VLA. [RT-2](https://arxiv.org/pdf/2307.15818) and [OpenVLA](https://openvla.github.io/) both came earlier and don't use an action head. Instead, they teach the VLM to predict actions directly.
