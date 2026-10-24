---
layout: post
title: "Summary of Research on Pretraining for Robotic Manipulation"
usemathjax: true
margin_footnotes: true
featured: true
featured_order: 0
---

_This is part of a series exploring interesting questions about robotics. I'll try to summarize existing research and, when needed, propose and run new experiments. Stay tuned!_

# Introduction

Recently, we've seen a wave of impressive generalist robotics demos for manipulation. Models can now complete complex manipulation tasks over long time horizons and learn in context from long human demonstrations.
Yet the compute required to train these generalist models is still within academia's reach.
The robotics race is starting to heat up, and industry pressures mean that robotics labs are publishing less and less.[^industry] 
So before the frontier pulls away from open knowledge, I think now is a great time to try to answer the question: **What's a good starting point for learning a generalist policy for manipulation?**

I'm going to spend the rest of this blog post convincing you that the literature currently supports the following (simplified) recipe:
1. Collect as much robot data as you can. It's ideal to have a ton of target-embodiment data, but even data from other embodiments will help.
2. Take a pretrained VLM and attach an action expert. This action expert should also be a transformer, conditioning on the VLM hidden states via attention. Be careful about how you let gradients flow from the action expert into the VLM. 
3. Train the architecture above with a behavior cloning objective on the dataset from Step 1.

It's worth noting that the strength of the support for each of these steps _varies_. Some steps (like 1 and 3) have strong support and seem unlikely to change soon, while we'll see that other steps (like 2) don't have nearly as much evidence.

Finally, we'll talk about the limits of the policy quality achievable with this recipe.


# Glossary

Definitions of the most common jargon you'll see throughout this post.

**Policy**: The "brain" of the robot. This is a function \\( \\pi(a | s) \\) that takes in some inputs \\( s \\) (e.g. camera data, a text prompt, the current position of the robot) and produces a probability distribution over actions.
The policy is said to produce an action distribution **conditioned** on its input.

**Task-Specific Policy**: Also referred to as a narrow policy, this is a policy trained to perform a single task. While it might be able to do this task robustly in many different environments, it isn't expected to perform any other tasks.

**Generalist Policy**: In contrast to a narrow policy, a generalist policy is trained to perform many different manipulation tasks.

**Training Distribution**: The space of tasks, objects, and environments the policy was trained on. We'll often refer to tasks as being _in distribution_ or _out of distribution_. This typically refers to whether or not the task was seen during training.
We might also claim to evaluate a policy under **Distribution Shift**. 
This means evaluating under conditions (like the environment, object appearance, and instruction text) that differ from those seen during training.

**Fine-Tuning**: We'll often fine-tune generalist policies on datasets used to train narrow policies. This just means that we'll do some extra training on the generalist policy to improve performance on a specific task.

**Zero-shot Evaluation**: When we evaluate a generalist policy on a task without collecting data and fine-tuning on that task first, we'll often call this _zero-shot evaluation_. All this means is that we're measuring the performance of the policy out of the box.

**Proprioception**: The robot's sense of its own body. In practice, this usually means measurements like joint angles, joint velocities, and how far the gripper is open. These are often extra inputs to the policy alongside images and a task prompt.

**End Effector**: The part at the end of a robot arm that interacts with the world. For the manipulation tasks in this post, this is usually a gripper or a robot hand. When we talk about its _pose_, we mean its position and orientation in 3D space.

**Teleoperation**: Having a human control the robot, often through a controller, VR headset, or another device that maps human movements to robot movements. We can record what the robot sees and what the human makes it do to collect demonstrations for behavior cloning.

**Inverse Kinematics**: Given the joint angles of a robot arm, we can work out where its end effector will be. Inverse kinematics goes the other way: given a desired end-effector pose, what joint angles would put it there? This lets a policy specify where the gripper should go without having to predict every joint angle itself.

**Hz / Control Frequency**: How often we send a new action command to the robot. Hz just means "times per second", so a control frequency of 50 Hz means one command every 20 ms. This doesn't mean the policy needs to run inference 50 times per second! With action chunking, one inference call can produce commands for multiple control steps.

**UMI**: Short for [Universal Manipulation Interface](https://umi-gripper.github.io/), a system for collecting demonstrations with handheld grippers fitted with cameras. A human uses the grippers to perform the task, and the system records images and gripper motion that can be used to train a robot policy. We can collect useful robot training data without needing a robot there during collection!

# Robot Data

As in the rest of deep learning, data is central to robot learning. Our goals directly shape the type of data we want to collect and the format we want to collect it in.

For the sake of our discussion, let's assume we know what we want our robot to do, and we know the hardware we want to use.
We'll refer to what we want the robot to do as the _target task or tasks_, and the robot body we want to deploy our policy on as the _target embodiment_.

We'll also have some known way of "prompting" our policy to execute some given task. Policies that can be prompted this way are called *goal-conditioned*.
There are various forms of goal conditioning, but the most common method you'll see is language.
We'll want this goal-conditioning (or something similar) to be present in our data as well.

In [Data Scaling Laws In Imitation Learning For Robotic Manipulation](https://arxiv.org/pdf/2410.18647) _Lin et al._ show that if you scale target-task data in diverse environments on the target embodiment[^1], you get a robust non-goal-conditioned narrow policy that can execute the target task at high success rates (~90%) in completely unseen environments!

<figure>
  <video controls playsinline preload="metadata" style="width: 100%;">
    <source src="{{ '/assets/images/vla/datascalingvideo.mp4' | relative_url }}?v=faststart" type="video/mp4">
    Your browser does not support embedded video. <a href="{{ '/assets/images/vla/datascalingvideo.mp4' | relative_url }}">Watch the data scaling video</a>.
  </video>
  <figcaption style="font-size: 0.85em; color: #777;">Demo from <a href="https://arxiv.org/pdf/2410.18647" style="color: inherit;">Lin et al.</a></figcaption>
</figure>

This is fantastic! We've solved robotics! Sure, these policies aren't generalists, but it seems likely that we can get there by scaling diverse data across many different tasks.

The only problem is that this is insanely expensive.

We're in _academia_! I'm an undergrad! Where am I going to find the time to record millions of demonstrations covering the full range of human manipulation skills?

Thankfully, academia has spent years collecting thousands of hours of data across various robots! Sure, it's not all on our target tasks and it's not on our target embodiment, but it must be better than nothing, right?

Right! There's a long line of evidence showing that training policies on large, diverse[^2] robot datasets helps make policies better across various axes!

[**Open X-Embodiment**](https://arxiv.org/pdf/2310.08864v5) was a huge effort across various academic institutions to create a large open-source dataset for training generalist policies.
They then tried to answer the question: _Does training on diverse robotics data across various embodiments improve task performance over training on narrow task-specific datasets?_

For tasks with small-scale datasets, the results are a resounding yes! Almost always, their generalist policy outperformed methods trained on just the targeted data.

For tasks with larger datasets, the results were mixed. When the generalist policy was matched against a policy with the same architecture but trained only on the target task, the single-task policy usually ended up on top. They posited that the architecture was too small to absorb the extra data and showed that a larger architecture trained on the full dataset did end up outperforming the smaller architecture trained on more targeted data.

[**Octo**](https://arxiv.org/pdf/2405.12213) went further and trained on a larger split of OXE which they cleaned and filtered. Again, they find that the resulting policy outperforms policies trained from scratch. They also show that fine-tuning Octo to perform a new task on a new embodiment is often much more data-efficient than training a policy from scratch! 

[**\\(\pi_0 \\)**](https://arxiv.org/pdf/2410.24164), a much larger model later developed by many of the same people behind Octo, was trained on an even larger dataset of diverse manipulation data (~ten thousand hours! Compare to the ~one thousand hours for Octo). Again, they show that the resulting policy performs better than policies trained from scratch, and that it's much more efficient to fine-tune pretrained policies.

![\\(\pi_0 \\) task progress after fine-tuning, compared with training from scratch and zero-shot evaluation.]({{ '/assets/images/vla/pi0-complex-tasks.png' | relative_url }})

To investigate these claims, the Large Behavior Model team at the Toyota Research Institute conducted a study titled [A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation](https://arxiv.org/pdf/2507.05331v1). They seek to compare the performance of generalist policies to those trained from scratch.
The paper is approachable and easy to read, and their claims are clear and well supported by the experimental results. At the same time, the experiments are done with a level of rigor not often seen in the literature!
I highly recommend reading this paper yourself, but some of their key findings are that:
1. Large Behavior Models (their name for generalist policies) perform similarly to narrow policies out of the box when evaluated on tasks seen during pretraining.
2. Fine-tuning generalist policies does produce a small performance benefit compared to training from scratch when in distribution.
3. Fine-tuned generalist policies are much more robust to distribution shift (i.e. new environments and objects) than policies trained on smaller target-task datasets!

![Results from Toyota Research Institute: success rates for single-task policies, Large Behavior Models, and fine-tuned Large Behavior Models, without distribution shift (top) and with distribution shift (bottom). Focus on the 'aggregate' columns to get a clear picture of the results.]({{ '/assets/images/vla/tri-results.png' | relative_url }})

Now that we're training on multiple embodiments, we need an action representation that makes experience across these embodiments transferable. [Demystifying Action Space Design for Robotic Manipulation Policies](https://arxiv.org/pdf/2602.23408) compares four approaches to action representation: absolute joint angle, absolute end-effector frame, delta joint angle, delta end-effector frame.
- **Absolute joint angle** asks the policy to output absolute joint angles, meaning we precisely specify the pose the robot should be in.
- **Absolute end-effector frame** asks the policy to output the position and rotation of the end-effector (usually the gripper in manipulation setups) in 3D space. Inverse kinematics is used to figure out how to move the joints to achieve that pose.
- **Delta joint angle** asks for the change in joint angle relative to the current pose. We specify how the current pose should change over the next time step.
- **Delta end-effector frame** asks the policy for the change in the position of the end-effector relative to its current position. Again, inverse kinematics can be used to figure out how to move the joints to the correct pose.

![Absolute commands specify target joint angles or end-effector poses; delta commands specify changes from the current state.]({{ '/assets/images/vla/action-representations-blog.png' | relative_url }})

They find that while delta joint angle produces the best policies when the data comes from a single embodiment, delta end-effector frame produces policies that perform much better on diverse embodiments and adapt well to new ones.[^angles]

Finally, a common trend amongst robotic manipulation papers that employ large-scale training is to train on a reweighted (meaning some samples have a higher likelihood of being repeated) subset of these large datasets.
While more data helps, we can't avoid the fundamental issue of garbage in, garbage out.

Though most of the dataset weighting reported in papers like \\(\pi_0 \\) and Octo was done by hand, [Re-Mix](https://arxiv.org/pdf/2408.14037) is a useful step toward systematically reweighting our data, using a min-max objective to choose dataset weights. They see sizable policy performance gains on the dataset they evaluated, though more work is needed.


So to summarize our findings on data:
- Modern methods can learn high-performing manipulation policies from demonstration data.
- In the absence of a large amount of target data, we can leverage non-target robot data to improve performance.
- There are various ways we can represent actions, and the method we choose has a significant impact on our policy's performance.
- Naively training on all the data out there won't necessarily produce the best policy. Various methods have been proposed for data selection, but the best such method is still an open area of research.

# Behavior Cloning, Action Chunking, and Flow Matching

The primary objective that pretraining methods use is **behavior cloning**.

Under behavior cloning, we're given some dataset of state-action[^5] pairs \\(S = \\{(s_{i}, a_{i})\\}\\) sampled from some reference policy 
(meaning \\( a_{i} \\sim \pi ( \cdot | s_{i}) \\)  where
\\( \pi \\) is our reference policy) and our goal is to match the reference policy as best as possible.

Behavior cloning is a natural objective to use because it allows us to learn from demonstration data, but there are two techniques that are crucial to its modern success.

## Action Chunking

The impact of [ALOHA](https://arxiv.org/pdf/2304.13705) can be felt throughout the robot learning space.
While it has two major contributions, the relevant one for our purposes is the demonstration that _action chunking_ combined with behavior cloning produces better policies than behavior cloning alone.

Under standard behavior cloning, policies consume one observation and then predict one action.
Under _action chunking_, policies consume one observation and then predict _multiple actions_.
Naively executing these policies results in jerky motions, so the paper proposes a fix for this called _temporal ensembling_.[^RTC]
Under temporal ensembling, at each timestep $t$ we sample the next $K$ actions. Instead of executing the current action at each step, we execute an exponential moving average of the past $K$ predictions for this timestep.

![Image from Zhao et al. describing the action chunking inference scheme]({{ '/assets/images/vla/act.png' | relative_url }})

It's still somewhat mysterious why action chunking works so well.
It was originally thought that by training policies to predict chunks, we reduce the horizon over which errors can accumulate.
But it was later noticed that this alone wasn't enough to explain the superior performance of the technique.
The [best investigation](https://arxiv.org/pdf/2608.02547) into this question that I know of argues that instead, it's the _ensembling_ of actions predicted using multiple observations that helps reduce compounding errors, often _more so_ than the chunking!

## Diffusion Policy

[Diffusion Policy](https://arxiv.org/pdf/2303.04137) is another groundbreaking paper that solved a central challenge at the time: What action should my policy pick when there are multiple good options?

Usually, we model actions as continuous $d$-dimensional vectors, so our policy must generate a continuous distribution over these $d$-dimensional vectors.
This is non-trivial, since the space of possible outputs is infinite!

The standard solution for a while was to predict a single action with a regression loss[^3], or model the action distribution with a single Gaussian. A Gaussian can change its mean and variance, but it always has just one peak.

This works well when good actions cluster around one choice. But when there are multiple distinct choices for the _same observation_, squared-error regression learns their _average_, which can be an invalid action. Imagine reaching around an obstacle: going left or right might work, but averaging the two sends the gripper straight into it! 

<figure>
  <img src="{{ '/assets/images/vla/bc-mean-trajectory.svg' | relative_url }}" alt="Two panels show demonstration paths from a start dot to a goal star around a round obstacle. When every demonstration goes over the obstacle, their average also goes over it. When half go over and half go under, their average runs straight into the obstacle." loading="lazy" width="1040" height="510" style="width: 100%; height: auto;">
  <figcaption class="image-caption">A toy example with a fixed start and goal. Blue lines are demonstrations; the dashed orange line is their pointwise average, which is what a squared-error policy learns to output. With one way around the obstacle, the average is a fine path. With two, it splits the difference and drives straight into the obstacle.</figcaption>
</figure>

Instead, Diffusion Policy leverages the fact that diffusion models[^4] can learn complex continuous distributions given only data sampled from that distribution.
So instead of trying to always pick the single correct action, we _sample_ one of the good actions from our policy's action distribution!

[Flow matching](https://arxiv.org/abs/2210.02747)[^7], similar to diffusion, also learns distributions from data. Flow matching starts with samples of standard Gaussian noise, then uses the model to repeatedly push them toward the target distribution. As those samples move, their distribution can develop multiple peaks. At the end, we have a sample from the learned action distribution.

So in summary:
- Behavior cloning is the natural objective when we want to learn policies from demonstration data.
- Naively predicting the next action based on the current observation results in brittle policies.
- There isn't strong consensus about why action chunking works, but empirically it provides strong gains.
- Action distributions can be _multimodal_. Squared-error regression averages distinct choices, potentially producing invalid actions. **Diffusion and flow matching** let us model multiple modes and sample actions from the learned distribution.

# Vision Language Action Models

Vision-language-action models (VLAs) are the dominant paradigm in generalist robotics.
For a while, they seemed to be the _only_ relevant architecture in generalist robotics.
As such, there isn't much work rigorously comparing them to other model classes.
So I can't claim that VLAs are the _best_ architecture we have for generalist robot learning, just that they're the _default_.

Recent research has proposed a new class of models -- world action models (WAMs) -- as an alternative. We'll talk about these later.

To motivate VLAs, let's take a look at the architecture of Octo, one of the earliest generalist policies and one of the last major generalist non-VLA policies.
At its core, Octo is a transformer that takes in a description of the current goal (via an image showing the desired end state or a text prompt telling the model what to do) and a sequence of observations (images in Octo's case), both encoded as tokens.

To turn images and text into tokens, Octo uses two separate encoders.
A pretrained language encoder produces language tokens, and a learned image encoder produces image tokens.

![Visualization of the Octo architecture]({{ '/assets/images/vla/octo.png' | relative_url }})

Now, what if instead of using separate encoders for language and vision so that the _policy_ has to learn to connect them (i.e. understand that when we refer to an apple, we mean 🍎), we used a pretrained encoder that has _already_ connected language and vision?

Vision-language models (VLMs) are large language models trained to predict text conditioned on images. They fit our encoder description almost perfectly.

This is the key idea behind VLAs.

\\(\pi_0 \\) (the canonical VLA) has two major components: a pretrained vision-language model (PaliGemma-3b) and an action head[^9].

<figure>
  <img src="{{ '/assets/images/vla/vla-attention-gradients.svg' | relative_url }}" alt="Action-transformer queries attend to keys and values from VLM hidden states at successive layers. Dashed gold arrows trace gradients from the flow-matching behavior-cloning loss back through the action transformer and attention into the VLM." loading="lazy" width="1000" height="660" style="width: 100%; height: auto;">
  <figcaption class="image-caption">Gray arrows show forward computation; dashed gold arrows show a training-gradient path. The action expert reads the VLM's hidden states, and action supervision trains both. This is a schematic of the information flow: \(\pi_0 \) implements these connections with shared masked attention.</figcaption>
</figure>

The VLM takes in a task prompt and images corresponding to the current observations. During inference, it does not generate text. Instead, the action transformer attends to the VLM's intermediate hidden states, while also conditioning on proprioception. The action head learns the action distribution via flow matching.

The flow-matching loss is the behavior-cloning objective. Its gradients propagate through the action head and attention connections into the VLM, encouraging its representations to be more useful for predicting actions.

## Training VLAs

Since the release of \\(\pi_0 \\) in 2024, we've learned a bit more about how to best train VLAs, and I want to mention a few of the most impactful parts of the updated recipe.

### Co-training

Available robot datasets are many orders of magnitude smaller than the pretraining datasets used for LLMs. It's also costly to spend time teleoperating and collecting more data.
As a result, there's been a substantial amount of effort to figure out how to learn good policies from alternative sources of data that are easier to collect.

Training on auxiliary data to improve robot performance is known as _co-training_.

There's strong evidence that your co-training data should include a mix of standard vision-language data and densely annotated human and robot video, and growing evidence that retargeted human egocentric data is of value.

The Toyota Research Institute has a great paper studying this exact question. Like the earlier TRI study, [A Systematic Study of Data Modalities and Strategies for Co-training Large Behavior Models for Robot Manipulation](https://arxiv.org/pdf/2602.01067v1) is _fantastic_. In it, the researchers try multiple co-training methods from the literature. They only find strong support for the following[^cotraining]:
1. Standard vision-language data: Data consisting of an image, some question about its contents, and an answer.
2. Human and robot video annotated by a vision-language model: A language model is given a video and asked to describe in detail what's happening in it. This can be seen as standard vision-language data with the question "describe this video".


Notably, neither form of data improves _in-distribution_ manipulation performance! Their benefit instead comes from improving performance _out of distribution_!

<!-- maybe an example of both forms -->
![Results from Toyota Research Institute: success rates for policies co-trained on each useful data modality, compared with a robot-data-only baseline. In simulation (top), co-training leaves performance on seen tasks unchanged but improves it under distribution shift (DS) and on unseen tasks. Real-world results (bottom) measure language following. Letters mark statistically distinguishable groups: bars that share a letter aren't significantly different.]({{ '/assets/images/vla/tri-cotraining-modalities.png' | relative_url }})

We get some validation of their findings from the [\\( \pi_{0.5} \\)](https://arxiv.org/pdf/2504.16054) report. Physical Intelligence finds that a mix of data that TRI would call "standard vision-language data" (Physical Intelligence calls it multimodal web data) greatly improves out-of-distribution success rates.

While there's less rigorous empirical support, there's been a substantial wave of interest in utilizing human egocentric data to improve policy performance.
Typically in egocentric pipelines, humans wear a camera rig that records their first-person view.
We later apply hand pose estimation models to get "actions" from these human videos. 

<figure>
  <img src="{{ '/assets/images/vla/ego-pipeline.svg' | relative_url }}" alt="A person wears a head-mounted camera that records first-person video of their hands. Hand pose estimation labels 3D wrist and finger keypoints in each frame, and the resulting hand trajectory, optionally retargeted to the robot hand, becomes the action label. Video frames and actions form observation-action pairs co-trained with robot data." loading="lazy" width="1000" height="560" style="width: 100%; height: auto;">
  <figcaption class="image-caption">A typical egocentric pipeline. Video frames become observations and estimated hand poses become actions. Some pipelines, like EgoScale, retarget the hand poses to the robot hand before training. Photos from <a href="https://arxiv.org/pdf/2602.16710" style="color: inherit;">EgoScale</a>.</figcaption>
</figure>

[EgoScale](https://arxiv.org/pdf/2602.16710) applies this pipeline to _thousands_ of hours of human video, and finds that at this scale, there's a clear benefit to scaling diverse human data -- with a few caveats.
1. As in the pipeline above, they estimate hand poses to extract actions from human video. They then _retarget_ the human hand poses to the robot hand space, trying to find the robot pose that best "matches" the human pose. It's this retargeted hand pose they train on.
2. They include EgoDex, a dataset captured with an Apple Vision Pro, which has built-in sensors that can label joints with better precision than images alone. They never ablate this addition. This is about 4% of their dataset.

Additionally, they find that adding human data that "matched" robot data substantially improved performance. 
By "matched", they mean that the human data was collected in the same environment under the same conditions as the robot data, and the human was performing the same task as the robot.
The matched human data they collected didn't use the retargeting pipeline used for the pretraining data. Instead, they used hardware to measure joint positions precisely and matched the camera setup to that of their target embodiment.


![EgoScale's scaling results. Left: human validation loss during training for 1k to 20k hours of human pretraining data. Center: the best validation loss falls log-linearly with data. Right: average task completion after fine-tuning rises from 0.30 to 0.71 as human data grows.]({{ '/assets/images/vla/egoscale-scaling.png' | relative_url }})

![EgoScale's main results on five dexterous tasks: task completion (left) and success rate (right) with no pretraining, mid-training only, human pretraining, and human pretraining plus mid-training. The matched data corresponds to the mid-training stage of this plot.]({{ '/assets/images/vla/egoscale-results.png' | relative_url }})

Their results are promising and scary at the same time. On the one hand, human pretraining greatly accelerates subsequent fine-tuning. On the other hand, the scale needed to even get to 50% success rates is mind-boggling.


### Gradient Management

Multiple groups have independently noticed that naively allowing action gradients to flow through cross-attention from the randomly initialized action head results in worse performance than more restrictive forms of information flow, such as conditioning the action head on the last token's hidden state and allowing gradients to flow only through that state[^bottleneck].

[Knowledge Insulation](https://arxiv.org/pdf/2505.23705v1), proposed by _Driess et al._, is Physical Intelligence's solution to this problem. They propose a simple fix: don't let gradients flow from the action head into the VLM at all!

This introduces a new problem: the VLM is no longer being trained to produce representations that are useful for predicting actions. To fix this, they propose the following training process:
1. Train the VLM on a mixture of standard vision-language data and action data.
2. When training on action data, have the VLM predict discrete action tokens[^dtoken] and have the action head generate continuous actions via flow matching based on the VLM's hidden states. This way, the VLM is still being trained to produce representations useful for predicting actions, but the action head isn't allowed to backpropagate into the VLM.

Driess et al. show that policies trained with knowledge insulation perform better than those trained with naive gradient flow _and_ generalize substantially better to out-of-distribution tasks.


## World Action Models
I'd be remiss not to at least mention the primary alternative to VLAs -- world action models.

The common thread between models that call themselves WAMs is that they rely on a _video-prediction backbone_.
Where VLAs use a VLM trained on vision and language data, WAMs use a model trained to predict future video frames given past video frames.
How this video-prediction backbone is used varies between models.

The intuition is that video is a much richer signal than static images and language. If we can learn a model that predicts what future actions will _look like_, we should be able to use this model to extract what future actions will _be_.

[DreamZero](https://arxiv.org/pdf/2602.15922) is the most famous example of this class of models. They take a pretrained Wan 2.1 14B model and fine-tune it to predict both future frames from the robot's cameras and the next action based on past observations and proprioceptive state.

<figure>
  <img src="{{ '/assets/images/vla/dreamzero-pipeline.svg' | relative_url }}" alt="Past camera frames, a language instruction, and the robot state are encoded and fed to a pretrained video diffusion model, Wan 2.1 14B. The model jointly denoises future video frames and an action chunk. The robot executes the action chunk, and the real frames it records replace the predicted frames in the model's KV cache." loading="lazy" width="1000" height="490" style="width: 100%; height: auto;">
  <figcaption class="image-caption">A simplified view of DreamZero. Future frames and the action chunk are denoised together with a single flow-matching objective. After each chunk is executed, real camera frames replace the predicted ones in the KV cache, so prediction errors don't compound. The frames shown are real DreamZero generations from the <a href="https://arxiv.org/pdf/2602.15922" style="color: inherit;">paper</a>.</figcaption>
</figure>

On the reported benchmarks, DreamZero outperforms \\(\pi_{0.5} \\) on a small set of manipulation tasks, but a major issue is the compute needed to train and run the model. Even with their optimizations, DreamZero takes about 350 ms to sample a chunk of 48 actions on 2 GB200(!) GPUs.
For comparison, \\(\pi_{0} \\) reports 73 ms for sampling a chunk of 50 actions on a consumer 4090 GPU.[^fastwam]

There is work to be done rigorously comparing VLAs and WAMs, but there seems to be growing adoption of WAMs in industry (see [Rhoda](https://www.rhoda.ai/), [Dyna](https://www.dyna.co/), [1X](https://www.1x.tech/)). I'd keep my eyes peeled for more research in this area.


In summary:
- The default architecture for most generalist policies is the **vision-language-action** model.
- Since VLMs are pretrained to understand both vision and language, VLAs use them as a backbone to turn this understanding into actions.
- A few subtle training choices help VLAs train well and make the most of your data.
- World action models are a promising alternative to VLAs, but they require more compute and there isn't much research comparing them to VLAs yet.

# Limitations of the Recipe

While we have abundant evidence that it's possible to pretrain robotic policies that outperform narrow policies, we don't have much evidence for various forms of _generalization_.

In this section, we try to find out what we can say about generalist policy performance *out of distribution*.

## Environment & Object Generalization

Environment and object generalization -- the ability to perform the same task in a new environment or with different objects -- is what people commonly refer to when they mention generalization. 

This is probably also the best-studied form of generalization. As mentioned earlier, the Toyota Research Institute has a [great paper](https://arxiv.org/pdf/2507.05331v1) studying the effects of generalist training on this type of generalization. To reiterate: Generalist policies are more robust to environment and object shifts than narrow policies (see the TRI results in the [Robot Data](#robot-data) section).

Though generalization improves over the baseline, these policies aren't reliable enough to deploy zero-shot in new environments and expect reliability of 99%, 99.9%, or higher.

Again, we _do_ have evidence that by scaling diverse data (via [Lin et al.](https://arxiv.org/pdf/2410.18647)), zero-shot generalization to new objects can be improved, but strong generalization remains to be seen in any open policy.

## Compositional Generalization

One capability we might want from a generalist policy is the ability to _compose_ atomic skills learned in pretraining or fine-tuning.

If we know how to open drawers and place things in open drawers, we should be able to open drawers *and then* place things in them, without ever seeing this combination in training.


[RoboHiMan](https://arxiv.org/pdf/2510.13149) sets out to investigate this question by constructing a hierarchy of four datasets to train on.
1. The first dataset consists of demonstrations of all the atomic skills they plan to evaluate, collected in roughly the same setting.
2. The second dataset adds _perturbed_ versions of the atomic skills to the first dataset, matching the lighting, objects, and environments used during evaluation.
3. This dataset adds some of the composite skills they evaluate on, collected in roughly the same setting.
4. This final dataset adds those same composite skills but collected under perturbed conditions.

They then evaluate success rates on atomic skills and composite skills.

The paper finds that policies trained under our recipe (\\(\pi_0 \\)) using datasets L1 or L2 can't learn to compose skills like those demonstrated in L3 and L4.

<figure>
  <img src="{{ '/assets/images/vla/robohiman-results.svg' | relative_url }}" alt="Two bar charts of success rate by training dataset L1 to L4, for pi0 and pi0.5 without a planner. Atomic-task success stays between 8 and 17 percent. Composite-task success is 0 percent at L1, L2 and L3, and under 1 percent at L4." loading="lazy" width="1040" height="540" style="width: 100%; height: auto;">
  <figcaption class="image-caption">End-to-end ("Vanilla") \(\pi_0 \) and \(\pi_{0.5} \) on RoboHiMan, from the appendix of <a href="https://arxiv.org/pdf/2510.13149" style="color: inherit;">the paper</a>. Each task type includes its perturbed variants. L3 and L4 add composite-task demonstrations, yet composite success never passes 1%.</figcaption>
</figure>

You'll notice that it also doesn't seem like the policies learned the composite skills _even when explicitly trained on them_. This raises the question: how much of the observed failure was due to poor policy performance versus a failure to compose?

[Atom-Bench](https://arxiv.org/pdf/2606.16826) uses a setup similar to RoboHiMan's: they have a set of atomic skills and composite skills built up from those atomic skills. They collect demonstrations on the atomic skills and attempt to see if the model can complete the composite skills zero-shot.


Similar to RoboHiMan, they find that the models can't compose skills zero-shot. In contrast with RoboHiMan, they come up with a metric to analyze what fraction of failures is due to poor learning versus a failure to compose.

They give each atomic skill a scoring rubric to reward the policy for partial progress. For example, a simple task like "pick up the fork and place it in the plate" might give one point for picking up the fork, and another for placing the fork on the plate.

Since all composite skills they test are made up of atomic skills, they can also combine these atomic rubrics into a composite rubric.

Now, in theory, if the policy had no issues composing skills, then the average score on a composite task should be the sum of the scores we get when we execute each of the atomic subskills in isolation. 

To measure the share of failure attributable to composition, they introduce the metric _Compositional Failure Share_ (CFS). CFS is 0% when the average composite score is identical to the sum of the average atomic scores, and 100% when all of the failure is due to composition, i.e. every atomic skill succeeds on its own but the composite task still falls short.

It's worth pointing out that CFS also depends on atomic ability. If a policy is really bad at the atomic skills, then it will have a low CFS because it fails to complete the atomic skills in isolation.

<figure>
  <img src="{{ '/assets/images/vla/atom-bench-cfs.svg' | relative_url }}" alt="Dumbbell chart of atomic score versus composite-task score for five policies on two robot platforms, with compositional failure share printed beside each. Franka Panda (atomic, composite, CFS): pi0.5 83%, 30%, 74%; Motus 69%, 27%, 49%; LingBot-VLA 60%, 12%, 54%; GR00T N1.6 66%, 12%, 61%; SmolVLA 33%, 7%, 28%. Cobot Magic: pi0.5 80%, 43%, 57%; Motus 69%, 32%, 48%; LingBot-VLA 47%, 24%, 29%; GR00T N1.6 48%, 14%, 39%; SmolVLA 33%, 6%, 27%." loading="lazy" width="1040" height="540" style="width: 100%; height: auto;">
  <figcaption class="image-caption">Atomic score (the average progress on a composite task's atomic skills when run on their own) against progress on the composite task itself, with Compositional Failure Share on the right. Data from <a href="https://arxiv.org/pdf/2606.16826" style="color: inherit;">ATOM-Bench</a>. Higher CFS means more of the composite-task failure comes from composing skills the policy can already execute in isolation. Policies with weak atomic scores, like SmolVLA, have little failure left over to blame on composition.</figcaption>
</figure>

As you see in the plot, the policies trained under our recipe have pretty high compositional failure shares, and \\(\pi_{0.5} \\) has the highest compositional failure share on both robots!

## Language Generalization

We now want to ask the question: Does the policy actually listen to what we tell it?

There aren't many comprehensive studies of language following alone, and most benchmarks fold it into a broader suite of robustness tests. The cleanest real-world experiment I know of comes from the [\\( \pi_{0.5} \\)](https://arxiv.org/pdf/2504.16054) report. The robot is placed in an unseen kitchen in front of five objects and told to move one of them, e.g. "put the scissors in the drawer". The target is always placed further away than the distractors, so a policy that ignores its instruction should grab the right object only about 20% of the time. They then measure the _follow rate_: how often the robot goes for the object it was asked for.

![Language following results from Figure 15 of the \\( \pi_{0.5} \\) report. Follow rate measures how often the robot goes for the object it was asked for; success rate also requires placing it correctly. \\( \pi_0 \\)-FAST+Flow is trained on the same robot data as \\( \pi_{0.5} \\), but without web data. A policy that ignores its instruction would have a follow rate of about 20%.]({{ '/assets/images/vla/pi05-language-following.png' | relative_url }})

With enough training, \\( \pi_0 \\) does no better than a policy that ignores language entirely, and it gets _worse_ the longer it trains!

Physical Intelligence attributes this gap to how the models are trained, not just what they're trained on. \\( \pi_0 \\)-FAST+Flow sees no web data, but like \\( \pi_{0.5} \\) it's trained with an early version of [Knowledge Insulation](https://arxiv.org/pdf/2505.23705v1): the VLM first learns to predict discrete action tokens, and the freshly initialized action head is only added at the end of training (without the stop-gradient). \\( \pi_0 \\), on the other hand, trains its action head from the start. Most of the gap is already closed by this change alone. Web data plays a smaller role. Removing it leaves the follow rate on familiar objects unchanged (about 86%), and lowers it on object categories never seen in robot data from about 94% to about 80%, though the error bars overlap.

Note that even \\( \pi_{0.5} \\), the best model in the comparison, still picks the wrong object roughly 14–25% of the time in a scene with only five candidates.

TRI made a similar observation in its study. From Appendix 10 of [A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation](https://arxiv.org/pdf/2507.05331v1):

> When inspecting single-task policies’ rollouts, the robot always performs the correct task but often misses the target manipuland or is unable to place it correctly. Pretrained and finetuned LBM have two common main failure modes: 1) the robot is unable to move away from its starting pose, as seen in Figure S13; 2) the robot performs a different task, as seen in Figures S11 and S12. In both cases, most rollouts end in failure. The finetuned LBM performs better than the pretrained LBM by staying static less often, and by performing the correct task more often.

So to summarize the limitations of our recipe:
- Generalist policies are more robust to new environments and objects than narrow policies, but not robust enough to deploy zero-shot without fine-tuning.
- Policies can't **compose** atomic skills they already know into new tasks, and a large share of their failures on composite tasks comes from composition rather than from executing the individual skills.
- Policies often ignore their language instructions. In a real-world test where ignoring the instruction yields only chance-level performance, \\( \pi_0 \\) picks the right object no more often than chance, and even \\( \pi_{0.5} \\) regularly picks the wrong object.
- Overall, pretraining buys robustness near the training distribution, but there's little evidence yet that it buys true out-of-distribution generalization.


# Conclusions 

So where does that leave us? Let's revisit the recipe from the start of the post, now with everything we've learned:
1. **Collect as much robot data as you can.** Target-embodiment data on your target tasks is best, but diverse data from other embodiments still helps, especially for robustness to distribution shift. Represent actions in a way that transfers across embodiments (delta end-effector frame is a good default), and be thoughtful about how you weight your data mixture.
2. **Take a pretrained VLM and attach an action expert.** This is the _default_, not a proven best choice. World action models are a serious contender, and we don't yet have rigorous comparisons between the two.
3. **Train with a behavior cloning objective.** Use action chunking, and model the action distribution with flow matching (or diffusion) so your policy can handle multiple good options. Co-train on vision-language data to help out of distribution, and be careful about letting gradients from a freshly initialized action head flow into your VLM.

If you follow this recipe, you should expect a policy that is more robust than a narrow policy and much cheaper to fine-tune on new tasks.
What you _shouldn't_ expect is a policy that works zero-shot in new environments, composes skills it already knows into new tasks, or reliably listens to what you tell it to do.

These limitations are, to me, the most exciting part. They point to open questions that are still within reach for academia:
- How should we choose what data to train on?
- Are VLAs or WAMs the better foundation for a generalist policy?
- And most importantly, what would it take for a policy to actually _compose_ its skills and _follow_ its instructions?

I hope to tackle some of these in future posts in this series.

There's much I didn't get to cover in this post, and I encourage you to read the papers I linked to for more details.

Please let me know if you find any errors or have any questions! Thanks for reading! :)

[^1]: They collect data using the Universal Manipulation Interface, so the data isn't _technically_ collected on the target embodiment, but UMI data looks enough like target data that we'll give it a pass.

[^2]: Emphasis on _diverse_. A key finding of Lin et al. is that smaller, more diverse datasets can outperform larger, more uniform ones.

[^3]: This is done by trying to minimize the \\(\text{L}_2\\) loss $\lVert\mathbf{a}_t - \hat{\mathbf{a}}_t\rVert^2$.

[^4]: I won't cover what diffusion models are or why they can do this, but understanding them is important for implementation. Please refer to this amazing work at [MIT](https://diffusion.csail.mit.edu/2026/) for an introduction.

[^5]: State $s_t$ is usually a collection of images taken from cameras attached to the robot + some info about the robot's current state. Currently, in robot learning, it's not common to (explicitly) predict actions based on a history of observations.


[^7]: Flow matching and diffusion are deeply related. While diffusion was popularized first, flow matching is much simpler conceptually and easier to implement. Most modern generalist robotics policies are trained with some variant of flow-matching loss.


[^9]: While most VLAs are some variant of the \\(\pi_0 \\) architecture, \\(\pi_0 \\) wasn't the first VLA. [RT-2](https://arxiv.org/pdf/2307.15818) and [OpenVLA](https://openvla.github.io/) both came earlier and don't use an action head. Instead, they teach the VLM to predict actions directly.

[^angles]: Note that some policies like \\(\pi_0 \\) train on multiple action representations (i.e. both relative end-effector and absolute joint angle)! This seems to work fine for them, but the paper argues that when possible, relative end-effector frame is the best single choice for training on diverse embodiments.

[^bottleneck]: The appendix of [A Systematic Study of Data Modalities and Strategies for Co-training Large Behavior Models for Robot Manipulation](https://arxiv.org/pdf/2602.01067v1) finds that conditioning the action head on the last token of the VLM's hidden states and only allowing gradients to flow through that token produces better results than allowing gradients to flow through cross-attention on hidden states.

[^dtoken]: The Knowledge Insulation paper uses an action discretization scheme known as [FAST](https://arxiv.org/pdf/2501.09747). Training with FAST tokens is often faster than training flow-matching policies, but flow matching is often faster to use at inference time.

[^fastwam]: There's also work like [FastWAM](https://arxiv.org/pdf/2603.16666) that tests variants of WAMs that are more inference-efficient while maintaining performance on simple tasks. They show that some architectures can get significant speedups over the style of WAM presented in DreamZero.

[^industry]: You might notice that a few industry lab reports are missing from this post. This is intentional. These companies have odd incentives to publish, and their audience is often investors, not researchers. As a result, they often don't provide enough information to compare their results to other methods in the literature.

[^cotraining]: The paper also finds "cross embodiment robot data" useful as a co-training modality. This is in line with the results of the [data](#robot-data) section, so I omit discussion of that here.

[^RTC]: Temporal ensembling works well as an inference scheme when your policy produces chunks faster than your sampling rate, but for modern large generalist policies this breaks down. [Real Time Chunking](https://arxiv.org/pdf/2506.07339) proposes an inference scheme that works when your policy is slower than your sampling rate. [VLASH](https://arxiv.org/pdf/2512.01031) is another great paper on asynchronous inference that allows for real-time control. I won't cover these here, but they're great papers and these details are important when evaluating these policies.
