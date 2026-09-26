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
So before the frontier pulls away from open knowledge, I think now is a great time to try to answer the question: **What is the best way to pretrain a generalist robotics policy for manipulation?**

I'm going to spend the rest of this blog post convincing you that the literature currently supports the following recipe:
1. Collect as much robot data as you can, obviously it's ideal to have a ton of target embodiment data, but even data from other embodiments will help.
2. Assemble a mix of visual reasoning data and finetune a pretrained vision language model (VLM).
3. Take the model generated in the last step and add an action transformer: this action transformer will use _flow matching_ to generate _action chunks_ conditioned on the hidden states of the VLM via cross-attention.
4. Now using flow matching, train the architecture above with a Behavior Cloning objective on the dataset from step 1.

I'll then discuss pathologies of our current method, and the open research directions trying to address them.

# Glossary

Definitions of the most common jargon you'll see throughout this post.

**Policy**: The "brain" of the robot. This is a function that takes in some inputs (e.g. camera data, a text prompt, the current position of the robot) and produces robot actions.
The policy is said to produce an action **conditioned** on it's input.

**Task Specific Policy**: Also referred to as narrow policies, these policies are trained to perform a single task. While they might be able to do this task robustly in many different environments, they aren't expected to perform any other tasks.

**Generalist Policy**: In contrast to narrow policies, these policies are trained to perform many different tasks across manipulation.

**Training Distribution**: The space of tasks, objects, and environments the policy was trained on. We'll often refer to tasks as being _in distribution_ or _out of distribution_. This typically refers to wheher or not the task was seen during training.
We might also claim to evaluate a policy under **Distribution Shift**. 
This just refers to when the conditions we evaluate under (like environment, object appearence, instruction text) are different than those that were seen during training.

**Fine-Tuning**: We'll often fine-tune generalist policies on datasets used to train narrow policies. This just means that we'll do some extra training on the generalist policy to improve performance on a specific task.

**Zero-shot Evaluation**: When we evaluate a generalist policy on a task without collecting data and finetuning on that task first, we'll often call this _zero-shot evaluation_. All this means is that we're measuring the performance of the policy out of the box.

# Robot Data

Before we talk more about data, we should note that how we plan to deploy our policy potentially has a huge impact on the data we'd want to train on.  
Imagine we only cared about our policy stacking blocks, we might make very different data decisions compared to the company that only cares about policies folding clothes!

Additionally, how we want to condition (i.e. prompt) our policy impacts our decisions as well.
If we want to use text to ask our policies to do different things, they're going to need examples of that in their training data!

For the sake of our discussion, let's assume we know what we want our robot to do, and we know the hardware we want the robot to do it on.
We'll refer to what we want the robot to do as the _target task_(s), and the robot body we want to deploy our policy on as the _target embodiment_.

In [Data Scaling Laws In Imitation Learning For Robotic Manipulation](https://arxiv.org/pdf/2410.18647) _Lin et al._ show that if you scale target task data in diverse environments on the target embodiment[^1], you get a robust single-task policy that can execute the target task at high sucess rates (90%) in completely unseen environments!

<figure>
  <video controls playsinline preload="metadata" style="width: 100%;">
    <source src="{{ '/assets/images/vla/datascalingvideo.mp4' | relative_url }}?v=faststart" type="video/mp4">
    Your browser does not support embedded video. <a href="{{ '/assets/images/vla/datascalingvideo.mp4' | relative_url }}">Watch the data scaling video</a>.
  </video>
  <figcaption style="font-size: 0.85em; color: #777;">Demo from <a href="https://arxiv.org/pdf/2410.18647" style="color: inherit;">Lin et al.</a></figcaption>
</figure>

This is fantastic! We've solved robotics! Sure these policies aren't generalists, but I'm confident we just need to collect a large diverse high quality demonstration dataset across every task we could ever think to use robots for!

The only problem is that this is insanely expensive.

We're in _academia_! I'm an undergrad! Where am I going to find the time to record millions of demonstrations across the gamut of human ability! It's over!

But wait! All hope is not lost!
Over the years, academia has collected thousands of hours of data across various robots! Sure it's not all on our target tasks and it's not on our target embodiment, but it must be better than nothing right?

Right! There's a long line of evidence showing that training policies on large diverse robot datasts helps makes policies better across various axes!

[Open X-Embodiment](https://arxiv.org/pdf/2310.08864v5) was a huge effort across various academic institutions to create a large open source dataset for training generalist policies. 
They then tried to answer the question: _Does training on diverse robotics data across various embodiments improve task performance over training on narrow task specific datasets?_ 
For tasks with small scale datasets, the results are a resounding yes! Almost always, their generalist policy out performed methods trained on just the targeted data.
For tasks with larger datasets, the results were mixed. When the generalist policy was matched against a policy with the same architecture but trained only on the target task, the single task policy usually ended up on top. They posited that the architecture was too small to absorb the extra data and showed that a larger architecture trained on the the full dataset did end outperforming the smaller architecture trained on more targeted data.

[Octo](https://arxiv.org/pdf/2405.12213) Went further and trained on a larger split of OXE which they cleaned and filtered. While they found that the resulting policy performed better than the policies trained in OXE out of the box, they also show that fine-tuning Octo to perform a new task on a new embodiment is often much more data-efficent than training a policy from scratch! 

[Pi0](https://arxiv.org/pdf/2410.24164), a much larger model later developed by many of the same people behind Octo, would train on an even larger dataset of diverse manipulation data (~ten thousand hours from online reports! Compare to the one thousand hours for octo) and strongly outperform octo on tasks seen during training! They also claim that pretraining improves data-efficency compared to training from scratch.

![Pi0 task progress after fine-tuning, compared with training from scratch and zero-shot evaluation.]({{ '/assets/images/vla/pi0-complex-tasks.png' | relative_url }})

To investigate these claims, the Large Behavior Model team at the Toyota Research Institute performed a [A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation](https://arxiv.org/pdf/2507.05331v1). They seek to compare the performance of policies like Octo, Pi-0 and those trained in Open X Embodiment to policies trained from scratch. 
What makes this paper notable is that it's incredibly well executed. 
The paper is approachable and easy to read, and their claims are clear and well supported by the experiment results. At the same time, the experiments are done with a level of rigour not often seen in the literature! 
I highly reccomend reading this paper yourself, but some of their key findings are that:
1. Large Behavior Models (their name for generalist policies) do perform similar to narrow polices out of the box when evaluated on tasks seen during pre-training.
2. Finetuning generalist policies does produce a small performance benefit compared to training from scratch when in distribution. But more importantly...
3. Finetuned generalist policies are much more robust to distribution shift than polices trained on smaller target data!

![Results from Toyota Research Institute: success rates for single-task policies, Large Behavior Models, and fine-tuned Large Behavior Models, without distribution shift (top) and with distribution shift (bottom). Focus on the 'aggregate' columns to get a clear picture of the results.]({{ '/assets/images/vla/tri-results.png' | relative_url }})

A subtle detail arises when we consider how to represent robot actions during training. [Demystifying Action Space Design for Robotic Manipulation Policies](https://arxiv.org/pdf/2602.23408) compares four approaches to action represention: absolute joint angle, absolute end effector frame, delta joint angle, delta end effector frame.
- **Absolute joint angle** asks the policy to output absolute joint angles, meaning we precisely specify the pose the robot should be in.
- **Absolute end effector frame** asks the policy to output the position and rotation of the end effector (usually the gripper in manipulation setups) in 3D space. Inverse Kinematics is used to figure out how to move the joints to said space
- **Delta joint angle** asks for the change in joint angle relative to the current pose. We specify how the current pose should change over the next time step
- **Delta end effector frame** asks the policy for the change in the position of the end effector relative to it's current position. Again, inverse kinematics can be used to figure out how to move the joints to the correct pose.

![Absolute commands specify target joint angles or end-effector poses; delta commands specify changes from the current state. These planar examples express position changes in the robot's base frame.]({{ '/assets/images/vla/action-representations-blog.png' | relative_url }})

They find that while delta joint angle produces the best policies when the data comes from a single embodiment, delta end effector frame produces policies that perform much better on diverse embodiments, and adapt well to new ones.

Finally, a common trend amongst robotic manipulation papers that employ large scale training is to train on a re-weighted (meaning some samples have a higher likelihood be being repeated) subset of these large datasets.
While more data helps, we can't avoid the fundamental issue of garbage in, garbage out.

Though most of the dataset weighting reported in papers like pi0 and Octo was done by hand, [Re-Mix](https://arxiv.org/pdf/2408.14037) is a great step in understanding how to systematically re-weight our data, using a min-max based objective to choose dataset weights. They see sizable gains in performance in policies trained on the dataset they evaluated on, though more work is needed.


So to summarize:
- Modern methods can extract high performance on manipulation tasks from demonstration data
- In the absence of a large amount target data, we can leverage non-target robot data to imrpove performance.
- There are various ways we can represent actions, and the method we choose has significant impact on our policies performance.
- Naively training on all the data out there won't necessarily produce the best policy. Various methods have been proposed for data selection, but the best such method is still an open area of research.

[^1]: They use Universial Manipulation Interface to collect data, so they aren't _technically_ on the target embodiment, but UMI data looks enough like target data that we'll give it a pass.

---

# Notes so far
open embodiment x: first massive dataset of labeled robot policies,
enabled the first wave of open research, but has quality issues.

first major sub question: can the deep learning playbook learn good policies from data?
- we have much evidence that: yes this is possible (as early as 2022! examples include Octo, RT-1)
- "good" is a bit loose here. Tasks we're evaluating on are a bit simple, and we have the recurring trend of papers selectively reporting evals and not contstructing evals where their policy fails
    - as a result, it's really hard to measure improvement over time
    - we can try to evaluate models ourselves on Robocasa365 (diverse simple robotics tasks) and RoboDojo (really stress tests generalization + harder tasks)
- RT-2 shows that if we want our policies to be language conditioned, we can *reuse* VLM backbones to improve generaliztion without needing to train much on robotics data
    - This improves over policies like Octo, and their struggle to connect visual cues to language
    - pi0 also does this but not retardedly
        - though the claim stuff like their datamix is harder for openvla than it is for pi0... why? we can't see the datamix... it's over
    - These models show some limited generalization ability
        - Both across language objectives and environments, limited evidence that skills transfer across embodiments
    - We have strong evidence that pretraining improves these models ability to learn from further data
    - These models SUCK at modern benches tho... which have tasks that don't seem that different than the ones in the paper, this is a point of confusion
    - SOTA recpies don't seem to have significantly diverged however, they just have bigger models and better data.
- refs for this section:
    - [octo](https://arxiv.org/pdf/2405.12213)
    - [openvla](https://arxiv.org/pdf/2406.09246)
    - [open-x-embodiment](https://arxiv.org/pdf/2310.08864v5)
    - [pi0](https://arxiv.org/pdf/2410.24164)
    - [rt2](https://arxiv.org/pdf/2307.15818)

next major question: the above gave us a recipie that _seems_ to kind of generalize, but what are the failures that we experience? 
    - first what kind of evidence did we get from the previous papers
        - again hard to really make any claims because evals aren't standardized!
        - octo shows that it gets non-zero generalization within the pretraining distribution when eval'd zero shot + when finetuned octo seems to be more efficent than training from scratch (on about ~100 demos)
        - pi0 then trains octo on their entire dataset, and it wets the bed even in distribution! what's going on!
            - best theory, the tasks that octo trains on are much much easier in the original paper than the tasks pi0 trains on! 
            - so worrying! the conclusions we made above may hold for low complexity tasks, but fall apart for higher complexity ones!
            - openVLA is also ass! the authors attribute this to a) the lack of action chunking b) low frequency control but don't test this!
        - this raises some worrying questions: when training on a new task out of the data distribution and a brand new environment, does FT provide any advantage over PT->FT?
            - pi0 does this experiment! it's results kind of bring context to the "improved finetuning" argument.
                - On the easiest task: Stack Bowls, the task is mostly a subtask of a pretraining task. The pretrained model seems to have a constant advantage over the scratch model. We see similar results with Towel Folding which is also easy (and note that it seems like the pretrained policy overfits less quickly!)  
                - On the next level up, the finetuning is basically a wash, neither policy can claim an advantage.
                - On the hardest level, the pretrained policy get's strcitly worse after being trained on more data and overfits much faster on paper towel replacement, but shows a major advantage over the policy trained from scratch. I'd argue that paper towel replacement uses a skill -- pick and place -- seen commonly in the dataset. While the objects are new, and yes placing the tube on the shaft requires precision, still pick and place
                    - ~0.5 implies that the policy is getting stuck after successfully removing the paper towel roll
                    - ~.75 implies that the policy is struggling to place the new roll on which i'll claim is the new part of the task
                - The hardest task shows exactly no advantage over finetuning from scratch. This is a task not seen at all in the training data on this embodiment, nor (from what I can tell) do any analogs exist in the training dataset.
                - I think we can weaken the claim somewhat: pretrained policies improve finetuning when the tasks that we're finetuning on are _just_ outside of the pretraining distribution, or partially in partially out.
    - Let's test that claim by digging into the literature!
        - The Toytoa Robotic Institute has a great study rigoursly evaluation almost the exact claims of the Pi0 paper on what they call Large Behavior Models
            - These models don't have a VLM backbone, and instead use pretrained image and language encoders for observation features
            - And to be fair, these may be the kind of easy tasks that octo was good at, no idea if these results hold on harder task distributions
            - They have a few interesting results. For instance: They show that without distribution shifts, you see a small, but significant boost to performance when finetuning LBMs. With distribution shifts, the benefit ot finetuning pretrained models becomes much more apparent. 
            - The main result we care about is the "unseen tasks" result. Given new tasks they claim that pretrained LBMs learn better than LBMs trained from scratch
                - Concerningly, they seem to expect no language generalization -- given a new language instruction in a familar scene, they don't expect the policy to be able to execute it
                    - They claim many times that their language encoder is _brittle_, and it's nowhere near the scale of VLMs. 
                    - A common issue is that the pretrained model would often complete the wrong task!
                - I claim that for both sim and real tasks, their results show that finetuning pretrained policies can help with adaptation to new tasks unseen in the datamix
                - But sadly, I think that this doesn't do much to confirm or deny my hypothesis. Since we don't have enough access to the datamix to see what the tasks present are, we can't see how different the tasks are from tasks present in the pretraining data, so we can't test how distance from this distribution changes finetuning performance!
            - As a result of the above, I'm still of the mind that finetuning is much less useful when the tasks are out of the pretraining distribution!
    - Okay, the TRI paper shows that we generalize (better) across new environments and objects on tasks that both were in the pretraining distrubtion and that weren't when training a LBM. 
        - I expect these results to hold for VLAs
        - What about other forms of generalization? Transfering skills across embodiments, composing skills with language, generalizing to rephrasings
            - Compositional Generalization: 
                - RoboHiMan is a relatively recent benchmark that seeks to answer the question: Can VLAs trained on task A and task B perform task A + B?
                    - The answer is kind of no?
                    - When finetuned directly on the compositional tasks in a variety of settings, pi0 + pi0.5 get shit performance
                    - Unsurprisingly, they fail to generalize to longer horizions.
                    - They find that VLA performance improves when another VLM is used as a high level planner
                    - Other interesting results we won't get into
                - Atom-Bench is another great paper that tests this question:
                    - They find that VLAs often get decent at performing the single tasks, but suck ass at composing skills together
                    - The even introduce a new metric to test how much of compositional task failure is due do poor skill on each atom (most is due to failure to compose)
                - I think it's reasonable to claim that with current methods and at current scales, we shouldn't expect compositional generalization 
            - Embodiment Generalization:
                - This question, whether the learning of skills on one embodiment transfers to learning on antother embodiment is hard to answer, few papers seeks to measure it directly.
                - LAP introduces a new method to imrpove zero-shot embodiment generalizsation and in so doing has to answer a weaker form of the question: does the learning of skills on one embodimetn transfer zero shot to another embodiment
                    - They introduce a method which they claim improves zero-shot transfer
                    - But compare against pi0 and pi0.5 as a baseline, so we can use this to kind of answer our original question
                    - They train on OXE + MolomoAct 
                    - TODO: analyse their data and see if it's reasonable to expect generalization
                    - We don't see _zero_ generalization, but we don't see very high performance either
                    - While Zero-Shot evaluation is bad, finetuning may change the question? Results are more mixed here.
            - Language Generalization: 
                - LIBERO-Plus shows that policies aren't super robust to rephrasings of the task. They find that a lot of the robustness that does exist can be explained by policies largly not relying on the language input to complete the task
                - LIBERO-X has a difficulty level that includes language generalization, and we see a non-negliglble drop in performance
                - Take it with a grain of salt as even though these are improved, LIBERO is a pretty shit benchmark to start off of
                - Conclusion: VLAs show some degree of robustness to langauge rephrasing, but not a ton
    - refs for this section:
        - [TRI LBM Paper](https://arxiv.org/pdf/2507.05331v1)
        - [RoboHiMan](https://arxiv.org/pdf/2510.13149)
        - [Atom-Bench](https://arxiv.org/pdf/2606.16826)
        - [LAP](https://arxiv.org/pdf/2602.10556)
        - [LIBERO-X](https://arxiv.org/pdf/2602.06556)
        - [LIBERO-Plus](https://arxiv.org/pdf/2510.13626)
    - Conclusion: At the current scale of our policies, we see limited amounts of generalization across various axes.
        - This seems like a lot of work to establish something pretty standard, but value here is getting an intution for what the current recipie can and cannot do at the current scale.
- 2025-26 brough a wave of new models!
    - models here were selected based on relevance at release, or things about them which I thought were noteworthy
    - We're going to report results on modern benchmarks, these papers evals suck
    - pi0.5, 0.6, 0.7
        - pi0.5: At a high level, the main contribution of 0.5 over 0 was 1) the scaling of pre-training datasets to incldue much more diverse scenes and task types 2) the introduction of high level goal prediction to improve performance over long tasks
            - the pretraining dataset is targeted to the task distribution we care about including a substantial amount (400hrs) of data on the embodiment we care about, and more data (how much more isn't specified) on embodiments we don't care about, and a substantial fraction of data on tasks that are less relevent on various embodiments we don't evaluate on.
                - ALL of these splits are vital for the performance pi0 gets
                - Notably, the robot's manipulation performance drops when robot data is excluded, but it's generalization performance drops when web data is excluded
                    - Implies model of manipulation skills are learned from robot data while the generalization of these skills is in large part driven by the VLA?
                - Both types of robot data missing will cause a drop, but the total drop of them both missing is smaller than the sum of the drops of each of their individual drops.
                    - Implies model of gains from pretraining data are proportional to the amount of pretraining data already collected?
                    - Or maybe "effective" dataset size when diversity is accounted for is smaller than the actual size.
            - High Level Prediction seems effective for improving perf @ first glance, but further inspection shows that most of the gains for high level perf can be attributed to a single post training dataset(!).
            - Diverse post training data improves performance when paired with sufficently diverse pretraining data
            - oddly pi0.5 performs really poorly on robocasa365 which should be a slam dunk?
                - should probably re-run
        - pi0.6: I don't know how I feel reporting on this. The pi0.6 paper is really about the RL method that they used to post train pi0.6, with minimal attention put on pi0.6. From the scant details they give about pi0.6, it looks like a scale up of both the model and data. The pretrain acheives a decent performance boost over pi0.5, though those gains are eclisped by the post training method introduced. 
    - gr00t series
        - n1: main depature from pi series is architecture (they have embodiment specific action encoder and decoders) and they include a massive amount of non robot data from various sources heavily leveraging co-training 
            - OXE, a humanoid dataset, and AgiBot-Alpha (OXE style dataset over various tasks)
            - They create various synthetic datasets to train on as well
                - Simualtion rearranging tasks scaled up with dexmimic-gen
                - Generated Image to video data labeled with latent actions (Neural Trajectories)
                - Human video labeled with latent actions
            - no real ablations for these datasets
        - n1.5: free the vlm during pretraining + change the adapter, most everything else is this same
    - xiaomi robotics series
        - XR-1: scaling of UMI data is the major depature from the pi0 recipe! So much UMI data...
    - ego-scale
        - shows human hand data annotated via ml models can be useful for training policies!
        - needed high quality mid training dataset to bridge the gap
        - will this work on non human hand data? will this help?
    - ling-bot va
- modern policies 
    - co training
        - groot n1's extensive use of co-training data motivates a systematic study into the efficacy of this type of data
        - luckily our goats @ TRI have got us again
        - in [Lin et al.](https://arxiv.org/pdf/2602.01067), they study the effects of various types of co training data on performance
        - They find diverse language annotated video and image data strongly improves generalization (!)
        - Everything else they tried was basically a wash
        - Misc Interesting Things:
            - The use only a single token's output layers to condition the flow transformer -- improves generalization. (I think this is one of those things that washes out with scale)
            - They don't use prio, wonder why (https://arxiv.org/pdf/2608.03052 finds that prio is usually helpful and leads to gains in perf)
    - umi
    - video backbones + generalization
        
- small details
    - diffusion
    - action chunking
    - surrogate objectives
    - joint vs eef
        - https://arxiv.org/pdf/2602.23408 talks abbout this
        - basically joint space mogs but leads to poor transfer
    - the vla you use
    - FAST tokens


talk about non settled things like ego data
video backbones

seems like this: 
1. gather all the robot data you can find literraly all of it
2. take a vla style architecture
3. co-train on good vision language annotated data + vision based tasks
4. train on litterally all of that data

there are some more details like:
- relative EEF delta seems like the best action space for transfer
- you want to use action chunking + flow matching
- what makes good co-training data
etc
