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
