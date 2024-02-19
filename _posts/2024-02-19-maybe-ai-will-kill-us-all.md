---
title: maybe ai will kill us all
cnt: "0x08"
layout: post
---
This is going to be my first long blog (and first blog on time) in a while! I'll talk about an interesting problem I learned about this week.
# productivity check in
Couple of goals for this week. 
- Finish the first 9 chapters of this [deep learning book](https://udlbook.github.io/udlbook/) I'm reading.
- Pick back up on my paper reading habit. 
- Finish half of the book I'm reading this month, Scout Mindset.
- Build a prototype for a productivity app idea I had. 
- Get to chapter 10 of AP CS A review book I'm using.

How I plan to do this?

- Read a chapter of the DL book a day. I'm already on chapter 5 so it shouldn't be too much of a problem. 
- Try and read before I go to bed each night. It takes about 2 hours to engage deeply with a paper for me. I've found a sub-field of AI I want to focus on for now, so I'll work on reading papers from there.
- I spend way to much time on Instagram reels. I'll cut back on this and spend more time with a book in my hands.
- I don't know how I'll accomplish this one, maybe with extra time I have during school. 
## obsidian review
I've spent the last week and a half using obsidian with the zettlekasten method and I'm absolutely in love. Besides the fact that it makes my notes look beautiful, the ability to *visualize* the links between ideas is amazing!

![](/assets/posts/0x08/obsidian-graph.png)

The zettlekasten method is awesome too. This is the first time that I've actually felt like reviewing my notes hasn't been a waste of time. I won't describe everything in full detail here, but I 100% recommend this system/app combo if you're trying to get into digital note taking. 
# messy-optimizer
The coolest (and scariest) thing I learned about this week was the **mesa-optimizer**. It's a key risk when we create AI's that pursue complex goals. 
## what's an optimizer??
An **optimizer** is a method of taking an existing solution and nudging it toward a better one. What defines better varies from environment to environment. The best example of an optimizer is **evolution**. Evolution optimizes for reproductive fitness, and builds an organism (like us) that maximizes this fitness. 

However, this organism *doesn't care about the optimizers goal*. We are optimized by evolution to maximize our reproductive fitness, yet (most) of us don't really want to pump out as many babies as we can before we die. Instead we optimize for different things, like love, joy, and fulfillment. In other words **The optimizer** (evolution) **built an optimizer** (us).
## why does this matter?
First some jargon. When an optimizer builds another optimizer, we call the resulting optimizer a **mesa-optimizer**. The mesa prefix meaning below, giving us a "below-optimizer", an optimizer beneath our previous optimizer. 

So optimizers might build mesa-optimizers. The mesa-objective of the optimizer should be aligned with the objective of our original optimizer right? Otherwise, why would the optimizer select it?

**Not quite.**

Going back to the evolution example, evolution optimized us for **reproductive fitness**. It was in evolution's favor for sex to feel pleasurable for humans, but for us, that's just a nice side effect. 

We don't optimize for evolutions goal (reproduction) we optimize for *our goal*, (pleasure). That's why in addition to sex, we also masturbate, even though through the lens of evolution that's sub-optimal. 

Similarly, we train our AI models via an optimizer. And that optimizer can (and will) build mesa-optimizers. The consequences of such a process can be drastic. 

## a simple example

![](https://miro.medium.com/v2/resize:fit:786/format:webp/0*1sPM_3emcfsCQJ21)

Above, you see a blue ball training to navigate an environment with these holographic spheres. There is a specific order the agent must visit the spheres in, and our optimizer rewards the agent for visiting the spheres in the proper order. 

This is the training environment for the agent, and provided with is is a red helper bot which knows the order and carries out the blue bot's task. Within this environment, the optimizer is happy. It's managed to train an agent that visits the spheres it the proper order.

![](https://miro.medium.com/v2/resize:fit:786/format:webp/0*0c--QeuEVfKmZL2y)

However, the environment above shows that the agent hasn't really learned the order of spheres, just that it should follow the red ball *regard less of if the red ball is correct or not*. While the optimizer optimizes for order, the *mesa-optimizer* optimizes for distance from the red ball.

Faults in the mesa-optimizer are revealed when there are distributional shifts between the testing environment and the training environment. This is a *serious* problem. What if a model trained to identify criminals actually is focused on identifying minorities? If a self driving car accelerates because a stop sign is painted green instead of red? There are a million ways the mesa-optimizer can be out of line the the optimizer, the question is, can we find them all? I'll let you know if I find the answer. 

*The above example is taken from [here](https://deepmindsafetyresearch.medium.com/goal-misgeneralisation-why-correct-specifications-arent-enough-for-correct-goals-cf96ebc60924)*