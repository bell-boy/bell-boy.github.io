---
layout: post
title: the struggle
cnt: "0x02"
---

Hi again! This week was kind of busy, yet kind of chill. Lets recap.

* Kinda fumbled on my goals
* Added a couple of features to bytecode
* Spent some time in MIT's Deep Learning course

## Last Weeks Goals

* I was **NOT** able to find a good bio text book, this is really important as I paid for that (expensive) AP exam.
* I did not train a model using MLPack.
* I (kind of) solved a cp problem per day.
* I did not start a kaggle competition.
* I was not able to make a commit per day. 

So what did I do?

I applied to college lol. 

But the goals I set in the previous post are kind of unrealistic. I don't feel bad for being over ambitious. I would much rather reign in my goals then finish a month thinking that I could've done more. So here's 2024, rewritten.

* [SERI MATS](https://www.matsprogram.org/)
* [Google Summer Of Code](https://summerofcode.withgoogle.com/)
* Reach Candiate Master on Codeforces
* Gain 30 lbs
* Get into an REU or Quant diversity program
* Build 4 projects throughout the year
* Commit code daily
* Finish 12 different non-fiction books 
* Get a 5 on all my AP exams

All of the motivations for these goals were explained in the [last](https://bell-boy.github.io/0x100/posts/00000001.html) post.

Continuing on, here's January, rewritten.

* Complete Half of MIT's intro to Machine Learning course (yearly goals 1, 2, & 5)
* Complete a CP problem a day (yearly goal 3)
* Participate in a contest per week (yearly goal 3)
* Gain 4lbs (yearly goal 4)
* Participate in a kaggle competition. (yearly goal 6)
* *Try* to read a paper per day, but optimize for understanding (yearly goals 1 & 5)
* Read *superforcasting* (yearly goal 7)
* Read (but not necessarily finish yet) *Linear Algebra Done Right* (yearly goals 1, 2, & 5)
* Get through half of MIT bio (yearly goal 8)
* Tutor on schoolhouse.world (just for fun :P)

finally, this week!

* Work on MIT machine learning at least 3 times per week
* Get tutor certified on schoolhouse.world
* Try to finish the weights section of mech interp lit review
* decide on model architecture for kaggle comp.
* Practice AP Comp Sci Exam

Most of the goals of the form *do x (weekly, daily, monthly)* are just going to be melded into my routine as I experiment more. We'll see.

## bytecode!

THANK GOD FOR TEST DRIVEN DEVELOPMENT!

But seriously, I don't even understand how a project is even possible without TDD. I implemented a couple of new features this week and in the process SOOOO many things broke that I wouldn't have caught without my tests. Seriously thinking of switching to a new testing framework. 

If you don't know, Test Driven Development is the process of writing tests that validate your code *before writing your code*. Then write the bare minimum code that works and passes your tests, then refactor and clean up your code! 

It's real strength was shown off after I'd accumulated a library of tests. If I wrote code that broke old code, my tests would catch it, I didn't even have to stress!

Going on to new features, I added in
* Unordered lists
* Links
* Escape characters
* A central post listing

Of these four features ESCAPE CHARACTERS were the trickiest to implement. It was hard to do so in a way that didn't nuke all other functionality, so my code is currently a mess. 

Links are strange because instead of following the \* &lt;-- open tag |content| --&gt; \* format, they're really two tags. Links look like this \[text\]\(hyperlink\), this is super annoying and I had to implement a solution I felt less than comfortable with. I'm thinking of getting my code reviewed on reddit to try and improve my solutions to these less obvious cases.

## MIT Machine Learning

*[This](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/course/) is the course I'll be referring to*

So week one was just a more general review of everything I'd learned over these past couple of weeks I've spent with ML. Linear algebra, tensor manipulation in python etc. The actual content started of with *Linear Classifiers*. A pretty basic ML model that can easily be perfectly optimized. In short, it's just a plane through an arbitrarily high dimensional space. Points are scattered through out the space, and what we learn is a plane that separates classes. A 1-D example would be a classifier trying to figure out if you're above a certain height by looking at the length of your femur. If your femur lies on a line above the LC, then you're over the height we trained for, if not, you're under! They go into a lot more mathematical detail that I'm still trying to understand myself. I'll let you know how it goes.
