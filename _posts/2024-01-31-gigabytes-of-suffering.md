---
layout: post
title: gigabytes of suffering
cnt: "0x05"
---

Hi! I've spent most of my free time building my final project for my engineering class. So after a quick productivity update, I'll spend most of this post detailing how that's going.  

# productivity

I wish I was uploading this on Monday, but it is currently Wednesday the 31st. So I'll take a moment to reflect on the goals I set for the month, see how I did.

## i did pretty poorly

Looking at my monthly goals...

* I did *not* read and explore AI papers daily
* I did *not* participate in a kaggle competition
* I did finish superforecasting!
* I did make the blog look slightly better!

The main reason I failed the first two was that I didn't really have a consistant time and place to do them in. Reading could be done with any bit of free time, and small updates to the blog can be done in a few lines of code. But reading a paper or building a model requires a long block of uninteruptted attention. During Febuary I want to have times and places where I do work everyday, regardless of what's going on. 

Another problem was I didn't have a concrete problem I was solving. In Febuary I want to focus on clarifiying what the problems within my field are, previous approaches to tackling them, and directions to go forward. 

Moving on to yearly goals...

* AI Research: I feel like I made little progress on this goal for the reasons stated previously.
* Becoming A Better Engineer: Working on my engineering project has pushed me to grow in unexpected ways, so I'm happy with this one.
* Getting 5s on my AP Exams: This is the goal I've probably made the least progress on. Like I said before, I just couldn't find the time to start studying. Hopefully that can change.
* Executing my engineering capstone project well: As I said before, It's gone well. However, I think I'll fold this goal into my second goal. 
* Gaining 30lbs: I cannot go to the gym consistantly. It's not possible without a car. This goal might have to be put on hold.
* Reading More: There are three types of books I read. Non-fiction techincal, Non-fiction non-techincal, fiction. I'm reading plenty of non-technical literature, but I don't think I've dedicated enough to for technical reading.

## goals for feb

* Finish [AI Safety Fundamentals](https://course.aisafetyfundamentals.com/alignment)
* Finish [MIT Bio](https://ocw.mit.edu/courses/7-01sc-fundamentals-of-biology-fall-2011/)
* Finish my engineering project
* Finish Scout Mindset by Julia Galef
* Finish 3 Blog TODO list items

## goals for this week

* Finish Session 1 of AISF
* Finish the Biochemistry unit of bio
* Finish my model architecture and get a music free prototype of the game working
* Give my markdown parser the ability to deal with YAML front matter

# the engineering project 

so the core functionality of my engineering project is a *monophonic music transcription system*. Which means that it takes in a musical recording from a single instrument and writes down the music.

The approach I chose is based on machine learning, but the hard part isn’t the machine learning. It’s getting the data in the proper format that makes the project difficult. 

The ML model is based on [this](https://ismir2018.ismir.net/doc/pdfs/87_Paper.pdf) paper which uses an architecture based on [DeepSpeech2](https://arxiv.org/pdf/1512.02595.pdf) a system that converts *speech* to text. I’ve made a couple of modifications to the architecture that i’ll go into detail about in a later post. 

What really shocked me about this whole process was that the researchers behind the first paper went through a long, difficult process to build the dataset they used to train the model and they *didn’t publish it*. Creating model is the easy part, just *getting the data* has been the hard part.

## gigabytes of struggle

The data comes mostly from the [RISM](https://opac.rism.info/main-menu-/kachelmenu/data) dataset. RISM (Répertoire International des Sources Musicales) is an “international, non-profit organization that aims to comprehensively document extant musical sources worldwide.” Being international, the website was not in english, and I lost my shit trying to navigate it.

Then, the actual data. It felt like it was impossible to find a link to download the actual dataset and I spent hours trying to work with their online system. I was so frustrated with it all. 

Until I found the english button on the website.

After I got over feeling stupid. I downloaded the data and struggled to understand it's format (the link to the formatting rules were in French). Most of the data in the dataset was irrelevent. So I had to write code to take the bits of music I cared about. Once I had all these bits, I converted them to audio recordings using a super fucked processing pipeline. After 10 hours of switching between coding and googling, I had about 160GB of audio data. Yet my suffering still isn't over, because I still have to convert the bits to a format the model can and process and understand. 

All this headache, all this suffering, because the researchers couldn't be bothered to publish their dataset. I have a lot more to do, so i'll let you know how it goes. 