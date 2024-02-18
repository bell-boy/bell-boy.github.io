---
layout: post
title: it's all just a linear map
cnt: "0x01"
---

Happy new year! If you're reading this, I finally got the markdown transpiler to work. This blog is going to be a bit more basic than I would've otherwise liked. Still haven't gotten around to implementing a couple key features (most noticeably lists), but that's a problem for next week.

A lot happened this week and a lot more is to come, but if you're feeling lazy...

<ul>
    <li>I implemented a transformer</li>
    <li>I finished the first version of my markdown transpiler</li>
    <li>I set goals for the new year</li>
</ul>

## Building GPT

I won't go over much about how transformers work, just the new ideas gained.

It's all just linear maps.

I don't think I could've picked a better applied liner algebra project if I tried. Before making GPT, I was iffy on many of the concepts presented in linear algebra, but I don't think I could forget anything I learned after this even if I tried. 

A core idea behind GPT is that tokens (usually learned subwords like ['coo', 'l', ' be' 'ans'] to represent the string "cool beans", but in my case I just used single letters) are represented as *directions in some high dimensional space*. What's cool about linear algebra is that it's all about *shifting and transforming* from one high dimensional space to another. A matrix is just a way to transform vectors between spaces!!

Another thing I was confused about was tensors, where do they come into the equation? If a 1-d array is a vector in space, and a 2-d array is a transformation between spaces, what the hell is a 3-d array, or an n-d array more generally?

The first step to resolving that question was understanding more specifically that a matrix is a *collection* of vectors. Usually, a collection of *basis* vectors. Basis vectors are vectors that fully describe a space. Some combination of these vectors can create any other vector within a subspace. Linear transformations ask the question, what if we took those basis vectors and moved them to other locations, where would all the other vectors move to?

But that's just one way to use a matrix. We can do anything we want with the collection of vectors, including scaling them, shifting them, transforming them with other matrices. They *can* be linear transformations, but only if we want them to be. More generally, tensors are just collections of collections, their meaning can vary. 

So understanding that was key, but I still had to write code. What's really annoying about python is that everything you do should probably be vectorized, meaning that instead of using the python to perform tasks, you use python to call some underlying c++ code that performs the task. This is super fucking annoying, because it's really easy to lose track of the shapes of your tensors, and since the tensors can be 4+ dimensions, it be hard to visualize what an operation is actually doing.

<a href="https://einops.rocks/">Einops</a> is cool because it lets you trade one headache for a slightly more manageable headache. It works using <a href="">einstein summation notation</a>, which physicists use to manipulate tensors. Over time, I developed a framework to work with einops. Step one is to determine the resultant shape you want from the operation, step two is determining what indices you want to multiply together, and then step three takes care of it self. Just leave out the indices left and einops will sum them over. (If none of what I just said made sense, <a href="https://rockt.github.io/2018/04/30/einsum">it didn't to me too at first</a>)

After I understood those concepts, everything else was a breeze. 

### Attention

The toughest module to write was Multi-Head Attention. Firstly, attention has been around since before transformers, so when the authors described their form of attention <a href="https://arxiv.org/abs/1706.03762">here</a>, they neglected to describe how the query, key, and value matrices are generated. So after a little googling, I figured out that it's just a learned linear map from the space of the residual stream, to the sub space of the attention heads. 

Attention relies on the fact that all tokens are represented as vectors, and takes the dot product between each pair of query, key vectors (each position has a query vector and a key vector). This generates a similarity score matrix of query by key. this in turn gives us an way to figure out how much attention to pay to each key. A couple more linear maps and a concatenation later, and we're back to the residual stream. If you're interested in an explanation that makes sense <a href="https://jalammar.github.io/illustrated-transformer/">This is a really solid guide</a>

Check out my code <a href="https://colab.research.google.com/drive/1OVkX4o_oyohDE0ze-C8LuXjlsCxt46Tj?usp=sharing">here</a>


## Bytecode

### Why do you even need a transpiler?

Writing raw html is annoying. noticeably about opening and closing tags, styles, and semantics makes writing that much harder. So I figured, I like markdown, why not use it to write my blogs?

While other transpilers exist, I figured I would learn the most from just writing my own. Writing my own gives me the additional benefit of fine grained control over how I want things to be styled.

### How did you do it?

I started this project like all good projects should start.

I asked ChatGPT. 

I wasn't expecting much, but surprisingly, ChatGPT is great at pointing you in the right direction. It outlined what my project would need, and gave me a couple of terms. After a brief stint on wikipedia, I got started.

The whole thing can be broken down into two parts: the **Abstract Syntax Tree** and the **Translator**

I call it an Abstract Syntax Tree but its really more of a mini-DOM. It's a tree that holds the structure of the website within it. As of writing this, there are 7 types of nodes. ***BoldItalic***, **Bold**, *Italic*, Header, Paragraph, Text, and Empty. Each node has 3 properties, a NodeValue labeled *value* stating which of the 7 types it belongs to, a string labeled *contents* the usage of which varies depending on the node, and a list of Nodes called children, which contains the nodes nested inside the current node.

The first three are pretty obvious as to what they do, and the foruth and fifth make sense, but the last two are non-obvious and I want to describe why I did it. Text nodes are just plain text, and empty nodes are just empty nodes. 

Text nodes were needed because Paragraph nodes can be split in half: &lt;p&gt;before italics *italics* after italics&lt;/p&gt; -> Paragraph node with three children: Text node ("before italics "), Italic node ("*italics*"), Text node (" after italics").

To explain Empty nodes, I'll have to explain the way nodes in general are created. There are *Line nodes* and there are *subline nodes*. A *Line node* is a node that wraps a whole line. The only Line nodes that exist are Header, Paragraph, and Empty nodes. A line is wrapped in a header when up to six "#" characters with a space next to them at the start of a line, A Paragraph node is when the line has text, but no header markings, and Empty nodes are for totally empty lines.

Subline nodes are the *Italics* and Text nodes. Each line is processed recursively. For example, the symbol for Italics is "*", and when the parser finds two of them (or the marking for any other node) in a line it then processes the section before the symbols, the section in between the symbols, and the section after the symbols. The section in between the symbols is appended to a new Italics node and returned in a list along side the elements in the suffix process and the prefix process. The base case is when there are no special symbols. If the string is empty, nothing is returned, else a text node with the text in it's contents string is returned.

So the file is parsed like this: each line gets a *Line node* then is recursively processed to return the *subline node* children of said Line node. However, consecutive paragraph lines should be one paragraph, not a mess of seperate ones. So there is a final step called compression to squeeze together consecutive Line node lines with the same type. So paragraph, paragraph, empty, paragraph, paragraph, paragraph becomes: paragraph, empty, paragraph. This is the slowest part of the parser and runs in roughly O(n^3) with a high constant factor. However, I doubt that I'd get to the point where my blog posts are long enough to take noticeably longer to process, so ill leave it for now.

Translation is markedly easier. To process a node, I run a recursive function turning nodes into strings. It adds outer tags depending on the node's value and concatenates the result of the translation function on its children inside said outer tags. The base case is when we reach a text or empty node, in which case text or nothing is returned.

After running this on the whole file, I insert the result into a template blog post and call it a day!

However, the code is vulnerable to injections. Upon opening this post, you saw an alert that said ":)", that's because the raw text of the markdown file is turned into HTML, meaning that if I write HTML it will be processed as HTML. I'm considering not getting rid of it though, I'm the only one planning to use bytecode and it allows me to use HTML features I haven't implemented yet.

<script>alert(":)")</script>

The line above contains the injection, but it won't be visible here. 

So there's a lot more to do, I'll work on it as I improve the blog.

## 2024

New years resolutions! I have a couple of goals.

Starting long term, I have three main goals: stay healthy, live comfortably, and change the world.

I have to thank the website <a href="https://80000hours.org/">80,000 Hours</a> for giving me suggestions and comprehensive career profiles on jobs and pressing issues to tackle with my career. Over the past month I've gotten closer to narrowing down the last two, and I already have a pretty good idea about the pathway to the first.

So in four years, at the end of 0x100, where do I see myself? Well for starters with a degree. But in what, and why? The easiest answer to question two is to become a <a href="https://80000hours.org/career-reviews/quantitative-trading/">quant</a>, which is going to require strong math and cs skills. The answer that seems to suit my fit for question three is to become an <a href="https://80000hours.org/career-reviews/ai-safety-researcher/">AI safety researcher</a>. Yet I haven't ruled out becoming a startup founder. <a href="https://www.ycombinator.com/library/4r-yc-and-hard-tech-startups">Hard Tech</a> seems like a great way to achieve goal three.

So we have three paths to travel. How to figure out which one is the right one?

Answer: try them all!

For the first two paths, it would probably be beneficial to get a Ph.D. So optimizing my undergrad for a CS Ph.D is probably the move. Along the way, internships will probably help to achieve paths 1 and 3. So I'll also focus on being able to achieve those as well.

From what I've <a href="https://docs.google.com/document/d/1RFo7_9JVmt0z8RPwUjB-mUMgCMoUQmsaj2CM5aHvxCw/edit#heading=h.hfj52k67ycog">read</a> research experience is the thing I need to optimize for when aiming for a top Ph.D. Similarly, a strong portfolio of projects and top-tier interview skills will be needed to land increasingly competitive internships.

So now we can get to this year. Trying to make my goals as <a herf="https://www.forbes.com/advisor/business/smart-goals/">SMART</a> as possible, so I'll aim for a specific outcome as a proxy for a broader goal (i.e. get into MATS as a proxy for gain research experience). In no particular order:

<ul>
    <li>Get into <a href="https://www.matsprogram.org/">SERI MATS</a></li>
    <li>Get into <a href="https://summerofcode.withgoogle.com/">Google Summer of Code</a></li>
    <li>Reach Candidate Master on Codeforces</li>
    <li>Gain 30lbs</li>
    <li>Get into an <a href="https://www.nsf.gov/crssprgm/reu/">REU</a> OR <a href="https://www.janestreet.com/join-jane-street/programs-and-events/jsip/">Quant diversity program</a></li>
    <li>Build 4 projects focusing on different fields</li>
    <li>Read 12 books, on a variety of different topics</li>
</ul>

I'll let you know how it goes.

But for now, I'm still in highschool. So my priorities are ending the school year strong. For the month of january...

<ul>
    <li>Pass a practice AP Bio Exam & a practice AP CS A Exam</li>
    <li>Make a contribution to <a href="https://www.mlpack.org/">mlpack</a></li>
    <li>Reach Pupil on Codeforces</li>
    <li>Gain 2.5 lbs</li>
    <li>Participate in a kaggle competition</li>
    <li>Read Linear Algebra Done Right</li>
    <li><a href="https://www.lesswrong.com/posts/B3z8PMqor3rivzgAv/deep-dives-my-advice-for-pursuing-work-in-research">Read a paper a day</a></li>
    <li>Commit Code Daily</li>
</ul>

Now to this week... 

<ul>
    <li>Find a good Bio textbook and get through chapter one</li>
    <li>Train an MNIST model using MLpack</li>
    <li>Solve a cp problem per day</li>
    <li>Participate in a demo kaggle competition</li>
    <li>Read Linear Algebra Done Right chapter 1</li>
</ul>