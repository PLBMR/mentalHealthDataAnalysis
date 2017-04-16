# Data And Mental Health: The OSMI Survey 2016 - Part 2

_By [Michael Rosenberg](mailto:rosenberg.michael.m@gmail.com)_

# Introduction

Hi everyone! After performing a data analysis on the [Open Sourcing Mental
Illness](https://osmihelp.org) (OSMI) survey [previously](
https://medium.com/@tfluffm/data-and-mental-health-the-osmi-survey-2016-39a3d308ac2f), 
I got a lot of good feedback on my techniques and decisions
within the data science community and the mental health landscape. I appreciate
all of your thoughts and opinions, and I hope to stay involved in the
conversation related to data and mental health.

After looking over other components of the [OSMI Survey on Mental Health](
https://osmihelp.org/projects/research), I recognized some other components of
the dataset that could be mined for meaningful results. In particular, I 
recognized that there were many assets of the survey that took into account the 
opinions of respondents towards the mental health landscape in the industry. 
Using this opinion data, I was able to construct the analysis here today.

As discussed in my previous analysis, I have some ground rules:

* If you would like clarification in this analysis, please leave a comment 
below! I want to be as clear as I can when writing a report.

* If you have any questions on techniques used or components studied, 
also leave a comment! I really appreciate conversations on data-mining 
strategies.

With that being said, let's dive into the data.

# The Dataset

I spent some time in my previous analysis describing the OSMI Survey in detail.
Hence, I would recommend that you read through
[that section](https://medium.com/@tfluffm/data-and-mental-health-the-osmi-survey-2016-39a3d308ac2f)
of the previous analysis if you want the full details on the survey.

As mentioned earlier, the OSMI Survey contains several questions pertaining to
how respondent perceive mental health support in the workplace. Some of these
questions pertain to:

* Whether or not an employer provides coverage for mental health conditions in
the workplace healthcare plan.

* Whether an employer is supportive of mental health in the workplace.

* Whether one's colleagues (e.g. team members, direct supervisors) are
supportive of mental health in the workplace.

* If one believes that mental health conditions are regarded as taboo in the
industry as a whole.

I am interested in whether or not there is a set of patterns to the answers of
these questions that might indicate narratives about mental health in the
technology industry. In particular, I wonder if there are clusters
of respondents that each have a different perspective on how mental health is
dealt with in the workplace. Once I have discovered these perspective clusters,
I hope to see whether the demographics of respondents predict these perspectives
in some meaningful manner.

# Data Exploration

One of the key limitations we have to deal with is that individuals who are
self-employed do not fill out most of the information related to their 
employer or colleagues. Thus, in order to analyze this dataset, we must remove
self-employed persons from the analysis. This drops our number of observations
from 1433 to 1146, which is a sizable drop. That being said, self-employment is
a very different style of work than the typical salaried individuals working for
companies. In this sense, including them in the analysis will lead to
complicated interpretations of our results. Thus, while it does lead to a loss
of data points, self-employed individuals should only be considered in future
analyses.

![figure1](../figures/figure1.png)

_Figure 1: Distribution of whether or not an employer provides mental health 
benefits as part of health coverage._

We see that while a sizable number of individuals have some form of mental
health benefits as part of their health coverage, a moderate amount of
individuals are either not sure about their coverage or do not have coverage on
mental health. This is immediately concerning for the tech industry, as it would
suggest that a lot of workplace health care plans are not adequate for taking
care of mental health.

![figure2](../figures/figure2.png)

_Figure 2: Distribution of whether or not an employer offers resources to
learn more about mental health concerns and options for seeking help._

We see that most respondents claim their employer does not provide meaningful
resources to educate on mental health and seek help. This again shows a trend
that employers are typically not providing the benefits nor alternative
resources to address mental health.


# Model Selection

# Inference

## Defining the Clusters

## Predicting the Clusters

# Discussion

# Limitations

# Future Work
