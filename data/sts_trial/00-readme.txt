
			       STS 2015

	       Semantic Textual Similarity for English SUBTASK

			    TRIAL DATASET
				   

This set of files describes the trial DATASET for the Semantic Textual
Similarity 2015 task for English. It comprises sample files for the
datasets which will be used for STS 2015 evaluation, plus all train
and test datasets released to date for STS.

The trial dataset contains the following:

  00-README.txt 		  this file

  correlation-noconfidence.pl     evaluation script

  sts2012-train.tgz               train data for sts 2012
  sts2012-test.tgz                test data for sts 2012
  sts2013-test.tgz                test data for sts 2013
  sts-en-gs-2014.zip              test data for sts 2014

  STS.input.image.txt             tab separated sample input file with 
                                  sentence pairs
  STS.input.headlines.txt         "
  STS.input.answers-students.txt  "
  STS.input.answers-forum.txt     "  
  STS.input.belief.txt            "

  STS.gs.image.txt                tab separated sample gold standard
  STS.gs.headlines.txt            "
  STS.gs.answers-students.txt     "
  STS.gs.answers-forum.txt        "  
  STS.gs.belief.txt               "


  STS.output.image.txt            Tab separated sample system output file
  STS.output.headlines.txt        "
  STS.output.answers-students.txt "
  STS.output.answers-forum.txt    "  
  STS.output.belief.txt           "

  


Introduction
------------

Given two sentences of text, s1 and s2, the systems participating in
this task should compute how similar s1 and s2 are, returning a
similarity score, and an optional confidence score.

The trial dataset comprises the 2012, 2013 and 2014 datasets, which
can be used to develop and train systems.

In addition, we include sample data for the test datasets, coming from
the following:

1) image description (image)
2) news headlines (headlines)
3) student answers paired with reference answers (answers-students)
4) answers to questions posted in stach exchange forums (answers-forum)
5) English discussion forum data exhibiting commited belief (belief)

The trial data is a small subset of the sentence pairs that will be
used as test data, with (dummy) gold standard scores. The goal of these
samples is to allow participants to have an idea of which kind of
sentences will occur in each of the test datasets.

The datasets have been derived as follows:

- STS.input.image.txt: The Image Descriptions data set is a subset of
  the Flickr dataset presented in (Rashtchian et al., 2010), which
  consisted on 8108 hand-selected images from Flickr, depicting
  actions and events of people or animals, with five captions per
  image. The image captions of the data set are released under a
  CreativeCommons Attribution-ShareAlike license.

- STS.input.headlines.txt: We used headlines mined from several news
  sources by European Media Monitor using their RSS feed from April 2,
  2013 to July 28, 2014. This period was selected to avoid overlap
  with STS 2014 data.
  http://emm.newsexplorer.eu/NewsExplorer/home/en/latest.html

- STS.input.answers-students.txt: The source of these pairs is the
  BEETLE corpus (Dzikovska et al., 2010), is a question-answer data
  set collected and annotated during the evaluation of the BEETLE II
  tutorial dialogue system. The BEETLE II system is an intelligent
  tutoring engine that teaches students in basic electricity and
  electronics. The corpus was used in the student response analysis
  task of semeval-2013. Given a question, a known correct "reference
  answer" and the "student answer", the goal of the task was to assess
  student answers as correct, contradictory or incorrect (partially
  correct, irrelevant or not in the domain). For STS, we selected
  pairs of answers made up by single sentences.

- STS.input.answers-forum.txt: This data set consists of paired
  answers collected from the Stack Exchange question and answer
  websites (http://stackexchange.com/). Some of the paired answers are
  in response to the same question, while others are in response to
  different questions. Each answer in the pair consists of a statement
  composed of a single sentence or sentence fragment. For
  multi-sentence answers, we extract the single sentence from the
  larger answer that appears to best summarize the answer. The Stack
  Exchange data license requires that we provide additional metadata
  that allows participants to recover the source of each paired
  answer. Systems submitted to the shared task must not make use of
  this meta-data in anyway to assign STS scores or to otherwise inform
  the operation of the system.

- STS.input.belief: The data is collected from DEFT Committed Belief
  Annotation dataset (LDC2014E55).  All source documents are English
  Discussion Forum data.



Input format
------------

The input file consist of two fields separated by tabs:

- first sentence (does not contain tabs)
- second sentence (does not contain tabs)

Please check any of STS.input.*.txt



Gold Standard
-------------

The gold standard contains a score between 0 and 5 for each pair of
sentences, with the following interpretation:

(5) The two sentences are completely equivalent, as they mean the same
    thing.  

      The bird is bathing in the sink.  
      Birdie is washing itself in the water basin.

(4) The two sentences are mostly equivalent, but some unimportant
    details differ.

      In May 2010, the troops attempted to invade Kabul.
      The US army invaded Kabul on May 7th last year, 2010.

(3) The two sentences are roughly equivalent, but some important
    information differs/missing.

      John said he is considered a witness but not a suspect.
      "He is not a suspect anymore." John said.

(2) The two sentences are not equivalent, but share some details.

      They flew out of the nest in groups.
      They flew into the nest together.

(1) The two sentences are not equivalent, but are on the same topic.

      The woman is playing the violin.
      The young lady enjoys listening to the guitar.

(0) The two sentences are on different topics.

      John went horse back riding at dawn with a whole group of friends.
      Sunrise at dawn is a magnificent view to take in if you wake up
      early enough for it.

Format: the gold standard file consist of one single field per line:

- a number between 0 and 5

The gold standard in the test data will be assembled using mechanical
turk, gathering 5 scores per sentence pair. The gold standard score
will the average of those 5 scores. In this trial dataset, this is
just a dummy number which you can ignore.

Please check any of STS.*.gs.txt



Answer format
--------------

The answer format is similar to the gold standard format, but includes
an optional confidence score. Each line has two fields separated by a
tab:

- a number between 0 and 5 (the similarity score)
- a number between 0 and 100 (the confidence score)

The use of confidence scores is experimental, and it is not required
for the official score.

Please check any of STS.*.output.txt


Scoring
-------

The official score is based on the average of Pearson correlation. The use of
confidence scores will be experimental, and it is not required for the
official scores. 

You can use correlation-noconfidence.pl (a perl program) as follows:

 $ ./correlation-noconfidence.pl STS.gs.images.txt  STS.output.images.txt 
 Pearson: 0.31589


Participation in the task
-------------------------

Participant teams will be allowed to submit three runs at most.



Other
-----

Please check http://alt.qcri.org/semeval2015/task2/ for more details.



Authors
-------

Eneko Agirre
Daniel Cer
Mona Diab
Aitor Gonzalez-Agirre
Weiwei Guo
German Rigau



References
----------


Dzikovska, M. O., Bental, D., Moore, J. D., Steinhauser, N. B.,
  Campbell, G. E., Farrow, E., and Callaway, C. B. Intelligent
  tutoring with natural language support in the beetle ii system. In
  Sustaining TEL: From Innovation to Learning and Practice, pages
  620-625. Springer. 2010.

Rashtchian, C., Young, P., Hodosh, M., and Hockenmaier, J.
  Collecting Image Annotations Using Amazon's Mechanical Turk.  In
  Proceedings of the NAACL HLT 2010 Workshop on Creating Speech and
  Language Data with Amazon's Mechanical Turk. 2010.
