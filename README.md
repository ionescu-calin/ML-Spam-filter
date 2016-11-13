# ML-Spam-filter

## Spam filter

This assignment is near-final modulo some small adjustments (6 Nov '16)
In this assignment, you will discover that in many practical machine learning problems implementing the learning algorithm is often only a small part of the overall system. Thus to get a high mark for this assignment you need to implement any of the more advanced classification techniques or clever pre-processing methods. You will find plenty of them on the internet. If you do not know where to look for them, ask Google ;-).

Here your task is to build the standard (i.e. multinomial) Naive Bayes text classifier described during the lectures. You should test your program using the automatic marking software (described below), so it is critically important that it follows the specifications in detail.

You will train your classifier on real-world e-mails, which you can download from here. Each training e-mail is stored in a separate file. The names of spam training e-mails start with spam, while the names of ham e-mails start with ham.

## Marking criteria

### Part 1 (40%):
Your program classifies the testing set with an accuracy significantly higher than random within 30 minutes
Use very simple data preprocessing so that the emails can be read into the Naive Bayes (remove everything else other than words from emails)
Write simple Naive Bayes multinomial classifier or use an implementation from a library of your choice
Classify the data
Report your results with a metric (e.g. accuracy) and method (e.g. cross validation) of your choice
Choose a baseline and compare your classifier against it
### Part 2 (30%):
Use some smart feature processing techniques to improve the classification results
Compare the classification results with and without these techniques
Analyse how the classification results depend on the parameters (if available) of chosen techniques
Compare (statistically) your results against any other algorithm of your choice (use can use any library); compare and contrast results, ensure fair comparison
### Part 3 (30%):
Calibration (15%): calibrate Naive Bayes probabilities, such that they result in low mean squared error
Naive Bayes extension (15%): modify the algorithm in some interesting way (e.g. weighted Naive Bayes)
Specifications

The program should be written in Java or Python, and the name of the main file should be filter.java or filter.py depending on the language that you choose.

If you choose to work with Java all files should compile using the command:

javac *.java
The program should take only one argument:

java filter testfile
for Java or:

python filter.py testfile
for Python; where testfile is the name of a file containing a single e-mail, which is to be classified.

Note that the argument may contain a relative or absolute path to the file, e.g. your code might be called as:

java filter ../test/spam1.txt
or

python filter.py ../test/spam1.txt
Since you do not specify the directory with the training files, the program needs to store its knowledge gained during training somewhere, e.g. in a separate file (which you can submit together with your code). Therefore, you need to implement training your classifier and storing the data for future classification (for example you could invoke this functionality with some special commandline flag).

The program should print the word ham and the new line character, if it classifies the e-mail as ham, and the word spam followed by the new line character, otherwise. IMPORTANT: The output of the program needs to be exactly as specified in the previous sentence: no spaces, no capital letters, and remember about the new line character!!!

Your filter should be case sensitive (e.g. Alice and alice should be treated as two different words). You can assume that none of the testing e-mails have equal likelihood of being spam and ham.

Testing your program with automarking

The program will be tested using automatic marking on a test dataset. The testing set will contain e-mails from the same sources as the training set. The tests will be performed on the departmental Linux machines (MVB 2.11), and your program needs to classify all testing e-mails within 30 minutes. A sample automarking script with a couple of testing e-mail can be downloaded from here (read readme.md in the archive for more details about the automarker).

The automarker does not work on Windows 7 (and possibly on other new operating systems). But importantly, it does work on computers in the Linux lab in MVB. Since your assignment will be tested on the Linux machines in that Lab, please test your submission with the automarker in the Linux lab (rather than on your own computer).

The program will display how many e-mails were tested correctly and some other important statistics.

This archive contains the differences between log likelihoods for the testing e-mails. Please compare these with the differences computed by your program. It is very useful to check them, because if yours are not computed correctly, your program is very likely to fail in some of the tests.

Suggestions and hints

Arithmetic underflow: Because the number of words is very large, the probabilities dealt with are very small. When multiplied together, the values may become so small they cannot be represented. The solution is to use logarithms. The formula for classification becomes:

ci = argmax (log pi + sum (log P(wj|ci))). 

The multiplications become additions because log(a*b)=log(a)+log(b).
Memory: You need to be slightly careful about how you store some information while preprocessing.
If you plan to do Part 3 it might be necessary to implement Naive Bayes yourself or pick an implementation that is easy to understand and modify.
Remember to submit a pretrained spam classifier! We won't be training it on any data before attempting classification (hence 30 minutes time limit).
IMPORTANT: Your program should work with the automatic marking system. If it does not, because your program does not follow the specification above, YOU WILL NOT GAIN ANY POINTS, hence you will fail the assignment!!!
Submission

Submit your code, any files it uses and your report via the online submission system.

The report should be in pdf format and should describe the work that you have done.

Part 1 should not exceed 1000 words (roughly 2 pages + figures); part 2: another 1000 words; part 3: another 2000 words (roughly another 4 pages + figures).

